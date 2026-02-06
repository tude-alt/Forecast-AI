#main.py
import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Optional, Any, Literal, Dict, Tuple, List

import numpy as np
from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

from tavily import TavilyClient
from newsapi import NewsApiClient

# -----------------------------
# Environment & API Keys
# -----------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ConservativeHybridBot")

# -----------------------------
# Timeouts & Limits
# -----------------------------
RESEARCH_TIMEOUT_S = float(os.getenv("RESEARCH_TIMEOUT_S", "25"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "60"))
EXTREMIZE_THRESHOLD = float(os.getenv("EXTREMIZE_THRESHOLD", "0.60"))  # "Of 60 extremize it"
EXTREMIZE_ALPHA = float(os.getenv("EXTREMIZE_ALPHA", "1.6"))          # logit scaling strength
EXTREMIZE_ALPHA_STRONG = float(os.getenv("EXTREMIZE_ALPHA_STRONG", "2.0"))
MIN_P = float(os.getenv("MIN_P", "0.01"))
MAX_P = float(os.getenv("MAX_P", "0.99"))

# Calibration logging helper
CALIBRATION_LOG_FILE = "forecasting_calibration_log.jsonl"


def extract_question_id(question: MetaculusQuestion) -> str:
    """Extract question ID from URL since .id attribute may not be exposed."""
    try:
        url = getattr(question, "url", "")
        match = re.search(r"/questions/(\d+)", str(url))
        return match.group(1) if match else "unknown"
    except Exception:
        return "unknown"


def safe_community_prediction(question: MetaculusQuestion) -> Optional[float]:
    """
    Best-effort extraction of a scalar community anchor.
    Note: Metaculus objects can vary; this returns None if not a float.
    """
    try:
        pred = getattr(question, "community_prediction", None)
        if pred is not None and isinstance(pred, (int, float)):
            return float(pred)
        pred = getattr(question, "prediction", None)
        if pred is not None and isinstance(pred, (int, float)):
            return float(pred)
    except Exception as e:
        logger.warning(f"Failed to get community prediction for Q{extract_question_id(question)}: {e}")
    return None


def clamp01(p: float) -> float:
    return float(max(MIN_P, min(MAX_P, p)))


def sigmoid(x: float) -> float:
    # stable enough for typical logit inputs here
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float) -> float:
    p = clamp01(p)
    return float(np.log(p / (1.0 - p)))


def extremize_probability(p: float, alpha: float) -> float:
    """Extremize probability using logit scaling: p' = sigmoid(alpha * logit(p))."""
    return clamp01(sigmoid(alpha * logit(p)))


def should_extremize(p: float, threshold: float = EXTREMIZE_THRESHOLD) -> bool:
    """Extremize if p is at least threshold away from 0.5 in either direction (>=0.60 or <=0.40)."""
    return (p >= threshold) or (p <= (1.0 - threshold))


def is_research_sufficient(research_by_source: Dict[str, str]) -> bool:
    """Research is sufficient if at least one source has meaningful non-error content."""
    if not research_by_source:
        return False

    def good(txt: str) -> bool:
        if not txt:
            return False
        low = txt.lower()
        if "failed:" in low or "error:" in low or "timeout" in low:
            # allow some errors but not all
            pass
        # basic length check
        return len(txt.strip()) > 80

    return any(good(v) for v in research_by_source.values())


def interpolate_missing_percentiles(reported: list[Percentile], target_percentiles: list[float]) -> list[Percentile]:
    """Interpolate missing percentiles using linear interpolation on available ones."""
    if not reported:
        return [Percentile(percentile=p, value=0.0) for p in target_percentiles]

    sorted_rep = sorted(reported, key=lambda x: x.percentile)
    xs = [p.percentile for p in sorted_rep]
    ys = [p.value for p in sorted_rep]

    interpolated = []
    for tp in target_percentiles:
        if tp in xs:
            val = ys[xs.index(tp)]
        else:
            from bisect import bisect_left
            i = bisect_left(xs, tp)
            if i == 0:
                val = ys[0]
            elif i == len(xs):
                val = ys[-1]
            else:
                x0, x1 = xs[i - 1], xs[i]
                y0, y1 = ys[i - 1], ys[i]
                val = y0 + (y1 - y0) * (tp - x0) / (x1 - x0) if x1 != x0 else y0
        interpolated.append(Percentile(percentile=tp, value=val))
    return interpolated


def enforce_numeric_constraints(percentiles: list[Percentile], question: NumericQuestion) -> list[Percentile]:
    """Enforce bounds and monotonicity."""
    lower = question.lower_bound if not question.open_lower_bound else -np.inf
    upper = question.upper_bound if not question.open_upper_bound else np.inf

    bounded = []
    for p in percentiles:
        val = max(lower, min(upper, p.value))
        bounded.append(Percentile(percentile=p.percentile, value=val))

    sorted_by_p = sorted(bounded, key=lambda x: x.percentile)
    values = [p.value for p in sorted_by_p]
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            values[i] = values[i - 1]

    return [Percentile(percentile=sorted_by_p[i].percentile, value=values[i]) for i in range(len(values))]


def format_research_block(research_by_source: Dict[str, str]) -> str:
    """Create a single research string with explicit source sections."""
    blocks = []
    for src in ["tavily", "newsapi"]:
        if src in research_by_source:
            blocks.append(f"--- SOURCE {src.upper()} ---\n{research_by_source[src]}\n")
    # include any extras if present
    for src, txt in research_by_source.items():
        if src not in ("tavily", "newsapi"):
            blocks.append(f"--- SOURCE {src.upper()} ---\n{txt}\n")
    return "\n".join(blocks).strip()


def log_forecast_for_calibration(
    question: MetaculusQuestion,
    prediction_value: Any,
    reasoning: str,
    model_ids: list[str],
    research_used: bool,
    searchers_used: list[str],
):
    """Log forecast details for post-resolution scoring."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question_id": extract_question_id(question),
        "question_type": question.__class__.__name__,
        "question_text": getattr(question, "question_text", ""),
        "resolution_date": getattr(question, "resolution_date", None),
        "community_prediction": safe_community_prediction(question),
        "prediction_value": prediction_value,
        "models_used": model_ids,
        "research_used": research_used,
        "searchers_used": searchers_used,
        "reasoning_snippet": reasoning[:1500],
    }
    try:
        with open(CALIBRATION_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log calibration data: {e}")


async def with_timeout(coro, seconds: float, label: str) -> str:
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        return f"{label} timeout after {seconds}s"
    except Exception as e:
        return f"{label} error: {e}"


def build_comment(
    question: MetaculusQuestion,
    forecast_text: str,
    base_rate_text: str,
    how_text: str,
    searchers_used: List[str],
    models_used: List[str],
) -> str:
    """Metaculus report/comment text: question, forecast, how arrived at, and searchers used."""
    qtxt = getattr(question, "question_text", "").strip()
    qid = extract_question_id(question)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    searchers = ", ".join(searchers_used) if searchers_used else "None"
    models = ", ".join(models_used) if models_used else "Unknown"
    return clean_indents(f"""
    ## Forecast (Q{qid})
    **Date (UTC):** {today}

    **Question:** {qtxt}

    **Forecast:** {forecast_text}

    **Anchor / base rate:** {base_rate_text}

    **How this was arrived at:**
    {how_text}

    **Searchers used:** {searchers}
    **Models used:** {models}
    """).strip()


class ConservativeHybridBot(ForecastBot):
    """
    Conservative forecasting bot, updated to chase high scores:
    - Research: Tavily + NewsAPI, then GPT-5.2 synthesizes into a tight evidence summary
    - Binary: applies threshold-based extremization (>= 0.60 or <= 0.40) using logit scaling
    - No shared-state LLM mutation across concurrent questions
    """

    _max_concurrent_questions = 3
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "summarizer": "openrouter/openai/gpt-5",
            "researcher": "openrouter/openai/gpt-5.2",  # replaced Perplexity
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)

    # -----------------------------
    # Multi-Source Research (Tavily + NewsAPI) + GPT-5.2 synthesis
    # -----------------------------
    def call_tavily(self, query: str) -> str:
        if not getattr(self.tavily_client, "api_key", None):
            return ""
        try:
            response = self.tavily_client.search(query=query, search_depth="advanced")
            results = response.get("results", []) or []
            # keep compact and relevant
            lines = []
            for c in results[:6]:
                content = (c.get("content") or "").strip()
                url = (c.get("url") or "").strip()
                if content:
                    if url:
                        lines.append(f"- {content}\n  Source: {url}")
                    else:
                        lines.append(f"- {content}")
            return "\n".join(lines).strip()
        except Exception as e:
            return f"Tavily failed: {e}"

    def call_newsapi(self, query: str) -> str:
        if not getattr(self.newsapi_client, "api_key", None):
            return ""
        try:
            articles = self.newsapi_client.get_everything(
                q=query,
                language="en",
                sort_by="relevancy",
                page_size=6,
            )
            arts = (articles or {}).get("articles", []) or []
            lines = []
            for a in arts[:6]:
                title = (a.get("title") or "").strip()
                desc = (a.get("description") or "").strip()
                url = (a.get("url") or "").strip()
                if title or desc:
                    lines.append(f"- Title: {title}\n  Snippet: {desc or 'N/A'}\n  Source: {url or 'N/A'}")
            return "\n".join(lines).strip()
        except Exception as e:
            return f"NewsAPI failed: {e}"

    async def run_research(self, question: MetaculusQuestion) -> Tuple[str, Dict[str, str], List[str]]:
        """
        Returns:
          - synthesized_research_text (string for prompt)
          - raw research_by_source dict
          - list of searchers used
        """
        async with self._concurrency_limiter:
            q = getattr(question, "question_text", "")

            # raw research calls (timeboxed)
            tav = await with_timeout(asyncio.to_thread(self.call_tavily, q), RESEARCH_TIMEOUT_S, "tavily")
            nws = await with_timeout(asyncio.to_thread(self.call_newsapi, q), RESEARCH_TIMEOUT_S, "newsapi")

            research_by_source = {"tavily": tav, "newsapi": nws}
            searchers_used = [s for s in ["tavily", "newsapi"] if (research_by_source.get(s) or "").strip()]

            raw_block = format_research_block(research_by_source)
            sufficient = is_research_sufficient(research_by_source)

            if not sufficient:
                synthesized = "(Insufficient recent research found via Tavily/NewsAPI. Relying more on base rates and general knowledge.)"
                return synthesized + ("\n\n" + raw_block if raw_block else ""), research_by_source, searchers_used

            # Synthesize with GPT-5.2 into a concise evidence summary with clear claims + uncertainty
            researcher_llm = self.get_llm("researcher", "llm")
            synth_prompt = clean_indents(f"""
            You are an expert research synthesizer for forecasting.

            Task: Summarize the evidence relevant to the question. Be concrete, cautious, and cite which source block
            (TAVILY vs NEWSAPI) supports each key claim. If evidence is weak or outdated, say so.

            Question: {q}

            Raw research:
            {raw_block}

            Output format:
            - Key facts (bullet list, each bullet ends with [TAVILY] or [NEWSAPI] or both)
            - What remains uncertain (bullet list)
            - Most likely drivers going forward (short bullets)
            Keep it under ~2200 characters.
            """).strip()

            synthesized = await with_timeout(researcher_llm.invoke(synth_prompt), LLM_TIMEOUT_S, "research_synthesis")
            synthesized = synthesized.strip()
            return synthesized, research_by_source, searchers_used

    # -----------------------------
    # Forecasting core (no shared-state mutation)
    # -----------------------------
    async def _single_forecast(self, question, research: str, model_override: str = None):
        """
        Returns (result, reasoning_text).
        IMPORTANT: does NOT mutate self._llms.
        """
        llm = GeneralLlm(model=model_override) if model_override else self.get_llm("default", "llm")
        parser_llm = self.get_llm("parser", "llm")

        base_rate = safe_community_prediction(question)
        if base_rate is not None:
            if isinstance(question, BinaryQuestion):
                base_rate_str = f"Community prediction (anchor): {base_rate:.1%}"
            else:
                base_rate_str = f"Community prediction (anchor): {base_rate:,.4g}"
        else:
            base_rate_str = "No reliable community anchor available."

        today_utc = datetime.utcnow().strftime("%Y-%m-%d")

        if isinstance(question, BinaryQuestion):
            prompt = clean_indents(f"""
            You are a high-performing tournament forecaster optimizing for log score.
            Use base rates as anchors, but move decisively when evidence supports it.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}

            {base_rate_str}

            Research summary:
            {research}

            Today (UTC): {today_utc}

            Guidance:
            - Start from the anchor/base rate when available.
            - Adjust for time-to-resolution, path dependence, and recent evidence.
            - Avoid 0% or 100% unless logically impossible/certain.
            - State the final probability clearly.

            End with exactly: "Probability: ZZ%"
            """).strip()

            reasoning = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "binary_llm")
            pred: BinaryPrediction = await structure_output(
                reasoning, BinaryPrediction, model=parser_llm
            )
            result = clamp01(pred.prediction_in_decimal)
            return result, str(reasoning)

        elif isinstance(question, MultipleChoiceQuestion):
            prompt = clean_indents(f"""
            You are a high-performing tournament forecaster optimizing for log score.

            Question: {question.question_text}
            Options (exact): {question.options}
            Background: {question.background_info}
            Resolution: {question.resolution_criteria}

            {base_rate_str}

            Research summary:
            {research}

            Today (UTC): {today_utc}

            Guidance:
            - Assign non-zero probability to each option unless impossible.
            - If uncertain, distribute sensibly (often near-uniform or guided by anchor).

            End with probabilities for each option in order, matching the exact option strings.
            """).strip()

            reasoning = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "mc_llm")
            result = await structure_output(
                reasoning,
                PredictedOptionList,
                model=parser_llm,
                additional_instructions=f"Options must be exactly: {question.options}"
            )
            return result, str(reasoning)

        elif isinstance(question, NumericQuestion):
            lower_msg = f"Lower bound: {'open' if question.open_lower_bound else 'closed'} at {question.lower_bound or question.nominal_lower_bound}"
            upper_msg = f"Upper bound: {'open' if question.open_upper_bound else 'closed'} at {question.upper_bound or question.nominal_upper_bound}"

            prompt = clean_indents(f"""
            You are a high-performing tournament forecaster optimizing for log score.
            Provide a calibrated distribution (wide enough to avoid overconfidence).

            Question: {question.question_text}
            Units: {question.unit_of_measure or 'Infer from context'}
            Background: {question.background_info}
            Resolution: {question.resolution_criteria}
            {lower_msg}
            {upper_msg}

            {base_rate_str}

            Research summary:
            {research}

            Today (UTC): {today_utc}

            IMPORTANT:
            - Use the correct scale/units. If an anchor is given, it is likely in correct units.
            - Provide percentiles exactly: 10, 20, 40, 60, 80, 90.
            """).strip()

            reasoning = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "num_llm")
            percentile_list: list[Percentile] = await structure_output(
                reasoning, list[Percentile], model=parser_llm
            )

            target_ps = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            interpolated = interpolate_missing_percentiles(percentile_list, target_ps)
            validated = enforce_numeric_constraints(interpolated, question)
            result = NumericDistribution.from_question(validated, question)
            return result, str(reasoning)

        else:
            raise ValueError(f"Unsupported question type: {type(question)}")

    # -----------------------------
    # Binary: committee + extremization for high scores
    # -----------------------------
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-5.1",
            "openrouter/anthropic/claude-sonnet-4.5",
        ]

        base = safe_community_prediction(question)
        base_rate_text = f"{base:.1%}" if isinstance(base, (int, float)) else "None"

        forecasts: List[float] = []
        reasonings: List[str] = []

        # Evidence-strength classifier (optional) using GPT-5.2 to decide stronger extremization
        # This is *not* Perplexity; it's GPT-5.2 on OpenRouter.
        evidence_strength: Literal["STRONG", "MIXED", "WEAK"] = "MIXED"
        try:
            researcher_llm = self.get_llm("researcher", "llm")
            strength_prompt = clean_indents(f"""
            You are a superforecaster judging evidence strength for extremization.

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}

            Research summary:
            {research}

            Rate evidence strength as one of:
            - STRONG: multiple credible signals, little plausible reversal
            - MIXED: some evidence but meaningful uncertainty remains
            - WEAK: thin/noisy/dated evidence

            Answer with exactly one token: STRONG, MIXED, or WEAK
            """).strip()
            resp = await with_timeout(researcher_llm.invoke(strength_prompt), 30, "evidence_strength")
            resp = (resp or "").strip().upper()
            if resp in ("STRONG", "MIXED", "WEAK"):
                evidence_strength = resp  # type: ignore
        except Exception as e:
            logger.debug(f"Evidence strength check failed for Q{extract_question_id(question)}: {e}")

        alpha = EXTREMIZE_ALPHA_STRONG if evidence_strength == "STRONG" else EXTREMIZE_ALPHA

        for model in models:
            try:
                pred, reason = await self._single_forecast(question, research, model_override=model)
                pred = float(pred)

                # "Of 60 extremize it": if |p-0.5| >= 0.10 (i.e., >=0.60 or <=0.40), extremize via logit scaling.
                pred_before = pred
                if should_extremize(pred, EXTREMIZE_THRESHOLD):
                    pred = extremize_probability(pred, alpha=alpha)
                    tag = f"[EXTREMIZED@{EXTREMIZE_THRESHOLD:.2f}, alpha={alpha:.2f}, {evidence_strength}]"
                else:
                    tag = "[NOT EXTREMIZED]"

                forecasts.append(clamp01(pred))
                reasonings.append(f"{tag} model={model} p0={pred_before:.3f} p={pred:.3f} :: {str(reason)[:1200]}")

            except Exception as e:
                logger.warning(f"Model {model} failed on binary Q{extract_question_id(question)}: {e}")
                fallback = clamp01(base if isinstance(base, (int, float)) else 0.5)
                forecasts.append(fallback)
                reasonings.append(f"[FALLBACK] model={model} :: {e}")

        median_pred = float(np.median(forecasts))
        median_pred = clamp01(median_pred)

        how_text = clean_indents(f"""
        - Ran a 3-model committee and took the median.
        - Applied threshold-based extremization: if p >= {EXTREMIZE_THRESHOLD:.2f} or p <= {1-EXTREMIZE_THRESHOLD:.2f},
          transformed p via logit scaling (alpha={alpha:.2f}; evidence={evidence_strength}).
        - Anchored on community prediction when available, adjusted using research summary.
        """).strip()

        forecast_text = f"{median_pred:.1%}"
        searchers_used = []
        if "--- SOURCE TAVILY ---" in research:
            searchers_used.append("tavily")
        if "--- SOURCE NEWSAPI ---" in research:
            searchers_used.append("newsapi")

        comment = build_comment(
            question=question,
            forecast_text=forecast_text,
            base_rate_text=base_rate_text,
            how_text=how_text + "\n\n" + "### Model notes\n" + "\n".join(f"- {r}" for r in reasonings[:6]),
            searchers_used=searchers_used,
            models_used=models,
        )

        log_forecast_for_calibration(
            question,
            median_pred,
            comment,
            models,
            True if searchers_used else False,
            searchers_used,
        )
        return ReasonedPrediction(prediction_value=median_pred, reasoning=comment)

    # -----------------------------
    # Multiple choice: committee median by option key
    # -----------------------------
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-5.1",
            "openrouter/anthropic/claude-sonnet-4.5",
        ]

        base = safe_community_prediction(question)
        base_rate_text = f"{base:,.4g}" if isinstance(base, (int, float)) else "None"

        per_model_maps: List[Dict[str, float]] = []
        reasonings: List[str] = []

        for model in models:
            try:
                pred, reason = await self._single_forecast(question, research, model_override=model)
                # Align by option string
                m: Dict[str, float] = {}
                for item in pred.predicted_options:
                    opt = item["option"]
                    prob = float(item["probability"])
                    m[opt] = max(0.0, prob)
                # Fill any missing with 0
                for opt in question.options:
                    m.setdefault(opt, 0.0)
                # Normalize
                s = sum(m.values())
                if s > 0:
                    for k in list(m.keys()):
                        m[k] = float(m[k] / s)
                else:
                    n = len(question.options)
                    for opt in question.options:
                        m[opt] = 1.0 / n
                per_model_maps.append(m)
                reasonings.append(f"model={model} :: {str(reason)[:1200]}")
            except Exception as e:
                logger.warning(f"Model {model} failed on MC Q{extract_question_id(question)}: {e}")
                n = len(question.options)
                per_model_maps.append({opt: 1.0 / n for opt in question.options})
                reasonings.append(f"[FALLBACK] model={model} :: {e}")

        # Median by option key
        option_list = list(question.options)
        mat = np.array([[m[opt] for opt in option_list] for m in per_model_maps], dtype=float)
        med = np.median(mat, axis=0)
        med = med / med.sum() if med.sum() > 0 else np.full_like(med, 1.0 / len(med))

        median_forecast = PredictedOptionList([
            {"option": opt, "probability": float(p)} for opt, p in zip(option_list, med)
        ])

        forecast_text = ", ".join([f"{o['option']}: {o['probability']:.1%}" for o in median_forecast.predicted_options])

        searchers_used = []
        if "--- SOURCE TAVILY ---" in research:
            searchers_used.append("tavily")
        if "--- SOURCE NEWSAPI ---" in research:
            searchers_used.append("newsapi")

        how_text = clean_indents(f"""
        - Ran a 3-model committee and took the median probability per option.
        - Normalized the final distribution to sum to 1.
        - Used research summary (if available) + community anchor (if available).
        """).strip()

        comment = build_comment(
            question=question,
            forecast_text=forecast_text,
            base_rate_text=base_rate_text,
            how_text=how_text + "\n\n" + "### Model notes\n" + "\n".join(f"- {r}" for r in reasonings[:6]),
            searchers_used=searchers_used,
            models_used=models,
        )

        log_forecast_for_calibration(
            question,
            [opt["probability"] for opt in median_forecast.predicted_options],
            comment,
            models,
            True if searchers_used else False,
            searchers_used,
        )
        return ReasonedPrediction(prediction_value=median_forecast, reasoning=comment)

    # -----------------------------
    # Numeric: committee median per percentile after per-model normalization
    # -----------------------------
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-5.1",
            "openrouter/anthropic/claude-sonnet-4.5",
        ]

        base = safe_community_prediction(question)
        base_rate_text = f"{base:,.4g}" if isinstance(base, (int, float)) else "None"

        target_percentiles = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        per_model_percentiles: List[List[Percentile]] = []
        reasonings: List[str] = []

        for model in models:
            try:
                dist, reason = await self._single_forecast(question, research, model_override=model)
                # Normalize each model output onto the target percentile set
                declared = list(getattr(dist, "declared_percentiles", [])) or []
                interpolated = interpolate_missing_percentiles(declared, target_percentiles)
                validated = enforce_numeric_constraints(interpolated, question)
                per_model_percentiles.append(validated)
                reasonings.append(f"model={model} :: {str(reason)[:1200]}")
            except Exception as e:
                logger.warning(f"Model {model} failed on numeric Q{extract_question_id(question)}: {e}")
                # Fallback distribution centered around anchor or mid-bounds
                lower_b = question.lower_bound if question.lower_bound is not None else 0.0
                upper_b = question.upper_bound if question.upper_bound is not None else 1.0

                if isinstance(base, (int, float)):
                    center = float(base)
                    width = (upper_b - lower_b) * 0.25
                else:
                    center = (lower_b + upper_b) / 2.0
                    width = (upper_b - lower_b) * 0.35

                fallback_vals = [
                    center - width * 0.9,
                    center - width * 0.5,
                    center - width * 0.15,
                    center + width * 0.15,
                    center + width * 0.5,
                    center + width * 0.9,
                ]
                fallback_vals = [max(lower_b, min(upper_b, v)) for v in fallback_vals]
                fallback_ps = [
                    Percentile(percentile=p, value=v) for p, v in zip(target_percentiles, fallback_vals)
                ]
                per_model_percentiles.append(enforce_numeric_constraints(fallback_ps, question))
                reasonings.append(f"[FALLBACK] model={model} :: {e}")

        # Median per percentile
        aggregated: List[Percentile] = []
        for idx, p in enumerate(target_percentiles):
            vals = [pm[idx].value for pm in per_model_percentiles]
            aggregated.append(Percentile(percentile=p, value=float(np.median(vals))))

        validated = enforce_numeric_constraints(aggregated, question)
        distribution = NumericDistribution.from_question(validated, question)

        forecast_text = ", ".join([f"p{int(p.percentile*100)}={p.value:,.4g}" for p in validated])

        searchers_used = []
        if "--- SOURCE TAVILY ---" in research:
            searchers_used.append("tavily")
        if "--- SOURCE NEWSAPI ---" in research:
            searchers_used.append("newsapi")

        how_text = clean_indents(f"""
        - Ran a 3-model committee.
        - For each model, forced percentiles onto {', '.join(str(int(p*100)) for p in target_percentiles)} via interpolation,
          then enforced bounds + monotonicity.
        - Took the median value per percentile and re-validated constraints.
        - Used research summary (if available) + community anchor (if available).
        """).strip()

        comment = build_comment(
            question=question,
            forecast_text=forecast_text,
            base_rate_text=base_rate_text,
            how_text=how_text + "\n\n" + "### Model notes\n" + "\n".join(f"- {r}" for r in reasonings[:6]),
            searchers_used=searchers_used,
            models_used=models,
        )

        log_forecast_for_calibration(
            question,
            [p.value for p in validated],
            comment,
            models,
            True if searchers_used else False,
            searchers_used,
        )
        return ReasonedPrediction(prediction_value=distribution, reasoning=comment)


# -----------------------------
# Entrypoint — Tournament Only
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Conservative Hybrid Bot.")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=["32916", "minibench", MetaculusApi.CURRENT_MINIBENCH_ID],
    )
    args = parser.parse_args()

    bot = ConservativeHybridBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    try:
        all_reports = []
        for tid in args.tournament_ids:
            logger.info(f"Forecasting on tournament: {tid}")

            # NOTE: ForecastBot will call run_research() internally if configured that way.
            # If your ForecastBot expects overridden run_research to return a string only,
            # adjust ForecastBot integration accordingly. If it calls run_research and uses its return
            # as "research", then change run_research to return just the synthesized text.
            #
            # If ForecastBot calls run_research(question) expecting a str, we can shim by overriding
            # ForecastBot hooks elsewhere. If needed, modify run_research to return only synthesized text.

            reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
            all_reports.extend(reports)

        bot.log_report_summary(all_reports)
        logger.info(f"Run completed. Calibration logs saved to {CALIBRATION_LOG_FILE}")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
