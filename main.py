# main.py
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

# Extremization & calibration controls
EXTREMIZE_THRESHOLD = float(os.getenv("EXTREMIZE_THRESHOLD", "0.60"))     # legacy threshold
EXTREMIZE_ALPHA = float(os.getenv("EXTREMIZE_ALPHA", "1.35"))            # default gentler alpha (ROI: less overconfidence)
EXTREMIZE_ALPHA_STRONG = float(os.getenv("EXTREMIZE_ALPHA_STRONG", "1.80"))
EXTREMIZE_DISPERSION_STD_MAX = float(os.getenv("EXTREMIZE_DISPERSION_STD_MAX", "0.06"))  # extremize only if committee agrees

# Crowd/model shrinkage (ROI: improves calibration)
CROWD_BLEND_WEAK = float(os.getenv("CROWD_BLEND_WEAK", "0.45"))    # weight on model when evidence weak
CROWD_BLEND_MIXED = float(os.getenv("CROWD_BLEND_MIXED", "0.65"))
CROWD_BLEND_STRONG = float(os.getenv("CROWD_BLEND_STRONG", "0.80"))

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
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float) -> float:
    p = clamp01(p)
    return float(np.log(p / (1.0 - p)))


def extremize_probability(p: float, alpha: float) -> float:
    """Extremize probability using logit scaling: p' = sigmoid(alpha * logit(p))."""
    return clamp01(sigmoid(alpha * logit(p)))


def should_extremize(p: float, threshold: float = EXTREMIZE_THRESHOLD) -> bool:
    """Legacy: extremize if p >= threshold or p <= 1-threshold."""
    return (p >= threshold) or (p <= (1.0 - threshold))


def is_research_sufficient(research_by_source: Dict[str, str]) -> bool:
    """Research is sufficient if at least one source has meaningful non-error content."""
    if not research_by_source:
        return False

    def good(txt: str) -> bool:
        if not txt:
            return False
        low = txt.lower()
        # Treat pure failures as bad
        if ("failed:" in low or "error:" in low or "timeout" in low) and len(txt.strip()) < 200:
            return False
        return len(txt.strip()) > 120

    return any(good(v) for v in research_by_source.values())


def interpolate_missing_percentiles(reported: list[Percentile], target_percentiles: list[float]) -> list[Percentile]:
    """Interpolate missing percentiles using linear interpolation on available ones."""
    if not reported:
        return [Percentile(percentile=p, value=0.0) for p in target_percentiles]

    sorted_rep = sorted(reported, key=lambda x: x.percentile)
    xs = [p.percentile for p in sorted_rep]
    ys = [p.value for p in sorted_rep]

    interpolated: List[Percentile] = []
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
        interpolated.append(Percentile(percentile=tp, value=float(val)))
    return interpolated


def enforce_numeric_constraints(percentiles: list[Percentile], question: NumericQuestion) -> list[Percentile]:
    """Enforce bounds and monotonicity."""
    lower = -np.inf if getattr(question, "open_lower_bound", False) else getattr(question, "lower_bound", None)
    upper = np.inf if getattr(question, "open_upper_bound", False) else getattr(question, "upper_bound", None)

    if lower is None:
        lower = getattr(question, "nominal_lower_bound", None)
    if upper is None:
        upper = getattr(question, "nominal_upper_bound", None)

    if lower is None:
        lower = -np.inf
    if upper is None:
        upper = np.inf

    bounded: List[Percentile] = []
    for p in percentiles:
        val = float(p.value)
        val = max(lower, min(upper, val))
        bounded.append(Percentile(percentile=float(p.percentile), value=float(val)))

    sorted_by_p = sorted(bounded, key=lambda x: x.percentile)
    values = [p.value for p in sorted_by_p]
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            values[i] = values[i - 1]

    return [Percentile(percentile=sorted_by_p[i].percentile, value=float(values[i])) for i in range(len(values))]


def derive_numeric_fallback_bounds(question: NumericQuestion, anchor: Optional[float]) -> Tuple[float, float]:
    """
    ROI: numeric questions often have open bounds; never default to 0..1.
    Prefer (closed bounds) > (nominal bounds) > (anchor-based bounds) > wide generic.
    """
    lb = getattr(question, "lower_bound", None)
    ub = getattr(question, "upper_bound", None)

    if lb is None:
        lb = getattr(question, "nominal_lower_bound", None)
    if ub is None:
        ub = getattr(question, "nominal_upper_bound", None)

    open_lb = getattr(question, "open_lower_bound", False)
    open_ub = getattr(question, "open_upper_bound", False)

    # If bounds are present but open flags are True, still treat as None for clipping range
    if open_lb:
        lb = None
    if open_ub:
        ub = None

    if lb is not None and ub is not None and ub > lb:
        return float(lb), float(ub)

    if isinstance(anchor, (int, float)):
        a = float(anchor)
        if a > 0:
            return float(a * 0.25), float(a * 3.0)
        return float(a - 1.0), float(a + 1.0)

    # last resort: extremely wide, but finite
    return -1e9, 1e9


def format_research_block(research_by_source: Dict[str, str]) -> str:
    """Create a single research string with explicit source sections."""
    blocks: List[str] = []
    for src in ["tavily", "newsapi"]:
        if src in research_by_source:
            blocks.append(f"--- SOURCE {src.upper()} ---\n{research_by_source[src]}\n")
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
    High-ROI tournament bot:
    - Fixes ForecastBot integration: run_research returns str only (prevents ResearchWithPredictions validation error)
    - Research: Tavily + NewsAPI (publishedAt), synthesized by GPT-5.2
    - Robust structured outputs: forecasters emit JSON ONLY (prevents "No JSON found" parser failures)
    - Binary: committee aggregation in logit space + (optional) extremization gated by agreement + evidence
      and crowd/model shrinkage blending (improves calibration/log score)
    - Numeric: reliable percentile JSON + sensible fallbacks (never 0..1 for open bounds)
    """

    _max_concurrent_questions = 3
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "summarizer": "openrouter/openai/gpt-5",
            "researcher": "openrouter/openai/gpt-5.2",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)

        # Store meta without breaking ForecastBot schema
        self._research_meta: Dict[str, Dict[str, Any]] = {}

    # -----------------------------
    # Multi-Source Research (Tavily + NewsAPI) + GPT-5.2 synthesis
    # -----------------------------
    def call_tavily(self, query: str) -> str:
        if not getattr(self.tavily_client, "api_key", None):
            return ""
        try:
            response = self.tavily_client.search(query=query, search_depth="advanced")
            results = response.get("results", []) or []
            lines: List[str] = []
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
            # ROI: recency is usually more important than "relevancy" for Metaculus
            articles = self.newsapi_client.get_everything(
                q=query,
                language="en",
                sort_by="publishedAt",
                page_size=8,
            )
            arts = (articles or {}).get("articles", []) or []
            lines: List[str] = []
            for a in arts[:6]:
                title = (a.get("title") or "").strip()
                desc = (a.get("description") or "").strip()
                url = (a.get("url") or "").strip()
                published = (a.get("publishedAt") or "").strip()
                if title or desc:
                    lines.append(
                        f"- Title: {title}\n"
                        f"  Published: {published or 'N/A'}\n"
                        f"  Snippet: {desc or 'N/A'}\n"
                        f"  Source: {url or 'N/A'}"
                    )
            return "\n".join(lines).strip()
        except Exception as e:
            return f"NewsAPI failed: {e}"

    async def _run_research_impl(self, question: MetaculusQuestion) -> Tuple[str, Dict[str, str], List[str]]:
        async with self._concurrency_limiter:
            q = getattr(question, "question_text", "") or ""

            tav = await with_timeout(asyncio.to_thread(self.call_tavily, q), RESEARCH_TIMEOUT_S, "tavily")
            nws = await with_timeout(asyncio.to_thread(self.call_newsapi, q), RESEARCH_TIMEOUT_S, "newsapi")

            research_by_source = {"tavily": tav, "newsapi": nws}

            # Only count as "used" if it has meaningful content
            def meaningful(txt: str) -> bool:
                if not txt:
                    return False
                low = txt.lower()
                if "failed:" in low or "error:" in low or "timeout" in low:
                    return False
                return len(txt.strip()) > 120

            searchers_used = [s for s in ["tavily", "newsapi"] if meaningful(research_by_source.get(s, ""))]

            raw_block = format_research_block(research_by_source)
            sufficient = is_research_sufficient(research_by_source)

            if not sufficient:
                synthesized = "(Insufficient recent research found via Tavily/NewsAPI. Relying more on base rates and general knowledge.)"
                return (synthesized + ("\n\n" + raw_block if raw_block else "")).strip(), research_by_source, searchers_used

            researcher_llm = self.get_llm("researcher", "llm")
            synth_prompt = clean_indents(f"""
            You are an expert research synthesizer for forecasting.

            Task: Summarize evidence relevant to the question. Be concrete, cautious, and include dates when present.
            Each key fact MUST end with [TAVILY] or [NEWSAPI] or both.
            If evidence is weak/outdated, say so.

            Question: {q}

            Raw research:
            {raw_block}

            Output format (plain text):
            - Key facts (bullets with dates where possible; end each with [TAVILY]/[NEWSAPI])
            - What remains uncertain (bullets)
            - Likely drivers going forward (bullets)
            Keep it under ~2200 characters.
            """).strip()

            synthesized = await with_timeout(researcher_llm.invoke(synth_prompt), LLM_TIMEOUT_S, "research_synthesis")
            synthesized = (synthesized or "").strip()
            if not synthesized:
                synthesized = "(Research synthesis returned empty. Relying more on base rates and general knowledge.)"
            return synthesized, research_by_source, searchers_used

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        ForecastBot expects a STRING research_report.
        We store meta on self._research_meta keyed by QID for logging.
        """
        synthesized, research_by_source, searchers_used = await self._run_research_impl(question)
        qid = extract_question_id(question)
        self._research_meta[qid] = {
            "synthesized": synthesized,
            "raw": research_by_source,
            "searchers_used": searchers_used,
        }
        # Optional: log type info to debug misclassification
        logger.info(
            f"Q{qid} class={type(question).__name__} "
            f"options={getattr(question,'options',None)} "
            f"bounds=({getattr(question,'lower_bound',None)},{getattr(question,'upper_bound',None)}) "
            f"open_bounds=({getattr(question,'open_lower_bound',None)},{getattr(question,'open_upper_bound',None)})"
        )
        return synthesized

    # -----------------------------
    # Forecasting core (no shared-state mutation)
    # -----------------------------
    async def _single_forecast(self, question, research: str, model_override: str = None):
        """
        Returns (result, raw_text).
        IMPORTANT: does NOT mutate shared llm state.
        Outputs are JSON-only to avoid parser failures (ROI).
        """
        llm = GeneralLlm(model=model_override) if model_override else self.get_llm("default", "llm")
        parser_llm = self.get_llm("parser", "llm")

        base_rate = safe_community_prediction(question)
        if base_rate is not None:
            if isinstance(question, BinaryQuestion):
                base_rate_str = f"Community prediction (anchor): {base_rate:.3f} (decimal probability)"
            else:
                base_rate_str = f"Community prediction (anchor, scalar): {base_rate:,.6g}"
        else:
            base_rate_str = "No reliable community anchor available."

        today_utc = datetime.utcnow().strftime("%Y-%m-%d")

        if isinstance(question, BinaryQuestion):
            prompt = clean_indents(f"""
            You are a tournament forecaster optimizing for log score.
            Use base rates as anchors, but update with evidence.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}

            {base_rate_str}

            Research summary:
            {research}

            Today (UTC): {today_utc}

            Output JSON ONLY. No extra text.
            Schema: {{"prediction_in_decimal": number}}
            Constraints:
            - 0.01 <= prediction_in_decimal <= 0.99
            """).strip()

            raw = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "binary_llm")
            pred: BinaryPrediction = await structure_output(raw, BinaryPrediction, model=parser_llm)
            result = clamp01(float(pred.prediction_in_decimal))
            return result, str(raw)

        if isinstance(question, MultipleChoiceQuestion):
            prompt = clean_indents(f"""
            You are a tournament forecaster optimizing for log score.

            Question: {question.question_text}
            Options (exact strings): {question.options}
            Background: {question.background_info}
            Resolution: {question.resolution_criteria}

            {base_rate_str}

            Research summary:
            {research}

            Today (UTC): {today_utc}

            Output JSON ONLY. No extra text.
            Schema:
            {{
              "predicted_options": [
                {{"option":"<exact option>","probability": number}},
                ...
              ]
            }}
            Constraints:
            - Include ALL options exactly once (match exact strings).
            - Probabilities sum to 1.
            - No probability is exactly 0 unless logically impossible.
            """).strip()

            raw = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "mc_llm")
            result = await structure_output(
                raw,
                PredictedOptionList,
                model=parser_llm,
                additional_instructions=f"Options must be exactly: {question.options}"
            )
            return result, str(raw)

        if isinstance(question, NumericQuestion):
            lower_ref = getattr(question, "lower_bound", None)
            upper_ref = getattr(question, "upper_bound", None)
            nlb = getattr(question, "nominal_lower_bound", None)
            nub = getattr(question, "nominal_upper_bound", None)
            lower_msg = f"Lower bound: {'open' if question.open_lower_bound else 'closed'} at {lower_ref if lower_ref is not None else nlb}"
            upper_msg = f"Upper bound: {'open' if question.open_upper_bound else 'closed'} at {upper_ref if upper_ref is not None else nub}"

            prompt = clean_indents(f"""
            You are a tournament forecaster optimizing for log score.
            Provide a calibrated distribution (avoid overconfidence).

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

            Output JSON ONLY. No extra text.
            Schema: a JSON array of objects with keys:
              - "percentile": one of [0.1,0.2,0.4,0.6,0.8,0.9]
              - "value": number
            Constraints:
            - Values must be non-decreasing with percentile.
            - Respect bounds if they exist.
            """).strip()

            raw = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "num_llm")
            percentile_list: list[Percentile] = await structure_output(raw, list[Percentile], model=parser_llm)

            target_ps = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            interpolated = interpolate_missing_percentiles(percentile_list, target_ps)
            validated = enforce_numeric_constraints(interpolated, question)

            # sanity: if everything is identical (degenerate), widen slightly within bounds
            vals = [p.value for p in validated]
            if len(set([round(v, 12) for v in vals])) == 1:
                lb, ub = derive_numeric_fallback_bounds(question, base_rate)
                mid = float(vals[0])
                width = (ub - lb) * 0.08 if np.isfinite(ub - lb) else max(1.0, abs(mid) * 0.25)
                widened = [
                    Percentile(0.1, mid - 1.2 * width),
                    Percentile(0.2, mid - 0.7 * width),
                    Percentile(0.4, mid - 0.2 * width),
                    Percentile(0.6, mid + 0.2 * width),
                    Percentile(0.8, mid + 0.7 * width),
                    Percentile(0.9, mid + 1.2 * width),
                ]
                validated = enforce_numeric_constraints(widened, question)

            result = NumericDistribution.from_question(validated, question)
            return result, str(raw)

        raise ValueError(f"Unsupported question type: {type(question)}")

    # -----------------------------
    # Binary: committee + agreement-gated extremization + crowd shrinkage
    # -----------------------------
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-5.1",
            "openrouter/anthropic/claude-sonnet-4.5",
        ]

        qid = extract_question_id(question)
        base = safe_community_prediction(question)
        base_rate_text = f"{base:.1%}" if isinstance(base, (int, float)) else "None"

        # evidence strength label (used for blending + extremization)
        evidence_strength: Literal["STRONG", "MIXED", "WEAK"] = "MIXED"
        try:
            researcher_llm = self.get_llm("researcher", "llm")
            strength_prompt = clean_indents(f"""
            Rate evidence strength for this forecast as exactly one token: STRONG, MIXED, or WEAK.

            STRONG: multiple credible signals, little plausible reversal
            MIXED: some evidence but meaningful uncertainty remains
            WEAK: thin/noisy/dated evidence

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}

            Research summary:
            {research}
            """).strip()
            resp = await with_timeout(researcher_llm.invoke(strength_prompt), 30, "evidence_strength")
            resp = (resp or "").strip().upper()
            if resp in ("STRONG", "MIXED", "WEAK"):
                evidence_strength = resp  # type: ignore
        except Exception as e:
            logger.debug(f"Evidence strength check failed for Q{qid}: {e}")

        # Collect committee forecasts
        raw_ps: List[float] = []
        model_notes: List[str] = []

        for model in models:
            try:
                p, raw = await self._single_forecast(question, research, model_override=model)
                p = float(p)
                raw_ps.append(clamp01(p))
                model_notes.append(f"- model={model} p={p:.3f}")
            except Exception as e:
                logger.warning(f"Model {model} failed on binary Q{qid}: {e}")
                fallback = clamp01(base if isinstance(base, (int, float)) else 0.5)
                raw_ps.append(fallback)
                model_notes.append(f"- model={model} FALLBACK p={fallback:.3f} err={e}")

        # Aggregate in logit space (better behaved)
        logits = np.array([logit(p) for p in raw_ps], dtype=float)
        logit_med = float(np.median(logits))
        p_agg = clamp01(sigmoid(logit_med))

        # Agreement metric
        p_std = float(np.std(raw_ps))
        agree = p_std <= EXTREMIZE_DISPERSION_STD_MAX

        # Optional extremization: only if (a) committee agrees AND (b) p is away from 0.5 AND (c) evidence not WEAK
        alpha = EXTREMIZE_ALPHA_STRONG if evidence_strength == "STRONG" else EXTREMIZE_ALPHA
        p_final = p_agg
        did_ext = False
        if agree and evidence_strength != "WEAK" and should_extremize(p_agg, EXTREMIZE_THRESHOLD):
            p_final = extremize_probability(p_agg, alpha=alpha)
            did_ext = True

        # Crowd shrinkage blending (ROI)
        if isinstance(base, (int, float)):
            w = CROWD_BLEND_MIXED
            if evidence_strength == "WEAK":
                w = CROWD_BLEND_WEAK
            elif evidence_strength == "STRONG":
                w = CROWD_BLEND_STRONG

            # also penalize w when models disagree
            if not agree:
                w = max(0.35, w - 0.20)

            p_final = clamp01(w * p_final + (1.0 - w) * float(base))

        forecast_text = f"{p_final:.1%}"

        # searchers used from cached meta (more accurate than string matching)
        meta = self._research_meta.get(qid, {})
        searchers_used = meta.get("searchers_used", []) if isinstance(meta.get("searchers_used", None), list) else []

        how_text = clean_indents(f"""
        - 3-model committee; aggregated in logit space (median of logits).
        - Agreement check: std={p_std:.3f} (extremize only if <= {EXTREMIZE_DISPERSION_STD_MAX:.2f}).
        - Extremization: {'YES' if did_ext else 'NO'} (threshold={EXTREMIZE_THRESHOLD:.2f}, alpha={alpha:.2f}, evidence={evidence_strength}).
        - Crowd/model blending: {'YES' if isinstance(base,(int,float)) else 'NO'} (evidence={evidence_strength}, agreement={agree}).
        """).strip()

        comment = build_comment(
            question=question,
            forecast_text=forecast_text,
            base_rate_text=base_rate_text,
            how_text=how_text + "\n\n### Model notes\n" + "\n".join(model_notes[:9]),
            searchers_used=searchers_used,
            models_used=models,
        )

        log_forecast_for_calibration(
            question,
            p_final,
            comment,
            models,
            True if searchers_used else False,
            searchers_used,
        )
        return ReasonedPrediction(prediction_value=p_final, reasoning=comment)

    # -----------------------------
    # Multiple choice: committee median by option key + prob floor
    # -----------------------------
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-5.1",
            "openrouter/anthropic/claude-sonnet-4.5",
        ]

        qid = extract_question_id(question)
        base = safe_community_prediction(question)
        base_rate_text = f"{base:,.4g}" if isinstance(base, (int, float)) else "None"

        per_model_maps: List[Dict[str, float]] = []
        model_notes: List[str] = []

        for model in models:
            try:
                pred, raw = await self._single_forecast(question, research, model_override=model)
                m: Dict[str, float] = {}
                for item in pred.predicted_options:
                    opt = item["option"]
                    prob = float(item["probability"])
                    m[opt] = max(0.0, prob)
                for opt in question.options:
                    m.setdefault(opt, 0.0)

                s = sum(m.values())
                if s <= 0:
                    n = len(question.options)
                    m = {opt: 1.0 / n for opt in question.options}
                else:
                    for k in list(m.keys()):
                        m[k] = float(m[k] / s)

                per_model_maps.append(m)
                model_notes.append(f"- model={model} ok")
            except Exception as e:
                logger.warning(f"Model {model} failed on MC Q{qid}: {e}")
                n = len(question.options)
                per_model_maps.append({opt: 1.0 / n for opt in question.options})
                model_notes.append(f"- model={model} FALLBACK err={e}")

        option_list = list(question.options)
        mat = np.array([[m[opt] for opt in option_list] for m in per_model_maps], dtype=float)
        med = np.median(mat, axis=0)

        # ROI: small probability floor to reduce catastrophic log loss when surprised
        n = len(option_list)
        floor = max(1e-6, 0.01 / n)
        med = np.maximum(med, floor)
        med = med / med.sum()

        median_forecast = PredictedOptionList([
            {"option": opt, "probability": float(p)} for opt, p in zip(option_list, med)
        ])
        forecast_text = ", ".join([f"{o['option']}: {o['probability']:.1%}" for o in median_forecast.predicted_options])

        meta = self._research_meta.get(qid, {})
        searchers_used = meta.get("searchers_used", []) if isinstance(meta.get("searchers_used", None), list) else []

        how_text = clean_indents(f"""
        - 3-model committee; median probability per option.
        - Applied probability floor={floor:.6f} then renormalized.
        """).strip()

        comment = build_comment(
            question=question,
            forecast_text=forecast_text,
            base_rate_text=base_rate_text,
            how_text=how_text + "\n\n### Model notes\n" + "\n".join(model_notes[:9]),
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
    # Numeric: committee median per percentile + safe fallbacks
    # -----------------------------
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-5.1",
            "openrouter/anthropic/claude-sonnet-4.5",
        ]

        qid = extract_question_id(question)
        base = safe_community_prediction(question)
        base_rate_text = f"{base:,.4g}" if isinstance(base, (int, float)) else "None"

        target_percentiles = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        per_model_percentiles: List[List[Percentile]] = []
        model_notes: List[str] = []

        for model in models:
            try:
                dist, raw = await self._single_forecast(question, research, model_override=model)
                declared = list(getattr(dist, "declared_percentiles", [])) or []
                interpolated = interpolate_missing_percentiles(declared, target_percentiles)
                validated = enforce_numeric_constraints(interpolated, question)
                per_model_percentiles.append(validated)
                model_notes.append(f"- model={model} ok")
            except Exception as e:
                logger.warning(f"Model {model} failed on numeric Q{qid}: {e}")

                lb, ub = derive_numeric_fallback_bounds(question, base)
                if isinstance(base, (int, float)):
                    center = float(base)
                    width = (ub - lb) * 0.25
                else:
                    center = (lb + ub) / 2.0
                    width = (ub - lb) * 0.35

                fallback_vals = [
                    center - width * 0.9,
                    center - width * 0.5,
                    center - width * 0.15,
                    center + width * 0.15,
                    center + width * 0.5,
                    center + width * 0.9,
                ]
                # clip to bounds
                fallback_vals = [max(lb, min(ub, v)) for v in fallback_vals]
                fallback_ps = [Percentile(percentile=p, value=v) for p, v in zip(target_percentiles, fallback_vals)]
                per_model_percentiles.append(enforce_numeric_constraints(fallback_ps, question))
                model_notes.append(f"- model={model} FALLBACK bounds=({lb:.3g},{ub:.3g}) err={e}")

        # Median per percentile
        aggregated: List[Percentile] = []
        for idx, p in enumerate(target_percentiles):
            vals = [pm[idx].value for pm in per_model_percentiles]
            aggregated.append(Percentile(percentile=p, value=float(np.median(vals))))

        validated = enforce_numeric_constraints(aggregated, question)
        distribution = NumericDistribution.from_question(validated, question)

        forecast_text = ", ".join([f"p{int(p.percentile*100)}={p.value:,.6g}" for p in validated])

        meta = self._research_meta.get(qid, {})
        searchers_used = meta.get("searchers_used", []) if isinstance(meta.get("searchers_used", None), list) else []

        how_text = clean_indents(f"""
        - 3-model committee.
        - Forced percentiles onto {', '.join(str(int(p*100)) for p in target_percentiles)} via interpolation,
          enforced bounds + monotonicity, then took median value per percentile.
        - Numeric fallbacks never assume 0..1; they derive bounds from question/nominals/anchor.
        """).strip()

        comment = build_comment(
            question=question,
            forecast_text=forecast_text,
            base_rate_text=base_rate_text,
            how_text=how_text + "\n\n### Model notes\n" + "\n".join(model_notes[:9]),
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
            reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
            all_reports.extend(reports)

        bot.log_report_summary(all_reports)
        logger.info(f"Run completed. Calibration logs saved to {CALIBRATION_LOG_FILE}")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
