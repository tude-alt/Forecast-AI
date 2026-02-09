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
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "70"))  # slightly higher for 5.2 JSON reliability

# Extremization & calibration controls
EXTREMIZE_THRESHOLD = float(os.getenv("EXTREMIZE_THRESHOLD", "0.60"))
EXTREMIZE_ALPHA = float(os.getenv("EXTREMIZE_ALPHA", "1.35"))
EXTREMIZE_ALPHA_STRONG = float(os.getenv("EXTREMIZE_ALPHA_STRONG", "1.80"))
EXTREMIZE_DISPERSION_STD_MAX = float(os.getenv("EXTREMIZE_DISPERSION_STD_MAX", "0.06"))

# Crowd/model shrinkage
CROWD_BLEND_WEAK = float(os.getenv("CROWD_BLEND_WEAK", "0.45"))
CROWD_BLEND_MIXED = float(os.getenv("CROWD_BLEND_MIXED", "0.65"))
CROWD_BLEND_STRONG = float(os.getenv("CROWD_BLEND_STRONG", "0.80"))

MIN_P = float(os.getenv("MIN_P", "0.01"))
MAX_P = float(os.getenv("MAX_P", "0.99"))

# Never forecast without research: require at least one meaningful source
REQUIRE_RESEARCH = os.getenv("REQUIRE_RESEARCH", "true").lower() in ("1", "true", "yes")

# Calibration logging helper
CALIBRATION_LOG_FILE = "forecasting_calibration_log.jsonl"


def extract_question_id(question: MetaculusQuestion) -> str:
    try:
        url = getattr(question, "url", "")
        match = re.search(r"/questions/(\d+)", str(url))
        return match.group(1) if match else "unknown"
    except Exception:
        return "unknown"


def safe_community_prediction(question: MetaculusQuestion) -> Optional[float]:
    """
    Best-effort extraction of a scalar community anchor.
    For binary this is typically a probability (0..1).
    For numeric/MC it may be non-scalar; we ignore those.
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
    return clamp01(sigmoid(alpha * logit(p)))


def should_extremize(p: float, threshold: float = EXTREMIZE_THRESHOLD) -> bool:
    return (p >= threshold) or (p <= (1.0 - threshold))


def is_meaningful_research_text(txt: str) -> bool:
    if not txt:
        return False
    low = txt.lower()
    if "failed:" in low or "error:" in low or "timeout" in low:
        return False
    return len(txt.strip()) > 120


def is_research_sufficient(research_by_source: Dict[str, str]) -> bool:
    if not research_by_source:
        return False
    return any(is_meaningful_research_text(v) for v in research_by_source.values())


def interpolate_missing_percentiles(reported: list[Percentile], target_percentiles: list[float]) -> list[Percentile]:
    if not reported:
        return [Percentile(percentile=p, value=0.0) for p in target_percentiles]

    sorted_rep = sorted(reported, key=lambda x: x.percentile)
    xs = [float(p.percentile) for p in sorted_rep]
    ys = [float(p.value) for p in sorted_rep]

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
        interpolated.append(Percentile(percentile=float(tp), value=float(val)))
    return interpolated


def enforce_numeric_constraints(percentiles: list[Percentile], question: NumericQuestion) -> list[Percentile]:
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
        v = float(p.value)
        v = max(lower, min(upper, v))
        bounded.append(Percentile(percentile=float(p.percentile), value=float(v)))

    sorted_by_p = sorted(bounded, key=lambda x: x.percentile)
    values = [p.value for p in sorted_by_p]
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            values[i] = values[i - 1]

    return [Percentile(percentile=sorted_by_p[i].percentile, value=float(values[i])) for i in range(len(values))]


def derive_numeric_fallback_bounds(question: NumericQuestion, anchor: Optional[float]) -> Tuple[float, float]:
    lb = getattr(question, "lower_bound", None)
    ub = getattr(question, "upper_bound", None)

    if lb is None:
        lb = getattr(question, "nominal_lower_bound", None)
    if ub is None:
        ub = getattr(question, "nominal_upper_bound", None)

    if getattr(question, "open_lower_bound", False):
        lb = None
    if getattr(question, "open_upper_bound", False):
        ub = None

    if lb is not None and ub is not None and float(ub) > float(lb):
        return float(lb), float(ub)

    if isinstance(anchor, (int, float)):
        a = float(anchor)
        if a > 0:
            return a * 0.25, a * 3.0
        return a - 1.0, a + 1.0

    return -1e9, 1e9


def format_research_block(research_by_source: Dict[str, str]) -> str:
    blocks: List[str] = []
    for src in ["tavily", "newsapi"]:
        if src in research_by_source and research_by_source[src]:
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
    Tournament bot (high-ROI):
    - run_research returns str only; meta stored separately
    - Research pipeline:
        1) GPT-5.2 generates query expansions ("web search brain")
        2) Tavily + NewsAPI runs on expanded queries
        3) GPT-5.2 synthesizes evidence summary
      => never forecast without at least one meaningful research source
    - Forecasters output JSON-only (stable parsing)
    - No GPT-5 as forecaster (removed); GPT-5.2 is the primary forecaster
    - Good Judgment principles + base rates: prompts + crowd/model blending
    """

    _max_concurrent_questions = 3
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            # Forecasters: gpt-5.2 primary
            "default": "openrouter/openai/gpt-5.2",
            "parser": "openrouter/openai/gpt-4.1-mini",
            # Researcher/synth/query-expander: gpt-5.2
            "researcher": "openrouter/openai/gpt-5.2",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)
        self._research_meta: Dict[str, Dict[str, Any]] = {}

    # -----------------------------
    # Research helpers
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
                    lines.append(f"- {content}\n  Source: {url or 'N/A'}")
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
                sort_by="publishedAt",  # recency-first for forecasting
                page_size=10,
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

    async def _expand_queries_with_gpt52(self, question_text: str) -> List[str]:
        """
        Use GPT-5.2 to propose search queries that improve recall/precision.
        Returns a short list of distinct query strings.
        """
        researcher_llm = self.get_llm("researcher", "llm")
        prompt = clean_indents(f"""
        Generate 5 concise web search queries to research this forecasting question.
        Mix: (1) exact entities, (2) key event terms, (3) likely resolution criteria terms, (4) newest updates.
        Avoid quotes unless needed. Do NOT include explanations.

        Output JSON ONLY: {{"queries":["...","..."]}}

        Question: {question_text}
        """).strip()
        raw = await with_timeout(researcher_llm.invoke(prompt), 35, "query_expand")
        try:
            data = json.loads(raw)
            qs = data.get("queries", [])
            qs = [str(x).strip() for x in qs if str(x).strip()]
            # include original question text as first query
            merged = [question_text.strip()] + qs
            # unique while preserving order
            seen = set()
            out = []
            for q in merged:
                key = q.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(q)
            return out[:6]
        except Exception:
            # fallback: just original question
            return [question_text.strip()]

    async def _run_research_impl(self, question: MetaculusQuestion) -> Tuple[str, Dict[str, str], List[str]]:
        async with self._concurrency_limiter:
            qtxt = (getattr(question, "question_text", "") or "").strip()
            if not qtxt:
                return "", {}, []

            # 1) GPT-5.2 query expansion ("web search brain")
            expanded_queries = await self._expand_queries_with_gpt52(qtxt)

            # 2) Run Tavily + NewsAPI across top queries (cap to avoid rate limits)
            tav_parts: List[str] = []
            nws_parts: List[str] = []

            # try up to 3 queries max for each tool to control spend/latency
            for i, q in enumerate(expanded_queries[:3]):
                tav = await with_timeout(asyncio.to_thread(self.call_tavily, q), RESEARCH_TIMEOUT_S, f"tavily_{i}")
                if tav and "failed:" not in tav.lower() and "error:" not in tav.lower():
                    tav_parts.append(f"## Query: {q}\n{tav}")

                nws = await with_timeout(asyncio.to_thread(self.call_newsapi, q), RESEARCH_TIMEOUT_S, f"newsapi_{i}")
                if nws and "failed:" not in nws.lower() and "error:" not in nws.lower():
                    nws_parts.append(f"## Query: {q}\n{nws}")

            tav_all = "\n\n".join(tav_parts).strip()
            nws_all = "\n\n".join(nws_parts).strip()

            research_by_source = {"tavily": tav_all, "newsapi": nws_all}
            searchers_used = [s for s in ["tavily", "newsapi"] if is_meaningful_research_text(research_by_source.get(s, ""))]

            sufficient = is_research_sufficient(research_by_source)
            raw_block = format_research_block(research_by_source)

            if REQUIRE_RESEARCH and not sufficient:
                # never forecast without research
                raise RuntimeError(f"Insufficient research for Q{extract_question_id(question)}; refusing to forecast.")

            # 3) GPT-5.2 synthesis
            researcher_llm = self.get_llm("researcher", "llm")
            synth_prompt = clean_indents(f"""
            You are an expert research synthesizer for forecasting.

            Task: Summarize evidence relevant to the question.
            Requirements:
            - Prioritize RECENT developments (include dates if present).
            - Each key fact MUST end with [TAVILY] or [NEWSAPI] or both.
            - Separate facts from speculation.
            - Explicitly list what would change the forecast.

            Question: {qtxt}

            Raw research:
            {raw_block}

            Output (plain text):
            - Key facts (bullets; include dates; end tags)
            - What remains uncertain (bullets)
            - Likely drivers / signposts (bullets)
            - What would change the forecast (bullets)
            Keep it under ~2400 characters.
            """).strip()

            synthesized = await with_timeout(researcher_llm.invoke(synth_prompt), LLM_TIMEOUT_S, "research_synthesis")
            synthesized = (synthesized or "").strip()
            if REQUIRE_RESEARCH and not synthesized:
                raise RuntimeError(f"Empty synthesis for Q{extract_question_id(question)}; refusing to forecast.")

            return synthesized, research_by_source, searchers_used

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        ForecastBot expects a STRING for research_report.
        Store meta separately.
        """
        synthesized, research_by_source, searchers_used = await self._run_research_impl(question)
        qid = extract_question_id(question)
        self._research_meta[qid] = {
            "synthesized": synthesized,
            "raw": research_by_source,
            "searchers_used": searchers_used,
        }
        logger.info(
            f"Q{qid} class={type(question).__name__} "
            f"bounds=({getattr(question,'lower_bound',None)},{getattr(question,'upper_bound',None)}) "
            f"open_bounds=({getattr(question,'open_lower_bound',None)},{getattr(question,'open_upper_bound',None)}) "
            f"searchers_used={searchers_used}"
        )
        return synthesized

    # -----------------------------
    # Forecasting core (JSON-only outputs)
    # -----------------------------
    async def _single_forecast(self, question, research: str, model_override: str = None):
        llm = GeneralLlm(model=model_override) if model_override else self.get_llm("default", "llm")
        parser_llm = self.get_llm("parser", "llm")

        # Never forecast without research (belt & suspenders)
        if REQUIRE_RESEARCH and (not research or len(research.strip()) < 80):
            raise RuntimeError(f"Missing/insufficient research text for Q{extract_question_id(question)}; refusing to forecast.")

        base_rate = safe_community_prediction(question)
        if base_rate is not None:
            if isinstance(question, BinaryQuestion):
                base_rate_str = f"Community prediction (anchor): {base_rate:.4f} (decimal probability)"
            else:
                base_rate_str = f"Community scalar anchor (if meaningful): {base_rate:,.6g}"
        else:
            base_rate_str = "No reliable community anchor available."

        today_utc = datetime.utcnow().strftime("%Y-%m-%d")

        good_judgment_guidance = clean_indents("""
        Good Judgment principles:
        - Start with an outside view (base rate/reference class), then update with evidence (inside view).
        - Consider time-to-resolution and path dependence.
        - Weigh evidence by credibility + recency; discount stale/noisy signals.
        - Prefer calibrated uncertainty: avoid needless overconfidence.
        """).strip()

        if isinstance(question, BinaryQuestion):
            prompt = clean_indents(f"""
            You are a tournament forecaster optimizing for log score.

            {good_judgment_guidance}

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}

            {base_rate_str}

            Research summary (must use this; do not guess beyond it):
            {research}

            Today (UTC): {today_utc}

            Output JSON ONLY. No extra text.
            Schema: {{"prediction_in_decimal": number}}
            Constraints:
            - 0.01 <= prediction_in_decimal <= 0.99
            """).strip()

            raw = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "binary_llm")
            pred: BinaryPrediction = await structure_output(raw, BinaryPrediction, model=parser_llm)
            return clamp01(float(pred.prediction_in_decimal)), str(raw)

        if isinstance(question, MultipleChoiceQuestion):
            prompt = clean_indents(f"""
            You are a tournament forecaster optimizing for log score.

            {good_judgment_guidance}

            Question: {question.question_text}
            Options (exact strings): {question.options}
            Background: {question.background_info}
            Resolution: {question.resolution_criteria}

            {base_rate_str}

            Research summary (must use this; do not guess beyond it):
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

            {good_judgment_guidance}

            Question: {question.question_text}
            Units: {question.unit_of_measure or 'Infer from context'}
            Background: {question.background_info}
            Resolution: {question.resolution_criteria}
            {lower_msg}
            {upper_msg}

            {base_rate_str}

            Research summary (must use this; do not guess beyond it):
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

            # If degenerate, widen within fallback bounds
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

            return NumericDistribution.from_question(validated, question), str(raw)

        raise ValueError(f"Unsupported question type: {type(question)}")

    # -----------------------------
    # Binary: committee + agreement-gated extremization + crowd shrinkage
    # -----------------------------
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        # Remove GPT-5; add GPT-5.2
        models = [
            "openrouter/openai/gpt-5.2",
            "openrouter/openai/gpt-5.1",
            "openrouter/anthropic/claude-sonnet-4.5",
        ]

        qid = extract_question_id(question)
        base = safe_community_prediction(question)
        base_rate_text = f"{base:.1%}" if isinstance(base, (int, float)) else "None"

        evidence_strength: Literal["STRONG", "MIXED", "WEAK"] = "MIXED"
        try:
            researcher_llm = self.get_llm("researcher", "llm")
            strength_prompt = clean_indents(f"""
            Rate evidence strength for this forecast as exactly one token: STRONG, MIXED, or WEAK.

            STRONG: multiple credible recent signals, little plausible reversal
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

        raw_ps: List[float] = []
        model_notes: List[str] = []

        for model in models:
            try:
                p, _raw = await self._single_forecast(question, research, model_override=model)
                p = float(p)
                raw_ps.append(clamp01(p))
                model_notes.append(f"- model={model} p={p:.3f}")
            except Exception as e:
                logger.warning(f"Model {model} failed on binary Q{qid}: {e}")
                # if research exists but model fails, fall back to crowd if available, else 0.5
                fallback = clamp01(float(base) if isinstance(base, (int, float)) else 0.5)
                raw_ps.append(fallback)
                model_notes.append(f"- model={model} FALLBACK p={fallback:.3f} err={e}")

        logits = np.array([logit(p) for p in raw_ps], dtype=float)
        p_agg = clamp01(sigmoid(float(np.median(logits))))

        p_std = float(np.std(raw_ps))
        agree = p_std <= EXTREMIZE_DISPERSION_STD_MAX

        alpha = EXTREMIZE_ALPHA_STRONG if evidence_strength == "STRONG" else EXTREMIZE_ALPHA
        p_final = p_agg
        did_ext = False
        if agree and evidence_strength != "WEAK" and should_extremize(p_agg, EXTREMIZE_THRESHOLD):
            p_final = extremize_probability(p_agg, alpha=alpha)
            did_ext = True

        # Blend with crowd anchor (base rate) if available
        if isinstance(base, (int, float)):
            w = CROWD_BLEND_MIXED
            if evidence_strength == "WEAK":
                w = CROWD_BLEND_WEAK
            elif evidence_strength == "STRONG":
                w = CROWD_BLEND_STRONG
            if not agree:
                w = max(0.35, w - 0.20)
            p_final = clamp01(w * p_final + (1.0 - w) * float(base))

        forecast_text = f"{p_final:.1%}"

        meta = self._research_meta.get(qid, {})
        searchers_used = meta.get("searchers_used", []) if isinstance(meta.get("searchers_used", None), list) else []

        how_text = clean_indents(f"""
        - Base rates + Good Judgment updating: start from crowd anchor when available; update using research evidence.
        - 3-model committee; aggregated in logit space (median logits).
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
    # Multiple choice: committee median by option key + probability floor
    # -----------------------------
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        models = [
            "openrouter/openai/gpt-5.2",
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
                pred, _raw = await self._single_forecast(question, research, model_override=model)
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
        - Base rates + Good Judgment updating: use research-driven inside-view adjustments with conservative uncertainty.
        - 3-model committee; median probability per option.
        - Applied probability floor={floor:.6f} then renormalized (reduces catastrophic log loss on surprises).
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
            "openrouter/openai/gpt-5.2",
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
                dist, _raw = await self._single_forecast(question, research, model_override=model)
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
                fallback_vals = [max(lb, min(ub, v)) for v in fallback_vals]
                fallback_ps = [Percentile(percentile=p, value=v) for p, v in zip(target_percentiles, fallback_vals)]
                per_model_percentiles.append(enforce_numeric_constraints(fallback_ps, question))
                model_notes.append(f"- model={model} FALLBACK bounds=({lb:.3g},{ub:.3g}) err={e}")

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
        - Base rates + Good Judgment updating: combine outside-view anchoring with research-driven updates.
        - 3-model committee; interpolate to fixed percentiles then enforce bounds + monotonicity; median per percentile.
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
        default=[
            "32916",
            "minibench",
            "market-pulse-26q1",
            MetaculusApi.CURRENT_MINIBENCH_ID,
        ],
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
