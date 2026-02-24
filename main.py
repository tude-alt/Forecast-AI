import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from typing import Optional, Any, Dict, Tuple, List

import numpy as np
from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
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

# Use Tavily for semantic search
from tavily import TavilyClient

# TinyFish is used as a live web automation API for surfacing recent news.  The
# TinyFish API does not have a dedicated news endpoint, but we can instruct an
# agent to search for recent articles and return structured results.  See the
# documentation example in TinyFish's Quick Start, which shows how to issue a
# request via `POST https://agent.tinyfish.ai/v1/automation/run-sse` with a
# `goal` describing the desired extraction【450391332860218†L107-L126】.
import requests
import urllib.parse

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TINYFISH_API_KEY = os.getenv("TINYFISH_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")  # retained for backward compatibility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Tude")

RESEARCH_TIMEOUT_S = float(os.getenv("RESEARCH_TIMEOUT_S", "25"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "70"))

MAX_CONCURRENT_QUESTIONS = int(os.getenv("MAX_CONCURRENT_QUESTIONS", "1"))
PUBLISH_SLEEP_S = float(os.getenv("PUBLISH_SLEEP_S", "3.0"))
TOURNAMENT_SLEEP_S = float(os.getenv("TOURNAMENT_SLEEP_S", "8.0"))
RETRY_MAX = int(os.getenv("RETRY_MAX", "6"))
RETRY_BASE_S = float(os.getenv("RETRY_BASE_S", "2.0"))
RETRY_MAX_S = float(os.getenv("RETRY_MAX_S", "60.0"))

EXTREMIZE_THRESHOLD = float(os.getenv("EXTREMIZE_THRESHOLD", "0.60"))
EXTREMIZE_ALPHA = float(os.getenv("EXTREMIZE_ALPHA", "1.35"))
EXTREMIZE_DISPERSION_STD_MAX = float(os.getenv("EXTREMIZE_DISPERSION_STD_MAX", "0.06"))

CROWD_BLEND_WEAK = float(os.getenv("CROWD_BLEND_WEAK", "0.45"))
CROWD_BLEND_MIXED = float(os.getenv("CROWD_BLEND_MIXED", "0.65"))

MIN_P = float(os.getenv("MIN_P", "0.01"))
MAX_P = float(os.getenv("MAX_P", "0.99"))

REQUIRE_RESEARCH = os.getenv("REQUIRE_RESEARCH", "true").lower() in ("1", "true", "yes")
CALIBRATION_LOG_FILE = "forecasting_calibration_log.jsonl"


def extract_question_id(question: MetaculusQuestion) -> str:
    """Extract a numeric question identifier from various attributes of a question."""
    try:
        qid = getattr(question, "id", None)
        if isinstance(qid, (int, str)) and str(qid).isdigit():
            return str(qid)
    except Exception:
        pass
    try:
        url = str(getattr(question, "url", "") or "")
        m = re.search(r"/questions/(\d+)(?:/|$)", url)
        if m:
            return m.group(1)
    except Exception:
        pass
    try:
        s = str(question)
        m = re.search(r"/questions/(\d+)(?:/|$)", s)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "unknown"


def safe_community_prediction(question: MetaculusQuestion) -> Optional[float]:
    """Get the community prediction from a Metaculus question if available."""
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
    """Heuristically decide if a research string contains meaningful content."""
    if not txt:
        return False
    low = txt.lower()
    if "failed:" in low or "error:" in low or "timeout" in low:
        return False
    return len(txt.strip()) > 120


def is_research_sufficient(research_by_source: Dict[str, str]) -> bool:
    return any(is_meaningful_research_text(v) for v in (research_by_source or {}).values())


def interpolate_missing_percentiles(reported: list[Percentile], target_percentiles: list[float]) -> list[Percentile]:
    if not reported:
        return [Percentile(percentile=p, value=0.0) for p in target_percentiles]
    sorted_rep = sorted(reported, key=lambda x: x.percentile)
    xs = [float(p.percentile) for p in sorted_rep]
    ys = [float(p.value) for p in sorted_rep]
    out: List[Percentile] = []
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
        out.append(Percentile(percentile=float(tp), value=float(val)))
    return out


def _finite(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except Exception:
        return False


def _safe_lb_ub(question: NumericQuestion) -> Tuple[float, float]:
    lb = getattr(question, "lower_bound", None)
    ub = getattr(question, "upper_bound", None)

    if getattr(question, "open_lower_bound", False):
        lb = None
    if getattr(question, "open_upper_bound", False):
        ub = None

    if not _finite(lb):
        lb = getattr(question, "nominal_lower_bound", None)
    if not _finite(ub):
        ub = getattr(question, "nominal_upper_bound", None)

    if _finite(lb) and _finite(ub) and float(ub) > float(lb):
        return float(lb), float(ub)

    if _finite(lb) and not _finite(ub):
        return float(lb), float(lb) + max(1.0, abs(float(lb)) * 0.5)
    if _finite(ub) and not _finite(lb):
        return float(ub) - max(1.0, abs(float(ub)) * 0.5), float(ub)

    return 0.0, 1.0


def enforce_numeric_constraints(percentiles: list[Percentile], question: NumericQuestion) -> list[Percentile]:
    lb, ub = _safe_lb_ub(question)

    bounded: List[Percentile] = []
    for p in percentiles:
        v = float(p.value)
        if not np.isfinite(v):
            v = lb
        v = max(lb, min(ub, v))
        bounded.append(Percentile(percentile=float(p.percentile), value=float(v)))

    srt = sorted(bounded, key=lambda x: float(x.percentile))
    for i in range(1, len(srt)):
        if srt[i].value < srt[i - 1].value:
            srt[i].value = srt[i - 1].value

    if len(srt) >= 2 and srt[-1].value > ub:
        srt[-1].value = ub
    if len(srt) >= 2 and srt[0].value < lb:
        srt[0].value = lb

    return [Percentile(percentile=float(p.percentile), value=float(p.value)) for p in srt]


def derive_numeric_fallback_percentiles(question: NumericQuestion, anchor: Optional[float]) -> List[Percentile]:
    lb, ub = _safe_lb_ub(question)
    if not np.isfinite(ub - lb) or ub <= lb:
        lb, ub = 0.0, 1.0

    if isinstance(anchor, (int, float)) and np.isfinite(float(anchor)):
        a = float(anchor)
        if a < lb:
            a = lb + 0.35 * (ub - lb)
        elif a > ub:
            a = lb + 0.65 * (ub - lb)
        center = a
    else:
        center = lb + 0.5 * (ub - lb)

    span = max(1e-6, (ub - lb))
    w = 0.18 * span
    vals = [
        max(lb, min(ub, center - 1.2 * w)),
        max(lb, min(ub, center - 0.7 * w)),
        max(lb, min(ub, center - 0.2 * w)),
        max(lb, min(ub, center + 0.2 * w)),
        max(lb, min(ub, center + 0.7 * w)),
        max(lb, min(ub, center + 1.2 * w)),
    ]
    pcts = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    out = [Percentile(percentile=p, value=v) for p, v in zip(pcts, vals)]
    out = enforce_numeric_constraints(out, question)
    return out


def format_research_block(research_by_source: Dict[str, str]) -> str:
    """Format the raw research by source into a single text block for summarization."""
    blocks: List[str] = []
    # Only include sources that actually returned data.  Iterate in a stable order.
    for src in ["tavily", "tinyfish"]:
        txt = (research_by_source or {}).get(src, "") or ""
        if txt.strip():
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
    """Record a calibration entry so the bot can be assessed over time."""
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
    """Run a coroutine with a timeout and return a human-readable status on failure."""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        return f"{label} timeout after {seconds}s"
    except Exception as e:
        return f"{label} error: {e}"


def backoff_sleep(attempt: int) -> None:
    base = min(RETRY_MAX_S, RETRY_BASE_S * (2 ** attempt))
    jitter = random.uniform(0.0, base * 0.25)
    time.sleep(base + jitter)


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


class Tude(ForecastBot):
    """
    Tude is a Metaculus forecasting bot that uses research from Tavily and TinyFish
    to update base rates using Good Judgment style heuristics.  It aggregates
    predictions from multiple language models to provide calibrated estimates.
    """
    _max_concurrent_questions = MAX_CONCURRENT_QUESTIONS
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5.2",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "researcher": "openrouter/openai/gpt-5.2",
            "summarizer": "openrouter/openai/gpt-5.2",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        # Save TinyFish API key for later use.  If not provided, the research
        # pipeline will silently skip TinyFish.
        self.tinyfish_api_key = TINYFISH_API_KEY
        self._research_meta: Dict[str, Dict[str, Any]] = {}

    # TinyFish integration ---------------------------------------------------
    def call_tinyfish(self, query: str) -> str:
        """
        Use TinyFish's automation API to search for recent news articles related to a
        query and return a formatted string of bullet points.  The function
        constructs a natural-language goal to instruct the agent to pull top
        articles from a news site and extract title, snippet, date and link.  If
        no API key is configured or an error occurs, an empty string or
        descriptive error is returned.
        """
        if not self.tinyfish_api_key:
            return ""
        try:
            # Use Google News as a starting page for searching the query.  We rely
            # on TinyFish to navigate and extract relevant articles according to
            # the goal.  To avoid injection, make sure the query is quoted.
            encoded_query = query.replace("\n", " ").strip()
            goal = (
                f"Find the top 5 recent news articles about \"{encoded_query}\". "
                "Return JSON with an 'articles' array where each item has keys: title, "
                "published, snippet, and url."
            )
            payload = {
                "url": "https://news.google.com",
                "goal": goal,
            }
            headers = {
                "X-API-Key": self.tinyfish_api_key,
                "Content-Type": "application/json",
            }
            # Prefer the synchronous endpoint for simplicity.  TinyFish also
            # supports streaming (run-sse) but we do not need streaming here.
            resp = requests.post(
                "https://agent.tinyfish.ai/v1/automation/run",
                json=payload,
                headers=headers,
                timeout=20,
            )
            if not resp.ok:
                return f"TinyFish API error: {resp.status_code}"
            data = resp.json() if resp.content else {}
            result = data.get("resultJson") or data.get("result") or {}
            articles = result.get("articles") or result.get("results") or []
            lines: List[str] = []
            for a in articles[:6]:
                if not isinstance(a, dict):
                    continue
                title = (a.get("title") or "").strip()
                desc = (a.get("snippet") or a.get("description") or "").strip()
                url = (a.get("url") or "").strip()
                published = (a.get("published") or a.get("date") or "").strip()
                if title or desc:
                    lines.append(
                        f"- Title: {title}\n"
                        f"  Published: {published or 'N/A'}\n"
                        f"  Snippet: {desc or 'N/A'}\n"
                        f"  Source: {url or 'N/A'}"
                    )
            return "\n".join(lines).strip()
        except Exception as e:
            return f"TinyFish API failed: {e}"

    # Tavily wrapper ---------------------------------------------------------
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

    async def _expand_queries_with_gpt52(self, question_text: str) -> List[str]:
        researcher_llm = self.get_llm("researcher", "llm")
        prompt = clean_indents(f"""
        Generate 5 concise web search queries to research this forecasting question.
        Mix: entities, resolution criteria terms, and newest-updates phrasing.
        Output JSON ONLY: {{"queries":["...","..."]}}

        Question: {question_text}
        """).strip()
        raw = await with_timeout(researcher_llm.invoke(prompt), 35, "query_expand")
        try:
            data = json.loads(raw)
            qs = [str(x).strip() for x in data.get("queries", []) if str(x).strip()]
            merged = [question_text.strip()] + qs
            seen = set()
            out = []
            for q in merged:
                k = q.lower()
                if k not in seen:
                    seen.add(k)
                    out.append(q)
            return out[:6]
        except Exception:
            return [question_text.strip()]

    async def _run_research_impl(self, question: MetaculusQuestion) -> Tuple[str, Dict[str, str], List[str]]:
        async with self._concurrency_limiter:
            qtxt = (getattr(question, "question_text", "") or "").strip()
            if not qtxt:
                raise RuntimeError("Missing question_text; refusing to forecast.")

            expanded_queries = await self._expand_queries_with_gpt52(qtxt)

            tav_parts: List[str] = []
            tni_parts: List[str] = []

            # Query both Tavily and TinyFish on the first few expanded queries.  Use
            # timeouts to prevent hanging.
            for i, q in enumerate(expanded_queries[:3]):
                tav = await with_timeout(asyncio.to_thread(self.call_tavily, q), RESEARCH_TIMEOUT_S, f"tavily_{i}")
                if is_meaningful_research_text(tav):
                    tav_parts.append(f"## Query: {q}\n{tav}")

                tni = await with_timeout(asyncio.to_thread(self.call_tinyfish, q), RESEARCH_TIMEOUT_S, f"tinyfish_{i}")
                if is_meaningful_research_text(tni):
                    tni_parts.append(f"## Query: {q}\n{tni}")

            tav_all = "\n\n".join(tav_parts).strip()
            tni_all = "\n\n".join(tni_parts).strip()

            research_by_source = {"tavily": tav_all, "tinyfish": tni_all}
            searchers_used = [s for s in ["tavily", "tinyfish"] if is_meaningful_research_text(research_by_source.get(s, ""))]

            if REQUIRE_RESEARCH and not is_research_sufficient(research_by_source):
                raise RuntimeError(f"Insufficient research for Q{extract_question_id(question)}; refusing to forecast.")

            raw_block = format_research_block(research_by_source)

            researcher_llm = self.get_llm("researcher", "llm")
            synth_prompt = clean_indents(f"""
            Summarize evidence for forecasting.
            - Prioritize RECENT items; include dates when present.
            - Each fact ends with [TAVILY] or [TINYFISH] or both.
            - Include a short "Signposts" section (what to watch).

            Question: {qtxt}

            Raw research:
            {raw_block}

            Output (plain text, <=2400 chars):
            - Key facts (...)
            - Uncertainties (...)
            - Signposts (...)
            """).strip()
            synthesized = await with_timeout(researcher_llm.invoke(synth_prompt), LLM_TIMEOUT_S, "research_synthesis")
            synthesized = (synthesized or "").strip()
            if REQUIRE_RESEARCH and not synthesized:
                raise RuntimeError(f"Empty synthesis for Q{extract_question_id(question)}; refusing to forecast.")

            return synthesized, research_by_source, searchers_used

    async def run_research(self, question: MetaculusQuestion) -> str:
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

    async def _single_forecast(self, question, research: str, model_override: str = None):
        if REQUIRE_RESEARCH and (not research or len(research.strip()) < 80):
            raise RuntimeError(f"Missing/insufficient research for Q{extract_question_id(question)}; refusing to forecast.")

        llm = GeneralLlm(model=model_override) if model_override else self.get_llm("default", "llm")
        parser_llm = self.get_llm("parser", "llm")
        today_utc = datetime.utcnow().strftime("%Y-%m-%d")

        base_rate = safe_community_prediction(question)
        base_rate_str = (
            f"Community anchor: {base_rate:.4f} (decimal)"
            if isinstance(base_rate, (int, float)) and isinstance(question, BinaryQuestion)
            else (f"Community anchor (scalar): {base_rate:,.6g}" if isinstance(base_rate, (int, float)) else "No reliable community anchor.")
        )

        good_judgment = clean_indents("""
        Good Judgment rules:
        - Outside view first (base rate), then inside view update with evidence.
        - Weight evidence by recency + credibility; discount stale/noisy.
        - Keep uncertainty calibrated (avoid needless overconfidence).
        """).strip()

        if isinstance(question, BinaryQuestion):
            prompt = clean_indents(f"""
            {good_judgment}

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}

            {base_rate_str}

            Research summary:
            {research}

            Today (UTC): {today_utc}

            Output JSON ONLY: {{"prediction_in_decimal": number}}
            """).strip()
            raw = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "binary_llm")
            pred: BinaryPrediction = await structure_output(raw, BinaryPrediction, model=parser_llm)
            return clamp01(float(pred.prediction_in_decimal)), str(raw)

        if isinstance(question, MultipleChoiceQuestion):
            prompt = clean_indents(f"""
            {good_judgment}

            Question: {question.question_text}
            Options (exact): {question.options}
            Resolution criteria: {question.resolution_criteria}

            {base_rate_str}

            Research summary:
            {research}

            Today (UTC): {today_utc}

            Output JSON ONLY:
            {{
              "predicted_options": [
                {{"option":"<exact option>","probability": number}},
                ...
              ]
            }}
            Constraints: include all options once; probabilities sum to 1.
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
            lower_ref = getattr(question, "lower_bound", None) or getattr(question, "nominal_lower_bound", None)
            upper_ref = getattr(question, "upper_bound", None) or getattr(question, "nominal_upper_bound", None)

            prompt = clean_indents(f"""
            {good_judgment}

            Question: {question.question_text}
            Units: {question.unit_of_measure or 'Infer from context'}
            Resolution criteria: {question.resolution_criteria}
            Bounds (may be open): lower={lower_ref}, upper={upper_ref}

            {base_rate_str}

            Research summary:
            {research}

            Today (UTC): {today_utc}

            Output JSON ONLY: a list of percentiles:
            [
              {{"percentile":0.1,"value":...}},
              {{"percentile":0.2,"value":...}},
              {{"percentile":0.4,"value":...}},
              {{"percentile":0.6,"value":...}},
              {{"percentile":0.8,"value":...}},
              {{"percentile":0.9,"value":...}}
            ]
            """).strip()

            raw = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "num_llm")
            try:
                percentile_list: list[Percentile] = await structure_output(raw, list[Percentile], model=parser_llm)
                target_ps = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
                interpolated = interpolate_missing_percentiles(percentile_list, target_ps)
                validated = enforce_numeric_constraints(interpolated, question)

                vals = [p.value for p in validated]
                if len(set([round(v, 12) for v in vals])) == 1:
                    validated = derive_numeric_fallback_percentiles(question, base_rate)

            except Exception:
                validated = derive_numeric_fallback_percentiles(question, base_rate)

            dist = NumericDistribution.from_question(validated, question)
            return dist, str(raw)

        raise ValueError(f"Unsupported question type: {type(question)}")

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        models = [
            "openrouter/openai/gpt-5.2",
            "openrouter/openai/gpt-5.1",
            "openrouter/anthropic/claude-sonnet-4.5",
        ]
        qid = extract_question_id(question)
        base = safe_community_prediction(question)
        base_rate_text = f"{base:.1%}" if isinstance(base, (int, float)) else "None"

        raw_ps: List[float] = []
        for model in models:
            try:
                p, _ = await self._single_forecast(question, research, model_override=model)
                raw_ps.append(float(p))
            except Exception as e:
                fallback = clamp01(float(base) if isinstance(base, (int, float)) else 0.5)
                raw_ps.append(fallback)
                logger.warning(f"Binary model fallback Q{qid} model={model}: {e}")

        logits = np.array([logit(clamp01(p)) for p in raw_ps], dtype=float)
        p_agg = clamp01(sigmoid(float(np.median(logits))))

        p_std = float(np.std(raw_ps))
        agree = p_std <= EXTREMIZE_DISPERSION_STD_MAX

        p_final = p_agg
        if agree and should_extremize(p_agg, EXTREMIZE_THRESHOLD):
            p_final = extremize_probability(p_agg, alpha=EXTREMIZE_ALPHA)

        if isinstance(base, (int, float)):
            w = CROWD_BLEND_MIXED if agree else CROWD_BLEND_WEAK
            p_final = clamp01(w * p_final + (1.0 - w) * float(base))

        meta = self._research_meta.get(qid, {})
        searchers_used = meta.get("searchers_used", []) if isinstance(meta.get("searchers_used", None), list) else []

        comment = build_comment(
            question,
            forecast_text=f"{p_final:.1%}",
            base_rate_text=base_rate_text,
            how_text=f"- Committee logits median; agree={agree} std={p_std:.3f}\n- Used research + base rates (Good Judgment updating).",
            searchers_used=searchers_used,
            models_used=models,
        )
        log_forecast_for_calibration(question, p_final, comment, models, True, searchers_used)
        return ReasonedPrediction(prediction_value=p_final, reasoning=comment)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        models = [
            "openrouter/openai/gpt-5.2",
            "openrouter/openai/gpt-5.1",
            "openrouter/anthropic/claude-sonnet-4.5",
        ]
        qid = extract_question_id(question)

        per_model_maps: List[Dict[str, float]] = []
        for model in models:
            try:
                pred, _ = await self._single_forecast(question, research, model_override=model)
                m = {o["option"]: max(0.0, float(o["probability"])) for o in pred.predicted_options}
                for opt in question.options:
                    m.setdefault(opt, 0.0)
                s = sum(m.values()) or 1.0
                for k in list(m.keys()):
                    m[k] = m[k] / s
                per_model_maps.append(m)
            except Exception as e:
                n = len(question.options)
                per_model_maps.append({opt: 1.0 / n for opt in question.options})
                logger.warning(f"MC model fallback Q{qid} model={model}: {e}")

        option_list = list(question.options)
        mat = np.array([[m[opt] for opt in option_list] for m in per_model_maps], dtype=float)
        med = np.median(mat, axis=0)

        n = len(option_list)
        floor = max(1e-6, 0.01 / n)
        med = np.maximum(med, floor)
        med = med / med.sum()

        out = PredictedOptionList([{"option": opt, "probability": float(p)} for opt, p in zip(option_list, med)])

        meta = self._research_meta.get(qid, {})
        searchers_used = meta.get("searchers_used", []) if isinstance(meta.get("searchers_used", None), list) else []

        comment = build_comment(
            question,
            forecast_text=", ".join([f"{x['option']}: {x['probability']:.1%}" for x in out.predicted_options]),
            base_rate_text="(see community chart on Metaculus)",
            how_text=f"- Committee median per option + floor={floor:.6f}\n- Used research + Good Judgment updating.",
            searchers_used=searchers_used,
            models_used=models,
        )
        log_forecast_for_calibration(
            question,
            [x["probability"] for x in out.predicted_options],
            comment,
            models,
            True,
            searchers_used,
        )
        return ReasonedPrediction(prediction_value=out, reasoning=comment)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        models = [
            "openrouter/openai/gpt-5.2",
            "openrouter/openai/gpt-5.1",
            "openrouter/anthropic/claude-sonnet-4.5",
        ]
        qid = extract_question_id(question)
        base = safe_community_prediction(question)
        base_rate_text = f"{base:,.4g}" if isinstance(base, (int, float)) else "None"

        target_ps = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        per_model_percentiles: List[List[Percentile]] = []

        for model in models:
            try:
                dist, _ = await self._single_forecast(question, research, model_override=model)
                declared = list(getattr(dist, "declared_percentiles", [])) or []
                interpolated = interpolate_missing_percentiles(declared, target_ps)
                validated = enforce_numeric_constraints(interpolated, question)
                if not validated or len(validated) != 6:
                    validated = derive_numeric_fallback_percentiles(question, base)
                per_model_percentiles.append(validated)
            except Exception as e:
                per_model_percentiles.append(derive_numeric_fallback_percentiles(question, base))
                logger.warning(f"Numeric model fallback Q{qid} model={model}: {e}")

        aggregated: List[Percentile] = []
        for i, p in enumerate(target_ps):
            aggregated.append(Percentile(p, float(np.median([pm[i].value for pm in per_model_percentiles]))))

        validated = enforce_numeric_constraints(aggregated, question)
        if not validated or len(validated) != 6:
            validated = derive_numeric_fallback_percentiles(question, base)

        dist = NumericDistribution.from_question(validated, question)

        meta = self._research_meta.get(qid, {})
        searchers_used = meta.get("searchers_used", []) if isinstance(meta.get("searchers_used", None), list) else []

        comment = build_comment(
            question,
            forecast_text=", ".join([f"p{int(p.percentile*100)}={p.value:,.6g}" for p in validated]),
            base_rate_text=base_rate_text,
            how_text="- Committee median per percentile; enforced constraints.\n- Used research + base rates (Good Judgment updating).",
            searchers_used=searchers_used,
            models_used=models,
        )
        log_forecast_for_calibration(question, [p.value for p in validated], comment, models, True, searchers_used)
        return ReasonedPrediction(prediction_value=dist, reasoning=comment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tude.")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    client = MetaculusClient()
    # Include the Metaculus cup competitions along with minibench.  Duplicate
    # entries are harmless, but we de-duplicate for clarity.
    default_ids = list(dict.fromkeys([
        "32916",
        "minibench",
        "market-pulse-26q1",
        getattr(client, "CURRENT_MINIBENCH_ID", "minibench"),
    ]))
    tournament_ids = args.tournament_ids or default_ids

    bot = Tude(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    try:
        all_reports = []
        for tid in tournament_ids:
            logger.info(f"Forecasting on tournament: {tid}")

            for attempt in range(RETRY_MAX):
                try:
                    reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
                    all_reports.extend(reports)
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if "too many requests" in msg or "cloudflare" in msg or "1015" in msg or "429" in msg:
                        logger.error(f"Rate-limited on tournament {tid} (attempt {attempt+1}/{RETRY_MAX}): {e}")
                        backoff_sleep(attempt)
                        continue
                    raise

            time.sleep(TOURNAMENT_SLEEP_S)

        bot.log_report_summary(all_reports)
        logger.info(f"Run completed. Calibration logs saved to {CALIBRATION_LOG_FILE}")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
