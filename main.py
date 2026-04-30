

import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List, Tuple

import httpx
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
    PredictedOption,
    ReasonedPrediction,
    clean_indents,
    structure_output,
    AskNewsSearcher,
    SmartSearcher,
)
from tavily import TavilyClient

# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TINYFISH_API_KEY = os.getenv("TINYFISH_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ASKNEWS_API_KEY = os.getenv("ASKNEWS_API_KEY")
SMART_SEARCHER_API_KEY = os.getenv("SMART_SEARCHER_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Tude")

RESEARCH_TIMEOUT_S = float(os.getenv("RESEARCH_TIMEOUT_S", "25"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "90"))

# FIX 4 – TinyFish now uses httpx; these knobs still apply
TINYFISH_HTTP_TIMEOUT_S = float(os.getenv("TINYFISH_HTTP_TIMEOUT_S", "60"))
TINYFISH_RETRY_MAX = int(os.getenv("TINYFISH_RETRY_MAX", "3"))
TINYFISH_RETRY_BASE_S = float(os.getenv("TINYFISH_RETRY_BASE_S", "2.0"))
TINYFISH_RETRY_MAX_S = float(os.getenv("TINYFISH_RETRY_MAX_S", "25.0"))

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
# FIX 7 – if committee and crowd disagree by more than this in log-odds, skip blending
CROWD_BLEND_DISAGREE_LOGODDS = float(os.getenv("CROWD_BLEND_DISAGREE_LOGODDS", "1.5"))

MIN_P = float(os.getenv("MIN_P", "0.01"))
MAX_P = float(os.getenv("MAX_P", "0.99"))

REQUIRE_RESEARCH = os.getenv("REQUIRE_RESEARCH", "true").lower() in ("1", "true", "yes")
CALIBRATION_LOG_FILE = "forecasting_calibration_log.jsonl"

# FIX 8 – single source of truth for committee models
# FIX 17 – replaced gpt-5-perplexity with actual Perplexity online model
COMMITTEE_MODELS: List[str] = [
    "openrouter/openai/gpt-5.4",
    "openrouter/openai/gpt-5-search",
    "openrouter/perplexity/sonar-pro",
    "openrouter/anthropic/claude-sonnet-4.6",
]

# ---------------------------------------------------------------------------
# Async calibration log writer (FIX 16)
# ---------------------------------------------------------------------------
_cal_queue: asyncio.Queue = asyncio.Queue()


async def _cal_writer_task() -> None:
    """Background task that drains _cal_queue and appends to the JSONL file."""
    while True:
        entry = await _cal_queue.get()
        if entry is None:  # sentinel
            break
        try:
            with open(CALIBRATION_LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            logger.warning("Calibration log write failed: %s", exc)
        finally:
            _cal_queue.task_done()


def log_forecast_for_calibration(
    question: MetaculusQuestion,
    prediction_value: Any,
    reasoning: str,
    model_ids: List[str],
    research_used: bool,
    searchers_used: List[str],
) -> None:
    """Enqueue a calibration record (non-blocking)."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
        _cal_queue.put_nowait(entry)
    except Exception as exc:
        logger.warning("Failed to enqueue calibration entry: %s", exc)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def extract_question_id(question: MetaculusQuestion) -> str:
    try:
        qid = getattr(question, "id", None)
        if isinstance(qid, (int, str)) and str(qid).isdigit():
            return str(qid)
    except Exception:
        pass
    for attr in ("url", "question_url", "page_url", "web_url", "permalink", "link"):
        try:
            url = str(getattr(question, attr, "") or "")
            m = re.search(r"/questions/(\d+)(?:/|$)", url)
            if m:
                return m.group(1)
        except Exception:
            continue
    try:
        m = re.search(r"/questions/(\d+)(?:/|$)", str(question))
        if m:
            return m.group(1)
    except Exception:
        pass
    return "unknown"


def _question_key(question: MetaculusQuestion) -> str:
    """
    FIX 6 – stable per-question key for _research_meta.
    Prefer page_url (always unique); fall back to numeric ID; last resort repr.
    """
    for attr in ("page_url", "url", "question_url", "web_url", "permalink"):
        val = getattr(question, attr, None)
        if val:
            return str(val)
    qid = extract_question_id(question)
    if qid != "unknown":
        return f"id:{qid}"
    return f"repr:{id(question)}"


def _resolution_date_info(question: MetaculusQuestion) -> Tuple[str, str]:
    """
    R6 – Return (resolution_date_str, days_remaining_str) for injection into prompts.
    Gives the model a concrete time anchor so it can reason about how much can
    change before resolution.
    """
    rd = getattr(question, "resolution_date", None) or getattr(question, "close_time", None)
    if rd is None:
        return "Unknown", "Unknown"
    try:
        if hasattr(rd, "date"):
            rd_dt = rd
        else:
            from datetime import datetime as _dt
            rd_dt = _dt.fromisoformat(str(rd).replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        if rd_dt.tzinfo is None:
            rd_dt = rd_dt.replace(tzinfo=timezone.utc)
        delta = rd_dt - now
        days = delta.days
        date_str = rd_dt.strftime("%Y-%m-%d")
        if days < 0:
            remaining = f"already passed ({abs(days)} days ago)"
        elif days == 0:
            remaining = "today"
        elif days == 1:
            remaining = "1 day"
        else:
            remaining = f"{days} days"
        return date_str, remaining
    except Exception:
        return str(rd), "Unknown"


def safe_community_prediction(question: MetaculusQuestion) -> Optional[float]:
    try:
        for attr in ("community_prediction", "prediction"):
            pred = getattr(question, attr, None)
            if pred is not None and isinstance(pred, (int, float)):
                return float(pred)
    except Exception as exc:
        logger.warning("Failed to get community prediction for Q%s: %s", extract_question_id(question), exc)
    return None


def clamp01(p: float) -> float:
    return float(max(MIN_P, min(MAX_P, p)))


# FIX 15 – use math (faster on scalars than numpy)
def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def logit(p: float) -> float:
    p = clamp01(p)
    return math.log(p / (1.0 - p))


def extremize_probability(p: float, alpha: float) -> float:
    return clamp01(sigmoid(alpha * logit(p)))


def should_extremize(p: float, threshold: float = EXTREMIZE_THRESHOLD) -> bool:
    return (p >= threshold) or (p <= (1.0 - threshold))


def is_meaningful_research_text(txt: str) -> bool:
    if not txt:
        return False
    s = txt.strip()
    low = s.lower()
    if len(s) < 40:
        return False
    if "source:" in low:
        return True
    if "title:" in low and "snippet:" in low:
        return True
    bullets = sum(1 for line in s.splitlines() if line.strip().startswith("-"))
    if bullets >= 2:
        return True
    return len(s) >= 120


def is_research_sufficient(research_by_source: Dict[str, str]) -> bool:
    return any(is_meaningful_research_text(v) for v in (research_by_source or {}).values())


def interpolate_missing_percentiles(
    reported: List[Percentile], target_percentiles: List[float]
) -> List[Percentile]:
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


def _finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
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


def enforce_numeric_constraints(
    percentiles: List[Percentile], question: NumericQuestion
) -> List[Percentile]:
    lb, ub = _safe_lb_ub(question)
    bounded: List[Percentile] = []
    for p in percentiles:
        v = float(p.value)
        if not math.isfinite(v):
            v = lb
        v = max(lb, min(ub, v))
        bounded.append(Percentile(percentile=float(p.percentile), value=v))
    srt = sorted(bounded, key=lambda x: float(x.percentile))
    for i in range(1, len(srt)):
        if srt[i].value < srt[i - 1].value:
            srt[i] = Percentile(percentile=srt[i].percentile, value=srt[i - 1].value)
    return [Percentile(percentile=float(p.percentile), value=float(p.value)) for p in srt]


def derive_numeric_fallback_percentiles(
    question: NumericQuestion, anchor: Optional[float]
) -> List[Percentile]:
    lb, ub = _safe_lb_ub(question)
    if not math.isfinite(ub - lb) or ub <= lb:
        lb, ub = 0.0, 1.0
    if isinstance(anchor, (int, float)) and math.isfinite(float(anchor)):
        a = float(anchor)
        center = max(lb + 0.1 * (ub - lb), min(lb + 0.9 * (ub - lb), a))
    else:
        center = lb + 0.5 * (ub - lb)
    span = max(1e-6, ub - lb)
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
    return enforce_numeric_constraints(out, question)


def format_research_block(research_by_source: Dict[str, str]) -> str:
    blocks: List[str] = []
    # FIX 17 – updated sources to include asknews and smart_searcher, replaced openrouter_perplexity with perplexity
    for src in ["tavily", "tinyfish", "openrouter_search", "perplexity", "asknews", "smart_searcher"]:
        txt = (research_by_source or {}).get(src, "") or ""
        if txt.strip():
            blocks.append(f"--- SOURCE {src.upper()} ---\n{txt}\n")
    return "\n".join(blocks).strip()


def build_comment(
    question: MetaculusQuestion,
    forecast_text: str,
    base_rate_text: str,
    how_text: str,
    searchers_used: List[str],
    models_used: List[str],
    model_reasonings: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    FIX 11 / R4 / R9 – individual model reasonings included in the published comment.
    model_reasonings is a list of (model_name, reasoning_text) tuples.

    R9: The most divergent model (by prediction, inferred from reasoning length as a
    proxy when predictions aren't re-passed here) gets its FULL reasoning included.
    All other models get a 600-char snippet. This surfaces the most novel perspective
    for calibration and transparency.
    """
    qtxt = getattr(question, "question_text", "").strip()
    qid = extract_question_id(question)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    resolution_date, days_remaining = _resolution_date_info(question)
    searchers = ", ".join(searchers_used) if searchers_used else "None"
    models = ", ".join(models_used) if models_used else "Unknown"

    reasoning_section = ""
    if model_reasonings:
        # R9 – heuristic: longest reasoning is most likely most divergent/detailed.
        # Include it in full; others get 600-char snippets.
        sorted_by_len = sorted(model_reasonings, key=lambda x: len(x[1]), reverse=True)
        parts = []
        for i, (model_name, reasoning_text) in enumerate(sorted_by_len):
            short_name = model_name.split("/")[-1]
            if i == 0:
                # Most detailed — include in full (up to 4000 chars)
                parts.append(
                    f"**{short_name} (full reasoning):**\n\n{reasoning_text[:4000].strip()}"
                )
            else:
                parts.append(f"**{short_name}:** {reasoning_text[:600].strip()}")
        reasoning_section = "\n\n---\n\n**Committee reasoning:**\n\n" + "\n\n".join(parts)

    return clean_indents(f"""
    ## Forecast (Q{qid})
    **Date (UTC):** {today}
    **Resolution date:** {resolution_date} ({days_remaining} remaining)

    **Question:** {qtxt}

    **Forecast:** {forecast_text}

    **Anchor / base rate:** {base_rate_text}

    **How this was arrived at:**
    {how_text}{reasoning_section}

    **Searchers used:** {searchers}
    **Models used:** {models}
    """).strip()


# ---------------------------------------------------------------------------
# Tournament detection for conservative forecasting strategy
# ---------------------------------------------------------------------------

def _get_tournament_id(question: MetaculusQuestion) -> str:
    """FIX 18 – Extract tournament ID from question for conservative forecast logic."""
    try:
        # Try common tournament attributes
        for attr in ("tournament_id", "tournament", "group_id", "group"):
            val = getattr(question, attr, None)
            if val:
                return str(val).lower()
        
        # Try to extract from URL
        for attr in ("page_url", "url", "question_url", "web_url", "permalink"):
            url = str(getattr(question, attr, "") or "")
            if "/tournament/" in url:
                parts = url.split("/tournament/")
                if len(parts) > 1:
                    tid = parts[1].split("/")[0]
                    return tid.lower()
    except Exception:
        pass
    return ""


def _is_aggressive_tournament(question: MetaculusQuestion) -> bool:
    """FIX 18 – Return True if question is from minibench or market-pulse (allow aggressive forecasts)."""
    tid = _get_tournament_id(question)
    aggressive_tournaments = ["minibench", "market-pulse", "market_pulse"]
    return any(aggressive in tid for aggressive in aggressive_tournaments)


# ---------------------------------------------------------------------------
# Async backoff helpers (FIX 13)
# ---------------------------------------------------------------------------

def _binary_how_text(
    committee_models: List[str],
    raw_ps: List[float],
    p_agg: float,
    p_std: float,
    agree: bool,
    base_rate_text: str,
    lo_gap_str: str,
    p_final: float,
) -> str:
    """Helper to build the how_text for binary forecasts without f-string nesting."""
    per_model = ", ".join(
        f"{m.split('/')[-1]}={p:.1%}" for m, p in zip(committee_models, raw_ps)
    )
    lines = [
        f"- 3-model committee: {per_model}",
        f"- Logit-median aggregate: {p_agg:.1%}; std={p_std:.3f}; models agree={agree}",
        f"- Extremized: {agree and should_extremize(p_agg)} (alpha={EXTREMIZE_ALPHA})",
        f"- Crowd blend: community={base_rate_text}; log-odds gap={lo_gap_str}",
        f"- Final: {p_final:.1%} (8-step superforecaster reasoning protocol)",
    ]
    return "\n".join(lines)


async def backoff_sleep_async(attempt: int) -> None:
    base = min(RETRY_MAX_S, RETRY_BASE_S * (2 ** attempt))
    jitter = random.uniform(0.0, base * 0.25)
    await asyncio.sleep(base + jitter)


async def _tinyfish_backoff_async(attempt: int) -> None:
    base = min(TINYFISH_RETRY_MAX_S, TINYFISH_RETRY_BASE_S * (2 ** attempt))
    jitter = random.uniform(0.0, base * 0.25)
    await asyncio.sleep(base + jitter)


# kept for the synchronous __main__ retry loop only
def backoff_sleep(attempt: int) -> None:
    base = min(RETRY_MAX_S, RETRY_BASE_S * (2 ** attempt))
    jitter = random.uniform(0.0, base * 0.25)
    time.sleep(base + jitter)


# ---------------------------------------------------------------------------
# with_timeout helper
# ---------------------------------------------------------------------------

async def with_timeout(coro, seconds: float, label: str) -> str:
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        return f"{label} timeout after {seconds}s"
    except Exception as exc:
        return f"{label} error: {exc}"


# ---------------------------------------------------------------------------
# Tude bot
# ---------------------------------------------------------------------------

class Tude(ForecastBot):
    # FIX 1 – class-level attribute only stores the *count*; semaphore created in __init__
    _max_concurrent_questions = MAX_CONCURRENT_QUESTIONS

    # FIX 8 – models defined once here, referenced everywhere
    COMMITTEE_MODELS: List[str] = COMMITTEE_MODELS

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # FIX 1 – semaphore created inside __init__ so it is bound to the running loop
        self._concurrency_limiter = asyncio.Semaphore(self._max_concurrent_questions)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
        self.tinyfish_api_key = TINYFISH_API_KEY
        self.openrouter_api_key = OPENROUTER_API_KEY
        # FIX 17 – initialize asknews and smart-searcher
        self.asknews = AskNewsSearcher(api_key=ASKNEWS_API_KEY) if ASKNEWS_API_KEY else None
        self.smart_searcher = SmartSearcher(api_key=SMART_SEARCHER_API_KEY) if SMART_SEARCHER_API_KEY else None
        # FIX 6 – keyed by stable question URL, not potentially-"unknown" ID string
        self._research_meta: Dict[str, Dict[str, Any]] = {}

        if REQUIRE_RESEARCH and not (self.tavily_client or self.tinyfish_api_key or self.openrouter_api_key or self.asknews or self.smart_searcher):
            raise RuntimeError(
                "REQUIRE_RESEARCH=true but no research providers are configured. "
                "Set TAVILY_API_KEY, TINYFISH_API_KEY, OPENROUTER_API_KEY, ASKNEWS_API_KEY, and/or SMART_SEARCHER_API_KEY."
            )

    # -------------------------------------------------------------------------
    # TinyFish — FIX 4: fully async httpx, no blocking time.sleep
    # -------------------------------------------------------------------------
    async def call_tinyfish_async(self, query: str) -> str:
        if not self.tinyfish_api_key:
            return ""
        encoded_query = query.replace("\n", " ").strip()
        goal = (
            f'Find the top 3 recent news articles about "{encoded_query}". '
            "Return JSON with an 'articles' array where each item has keys: "
            "title, published, snippet, and url."
        )
        payload = {"url": "https://news.google.com", "goal": goal}
        headers = {
            "X-API-Key": self.tinyfish_api_key,
            "Content-Type": "application/json",
        }
        last_err: str = "TinyFish API failed: unknown error"
        async with httpx.AsyncClient(timeout=TINYFISH_HTTP_TIMEOUT_S) as client:
            for attempt in range(TINYFISH_RETRY_MAX):
                try:
                    resp = await client.post(
                        "https://agent.tinyfish.ai/v1/automation/run",
                        json=payload,
                        headers=headers,
                    )
                    if not resp.is_success:
                        last_err = f"TinyFish API error: {resp.status_code}"
                        await _tinyfish_backoff_async(attempt)
                        continue
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
                except Exception as exc:
                    last_err = f"TinyFish API failed: {exc}"
                    await _tinyfish_backoff_async(attempt)
        return last_err

    def call_tavily(self, query: str) -> str:
        if not self.tavily_client:
            return ""
        try:
            response = self.tavily_client.search(query=query, search_depth="advanced")
            results = response.get("results", []) or []
            lines: List[str] = []
            for r in results[:10]:
                title = (r.get("title") or "").strip()
                content = (r.get("content") or "").strip()
                snippet = (r.get("snippet") or "").strip()
                url = (r.get("url") or "").strip()
                body = content or snippet
                if title or body:
                    lines.append(
                        f"- {title + ': ' if title else ''}{body}\n"
                        f"  Source: {url or 'N/A'}"
                    )
            return "\n".join(lines).strip()
        except Exception as exc:
            return f"Tavily error: {exc}"

    async def call_openrouter_async(self, query: str, model: str) -> str:
        if not self.openrouter_api_key:
            return ""
        encoded_query = query.replace("\n", " ").strip()
        prompt = clean_indents(f"""
        You are a research assistant. For the query below, return the top 5 most relevant recent facts,
        headlines, or source summaries that would help answer the question. Use bullet points and include
        short citations or source descriptors where possible.

        Query: {encoded_query}
        """).strip()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 800,
        }
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        last_err = f"OpenRouter {model} API failed: unknown error"
        async with httpx.AsyncClient(timeout=TINYFISH_HTTP_TIMEOUT_S) as client:
            for attempt in range(TINYFISH_RETRY_MAX):
                try:
                    resp = await client.post(
                        "https://openrouter.ai/v1/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    if not resp.is_success:
                        last_err = f"OpenRouter {model} API error: {resp.status_code}"
                        await _tinyfish_backoff_async(attempt)
                        continue
                    data = resp.json() if resp.content else {}
                    choices = data.get("choices") or []
                    if choices and isinstance(choices, list):
                        text = (choices[0].get("message", {}).get("content") or "").strip()
                        return text
                    return data.get("result", "") or ""
                except Exception as exc:
                    last_err = f"OpenRouter {model} API failed: {exc}"
                    await _tinyfish_backoff_async(attempt)
        return last_err

    async def call_asknews_async(self, query: str) -> str:
        """FIX 17 – Call AskNews searcher in parallel."""
        if not self.asknews:
            return ""
        try:
            results = await asyncio.to_thread(self.asknews.search, query)
            if not results:
                return ""
            lines: List[str] = []
            for item in (results if isinstance(results, list) else [results])[:10]:
                if isinstance(item, dict):
                    title = (item.get("title") or "").strip()
                    snippet = (item.get("snippet") or item.get("description") or "").strip()
                    url = (item.get("url") or "").strip()
                    source = (item.get("source") or "").strip()
                else:
                    title = getattr(item, "title", "")
                    snippet = getattr(item, "snippet", None) or getattr(item, "description", "")
                    url = getattr(item, "url", "")
                    source = getattr(item, "source", "")
                if title or snippet:
                    lines.append(
                        f"- {title + ': ' if title else ''}{snippet}\n"
                        f"  Source: {source or url or 'N/A'}"
                    )
            return "\n".join(lines).strip()
        except Exception as exc:
            return f"AskNews error: {exc}"

    async def call_smart_searcher_async(self, query: str) -> str:
        """FIX 17 – Call SmartSearcher in parallel."""
        if not self.smart_searcher:
            return ""
        try:
            results = await asyncio.to_thread(self.smart_searcher.search, query)
            if not results:
                return ""
            lines: List[str] = []
            for item in (results if isinstance(results, list) else [results])[:10]:
                if isinstance(item, dict):
                    title = (item.get("title") or "").strip()
                    snippet = (item.get("snippet") or item.get("description") or "").strip()
                    url = (item.get("url") or "").strip()
                    source = (item.get("source") or "").strip()
                else:
                    title = getattr(item, "title", "")
                    snippet = getattr(item, "snippet", None) or getattr(item, "description", "")
                    url = getattr(item, "url", "")
                    source = getattr(item, "source", "")
                if title or snippet:
                    lines.append(
                        f"- {title + ': ' if title else ''}{snippet}\n"
                        f"  Source: {source or url or 'N/A'}"
                    )
            return "\n".join(lines).strip()
        except Exception as exc:
            return f"SmartSearcher error: {exc}"

    # -------------------------------------------------------------------------
    # Query expansion
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    async def _expand_queries(
        self, question_text: str, resolution_criteria: str = ""
    ) -> List[str]:
        """
        R1 – Generates three distinct query categories to ensure comprehensive coverage:
          1. Entity/current-status queries: who/what is involved right now.
          2. Resolution-criteria queries: what specific thresholds/events would resolve this.
          3. Base-rate/historical queries: historical precedents for this type of event.

        Including resolution_criteria as input (not just question text) improves
        query diversity by targeting what actually determines the outcome.
        """
        researcher_llm = self.get_llm("researcher", "llm")
        criteria_snippet = (resolution_criteria or "")[:400]
        prompt = clean_indents(f"""
        Generate 9 concise web search queries to research this forecasting question.
        Produce EXACTLY 3 queries of each type — do not mix types within a group:

        TYPE A — Entity & current-status (who/what is involved, latest news):
        TYPE B — Resolution-criteria focused (searches targeting the specific thresholds
                  or events that would resolve the question Yes or No):
        TYPE C — Base-rate & historical (searches for historical frequency of similar
                  events, analogous past cases, or base rates for this outcome type):

        Output JSON ONLY: {{"queries":[
          {{"type":"A","q":"..."}},{{"type":"A","q":"..."}},{{"type":"A","q":"..."}},
          {{"type":"B","q":"..."}},{{"type":"B","q":"..."}},{{"type":"B","q":"..."}},
          {{"type":"C","q":"..."}},{{"type":"C","q":"..."}},{{"type":"C","q":"..."}}
        ]}}

        Question: {question_text}
        Resolution criteria: {criteria_snippet}
        """).strip()
        raw = await with_timeout(researcher_llm.invoke(prompt), 40, "query_expand")
        try:
            data = json.loads(raw)
            items = data.get("queries", [])
            qs = []
            for item in items:
                if isinstance(item, dict):
                    q = str(item.get("q") or "").strip()
                elif isinstance(item, str):
                    q = item.strip()
                else:
                    continue
                if q:
                    qs.append(q)
            merged = [question_text.strip()] + qs
            seen: set = set()
            out: List[str] = []
            for q in merged:
                k = q.lower()
                if k not in seen:
                    seen.add(k)
                    out.append(q)
            return out[:12]
        except Exception:
            return [question_text.strip()]

    # -------------------------------------------------------------------------
    # Research — FIX 3: all queries run concurrently
    # -------------------------------------------------------------------------
    async def _run_research_impl(
        self, question: MetaculusQuestion
    ) -> Tuple[str, Dict[str, str], List[str]]:
        async with self._concurrency_limiter:
            qtxt = (getattr(question, "question_text", "") or "").strip()
            if not qtxt:
                raise RuntimeError("Missing question_text; refusing to forecast.")

            resolution_criteria = getattr(question, "resolution_criteria", "") or ""
            expanded_queries = await self._expand_queries(qtxt, resolution_criteria)  # R1
            max_queries = min(8, len(expanded_queries))  # allow more queries from 3-category expansion
            queries = expanded_queries[:max_queries]

            # FIX 3 – fire all (query × provider) requests concurrently
            async def _fetch_tavily(q: str) -> Tuple[str, str]:
                raw = await with_timeout(
                    asyncio.to_thread(self.call_tavily, q), RESEARCH_TIMEOUT_S, f"tavily:{q[:40]}"
                )
                return q, (raw or "").strip()

            async def _fetch_tinyfish(q: str) -> Tuple[str, str]:
                raw = await with_timeout(
                    self.call_tinyfish_async(q), RESEARCH_TIMEOUT_S, f"tinyfish:{q[:40]}"
                )
                return q, (raw or "").strip()

            async def _fetch_openrouter_search(q: str) -> Tuple[str, str]:
                raw = await with_timeout(
                    self.call_openrouter_async(q, "openrouter/openai/gpt-5-search"),
                    RESEARCH_TIMEOUT_S,
                    f"openrouter_search:{q[:40]}",
                )
                return q, (raw or "").strip()

            async def _fetch_openrouter_perplexity(q: str) -> Tuple[str, str]:
                # FIX 17 – use actual Perplexity online model
                raw = await with_timeout(
                    self.call_openrouter_async(q, "openrouter/perplexity/sonar-pro"),
                    RESEARCH_TIMEOUT_S,
                    f"perplexity:{q[:40]}",
                )
                return q, (raw or "").strip()

            async def _fetch_asknews(q: str) -> Tuple[str, str]:
                raw = await with_timeout(
                    self.call_asknews_async(q), RESEARCH_TIMEOUT_S, f"asknews:{q[:40]}"
                )
                return q, (raw or "").strip()

            async def _fetch_smart_searcher(q: str) -> Tuple[str, str]:
                raw = await with_timeout(
                    self.call_smart_searcher_async(q), RESEARCH_TIMEOUT_S, f"smart_searcher:{q[:40]}"
                )
                return q, (raw or "").strip()

            tav_coros = [_fetch_tavily(q) for q in queries]
            tni_coros = [_fetch_tinyfish(q) for q in queries]
            ors_coros = [_fetch_openrouter_search(q) for q in queries]
            orp_coros = [_fetch_openrouter_perplexity(q) for q in queries]
            asn_coros = [_fetch_asknews(q) for q in queries]
            sms_coros = [_fetch_smart_searcher(q) for q in queries]
            all_results = await asyncio.gather(
                *tav_coros, *tni_coros, *ors_coros, *orp_coros, *asn_coros, *sms_coros, return_exceptions=True
            )

            tav_parts: List[str] = []
            tni_parts: List[str] = []
            ors_parts: List[str] = []
            orp_parts: List[str] = []
            asn_parts: List[str] = []
            sms_parts: List[str] = []
            tav_errs: List[str] = []
            tni_errs: List[str] = []
            ors_errs: List[str] = []
            orp_errs: List[str] = []
            asn_errs: List[str] = []
            sms_errs: List[str] = []

            idx = 0
            for res in all_results[idx: idx + max_queries]:
                idx += max_queries
                if isinstance(res, Exception):
                    tav_errs.append(str(res))
                    continue
                q, raw = res
                if is_meaningful_research_text(raw):
                    tav_parts.append(f"## Query: {q}\n{raw}")
                elif raw:
                    tav_errs.append(f"[tavily] {raw[:500]}")

            for res in all_results[idx: idx + max_queries]:
                idx += max_queries
                if isinstance(res, Exception):
                    tni_errs.append(str(res))
                    continue
                q, raw = res
                if is_meaningful_research_text(raw):
                    tni_parts.append(f"## Query: {q}\n{raw}")
                elif raw:
                    tni_errs.append(f"[tinyfish] {raw[:500]}")

            for res in all_results[idx: idx + max_queries]:
                idx += max_queries
                if isinstance(res, Exception):
                    ors_errs.append(str(res))
                    continue
                q, raw = res
                if is_meaningful_research_text(raw):
                    ors_parts.append(f"## Query: {q}\n{raw}")
                elif raw:
                    ors_errs.append(f"[openrouter_search] {raw[:500]}")

            for res in all_results[idx: idx + max_queries]:
                idx += max_queries
                if isinstance(res, Exception):
                    orp_errs.append(str(res))
                    continue
                q, raw = res
                if is_meaningful_research_text(raw):
                    orp_parts.append(f"## Query: {q}\n{raw}")
                elif raw:
                    orp_errs.append(f"[perplexity] {raw[:500]}")

            for res in all_results[idx: idx + max_queries]:
                idx += max_queries
                if isinstance(res, Exception):
                    asn_errs.append(str(res))
                    continue
                q, raw = res
                if is_meaningful_research_text(raw):
                    asn_parts.append(f"## Query: {q}\n{raw}")
                elif raw:
                    asn_errs.append(f"[asknews] {raw[:500]}")

            for res in all_results[idx: idx + max_queries]:
                idx += max_queries
                if isinstance(res, Exception):
                    sms_errs.append(str(res))
                    continue
                q, raw = res
                if is_meaningful_research_text(raw):
                    sms_parts.append(f"## Query: {q}\n{raw}")
                elif raw:
                    sms_errs.append(f"[smart_searcher] {raw[:500]}")

            tav_all = "\n\n".join(tav_parts).strip()
            tni_all = "\n\n".join(tni_parts).strip()
            ors_all = "\n\n".join(ors_parts).strip()
            orp_all = "\n\n".join(orp_parts).strip()
            asn_all = "\n\n".join(asn_parts).strip()
            sms_all = "\n\n".join(sms_parts).strip()
            research_by_source = {
                "tavily": tav_all,
                "tinyfish": tni_all,
                "openrouter_search": ors_all,
                "perplexity": orp_all,
                "asknews": asn_all,
                "smart_searcher": sms_all,
            }
            searchers_used = [
                s for s in ["tavily", "tinyfish", "openrouter_search", "perplexity", "asknews", "smart_searcher"]
                if is_meaningful_research_text(research_by_source.get(s, ""))
            ]

            if REQUIRE_RESEARCH and not is_research_sufficient(research_by_source):
                qid = extract_question_id(question)
                diag = []
                if tav_errs:
                    diag.append("Tavily diagnostics:\n" + "\n".join(tav_errs[:4]))
                if tni_errs:
                    diag.append("TinyFish diagnostics:\n" + "\n".join(tni_errs[:4]))
                if ors_errs:
                    diag.append("OpenRouter search diagnostics:\n" + "\n".join(ors_errs[:4]))
                if orp_errs:
                    diag.append("Perplexity diagnostics:\n" + "\n".join(orp_errs[:4]))
                if asn_errs:
                    diag.append("AskNews diagnostics:\n" + "\n".join(asn_errs[:4]))
                if sms_errs:
                    diag.append("SmartSearcher diagnostics:\n" + "\n".join(sms_errs[:4]))
                diag_text = ("\n\n".join(diag)).strip() or "No provider output captured."
                raise RuntimeError(
                    f"Insufficient research for Q{qid}; refusing to forecast. "
                    f"Providers configured: tavily={'yes' if TAVILY_API_KEY else 'no'}, "
                    f"tinyfish={'yes' if self.tinyfish_api_key else 'no'}, "
                    f"openrouter={'yes' if self.openrouter_api_key else 'no'}, "
                    f"asknews={'yes' if self.asknews else 'no'}, "
                    f"smart_searcher={'yes' if self.smart_searcher else 'no'}. "
                    f"Research lengths: tavily={len(tav_all)}, tinyfish={len(tni_all)}, "
                    f"openrouter_search={len(ors_all)}, perplexity={len(orp_all)}, "
                    f"asknews={len(asn_all)}, smart_searcher={len(sms_all)}.\n\n"
                    f"{diag_text}"
                )

            raw_block = format_research_block(research_by_source)
            researcher_llm = self.get_llm("researcher", "llm")

            # FIX 9 – rich 6-section synthesis prompt, cap raised to 6000 chars
            resolution_criteria = getattr(question, "resolution_criteria", "") or ""
            fine_print = getattr(question, "fine_print", "") or ""
            synth_prompt = clean_indents(f"""
            You are a research analyst supporting an elite superforecaster.
            Produce a structured intelligence brief from the raw research below.
            Do NOT produce a probability or final forecast. Be factual and precise.

            Question: {qtxt}
            Resolution criteria: {resolution_criteria}
            {fine_print}

            Raw research:
            {raw_block}

            Produce your brief in exactly these six sections:

            ## 1. Current Status
            The factual state of play RIGHT NOW, directly relevant to this question.
            Include concrete numbers, dates, named actors, and recent developments.
            Cite recency (e.g. "as of [date]") for any time-sensitive claim.

            ## 2. Resolution-Relevant Signals
            What specific indicators, thresholds, or events would directly trigger
            a YES or NO resolution under the stated criteria? Be precise —
            e.g. "X must exceed Y by date Z". Flag if thresholds are ambiguous.

            ## 3. Key Evidence & Source Quality
            List the 5-7 most important pieces of evidence. For each: what it says,
            how strong/reliable it is, how recent it is. Flag missing or contested evidence.

            ## 4. Trend & Trajectory
            Direction the relevant variable/situation is moving. Accelerating,
            decelerating, or stable? Any recent inflection points?

            ## 5. Expert, Market Consensus & Base Rates
            What do domain experts, official forecasts, prediction markets, or
            institutional projections say? Include specific numbers if available.
            CRITICALLY: What is the BASE RATE for this type of event?
            - Historical frequency: how often do similar questions/events resolve YES?
            - Reference class: what is the most apt comparison population?
            - Any prediction market prices or superforecaster aggregates available?
            State a concrete base rate estimate (e.g. "~30% of similar X questions
            resolve YES within 6 months") if one can be derived from the evidence.

            ## 6. Key Uncertainties
            The 3-4 things you most do NOT know that matter most for resolution.
            Include any data gaps, model disagreements, or geopolitical wildcards.

            ## 7. Preliminary Probability Range
            Based purely on the above evidence (not a final forecast), state the
            plausible range: e.g. "Evidence suggests 20-40% probability of YES."
            This is a research anchor, not the final model forecast.

            Output plain text only. Be precise and cite sources where possible.
            """).strip()

            synthesized = await with_timeout(
                researcher_llm.invoke(synth_prompt), LLM_TIMEOUT_S, "research_synthesis"
            )
            synthesized = (synthesized or "").strip()

            # FIX 5 – fall back to raw block rather than raising on empty synthesis
            if not synthesized:
                logger.warning(
                    "Empty synthesis for Q%s — falling back to raw research block.",
                    extract_question_id(question),
                )
                synthesized = raw_block or "(no research available)"

            return synthesized, research_by_source, searchers_used

    async def run_research(self, question: MetaculusQuestion) -> str:
        synthesized, research_by_source, searchers_used = await self._run_research_impl(question)
        # FIX 6 – use stable URL-based key
        key = _question_key(question)
        self._research_meta[key] = {
            "synthesized": synthesized,
            "raw": research_by_source,
            "searchers_used": searchers_used,
        }
        logger.info(
            "Q%s class=%s bounds=(%s,%s) open_bounds=(%s,%s) searchers_used=%s",
            extract_question_id(question),
            type(question).__name__,
            getattr(question, "lower_bound", None),
            getattr(question, "upper_bound", None),
            getattr(question, "open_lower_bound", None),
            getattr(question, "open_upper_bound", None),
            searchers_used,
        )
        return synthesized

    # -------------------------------------------------------------------------
    # Single model forecast with full chain-of-thought reasoning scaffold
    # FIX 10 – all prompts now have explicit CoT steps
    # FIX 11 – raw reasoning returned alongside prediction
    # -------------------------------------------------------------------------
    async def _single_forecast(
        self,
        question: MetaculusQuestion,
        research: str,
        model_override: Optional[str] = None,
    ) -> Tuple[Any, str]:
        if REQUIRE_RESEARCH and (not research or len(research.strip()) < 80):
            raise RuntimeError(
                f"Missing/insufficient research for Q{extract_question_id(question)}; refusing to forecast."
            )

        llm = GeneralLlm(model=model_override) if model_override else self.get_llm("default", "llm")
        parser_llm = self.get_llm("parser", "llm")
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        base_rate = safe_community_prediction(question)
        base_rate_str = (
            f"Metaculus community prediction: {base_rate:.1%}"
            if isinstance(base_rate, (int, float))
            else "No community prediction available."
        )

        # R2, R5, R6, R8 – enriched reasoning scaffold
        resolution_date_str, days_remaining_str = _resolution_date_info(question)
        reasoning_scaffold = clean_indents(f"""
        You are an elite superforecaster with a track record of calibrated predictions.
        Think step by step. Show all your work — do not skip steps.

        CONTEXT:
        - Today (UTC): {today_utc}
        - Resolution date: {resolution_date_str}  ({days_remaining_str} remaining)
        - Time pressure: the shorter the window, the harder it is for the world to change.

        REASONING PROTOCOL — complete every step explicitly and in order:

        Step 1 — OUTSIDE VIEW & BASE RATE
          a. Name the reference class (the population of similar events this belongs to).
          b. State the historical base rate for this type of event resolving YES.
          c. Cite a specific analogous past case with its outcome.
          d. Write your outside-view probability estimate before looking at any specifics.

        Step 2 — EVIDENCE REVIEW
          For each of the 4-6 most decision-relevant facts from the research brief:
          - State the fact.
          - Rate it: [Strong / Moderate / Weak] and [Recent / Dated / Unknown age].
          - State whether it pushes probability UP, DOWN, or NEUTRAL vs. the base rate.
          Explicitly flag any evidence that is old (>6 months) or from a single source.

        Step 3 — INSIDE VIEW
          How does THIS specific situation differ from the base rate reference class?
          List up to 3 factors that make it more likely than the base rate.
          List up to 3 factors that make it less likely than the base rate.
          Quantify each adjustment (e.g. "+5pp because...", "-10pp because...").

        Step 4 — TIME & TRAJECTORY
          Given {days_remaining_str} until resolution:
          - What is the current trajectory? (improving / deteriorating / stable)
          - Is there enough time for the situation to change significantly?
          - What is the single most important thing that could change before resolution?

        Step 5 — SCENARIO ANALYSIS (assign explicit probabilities to each)
          A. Status quo / most likely scenario — probability: ____%
          B. Accelerating / upside scenario — probability: ____%
          C. Reversal / downside scenario — probability: ____%
          D. Wild card / tail risk — probability: ____%
          (These must sum to 100%.)

        Step 6 — STEELMAN THE MINORITY POSITION
          Write the strongest possible argument for the outcome you currently find LESS
          likely. After reading your own steelman, does your probability estimate change?
          If yes, update it now and explain why.

        Step 7 — RESOLUTION CRITERIA STRESS TEST
          Read the exact resolution criteria word by word.
          State precisely what must be TRUE on {resolution_date_str} for YES resolution.
          State precisely what must be TRUE on {resolution_date_str} for NO resolution.
          Are there any ambiguities in the criteria that create meaningful uncertainty?

        Step 8 — FINAL SYNTHESIS & BIAS CHECK
          Combine Steps 1-7 into a single probability.
          Explicitly check for these common LLM biases and correct if needed:
          - Recency bias: am I overweighting the most recent news?
          - Availability bias: am I overweighting dramatic/memorable events?
          - Overconfidence: is my interval too narrow given genuine uncertainty?
          - Anchoring: am I anchoring too strongly to the community prediction?
          State your final probability and a one-sentence justification.
        """).strip()

        resolution_criteria = getattr(question, "resolution_criteria", "") or ""
        background_info = getattr(question, "background_info", "") or ""
        fine_print = getattr(question, "fine_print", "") or ""

        # ── Binary ── (R6: resolution_date injected via scaffold)
        if isinstance(question, BinaryQuestion):
            prompt = clean_indents(f"""
            {reasoning_scaffold}

            ═══ QUESTION ═══
            {question.question_text}

            Background: {background_info}
            Resolution criteria: {resolution_criteria}
            Fine print: {fine_print}

            {base_rate_str}

            ═══ RESEARCH BRIEF (7-section intelligence report) ═══
            {research}

            ═══ YOUR TASK ═══
            Work through Steps 1-8 above in full. Label each step clearly.
            Show all reasoning. Do not skip steps.

            After completing Step 8, output your final answer on the very LAST line as JSON:
            {{"prediction_in_decimal": <number between 0.01 and 0.99>}}
            """).strip()

            raw = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "binary_llm")
            pred: BinaryPrediction = await structure_output(raw, BinaryPrediction, model=parser_llm)
            return clamp01(float(pred.prediction_in_decimal)), str(raw)

        # ── Multiple choice ──
        if isinstance(question, MultipleChoiceQuestion):
            options_str = "\n".join(f"  - {o}" for o in question.options)
            prompt = clean_indents(f"""
            {reasoning_scaffold}

            ═══ QUESTION ═══
            {question.question_text}

            Options (use exact spelling):
            {options_str}

            Background: {background_info}
            Resolution criteria: {resolution_criteria}
            Fine print: {fine_print}

            {base_rate_str}

            ═══ RESEARCH BRIEF (7-section intelligence report) ═══
            {research}

            ═══ YOUR TASK ═══
            Work through Steps 1-8 above. For Step 5 (Scenario Analysis), map each
            scenario to one or more of the options above with an explicit probability.
            For Step 7 (Resolution Criteria), spell out what would need to be true
            for each option to resolve.

            Assign non-zero probability to every option — even unlikely ones deserve
            at least 1% to account for unknown unknowns. Probabilities MUST sum to 1.0.

            After completing Step 8, output your final answer on the very LAST line as JSON:
            {{
              "predicted_options": [
                {{"option_name": "<exact option text>", "probability": <number>}},
                ...
              ]
            }}
            Use "option_name" (NOT "option") as the field name.
            """).strip()

            raw = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "mc_llm")
            result = await structure_output(
                raw,
                PredictedOptionList,
                model=parser_llm,
                additional_instructions=f"Options must be exactly: {question.options}",
            )
            # FIX 12 – use PredictedOption directly instead of raw dicts
            normalized = PredictedOptionList(
                predicted_options=[
                    PredictedOption(
                        option_name=(
                            o.get("option_name") or o.get("option", "")
                            if isinstance(o, dict)
                            else getattr(o, "option_name", None) or getattr(o, "option", "")
                        ),
                        probability=float(
                            o.get("probability", 0) if isinstance(o, dict) else getattr(o, "probability", 0)
                        ),
                    )
                    for o in (getattr(result, "predicted_options", []) or [])
                ]
            )
            return normalized, str(raw)

        # ── Numeric ── (R5: units sanity check, R8: Fermi estimation step)
        if isinstance(question, NumericQuestion):
            lb = getattr(question, "lower_bound", None) or getattr(question, "nominal_lower_bound", None)
            ub = getattr(question, "upper_bound", None) or getattr(question, "nominal_upper_bound", None)
            unit = getattr(question, "unit_of_measure", "") or "Infer from context"
            prompt = clean_indents(f"""
            {reasoning_scaffold}

            ═══ QUESTION ═══
            {question.question_text}

            Units: {unit}
            Hard bounds (values outside these CANNOT be correct): lower={lb}, upper={ub}
            Background: {background_info}
            Resolution criteria: {resolution_criteria}
            Fine print: {fine_print}

            {base_rate_str}

            ═══ RESEARCH BRIEF (7-section intelligence report) ═══
            {research}

            ═══ YOUR TASK ═══
            Work through Steps 1-8, with these numeric-specific additions:

            Step 1 (OUTSIDE VIEW): State the historical DISTRIBUTION of this metric,
              not just a point estimate. What is the typical range? What are historical
              highs and lows? What does the median look like across similar cases?

            Step 4 (TRAJECTORY): Where is this metric headed? State a trend rate
              (e.g. "+2% per quarter") if one can be inferred from the research.

            FERMI CHECK (between Steps 7 and 8):
              Build an independent Fermi estimate from first principles.
              E.g. for an economic figure: start from a known related quantity and
              scale it. Does your Fermi estimate match your research-derived estimate?
              If they disagree by more than 2×, explain why and which you trust more.

            UNITS SANITY CHECK (before outputting):
              Confirm your values are in the correct units ({unit}).
              Confirm all values are within [{lb}, {ub}].
              Confirm values are non-decreasing from p10 to p90.

            After completing Step 8, output your final answer on the very LAST line
            as a JSON array with exactly 6 percentiles:
            [
              {{"percentile": 0.1, "value": <number>}},
              {{"percentile": 0.2, "value": <number>}},
              {{"percentile": 0.4, "value": <number>}},
              {{"percentile": 0.6, "value": <number>}},
              {{"percentile": 0.8, "value": <number>}},
              {{"percentile": 0.9, "value": <number>}}
            ]
            Values must be strictly non-decreasing and within [{lb}, {ub}].
            """).strip()

            raw = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "num_llm")
            try:
                percentile_list: List[Percentile] = await structure_output(
                    raw, list[Percentile], model=parser_llm
                )
                target_ps = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
                interpolated = interpolate_missing_percentiles(percentile_list, target_ps)
                validated = enforce_numeric_constraints(interpolated, question)
                vals = [p.value for p in validated]
                if len(set(round(v, 12) for v in vals)) == 1:
                    validated = derive_numeric_fallback_percentiles(question, base_rate)
            except Exception:
                validated = derive_numeric_fallback_percentiles(question, base_rate)

            dist = NumericDistribution.from_question(validated, question)
            return dist, str(raw)

        raise ValueError(f"Unsupported question type: {type(question)}")

    # -------------------------------------------------------------------------
    # Binary forecast — FIX 2: committee runs concurrently; FIX 7: crowd blending
    # -------------------------------------------------------------------------
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        qid = extract_question_id(question)
        base = safe_community_prediction(question)
        base_rate_text = f"{base:.1%}" if isinstance(base, (int, float)) else "None"

        # FIX 2 – run all committee models concurrently
        async def _safe_forecast(model: str) -> Tuple[float, str]:
            try:
                p, reasoning = await self._single_forecast(question, research, model_override=model)
                return clamp01(float(p)), reasoning
            except Exception as exc:
                fallback = clamp01(float(base) if isinstance(base, (int, float)) else 0.5)
                logger.warning("Binary model fallback Q%s model=%s: %s", qid, model, exc)
                return fallback, f"(fallback due to error: {exc})"

        results = await asyncio.gather(*[_safe_forecast(m) for m in self.COMMITTEE_MODELS])
        raw_ps = [r[0] for r in results]
        # FIX 11 – capture individual reasoning snippets
        model_reasonings = [(m, r[1]) for m, r in zip(self.COMMITTEE_MODELS, results)]

        logits_arr = np.array([logit(p) for p in raw_ps], dtype=float)
        p_agg = clamp01(sigmoid(float(np.median(logits_arr))))
        p_std = float(np.std(raw_ps))
        agree = p_std <= EXTREMIZE_DISPERSION_STD_MAX

        p_final = p_agg
        if agree and should_extremize(p_agg, EXTREMIZE_THRESHOLD):
            p_final = extremize_probability(p_agg, alpha=EXTREMIZE_ALPHA)

        # FIX 7 – skip crowd blending when committee strongly disagrees with crowd
        if isinstance(base, (int, float)):
            lo_gap = abs(logit(p_final) - logit(clamp01(float(base))))
            if lo_gap <= CROWD_BLEND_DISAGREE_LOGODDS:
                w = CROWD_BLEND_MIXED if agree else CROWD_BLEND_WEAK
                p_final = clamp01(w * p_final + (1.0 - w) * float(base))
            else:
                logger.info(
                    "Q%s: skipping crowd blend (log-odds gap=%.2f > threshold=%.2f)",
                    qid, lo_gap, CROWD_BLEND_DISAGREE_LOGODDS,
                )

        # Enforce minibench extremization behavior: avoid weak middle forecasts.
        # If the committee is aligned and the aggregate is near 50%, push to
        # a decisive extreme. Otherwise, fall back to an honest 50/50.
        # FIX 18 – apply conservative adjustments for non-minibench tournaments
        is_aggressive = _is_aggressive_tournament(question)
        if is_aggressive:
            # Minibench / Market Pulse: extremize middle forecasts
            if 0.45 <= p_final <= 0.55:
                if agree and p_agg != 0.5:
                    p_final = 0.02 if p_agg < 0.5 else 0.98
                else:
                    p_final = 0.50
        else:
            # Other tournaments: conservative adjustments
            # Dampen extremization and blend more strongly with crowd/base rate
            if 0.45 <= p_final <= 0.55:
                p_final = 0.50
            else:
                # Pull extreme forecasts back toward 50% (more conservative)
                if p_final > 0.5:
                    p_final = clamp01(0.5 + (p_final - 0.5) * 0.7)
                else:
                    p_final = clamp01(0.5 - (0.5 - p_final) * 0.7)

        meta = self._research_meta.get(_question_key(question), {})
        searchers_used: List[str] = meta.get("searchers_used", []) if isinstance(meta.get("searchers_used"), list) else []

        lo_gap_str = (
            f"{abs(logit(p_final) - logit(clamp01(float(base)))):.2f}" 
            if isinstance(base, (int, float)) else "N/A"
        )
        comment = build_comment(
            question,
            forecast_text=f"{p_final:.1%}",
            base_rate_text=base_rate_text,
            how_text=_binary_how_text(
                self.COMMITTEE_MODELS, raw_ps, p_agg, p_std, agree,
                base_rate_text, lo_gap_str, p_final,
            ),
            searchers_used=searchers_used,
            models_used=self.COMMITTEE_MODELS,
            model_reasonings=model_reasonings,
        )
        log_forecast_for_calibration(question, p_final, comment, self.COMMITTEE_MODELS, True, searchers_used)
        return ReasonedPrediction(prediction_value=p_final, reasoning=comment)

    # -------------------------------------------------------------------------
    # Multiple choice — FIX 2: concurrent; FIX 12: PredictedOption directly
    # -------------------------------------------------------------------------
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        qid = extract_question_id(question)

        async def _safe_mc(model: str) -> Tuple[Dict[str, float], str]:
            try:
                pred, reasoning = await self._single_forecast(question, research, model_override=model)
                mapped: Dict[str, float] = {}
                for o in getattr(pred, "predicted_options", []) or []:
                    opt = getattr(o, "option_name", None) or (o.get("option_name") if isinstance(o, dict) else None)
                    prob = getattr(o, "probability", None) or (o.get("probability") if isinstance(o, dict) else None)
                    if opt is None:
                        continue
                    mapped[str(opt)] = max(0.0, float(prob) if prob is not None else 0.0)
                for opt in question.options:
                    mapped.setdefault(opt, 0.0)
                s = sum(mapped.values()) or 1.0
                return {k: v / s for k, v in mapped.items()}, reasoning
            except Exception as exc:
                n = len(question.options)
                logger.warning("MC model fallback Q%s model=%s: %s", qid, model, exc)
                return {opt: 1.0 / n for opt in question.options}, f"(fallback: {exc})"

        # FIX 2 – concurrent
        results = await asyncio.gather(*[_safe_mc(m) for m in self.COMMITTEE_MODELS])
        per_model_maps = [r[0] for r in results]
        model_reasonings = [(m, r[1]) for m, r in zip(self.COMMITTEE_MODELS, results)]

        option_list = list(question.options)
        mat = np.array([[m[opt] for opt in option_list] for m in per_model_maps], dtype=float)
        med = np.median(mat, axis=0)
        floor = max(1e-6, 0.01 / len(option_list))
        med = np.maximum(med, floor)
        med = med / med.sum()

        # FIX 18 – apply conservative adjustments for non-minibench tournaments
        is_aggressive = _is_aggressive_tournament(question)
        if not is_aggressive:
            # Conservative: dampen strong predictions, blend toward uniform distribution
            uniform = np.array([1.0 / len(option_list)] * len(option_list))
            med = clamp01(0.7 * med + 0.3 * uniform)
            med = med / med.sum()

        # FIX 12 – use PredictedOption directly
        out = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=opt, probability=float(p))
                for opt, p in zip(option_list, med)
            ]
        )

        meta = self._research_meta.get(_question_key(question), {})
        searchers_used: List[str] = meta.get("searchers_used", []) if isinstance(meta.get("searchers_used"), list) else []

        forecast_text = ", ".join(
            f"{o.option_name}: {o.probability:.1%}" for o in out.predicted_options
        )
        comment = build_comment(
            question,
            forecast_text=forecast_text,
            base_rate_text="(see community chart on Metaculus)",
            how_text=f"- Committee median per option + floor={floor:.6f}\n- Used research + Good Judgment updating.",
            searchers_used=searchers_used,
            models_used=self.COMMITTEE_MODELS,
            model_reasonings=model_reasonings,
        )
        log_forecast_for_calibration(
            question,
            [{"option_name": o.option_name, "probability": o.probability} for o in out.predicted_options],
            comment,
            self.COMMITTEE_MODELS,
            True,
            searchers_used,
        )
        return ReasonedPrediction(prediction_value=out, reasoning=comment)

    # -------------------------------------------------------------------------
    # Numeric — FIX 2: concurrent
    # -------------------------------------------------------------------------
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        qid = extract_question_id(question)
        base = safe_community_prediction(question)
        base_rate_text = f"{base:,.4g}" if isinstance(base, (int, float)) else "None"
        target_ps = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

        async def _safe_num(model: str) -> Tuple[List[Percentile], str]:
            try:
                dist, reasoning = await self._single_forecast(question, research, model_override=model)
                declared = list(getattr(dist, "declared_percentiles", [])) or []
                interpolated = interpolate_missing_percentiles(declared, target_ps)
                validated = enforce_numeric_constraints(interpolated, question)
                if not validated or len(validated) != 6:
                    validated = derive_numeric_fallback_percentiles(question, base)
                return validated, reasoning
            except Exception as exc:
                logger.warning("Numeric model fallback Q%s model=%s: %s", qid, model, exc)
                return derive_numeric_fallback_percentiles(question, base), f"(fallback: {exc})"

        # FIX 2 – concurrent
        results = await asyncio.gather(*[_safe_num(m) for m in self.COMMITTEE_MODELS])
        per_model_percentiles = [r[0] for r in results]
        model_reasonings = [(m, r[1]) for m, r in zip(self.COMMITTEE_MODELS, results)]

        aggregated: List[Percentile] = [
            Percentile(
                percentile=p,
                value=float(np.median([pm[i].value for pm in per_model_percentiles])),
            )
            for i, p in enumerate(target_ps)
        ]
        validated = enforce_numeric_constraints(aggregated, question)
        if not validated or len(validated) != 6:
            validated = derive_numeric_fallback_percentiles(question, base)

        # FIX 18 – apply conservative adjustments for non-minibench tournaments
        is_aggressive = _is_aggressive_tournament(question)
        if not is_aggressive and isinstance(base, (int, float)):
            # Conservative: narrow distribution around base rate estimate
            validated_conserv = []
            base_val = float(base)
            for p in validated:
                if p.percentile < 0.5:
                    # For lower percentiles, increase toward base (less extreme)
                    new_val = p.value + (base_val - p.value) * 0.3
                    validated_conserv.append(Percentile(percentile=p.percentile, value=new_val))
                elif p.percentile > 0.5:
                    # For upper percentiles, decrease toward base (less extreme)
                    new_val = p.value + (base_val - p.value) * 0.3
                    validated_conserv.append(Percentile(percentile=p.percentile, value=new_val))
                else:
                    validated_conserv.append(p)
            validated = enforce_numeric_constraints(validated_conserv, question)

        dist = NumericDistribution.from_question(validated, question)

        meta = self._research_meta.get(_question_key(question), {})
        searchers_used: List[str] = meta.get("searchers_used", []) if isinstance(meta.get("searchers_used"), list) else []

        comment = build_comment(
            question,
            forecast_text=", ".join(f"p{int(p.percentile * 100)}={p.value:,.6g}" for p in validated),
            base_rate_text=base_rate_text,
            how_text="- Committee median per percentile; enforced constraints.\n- Used research + base rates (Good Judgment updating).",
            searchers_used=searchers_used,
            models_used=self.COMMITTEE_MODELS,
            model_reasonings=model_reasonings,
        )
        log_forecast_for_calibration(
            question, [p.value for p in validated], comment, self.COMMITTEE_MODELS, True, searchers_used
        )
        return ReasonedPrediction(prediction_value=dist, reasoning=comment)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tude.")
    parser.add_argument("--tournament-ids", nargs="+", type=str, default=None)
    args = parser.parse_args()

    client = MetaculusClient()
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

    async def _main() -> None:
        # Start background calibration log writer (FIX 16)
        writer_task = asyncio.create_task(_cal_writer_task())

        try:
            all_reports = []
            for tid in tournament_ids:
                logger.info("Forecasting on tournament: %s", tid)
                for attempt in range(RETRY_MAX):
                    try:
                        reports = await bot.forecast_on_tournament(tid, return_exceptions=True)
                        all_reports.extend(reports)
                        break
                    except Exception as exc:
                        msg = str(exc).lower()
                        if any(k in msg for k in ("too many requests", "cloudflare", "1015", "429")):
                            logger.error(
                                "Rate-limited on tournament %s (attempt %d/%d): %s",
                                tid, attempt + 1, RETRY_MAX, exc,
                            )
                            await backoff_sleep_async(attempt)  # FIX 13
                            continue
                        raise
                await asyncio.sleep(TOURNAMENT_SLEEP_S)  # FIX 13

            bot.log_report_summary(all_reports)
            logger.info("Run completed. Calibration logs saved to %s", CALIBRATION_LOG_FILE)
        finally:
            # Flush the calibration log queue before exiting
            await _cal_queue.join()
            _cal_queue.put_nowait(None)  # sentinel
            await writer_task

    try:
        asyncio.run(_main())
    except Exception as exc:
        logger.error("Critical error: %s", exc, exc_info=True)
