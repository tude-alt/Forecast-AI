# main.py
# Conservative Hybrid Forecasting Bot — Tournament-Only, OpenRouter-Only
# Now with Confident-Conservative mode for high-scoring binary forecasts

import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Literal, Optional, Any

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


def extract_question_id(question: MetaculusQuestion) -> str:
    """Extract question ID from URL since .id attribute may not be exposed."""
    try:
        url = getattr(question, 'url', '')
        match = re.search(r'/questions/(\d+)', str(url))
        return match.group(1) if match else "unknown"
    except Exception:
        return "unknown"


def safe_community_prediction(question: MetaculusQuestion) -> Optional[float]:
    """Safely extract community prediction; return None if unavailable."""
    try:
        pred = getattr(question, 'community_prediction', None)
        if pred is not None and isinstance(pred, (int, float)):
            return float(pred)
        pred = getattr(question, 'prediction', None)
        if pred is not None and isinstance(pred, (int, float)):
            return float(pred)
    except Exception as e:
        logger.warning(f"Failed to get community prediction for Q{extract_question_id(question)}: {e}")
    return None


def is_research_sufficient(research: str) -> bool:
    """Heuristic: research is sufficient if it contains non-error content from at least one source."""
    if not research or "failed" in research.lower():
        return False
    cleaned = research.replace("--- SOURCE", "").replace("Tavily failed", "").replace("NewsAPI failed", "")
    return len(cleaned.strip()) > 50


def interpolate_missing_percentiles(
    reported: list[Percentile],
    target_percentiles: list[float]
) -> list[Percentile]:
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
                x0, x1 = xs[i-1], xs[i]
                y0, y1 = ys[i-1], ys[i]
                val = y0 + (y1 - y0) * (tp - x0) / (x1 - x0) if x1 != x0 else y0
        interpolated.append(Percentile(percentile=tp, value=val))
    return interpolated


def enforce_numeric_constraints(
    percentiles: list[Percentile],
    question: NumericQuestion
) -> list[Percentile]:
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
        if values[i] < values[i-1]:
            values[i] = values[i-1]

    return [Percentile(percentile=sorted_by_p[i].percentile, value=values[i]) for i in range(len(values))]


# Calibration logging helper
CALIBRATION_LOG_FILE = "forecasting_calibration_log.jsonl"

def log_forecast_for_calibration(
    question: MetaculusQuestion,
    prediction_value: Any,
    reasoning: str,
    model_ids: list[str],
    research_used: bool
):
    """Log forecast details for post-resolution scoring."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question_id": extract_question_id(question),
        "question_type": question.__class__.__name__,
        "question_text": question.question_text,
        "resolution_date": getattr(question, 'resolution_date', None),
        "community_prediction": safe_community_prediction(question),
        "prediction_value": prediction_value,
        "models_used": model_ids,
        "research_used": research_used,
        "reasoning_snippet": reasoning[:500]
    }
    try:
        with open(CALIBRATION_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log calibration data: {e}")


class ConservativeHybridBot(ForecastBot):
    """
    Conservative forecasting bot with Confident-Conservative mode:
    - Uses base rates, research, and ensemble median
    - For binary: goes extreme (>95% or <5%) ONLY when strong evidence exists
    - Otherwise stays moderate to avoid log-score penalties
    """

    _max_concurrent_questions = 3
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "summarizer": "openrouter/openai/gpt-5",
            "researcher": "openrouter/perplexity/llama-3.1-sonar-large-128k-online",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)

    # -----------------------------
    # Multi-Source Research
    # -----------------------------
    def call_tavily(self, query: str) -> str:
        if not self.tavily_client.api_key: return ""
        try:
            response = self.tavily_client.search(query=query, search_depth="advanced")
            return "\n".join([f"- {c['content']}" for c in response['results']])
        except Exception as e: return f"Tavily failed: {e}"

    def call_newsapi(self, query: str) -> str:
        if not self.newsapi_client.api_key: return ""
        try:
            articles = self.newsapi_client.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)
            if not articles or not articles.get('articles'): return ""
            return "\n".join([f"- Title: {a['title']}\n  Snippet: {a.get('description', 'N/A')}" for a in articles['articles']])
        except Exception as e: return f"NewsAPI failed: {e}"

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            loop = asyncio.get_running_loop()
            tasks = {
                "tavily": loop.run_in_executor(None, self.call_tavily, question.question_text),
                "newsapi": loop.run_in_executor(None, self.call_newsapi, question.question_text),
                "perplexity": self.get_llm("researcher", "llm").invoke(question.question_text)
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            raw_research = ""
            for i, result in enumerate(results):
                src = list(tasks.keys())[i]
                content = str(result) if not isinstance(result, Exception) else f"{src} error: {result}"
                raw_research += f"--- SOURCE {src.upper()} ---\n{content}\n\n"
            return raw_research

    # -----------------------------
    # Conservative Forecasting with Committee
    # -----------------------------
    async def _single_forecast(self, question, research: str, model_override: str = None):
        if model_override:
            self._llms["default"] = GeneralLlm(model=model_override)
            self._llms["parser"] = GeneralLlm(model="openrouter/openai/gpt-4.1-mini")

        base_rate = safe_community_prediction(question)
        if base_rate is not None:
            if isinstance(question, BinaryQuestion):
                base_rate_str = f"Community prediction (base rate): {base_rate:.1%}"
            else:
                base_rate_str = f"Community prediction (base rate): {base_rate:,.2f}"
        else:
            base_rate_str = "No reliable base rate available."

        research_sufficient = is_research_sufficient(research)
        if not research_sufficient:
            research = "(Insufficient recent research available. Relying on base rates and general knowledge.)\n" + research

        if isinstance(question, BinaryQuestion):
            prompt = clean_indents(f"""
            You are a professional forecaster known for conservative, well-calibrated predictions.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            {base_rate_str}
            Research: {research}
            Today: {datetime.now().strftime("%Y-%m-%d")}

            Consider:
            (a) Time until resolution
            (b) Status quo (world changes slowly)
            (c) Base rates — anchor to them unless strong evidence overrides

            Be humble. Avoid overconfidence. If research is weak, defer to base rate.

            End with: "Probability: ZZ%"
            """)
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            pred: BinaryPrediction = await structure_output(reasoning, BinaryPrediction, model=self.get_llm("parser", "llm"))
            result = max(0.01, min(0.99, pred.prediction_in_decimal))

        elif isinstance(question, MultipleChoiceQuestion):
            prompt = clean_indents(f"""
            Conservative forecaster mode.

            Question: {question.question_text}
            Options: {question.options}
            Background: {question.background_info}
            Resolution: {question.resolution_criteria}
            {base_rate_str}
            Research: {research}
            Today: {datetime.now().strftime("%Y-%m-%d")}

            Assign probabilities. Do not assign 0% to any option unless logically impossible.
            If uncertain, distribute probability evenly or according to base rates.

            End with probabilities for each option in order.
            """)
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            result = await structure_output(
                reasoning, PredictedOptionList, model=self.get_llm("parser", "llm"),
                additional_instructions=f"Options must be exactly: {question.options}"
            )

        elif isinstance(question, NumericQuestion):
            lower_msg = f"Lower bound: {'open' if question.open_lower_bound else 'closed'} at {question.lower_bound or question.nominal_lower_bound}"
            upper_msg = f"Upper bound: {'open' if question.open_upper_bound else 'closed'} at {question.upper_bound or question.nominal_upper_bound}"
            prompt = clean_indents(f"""
            Conservative forecaster. Set wide 90/10 intervals.

            Question: {question.question_text}
            Units: {question.unit_of_measure or 'Infer from context'}
            Background: {question.background_info}
            Resolution: {question.resolution_criteria}
            {lower_msg}
            {upper_msg}
            {base_rate_str}
            Research: {research}
            Today: {datetime.now().strftime("%Y-%m-%d")}

            IMPORTANT: The answer must be in the correct scale (e.g., if forecasting NVIDIA revenue in USD, expect values in BILLIONS, not millions or percentages).
            If the community prediction is available, it is given in the correct units—use it as an anchor.

            Provide percentiles: 10, 20, 40, 60, 80, 90.
            """)
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            percentile_list: list[Percentile] = await structure_output(reasoning, list[Percentile], model=self.get_llm("parser", "llm"))
            
            target_ps = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            interpolated = interpolate_missing_percentiles(percentile_list, target_ps)
            validated = enforce_numeric_constraints(interpolated, question)
            result = NumericDistribution.from_question(validated, question)

        if model_override:
            self._llms["default"] = GeneralLlm(model="openrouter/openai/gpt-5")
            self._llms["parser"] = GeneralLlm(model="openrouter/openai/gpt-4.1-mini")

        return result, reasoning

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-5.1",
            "openrouter/anthropic/claude-sonnet-4.5"
        ]
        forecasts = []
        reasonings = []

        for model in models:
            try:
                pred, reason = await self._single_forecast(question, research, model_override=model)
                
                # --- CONFIDENT-CONSERVATIVE ADJUSTMENT ---
                base_rate = safe_community_prediction(question) or 0.5
                research_sufficient = is_research_sufficient(research)
                should_be_extreme = False
                direction = None

                if research_sufficient:
                    extremity_prompt = clean_indents(f"""
                    You are a superforecaster evaluating evidence strength.

                    Question: {question.question_text}
                    Background: {question.background_info}
                    Resolution criteria: {question.resolution_criteria}
                    Current forecast (pre-adjustment): {pred:.1%}
                    Community prediction: {base_rate:.1%}
                    Research: {research}
                    Today: {datetime.now().strftime("%Y-%m-%d")}

                    Is there STRONG, UNAMBIGUOUS evidence that the outcome will be YES or NO?
                    - "Strong" = multiple credible sources, official announcements, irreversible trends
                    - "Unambiguous" = no plausible counter-scenarios

                    Answer ONLY: "YES", "NO", or "UNCERTAIN"
                    """)
                    try:
                        extremity_llm = self.get_llm("parser", "llm")
                        extremity_response = await extremity_llm.invoke(extremity_prompt)
                        extremity_response = extremity_response.strip().upper()
                        if extremity_response == "YES":
                            should_be_extreme = True
                            direction = "yes"
                        elif extremity_response == "NO":
                            should_be_extreme = True
                            direction = "no"
                    except Exception as e:
                        logger.debug(f"Extremity check failed for Q{extract_question_id(question)}: {e}")
                
                # Apply adjustment
                final_pred = pred
                if should_be_extreme:
                    if direction == "yes":
                        final_pred = max(pred, 0.95)
                    else:  # "no"
                        final_pred = min(pred, 0.05)
                    adjusted_reason = f"[EXTREME ADJUSTMENT: {direction.upper()}] " + reason
                else:
                    adjusted_reason = "[MODERATE] " + reason
                
                forecasts.append(max(0.01, min(0.99, final_pred)))
                reasonings.append(adjusted_reason)

            except Exception as e:
                logger.warning(f"Model {model} failed on binary Q{extract_question_id(question)}: {e}")
                base = safe_community_prediction(question) or 0.5
                forecasts.append(max(0.01, min(0.99, base)))
                reasonings.append(f"FALLBACK due to error: {e}")

        median_pred = float(np.median(forecasts))
        final_reasoning = " | ".join(reasonings)
        log_forecast_for_calibration(question, median_pred, final_reasoning, models, is_research_sufficient(research))
        return ReasonedPrediction(prediction_value=median_pred, reasoning=final_reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-5.1",
            "openrouter/anthropic/claude-sonnet-4.5"
        ]
        forecasts = []
        reasonings = []
        for model in models:
            try:
                pred, reason = await self._single_forecast(question, research, model_override=model)
                forecasts.append(pred)
                reasonings.append(reason)
            except Exception as e:
                logger.warning(f"Model {model} failed on MC Q{extract_question_id(question)}: {e}")
                n = len(question.options)
                uniform = PredictedOptionList([
                    {"option": opt, "probability": 1.0 / n} for opt in question.options
                ])
                forecasts.append(uniform)
                reasonings.append(f"FALLBACK due to error: {e}")

        all_probs = np.array([[opt["probability"] for opt in f.predicted_options] for f in forecasts])
        median_probs = np.median(all_probs, axis=0)
        if median_probs.sum() > 0:
            median_probs = median_probs / median_probs.sum()
        else:
            median_probs = np.full_like(median_probs, 1.0 / len(median_probs))
        options = forecasts[0].predicted_options
        median_forecast = PredictedOptionList([
            {"option": opt["option"], "probability": float(p)} for opt, p in zip(options, median_probs)
        ])
        final_reasoning = " | ".join(reasonings)
        log_forecast_for_calibration(question, [opt["probability"] for opt in median_forecast.predicted_options], final_reasoning, models, is_research_sufficient(research))
        return ReasonedPrediction(prediction_value=median_forecast, reasoning=final_reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-5.1",
            "openrouter/anthropic/claude-sonnet-4.5"
        ]
        forecasts = []
        reasonings = []
        for model in models:
            try:
                pred, reason = await self._single_forecast(question, research, model_override=model)
                forecasts.append(pred)
                reasonings.append(reason)
            except Exception as e:
                logger.warning(f"Model {model} failed on numeric Q{extract_question_id(question)}: {e}")
                base = safe_community_prediction(question)
                lower_b = question.lower_bound or 0
                upper_b = question.upper_bound or 1
                if base is not None:
                    center = float(base)
                    width = (upper_b - lower_b) * 0.2
                else:
                    center = (lower_b + upper_b) / 2
                    width = (upper_b - lower_b) * 0.3

                fallback_vals = [
                    center - width * 0.8,
                    center - width * 0.4,
                    center - width * 0.1,
                    center + width * 0.1,
                    center + width * 0.4,
                    center + width * 0.8,
                ]
                fallback_vals = [max(lower_b, min(upper_b, v)) for v in fallback_vals]
                
                fallback_ps = [
                    Percentile(percentile=p, value=v)
                    for p, v in zip([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], fallback_vals)
                ]
                fallback_dist = NumericDistribution.from_question(fallback_ps, question)
                forecasts.append(fallback_dist)
                reasonings.append(f"FALLBACK due to error: {e}")

        target_percentiles = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        aggregated = []
        for p in target_percentiles:
            values = []
            for f in forecasts:
                for item in f.declared_percentiles:
                    if abs(item.percentile - p) < 0.01:
                        values.append(item.value)
                        break
                else:
                    values.append((question.lower_bound + question.upper_bound) / 2)
            median_val = float(np.median(values))
            aggregated.append(Percentile(percentile=p, value=median_val))

        validated = enforce_numeric_constraints(aggregated, question)
        distribution = NumericDistribution.from_question(validated, question)
        final_reasoning = " | ".join(reasonings)
        log_forecast_for_calibration(question, [p.value for p in validated], final_reasoning, models, is_research_sufficient(research))
        return ReasonedPrediction(prediction_value=distribution, reasoning=final_reasoning)


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
