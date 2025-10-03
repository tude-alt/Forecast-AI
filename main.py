# main.py
# A forecasting bot using a hybrid of tailored, median-aggregated analytical techniques.

import argparse
import asyncio
import json
import logging
import os
import warnings
from datetime import datetime
from typing import Literal, List, Dict, Any

import numpy as np
import requests
from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    ReasonedPrediction,
    clean_indents,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
    NumericDistribution,
    Percentile,
)

from tavily import TavilyClient
from newsapi import NewsApiClient

# -----------------------------
# Environment & API Keys
# -----------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HybridPreMortemBot")

# -----------------------------
# Suppress noisy warnings
# -----------------------------
warnings.filterwarnings("ignore", message=".*does not support cost tracking.*")


class HybridPreMortemBot(ForecastBot):
    """
    This bot uses a hybrid of analytical techniques tailored to each question type.
    The final prediction for ALL types is the median of the synthesizer committee's judgments.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        """Provide defaults for all LLM roles used internally or expected by base class."""
        return {
            # Base class expected roles (to suppress warnings)
            "summarizer": "openrouter/openai/gpt-4o-mini",
            "researcher": "openrouter/openai/gpt-4o-search-preview",
            # Custom pipeline roles
            "online_researcher": "openrouter/perplexity/llama-3-sonar-large-32k-online",
            "research_synthesizer": "openrouter/openai/gpt-4o",
            "pre_mortem_agent": "openrouter/openai/gpt-5",
            "pre_parade_agent": "openrouter/openai/gpt-4.1",
            "advocate_agent": "openrouter/openai/gpt-4.1-nano",
            "risk_synthesizer": "openrouter/openai/gpt-5",
            "synthesizer_1": "openrouter/openai/o3",
            "synthesizer_2": "openrouter/openai/gpt-5",
            "synthesizer_3": "openrouter/anthropic/claude-sonnet-4",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)
        self.serpapi_key = SERPAPI_API_KEY
        self.synthesizer_keys = [k for k in self._llms.keys() if k.startswith("synthesizer")]
        if not self.synthesizer_keys:
            raise ValueError("No synthesizer models found in LLM configuration.")
        logger.info(f"Initialized with Hybrid analysis pipeline and a committee of {len(self.synthesizer_keys)} synthesizers.")

    # -----------------------------
    # External Research Methods
    # -----------------------------
    def call_tavily(self, query: str) -> str:
        if not self.tavily_client.api_key: return "Tavily search not performed."
        try:
            response = self.tavily_client.search(query=query, search_depth="advanced")
            return "\n".join([f"- {c['content']}" for c in response['results']])
        except Exception as e: return f"Tavily search failed: {e}"

    def call_newsapi(self, query: str) -> str:
        if not self.newsapi_client.api_key: return "NewsAPI search not performed."
        try:
            articles = self.newsapi_client.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)
            if not articles or not articles.get('articles'): return "No recent news articles found."
            return "\n".join([f"- Title: {a['title']}\n  Snippet: {a.get('description', 'N/A')}" for a in articles['articles']])
        except Exception as e: return f"NewsAPI search failed: {e}"

    def call_serpapi(self, query: str) -> str:
        if not self.serpapi_key: return "SerpApi search not performed."
        url = "https://serpapi.com/search.json"  # Fixed: no trailing space
        params = {"q": query, "api_key": self.serpapi_key}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            snippets = [result.get('snippet', '') for result in data.get('organic_results', [])]
            return "\n".join([f"- {s}" for s in snippets if s])
        except Exception as e: return f"SerpApi search failed: {e}"

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            logger.info(f"--- Starting All-Source Research for: {question.question_text} ---")
            loop = asyncio.get_running_loop()
            tasks = {
                "tavily": loop.run_in_executor(None, self.call_tavily, question.question_text),
                "newsapi": loop.run_in_executor(None, self.call_newsapi, question.question_text),
                "serpapi": loop.run_in_executor(None, self.call_serpapi, question.question_text),
                "perplexity": self.get_llm("online_researcher", "llm").invoke(question.question_text)
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            source_names = list(tasks.keys())
            raw_research_dump = ""
            for i, result in enumerate(results):
                raw_research_dump += f"--- RAW DATA FROM: {source_names[i].upper()} ---\n{result}\n\n"
            synthesis_prompt = clean_indents(f"""
            You are a senior intelligence analyst. Your job is to synthesize raw, potentially redundant 
            data from multiple sources into a single, clean, and comprehensive briefing for a team of superforecasters.  

            Raw Data Dump:
            {raw_research_dump}

            Synthesized Intelligence Briefing:
            """)
            final_briefing = await self.get_llm("research_synthesizer", "llm").invoke(synthesis_prompt)
            logger.info(f"--- All-Source Research Complete for Q {question.page_url} ---")
            return final_briefing

    # -----------------------------
    # Single Synthesizer Forecasters
    # -----------------------------
    async def _single_binary_forecast(self, synthesizer_key: str, question: BinaryQuestion, research: str) -> Dict[str, Any]:
        prompt = clean_indents(f"""
        You are a superforecaster. Based on the intelligence briefing below, estimate the probability (between 0.0 and 1.0) 
        that the following event will occur.

        Question: {question.question_text}
        Briefing: {research}

        Respond ONLY with a JSON object in this exact format:
        {{"probability": 0.75, "reasoning": "Concise justification..."}}
        """)
        llm = self.get_llm(synthesizer_key, "llm")
        response = await llm.invoke(prompt)
        try:
            parsed = json.loads(response.strip())
            prob = float(parsed["probability"])
            prob = max(0.0, min(1.0, prob))
            return {"probability": prob, "reasoning": parsed.get("reasoning", "")}
        except Exception as e:
            logger.warning(f"Synthesizer {synthesizer_key} returned invalid JSON: {e}. Using 0.5 as fallback.")
            return {"probability": 0.5, "reasoning": "Fallback due to parsing error."}

    async def _single_mc_forecast(self, synthesizer_key: str, question: MultipleChoiceQuestion, research: str) -> Dict[str, Any]:
        options_str = "\n".join([f"- {opt}" for opt in question.options])
        prompt = clean_indents(f"""
        You are a superforecaster. Assign probabilities to each option below so they sum to 1.0.

        Question: {question.question_text}
        Options:
        {options_str}
        Briefing: {research}

        Respond ONLY with a JSON object like:
        {{"probabilities": [0.2, 0.5, 0.3], "reasoning": "..."}}
        """)
        llm = self.get_llm(synthesizer_key, "llm")
        response = await llm.invoke(prompt)
        try:
            parsed = json.loads(response.strip())
            probs = [float(p) for p in parsed["probabilities"]]
            total = sum(probs)
            if total == 0:
                probs = [1.0 / len(probs)] * len(probs)
            else:
                probs = [p / total for p in probs]
            return {"probabilities": probs, "reasoning": parsed.get("reasoning", "")}
        except Exception as e:
            logger.warning(f"Synthesizer {synthesizer_key} MC parsing failed: {e}. Uniform fallback.")
            uniform = [1.0 / len(question.options)] * len(question.options)
            return {"probabilities": uniform, "reasoning": "Fallback due to parsing error."}

    async def _single_numeric_forecast(self, synthesizer_key: str, question: NumericQuestion, research: str) -> Dict[str, Any]:
        # Safely get unit info; fallback if not available
        unit_info = getattr(question, 'unit', getattr(question, 'units', 'Not specified'))

        prompt = clean_indents(f"""
        You are a superforecaster. Provide 10th, 50th (median), and 90th percentiles for the numeric outcome.

        Question: {question.question_text}
        Expected unit/format: {unit_info}
        Research briefing: {research}

        Respond ONLY with a JSON object like:
        {{"percentiles": [{{"percentile": 10, "value": -2.1}}, {{"percentile": 50, "value": 1.5}}, {{"percentile": 90, "value": 5.8}}], "reasoning": "..."}}
        """)
        llm = self.get_llm(synthesizer_key, "llm")
        response = await llm.invoke(prompt)
        try:
            parsed = json.loads(response.strip())
            percentiles = []
            for p in parsed["percentiles"]:
                percentiles.append({"percentile": int(p["percentile"]), "value": float(p["value"])})
            return {"percentiles": percentiles, "reasoning": parsed.get("reasoning", "")}
        except Exception as e:
            logger.warning(f"Synthesizer {synthesizer_key} numeric parsing failed: {e}. Using symmetric default.")
            if "exceed" in question.question_text.lower():
                return {
                    "percentiles": [
                        {"percentile": 10, "value": -5.0},
                        {"percentile": 50, "value": 0.0},
                        {"percentile": 90, "value": 5.0}
                    ],
                    "reasoning": "Fallback for relative return question."
                }
            else:
                return {
                    "percentiles": [
                        {"percentile": 10, "value": 0.1},
                        {"percentile": 50, "value": 0.5},
                        {"percentile": 90, "value": 0.9}
                    ],
                    "reasoning": "Generic fallback."
                }

    # -----------------------------
    # Median-Aggregated Forecast Pipelines
    # -----------------------------
    async def _forecast_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction:
        forecasts = await asyncio.gather(*[
            self._single_binary_forecast(key, question, research)
            for key in self.synthesizer_keys
        ])
        probabilities = [f["probability"] for f in forecasts]
        median_prob = float(np.median(probabilities))
        combined_reasoning = " | ".join([f["reasoning"] for f in forecasts])
        forecast = BinaryPrediction(probability_yes=median_prob)
        return ReasonedPrediction(forecast=forecast, reasoning=combined_reasoning)

    async def _forecast_mc(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction:
        forecasts = await asyncio.gather(*[
            self._single_mc_forecast(key, question, research)
            for key in self.synthesizer_keys
        ])
        all_probs = np.array([f["probabilities"] for f in forecasts])
        median_probs = np.median(all_probs, axis=0)
        total = median_probs.sum()
        if total > 0:
            median_probs = median_probs / total
        else:
            median_probs = np.full_like(median_probs, 1.0 / len(median_probs))
        forecast_list = [
            {"option": opt, "probability": float(p)}
            for opt, p in zip(question.options, median_probs)
        ]
        forecast = PredictedOptionList(forecast_list)
        combined_reasoning = " | ".join([f["reasoning"] for f in forecasts])
        return ReasonedPrediction(forecast=forecast, reasoning=combined_reasoning)

    async def _forecast_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction:
        forecasts = await asyncio.gather(*[
            self._single_numeric_forecast(key, question, research)
            for key in self.synthesizer_keys
        ])
        target_percentiles = [10, 50, 90]  # LLM uses 10/50/90 (human-readable)
        aggregated = []
        for p in target_percentiles:
            values = []
            for f in forecasts:
                for item in f["percentiles"]:
                    if item["percentile"] == p:
                        values.append(item["value"])
                        break
                else:
                    values.append(0.0)
            median_val = float(np.median(values))
            # 🔥 CRITICAL FIX: Convert 10 → 0.10, 50 → 0.50, 90 → 0.90
            aggregated.append(Percentile(percentile=p / 100.0, value=median_val))
        distribution = NumericDistribution(aggregated)
        combined_reasoning = " | ".join([f["reasoning"] for f in forecasts])
        return ReasonedPrediction(forecast=distribution, reasoning=combined_reasoning)

    # -----------------------------
    # Abstract method implementations
    # -----------------------------
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction:
        return await self._forecast_binary(question, research)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction:
        return await self._forecast_mc(question, research)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction:
        return await self._forecast_numeric(question, research)


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Hybrid Pre-Mortem Bot.")
    parser.add_argument("--mode", type=str, choices=["tournament", "test_questions"], default="tournament")
    parser.add_argument("--tournament-ids", nargs='+', type=str)
    args = parser.parse_args()
    run_mode: Literal["tournament", "test_questions"] = args.mode

    all_source_bot = HybridPreMortemBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        # llms dict is optional since _llm_config_defaults covers all
    )

    try:
        if run_mode == "tournament":
            logger.info("Running in tournament mode...")
            tournament_ids_to_run = args.tournament_ids or ['32813', 'market-pulse-25q4', MetaculusApi.CURRENT_MINIBENCH_ID]
            logger.info(f"Targeting tournaments: {tournament_ids_to_run}")
            all_reports = []
            for tournament_id in tournament_ids_to_run:
                reports = asyncio.run(all_source_bot.forecast_on_tournament(tournament_id, return_exceptions=True))
                all_reports.extend(reports)
            forecast_reports = all_reports
        elif run_mode == "test_questions":
            logger.info("Running in test questions mode...")
            # FIXED: No trailing spaces; includes extinction + Q4 2025 questions
            EXAMPLE_QUESTIONS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
                "https://www.metaculus.com/questions/40046/",
                "https://www.metaculus.com/questions/40045/",
                "https://www.metaculus.com/questions/40044/",
                "https://www.metaculus.com/questions/40043/",
                "https://www.metaculus.com/questions/40042/",
                "https://www.metaculus.com/questions/40041/",
                "https://www.metaculus.com/questions/40040/",
                "https://www.metaculus.com/questions/40038/",
            ]
            questions = [MetaculusApi.get_question_by_url(url.strip()) for url in EXAMPLE_QUESTIONS]
            forecast_reports = asyncio.run(all_source_bot.forecast_questions(questions, return_exceptions=True))
        all_source_bot.log_report_summary(forecast_reports)
        logger.info("Run finished successfully.")
    except Exception as e:
        logger.error(f"Run failed with a critical error: {e}", exc_info=True)
