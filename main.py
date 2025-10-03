# A forecasting bot using a hybrid of tailored, median-aggregated analytical techniques.
import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

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
    structure_output,
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


class HybridPreMortemBot(ForecastBot):
    """
    This bot uses a hybrid of analytical techniques tailored to each question type.
    The final prediction for ALL types is the median of the synthesizer committee's judgments.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        defaults = super()._llm_config_defaults()
        defaults.update({
            "online_researcher": "openrouter/perplexity/llama-3-sonar-large-32k-online",
            "research_synthesizer": "openrouter/openai/gpt-4o",
            "pre_mortem_agent": "openrouter/anthropic/claude-3-opus-20240229",
            "pre_parade_agent": "openrouter/qwen/qwen-2-72b-instruct",
            "advocate_agent": "openrouter/mistralai/mistral-large-latest",
            "risk_synthesizer": "openrouter/openai/gpt-5",
            "synthesizer_1": "openrouter/openai/o3",
            "synthesizer_2": "openrouter/openai/o4-mini",
            "synthesizer_3": "openrouter/anthropic/claude-3-opus-20240229",
        })
        return defaults

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.newsapi_client = NewsApiClient(api_key=NEWSAPI_API_KEY)
        self.serpapi_key = SERPAPI_API_KEY
        self.synthesizer_keys = [k for k in self.llms.keys() if k.startswith("synthesizer")]
        if not self.synthesizer_keys:
            raise ValueError("No synthesizer models found in LLM configuration.")
        logger.info(f"Initialized with Hybrid analysis pipeline and a committee of {len(self.synthesizer_keys)} synthesizers.")

    # --- All-Source Research (Used by all question types) ---
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
        url = "https://serpapi.com/search.json"
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
            tasks = { "tavily": loop.run_in_executor(None, self.call_tavily, question.question_text), "newsapi": loop.run_in_executor(None, self.call_newsapi, question.question_text), "serpapi": loop.run_in_executor(None, self.call_serpapi, question.question_text), "perplexity": self.get_llm("online_researcher", "llm").invoke(question.question_text) }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            source_names = list(tasks.keys())
            raw_research_dump = ""
            for i, result in enumerate(results): raw_research_dump += f"--- RAW DATA FROM: {source_names[i].upper()} ---\n{result}\n\n"
            synthesis_prompt = clean_indents(f"""You are a senior intelligence analyst. Your job is to synthesize raw, potentially redundant data from multiple sources into a single, clean, and comprehensive briefing for a team of superforecasters. Raw Data Dump: {raw_research_dump} Synthesized Intelligence Briefing:""")
            final_briefing = await self.get_llm("research_synthesizer", "llm").invoke(synthesis_prompt)
            logger.info(f"--- All-Source Research Complete for Q {question.page_url} ---")
            return final_briefing

    # --- FORECASTING LOGIC FOR EACH QUESTION TYPE ---

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        logger.info(f"--- Starting Pre-Mortem Analysis for Binary Q: {question.page_url} ---")
        pre_mortem_prompt = clean_indents(f"""It is the day after this question resolves: "{question.question_text}". Against expectations, the final outcome was NO. As a journalist write a plausible, detailed story explaining the failure. Research: {research}""")
        failure_narrative = await self.get_llm("pre_mortem_agent", "llm").invoke(pre_mortem_prompt)
        pre_parade_prompt = clean_indents(f"""It is the day after this question resolves: "{question.question_text}". Against expectations, the final outcome was YES. As a journalist write a plausible, detailed story explaining the success. Research: {research}""")
        success_narrative = await self.get_llm("pre_parade_agent", "llm").invoke(pre_parade_prompt)
        risk_synthesis_prompt = clean_indents(f"""Analyze the two narratives. Extract a structured list of key insights. Output ONLY two bulleted lists: - IDENTIFIED RISKS (from failure story) - IDENTIFIED OPPORTUNITIES (from success story). FAILURE NARRATIVE: {failure_narrative} SUCCESS NARRATIVE: {success_narrative}""")
        risk_opportunity_list = await self.get_llm("risk_synthesizer", "llm").invoke(risk_synthesis_prompt)
        final_judgment_prompt = clean_indents(f"""You are a superforecaster on a final review committee. Question: "{question.question_text}" Initial Research: {research} Scenario Analysis: {risk_opportunity_list} Write your final rationale, and conclude with your final probability as: "Probability: ZZ%".""")
        tasks = [self.get_llm(key, "llm").invoke(final_judgment_prompt) for key in self.synthesizer_keys]
        reasonings = await asyncio.gather(*tasks, return_exceptions=True)
        synthesizer_reasonings_dict = dict(zip(self.synthesizer_keys, reasonings))
        parsing_tasks = [structure_output(r, BinaryPrediction, self.get_llm("parser", "llm")) for r in reasonings if not isinstance(r, Exception)]
        predictions = await asyncio.gather(*parsing_tasks, return_exceptions=True)
        valid_preds = [p.prediction_in_decimal for p in predictions if not isinstance(p, Exception)]
        if not valid_preds: raise ValueError("All synthesizer predictions failed parsing for binary question.")
        median_pred = float(np.median(valid_preds))
        final_pred = max(0.01, min(0.99, median_pred))
        combined_comment = self._format_pre_mortem_comment(failure_narrative, success_narrative, risk_opportunity_list, synthesizer_reasonings_dict)
        logger.info(f"Forecasted Binary Q {question.page_url} with median prediction: {final_pred}")
        return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_comment)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        logger.info(f"--- Starting 'Case for Each Option' Analysis for MC Q: {question.page_url} ---")
        advocate_tasks = []
        for option in question.options:
            prompt = clean_indents(f"""You are an advocate for one outcome. Question: "{question.question_text}" Your assigned option is: "{option}". Based on the research, construct the strongest case for why "{option}" is the most likely outcome. Research: {research}""")
            advocate_tasks.append(self.get_llm("advocate_agent", "llm").invoke(prompt))
        arguments = await asyncio.gather(*advocate_tasks, return_exceptions=True)
        all_arguments_text = ""
        for i, arg in enumerate(arguments): all_arguments_text += f"--- CASE FOR OPTION '{question.options[i]}' ---\n{arg}\n\n"
        final_judgment_prompt = clean_indents(f"""You are on a final review committee. Question: "{question.question_text}" Options: {question.options} Research: {research} Advocates have made a case for each option: {all_arguments_text} Write your rationale, and conclude with your final probabilities for all options in the format: Option A: Prob A%, Option B: Prob B%...""")
        tasks = [self.get_llm(key, "llm").invoke(final_judgment_prompt) for key in self.synthesizer_keys]
        reasonings = await asyncio.gather(*tasks, return_exceptions=True)
        synthesizer_reasonings_dict = dict(zip(self.synthesizer_keys, reasonings))
        parsing_tasks = [structure_output(r, PredictedOptionList, self.get_llm("parser", "llm")) for r in reasonings if not isinstance(r, Exception)]
        predictions = await asyncio.gather(*parsing_tasks, return_exceptions=True)
        valid_preds = [p for p in predictions if not isinstance(p, Exception)]
        if not valid_preds: raise ValueError("All synthesizer predictions failed parsing for MC question.")
        
        # --- CORRECTED LOGIC: Use MEDIAN instead of MEAN ---
        median_probs = {}
        for option in question.options:
            probs_for_option = [p.get_prob(option) for p in valid_preds if p.get_prob(option) is not None]
            if probs_for_option:
                median_probs[option] = np.median(probs_for_option)
            else:
                median_probs[option] = 0.0

        # Normalize the median probabilities to sum to 1.0
        total_prob = sum(median_probs.values())
        if total_prob == 0: # Avoid division by zero if all medians are 0
             final_probs = {option: 1.0 / len(question.options) for option in question.options}
        else:
            final_probs = {option: prob / total_prob for option, prob in median_probs.items()}
        
        final_prediction = PredictedOptionList(list(final_probs.items()))
        combined_comment = self._format_mc_comment(all_arguments_text, synthesizer_reasonings_dict)
        logger.info(f"Forecasted MC Q {question.page_url} with median prediction: {final_prediction}")
        return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_comment)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        logger.info(f"--- Starting High/Low Scenario Analysis for Numeric Q: {question.page_url} ---")
        low_scenario_prompt = clean_indents(f"""It is the day after this question resolves: "{question.question_text}". The final number was surprisingly LOW. Write a plausible story explaining the factors that suppressed the value. Research: {research}""")
        low_narrative = await self.get_llm("pre_mortem_agent", "llm").invoke(low_scenario_prompt)
        high_scenario_prompt = clean_indents(f"""It is the day after this question resolves: "{question.question_text}". The final number was surprisingly HIGH. Write a plausible story explaining the factors that inflated the value. Research: {research}""")
        high_narrative = await self.get_llm("pre_parade_agent", "llm").invoke(high_scenario_prompt)
        risk_synthesis_prompt = clean_indents(f"""Analyze the two narratives. Extract a structured list of factors. Output ONLY two bulleted lists: - DOWNSIDE FACTORS (from low story) - UPSIDE FACTORS (from high story). LOW NARRATIVE: {low_narrative} HIGH NARRATIVE: {high_narrative}""")
        upside_downside_list = await self.get_llm("risk_synthesizer", "llm").invoke(risk_synthesis_prompt)
        final_judgment_prompt = clean_indents(f"""You are a superforecaster on a final review committee. Question: "{question.question_text}" Research: {research} Scenario analysis yielded: {upside_downside_list} Write your final rationale, and conclude with your final distribution in the format: Percentile 10: XX, Percentile 50: XX, Percentile 90: XX.""")
        tasks = [self.get_llm(key, "llm").invoke(final_judgment_prompt) for key in self.synthesizer_keys]
        reasonings = await asyncio.gather(*tasks, return_exceptions=True)
        synthesizer_reasonings_dict = dict(zip(self.synthesizer_keys, reasonings))
        parsing_tasks = [structure_output(r, list[Percentile], self.get_llm("parser", "llm")) for r in reasonings if not isinstance(r, Exception)]
        predictions = await asyncio.gather(*parsing_tasks, return_exceptions=True)
        valid_preds = [p for p in predictions if not isinstance(p, Exception)]
        if not valid_preds: raise ValueError("All synthesizer predictions failed parsing for numeric question.")
        
        # --- CORRECTED LOGIC: Use MEDIAN for each percentile level ---
        median_percentiles = []
        percentile_levels = sorted({p.percentile for pred_list in valid_preds for p in pred_list})
        for level in percentile_levels:
            values = [p.value for pred_list in valid_preds for p in pred_list if p.percentile == level]
            if values: median_percentiles.append(Percentile(percentile=level, value=np.median(values)))
        
        final_prediction = NumericDistribution.from_question(median_percentiles, question)
        combined_comment = self._format_numeric_comment(low_narrative, high_narrative, upside_downside_list, synthesizer_reasonings_dict)
        logger.info(f"Forecasted Numeric Q {question.page_url} with median prediction: {final_prediction.declared_percentiles}")
        return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_comment)

    # --- Comment Formatting Helpers ---
    def _format_pre_mortem_comment(self, failure_narrative, success_narrative, risk_list, synth_reasonings) -> str:
        comment = "--- SCENARIO ANALYSIS STAGE ---\n\n"
        comment += f"--- Pre-Mortem Narrative (Case for NO) from {self.get_llm('pre_mortem_agent', 'model_name')} ---\n\n{failure_narrative}\n\n"
        comment += f"--- Pre-Parade Narrative (Case for YES) from {self.get_llm('pre_parade_agent', 'model_name')} ---\n\n{success_narrative}\n\n"
        comment += f"--- Synthesized Risks & Opportunities from {self.get_llm('risk_synthesizer', 'model_name')} ---\n\n{risk_list}\n\n"
        comment += "--- FINAL JUDGMENT STAGE ---\n\n"
        for agent_key, reasoning in synth_reasonings.items():
            model_name = self.get_llm(agent_key, "model_name")
            comment += f"--- Final Analysis from {agent_key} ({model_name}) ---\n\n"
            comment += f"ERROR: {reasoning}\n\n" if isinstance(reasoning, Exception) else f"{reasoning}\n\n"
        return comment
    def _format_mc_comment(self, all_arguments, synth_reasonings) -> str:
        comment = "--- ADVOCATE ANALYSIS STAGE ---\n\n"
        comment += all_arguments
        comment += "--- FINAL JUDGMENT STAGE ---\n\n"
        for agent_key, reasoning in synth_reasonings.items():
            model_name = self.get_llm(agent_key, "model_name")
            comment += f"--- Final Analysis from {agent_key} ({model_name}) ---\n\n"
            comment += f"ERROR: {reasoning}\n\n" if isinstance(reasoning, Exception) else f"{reasoning}\n\n"
        return comment
    def _format_numeric_comment(self, low_narrative, high_narrative, upside_downside_list, synth_reasonings) -> str:
        comment = "--- HIGH/LOW SCENARIO STAGE ---\n\n"
        comment += f"--- Low Outcome Narrative from {self.get_llm('pre_mortem_agent', 'model_name')} ---\n\n{low_narrative}\n\n"
        comment += f"--- High Outcome Narrative from {self.get_llm('pre_parade_agent', 'model_name')} ---\n\n{high_narrative}\n\n"
        comment += f"--- Synthesized Factors from {self.get_llm('risk_synthesizer', 'model_name')} ---\n\n{upside_downside_list}\n\n"
        comment += "--- FINAL JUDGMENT STAGE ---\n\n"
        for agent_key, reasoning in synth_reasonings.items():
            model_name = self.get_llm(agent_key, "model_name")
            comment += f"--- Final Analysis from {agent_key} ({model_name}) ---\n\n"
            comment += f"ERROR: {reasoning}\n\n" if isinstance(reasoning, Exception) else f"{reasoning}\n\n"
        return comment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Hybrid Pre-Mortem Bot.")
    parser.add_argument( "--mode", type=str, choices=["tournament", "test_questions"], default="tournament")
    parser.add_argument( "--tournament-ids", nargs='+', type=str)
    args = parser.parse_args()
    run_mode: Literal["tournament", "test_questions"] = args.mode
    all_source_bot = HybridPreMortemBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(model="openrouter/openai/gpt-4o-mini"),
            "parser": GeneralLlm(model="openrouter/openai/gpt-4o"),
            "online_researcher": GeneralLlm(model="openrouter/perplexity/llama-3-sonar-large-32k-online"),
            "research_synthesizer": GeneralLlm(model="openrouter/openai/gpt-4o", temperature=0.1),
            "pre_mortem_agent": GeneralLlm(model="openrouter/anthropic/claude-3-opus-20240229", temperature=0.5),
            "pre_parade_agent": GeneralLlm(model="openrouter/qwen/qwen-2-72b-instruct", temperature=0.5),
            "advocate_agent": GeneralLlm(model="openrouter/mistralai/mistral-large-latest", temperature=0.4),
            "risk_synthesizer": GeneralLlm(model="openrouter/openai/o3-mini", temperature=0.1),
            "synthesizer_1": GeneralLlm(model="openrouter/openai/o3", temperature=0.2),
            "synthesizer_2": GeneralLlm(model="openrouter/openai/o4-mini", temperature=0.2),
            "synthesizer_3": GeneralLlm(model="openrouter/anthropic/claude-3-opus-20240229", temperature=0.2),
        },
    )
    try:
        if run_mode == "tournament":
            logger.info("Running in tournament mode...")
            tournament_ids_to_run = args.tournament_ids or ['32813', MetaculusApi.CURRENT_MINIBENCH_ID]
            logger.info(f"Targeting tournaments: {tournament_ids_to_run}")
            all_reports = []
            for tournament_id in tournament_ids_to_run:
                reports = asyncio.run(all_source_bot.forecast_on_tournament(tournament_id, return_exceptions=True))
                all_reports.extend(reports)
            forecast_reports = all_reports
        elif run_mode == "test_questions":
            logger.info("Running in test questions mode...")
            EXAMPLE_QUESTIONS = ["https://www.metaculus.com/questions/578/human-extinction-by-2100/"]
            questions = [MetaculusApi.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
            forecast_reports = asyncio.run(all_source_bot.forecast_questions(questions, return_exceptions=True))
        all_source_bot.log_report_summary(forecast_reports)
        logger.info("Run finished successfully.")
    except Exception as e:
        logger.error(f"Run failed with a critical error: {e}", exc_info=True)


