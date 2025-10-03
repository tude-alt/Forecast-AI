# main.py
# Hybrid Pre-Mortem Bot — Tournament-Only, OpenRouter-Only Models

import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal, List, Any

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

logger = logging.getLogger(__name__)


class HybridPreMortemBot(ForecastBot):
    """
    Tournament-only bot using only OpenRouter models:
    - gpt-5 (default, parser, summarizer)
    - perplexity/llama-3-sonar-large-32k-online (researcher)
    - claude-4 (committee member)
    Uses median aggregation across 3 forecasts per question.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-5",
            "summarizer": "openrouter/openai/gpt-5",
            "researcher": "openrouter/perplexity/llama-3-sonar-large-32k-online",
        }

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )
            research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number}."
        return upper_bound_message, lower_bound_message

    # -----------------------------
    # Single Forecast Functions (used by committee)
    # -----------------------------
    async def _single_binary_forecast(self, question: BinaryQuestion, research: str) -> float:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        return max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

    async def _single_mc_forecast(self, question: MultipleChoiceQuestion, research: str) -> PredictedOptionList:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        return await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )

    async def _single_numeric_forecast(self, question: NumericQuestion, research: str) -> NumericDistribution:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        return NumericDistribution.from_question(percentile_list, question)

    # -----------------------------
    # Committee Aggregation
    # -----------------------------
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        # Run 3 forecasts: 2× gpt-5, 1× claude-4
        forecasts = []
        for i in range(3):
            if i == 2:
                # Temporarily override to claude-4 for third forecast
                self._llms["default"] = GeneralLlm(model="openrouter/anthropic/claude-4")
                self._llms["parser"] = GeneralLlm(model="openrouter/anthropic/claude-4")
            pred = await self._single_binary_forecast(question, research)
            forecasts.append(pred)
            if i == 2:
                # Restore gpt-5
                self._llms["default"] = GeneralLlm(model="openrouter/openai/gpt-5")
                self._llms["parser"] = GeneralLlm(model="openrouter/openai/gpt-5")
        median_pred = float(np.median(forecasts))
        reasoning = f"Median of {len(forecasts)} forecasts: {forecasts}"
        return ReasonedPrediction(prediction_value=median_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        forecasts = []
        for i in range(3):
            if i == 2:
                self._llms["default"] = GeneralLlm(model="openrouter/anthropic/claude-4")
                self._llms["parser"] = GeneralLlm(model="openrouter/anthropic/claude-4")
            pred = await self._single_mc_forecast(question, research)
            forecasts.append(pred)
            if i == 2:
                self._llms["default"] = GeneralLlm(model="openrouter/openai/gpt-5")
                self._llms["parser"] = GeneralLlm(model="openrouter/openai/gpt-5")
        # Median per option
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
        reasoning = f"Median of {len(forecasts)} forecasts"
        return ReasonedPrediction(prediction_value=median_forecast, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        forecasts = []
        for i in range(3):
            if i == 2:
                self._llms["default"] = GeneralLlm(model="openrouter/anthropic/claude-4")
                self._llms["parser"] = GeneralLlm(model="openrouter/anthropic/claude-4")
            pred = await self._single_numeric_forecast(question, research)
            forecasts.append(pred)
            if i == 2:
                self._llms["default"] = GeneralLlm(model="openrouter/openai/gpt-5")
                self._llms["parser"] = GeneralLlm(model="openrouter/openai/gpt-5")
        # Aggregate percentiles
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
                    values.append(0.0)
            median_val = float(np.median(values))
            aggregated.append(Percentile(percentile=p, value=median_val))
        # Reconstruct distribution using question bounds
        distribution = NumericDistribution.from_question(aggregated, question)
        reasoning = f"Median of {len(forecasts)} numeric forecasts"
        return ReasonedPrediction(prediction_value=distribution, reasoning=reasoning)


# -----------------------------
# Entrypoint — Tournament Only
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run Hybrid Pre-Mortem Bot on tournaments.")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=["32813", "market-pulse-25q4", MetaculusApi.CURRENT_MINIBENCH_ID],
        help="Tournament IDs to forecast on",
    )
    args = parser.parse_args()

    bot = HybridPreMortemBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # We handle committee internally
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    try:
        logger.info("Starting tournament-mode forecasting...")
        all_reports = []
        for tid in args.tournament_ids:
            logger.info(f"Forecasting on tournament: {tid}")
            reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
            all_reports.extend(reports)
        bot.log_report_summary(all_reports)
        logger.info("Run completed successfully.")
    except Exception as e:
        logger.error(f"Critical error during run: {e}", exc_info=True)
