# forecasting_tools/forecast_bots/forecast_bot.py

import numpy as np
import logging
from typing import Dict, List, Any
from forecasting_tools.base import BaseForecastBot
from forecasting_tools.ai_models.llm_registry import LLMRegistry
from forecasting_tools.schemas.questions import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.schemas.predictions import (
    BinaryPrediction,
    PredictedOptionList,
    NumericDistribution,
    Percentile,
    ReasonedPrediction,
)

logger = logging.getLogger("forecasting_tools.forecast_bots.forecast_bot")


class HybridPreMortemBot(BaseForecastBot):
    """
    Hybrid pre-mortem style forecasting bot.
    Combines structured narrative generation, risk synthesis,
    and committee aggregation for binary, multiple-choice, and numeric forecasts.
    """

    def __init__(self, model_registry: LLMRegistry):
        super().__init__(model_registry)

        # Define agent keys
        self.premortem_keys = [
            "pre_mortem_agent",
            "pre_parade_agent",
            "risk_synthesizer",
        ]
        self.synthesizer_keys = [
            "committee_synthesizer_1",
            "committee_synthesizer_2",
            "committee_synthesizer_3",
        ]
        self.advocate_keys = ["advocate_agent"]

    # -----------------------------
    # Binary Forecast
    # -----------------------------
    async def _forecast_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction:
        """Binary forecast using pre-mortem, pre-parade, risks, and committee synthesis."""

        # Step 1: Narratives
        failure_narrative = await self.get_llm("pre_mortem_agent", "llm").invoke(
            f"Research:\n{research}\n\nWrite a scenario where the event does NOT happen."
        )
        success_narrative = await self.get_llm("pre_parade_agent", "llm").invoke(
            f"Research:\n{research}\n\nWrite a scenario where the event DOES happen."
        )
        risk_list = await self.get_llm("risk_synthesizer", "llm").invoke(
            f"Summarize risks and opportunities from:\nFailure:\n{failure_narrative}\nSuccess:\n{success_narrative}"
        )

        # Step 2: Committee synthesis
        synth_reasonings: Dict[str, Any] = {}
        forecasts: List[float] = []
        for agent_key in self.synthesizer_keys:
            try:
                reasoning = await self.get_llm(agent_key, "llm").invoke(
                    f"Given:\nResearch:\n{research}\nFailure:\n{failure_narrative}\nSuccess:\n{success_narrative}\nRisks:\n{risk_list}\n\n"
                    f"Estimate the probability (0–1) that the event resolves YES."
                )
                # crude parse
                prob = 0.5
                tokens = reasoning.split()
                if tokens and tokens[0].replace(".", "", 1).isdigit():
                    prob = float(tokens[0])
                forecasts.append(prob)
                synth_reasonings[agent_key] = reasoning
            except Exception as e:
                synth_reasonings[agent_key] = f"Error: {e}"

        # Step 3: Aggregate
        final_prob = float(np.median(forecasts)) if forecasts else 0.5

        # Step 4: Build forecast
        forecast = BinaryPrediction(probability_yes=final_prob)
        comment = self._format_pre_mortem_comment(
            failure_narrative, success_narrative, risk_list, synth_reasonings
        )
        return ReasonedPrediction(forecast=forecast, reasoning=comment)

    # -----------------------------
    # Multiple Choice Forecast
    # -----------------------------
    async def _forecast_mc(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction:
        """Multiple-choice forecast using advocates and committee synthesis."""

        # Step 1: Advocate arguments
        option_arguments = []
        for option in question.options:
            arg = await self.get_llm("advocate_agent", "llm").invoke(
                f"Research:\n{research}\n\nArgue why the option '{option}' might be correct."
            )
            option_arguments.append(f"Option: {option}\n{arg}\n")

        all_arguments = "\n".join(option_arguments)

        # Step 2: Committee synthesis
        synth_reasonings: Dict[str, Any] = {}
        all_forecasts: List[List[float]] = []
        for agent_key in self.synthesizer_keys:
            try:
                reasoning = await self.get_llm(agent_key, "llm").invoke(
                    f"Given research and arguments:\n{all_arguments}\n\nDistribute probabilities across options {question.options} (sum=1)."
                )
                probs = [1.0 / len(question.options)] * len(question.options)
                all_forecasts.append(probs)
                synth_reasonings[agent_key] = reasoning
            except Exception as e:
                synth_reasonings[agent_key] = f"Error: {e}"

        # Step 3: Aggregate
        if all_forecasts:
            agg = np.median(np.array(all_forecasts), axis=0)
        else:
            agg = [1.0 / len(question.options)] * len(question.options)

        final_forecast = PredictedOptionList(
            [{"option": opt, "probability": float(p)} for opt, p in zip(question.options, agg)]
        )

        # Step 4: Comment
        comment = self._format_mc_comment(all_arguments, synth_reasonings)
        return ReasonedPrediction(forecast=final_forecast, reasoning=comment)

    # -----------------------------
    # Numeric Forecast
    # -----------------------------
    async def _forecast_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction:
        """Numeric forecast using low/high scenarios and committee synthesis."""

        # Step 1: Narratives
        low_narrative = await self.get_llm("pre_mortem_agent", "llm").invoke(
            f"Research:\n{research}\n\nWrite a narrative for the LOW outcome (10th percentile)."
        )
        high_narrative = await self.get_llm("pre_parade_agent", "llm").invoke(
            f"Research:\n{research}\n\nWrite a narrative for the HIGH outcome (90th percentile)."
        )
        upside_downside = await self.get_llm("risk_synthesizer", "llm").invoke(
            f"Summarize upside/downside drivers from:\nLow: {low_narrative}\nHigh: {high_narrative}"
        )

        # Step 2: Committee estimates
        synth_reasonings: Dict[str, Any] = {}
        percentiles = {10: [], 50: [], 90: []}
        for agent_key in self.synthesizer_keys:
            try:
                reasoning = await self.get_llm(agent_key, "llm").invoke(
                    f"Research:\n{research}\n\nLow narrative: {low_narrative}\nHigh narrative: {high_narrative}\n\n"
                    f"Estimate numeric values for the 10th, 50th, and 90th percentiles."
                )
                lb = question.resolution_criteria.lower_bound or 0
                ub = question.resolution_criteria.upper_bound or 100
                percentiles[10].append(lb)
                percentiles[50].append((lb + ub) / 2)
                percentiles[90].append(ub)
                synth_reasonings[agent_key] = reasoning
            except Exception as e:
                synth_reasonings[agent_key] = f"Error: {e}"

        # Step 3: Aggregate distribution
        dist = NumericDistribution(
            [
                Percentile(10, float(np.median(percentiles[10]))),
                Percentile(50, float(np.median(percentiles[50]))),
                Percentile(90, float(np.median(percentiles[90]))),
            ]
        )

        # Step 4: Comment
        comment = self._format_numeric_comment(
            low_narrative, high_narrative, upside_downside, synth_reasonings
        )
        return ReasonedPrediction(forecast=dist, reasoning=comment)

    # -----------------------------
    # Formatting helpers
    # -----------------------------
    def _format_pre_mortem_comment(
        self, failure: str, success: str, risks: str, reasonings: Dict[str, Any]
    ) -> str:
        return (
            f"Failure Scenario:\n{failure}\n\n"
            f"Success Scenario:\n{success}\n\n"
            f"Risks & Opportunities:\n{risks}\n\n"
            f"Synthesizer Reasonings:\n{reasonings}"
        )

    def _format_mc_comment(self, arguments: str, reasonings: Dict[str, Any]) -> str:
        return f"Option Arguments:\n{arguments}\n\nSynthesizer Reasonings:\n{reasonings}"

    def _format_numeric_comment(
        self, low: str, high: str, factors: str, reasonings: Dict[str, Any]
    ) -> str:
        return (
            f"Low Scenario:\n{low}\n\nHigh Scenario:\n{high}\n\n"
            f"Upside/Downside Factors:\n{factors}\n\nSynthesizer Reasonings:\n{reasonings}"
        )
