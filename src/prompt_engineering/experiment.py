"""Main experiment runner with parallel processing support."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from prompt_engineering.data.dataset import Dataset, DatasetLoader
from prompt_engineering.data.models import (
    EvaluationResult,
    PromptResult,
    QuestionType,
    StrategyStatistics,
)
from prompt_engineering.evaluation.metrics import MetricsCalculator
from prompt_engineering.prompts.strategies import PromptStrategy
from prompt_engineering.utils.config import Settings, get_settings
from prompt_engineering.utils.llm_client import LLMClient
from prompt_engineering.visualization.plots import ResultVisualizer

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates prompt engineering experiments.

    This is the main coordinator that runs experiments across different
    prompting strategies and collects results for analysis.
    """

    def __init__(
        self,
        strategies: list[PromptStrategy],
        dataset: Dataset,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the experiment runner.

        Args:
            strategies: List of prompting strategies to test
            dataset: Dataset of questions to evaluate
            settings: Application settings
        """
        self._strategies = strategies
        self._dataset = dataset
        self._settings = settings or get_settings()
        self._llm_client = LLMClient(self._settings)
        self._metrics_calculator = MetricsCalculator()
        self._visualizer = ResultVisualizer()

        self._results: dict[str, list[PromptResult]] = {}
        self._evaluations: dict[str, list[EvaluationResult]] = {}
        self._statistics: dict[str, StrategyStatistics] = {}

    def run_experiment(self, use_parallel: bool = True) -> None:
        """Run the complete experiment.

        Args:
            use_parallel: Whether to use parallel processing
        """
        logger.info(f"Starting experiment with {len(self._strategies)} strategies")
        logger.info(f"Dataset size: {len(self._dataset)} questions")

        for strategy in self._strategies:
            logger.info(f"Running strategy: {strategy.name}")

            if use_parallel:
                results = self._run_strategy_parallel(strategy)
            else:
                results = self._run_strategy_sequential(strategy)

            self._results[strategy.name] = results

            evaluations = self._metrics_calculator.evaluate_batch(results)
            self._evaluations[strategy.name] = evaluations

            execution_times = [r.execution_time_ms for r in results]
            statistics = self._metrics_calculator.calculate_statistics(
                evaluations, execution_times
            )
            self._statistics[strategy.name] = statistics

            logger.info(
                f"Strategy {strategy.name}: "
                f"mean_similarity={statistics.mean_semantic_similarity:.3f}, "
                f"exact_match_rate={statistics.exact_match_rate:.2%}"
            )

    def _run_strategy_parallel(self, strategy: PromptStrategy) -> list[PromptResult]:
        """Run strategy using parallel processing.

        Args:
            strategy: The prompting strategy to run

        Returns:
            List of prompt results
        """
        results: list[PromptResult] = []

        with ThreadPoolExecutor(max_workers=self._settings.max_workers) as executor:
            futures = {
                executor.submit(self._execute_single, strategy, qa): qa
                for qa in self._dataset
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Running {strategy.name}",
            ):
                result = future.result()
                results.append(result)

        return results

    def _run_strategy_sequential(self, strategy: PromptStrategy) -> list[PromptResult]:
        """Run strategy sequentially.

        Args:
            strategy: The prompting strategy to run

        Returns:
            List of prompt results
        """
        results = []

        for qa in tqdm(self._dataset, desc=f"Running {strategy.name}"):
            result = self._execute_single(strategy, qa)
            results.append(result)

        return results

    def _execute_single(self, strategy: PromptStrategy, qa) -> PromptResult:
        """Execute a single prompt and return the result.

        Args:
            strategy: The prompting strategy
            qa: The question-answer pair

        Returns:
            Prompt result with response and timing
        """
        prompt = strategy.build_prompt(qa)
        system_prompt = strategy.get_system_prompt()

        response, execution_time = self._llm_client.complete(prompt, system_prompt)

        return PromptResult(
            question_id=qa.id,
            strategy_name=strategy.name,
            prompt_used=prompt,
            model_response=response,
            expected_answer=qa.expected_answer,
            execution_time_ms=execution_time,
        )

    def get_statistics(self) -> dict[str, StrategyStatistics]:
        """Get computed statistics for all strategies."""
        return self._statistics

    def get_evaluations(self) -> dict[str, list[EvaluationResult]]:
        """Get evaluation results for all strategies."""
        return self._evaluations

    def get_statistics_by_type(
        self,
    ) -> dict[QuestionType, dict[str, StrategyStatistics]]:
        """Get statistics broken down by question type.

        Returns:
            Dict mapping question type to strategy statistics
        """
        results: dict[QuestionType, dict[str, StrategyStatistics]] = {}

        for question_type in QuestionType:
            filtered_dataset = self._dataset.filter_by_type(question_type)
            if len(filtered_dataset) == 0:
                continue

            results[question_type] = {}
            question_ids = {qa.id for qa in filtered_dataset}

            for strategy_name, evaluations in self._evaluations.items():
                filtered_evals = [e for e in evaluations if e.question_id in question_ids]
                filtered_times = [
                    r.execution_time_ms
                    for r in self._results[strategy_name]
                    if r.question_id in question_ids
                ]

                if filtered_evals:
                    stats = self._metrics_calculator.calculate_statistics(
                        filtered_evals, filtered_times, question_type
                    )
                    results[question_type][strategy_name] = stats

        return results

    def save_results(self, output_dir: Optional[Path] = None) -> None:
        """Save all results to files.

        Args:
            output_dir: Directory to save results
        """
        output_dir = output_dir or self._settings.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        import json

        # Save statistics
        stats_data = {
            name: stat.model_dump() for name, stat in self._statistics.items()
        }
        with open(output_dir / "statistics.json", "w") as f:
            json.dump(stats_data, f, indent=2, default=str)

        # Save evaluations
        evals_data = {
            name: [e.model_dump() for e in evals]
            for name, evals in self._evaluations.items()
        }
        with open(output_dir / "evaluations.json", "w") as f:
            json.dump(evals_data, f, indent=2)

        logger.info(f"Results saved to {output_dir}")
