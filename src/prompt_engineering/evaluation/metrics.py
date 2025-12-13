"""Metrics calculation for prompt evaluation."""

import logging
from typing import Optional

import numpy as np

from prompt_engineering.data.models import (
    EvaluationResult,
    PromptResult,
    QuestionType,
    StrategyStatistics,
)
from prompt_engineering.evaluation.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate evaluation metrics for prompt results.

    This building block computes vector distances, similarity scores,
    and aggregated statistics for comparing prompting strategies.
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None) -> None:
        """Initialize the metrics calculator.

        Args:
            embedding_service: Service for computing embeddings
        """
        self._embedding_service = embedding_service or EmbeddingService()

    def evaluate_result(self, result: PromptResult) -> EvaluationResult:
        """Evaluate a single prompt result.

        Args:
            result: The prompt result to evaluate

        Returns:
            Evaluation metrics for the result
        """
        response_embedding = self._embedding_service.get_embedding(result.model_response)
        expected_embedding = self._embedding_service.get_embedding(result.expected_answer)

        cosine_dist = self._embedding_service.cosine_distance(
            response_embedding, expected_embedding
        )
        euclidean_dist = self._embedding_service.euclidean_distance(
            response_embedding, expected_embedding
        )
        semantic_sim = self._embedding_service.cosine_similarity(
            response_embedding, expected_embedding
        )

        exact_match = self._check_exact_match(result.model_response, result.expected_answer)

        normalized_score = self._calculate_normalized_score(semantic_sim, exact_match)

        return EvaluationResult(
            question_id=result.question_id,
            strategy_name=result.strategy_name,
            cosine_distance=cosine_dist,
            euclidean_distance=euclidean_dist,
            semantic_similarity=semantic_sim,
            exact_match=exact_match,
            normalized_score=normalized_score,
        )

    def evaluate_batch(self, results: list[PromptResult]) -> list[EvaluationResult]:
        """Evaluate multiple prompt results.

        Args:
            results: List of prompt results to evaluate

        Returns:
            List of evaluation results
        """
        logger.info(f"Evaluating batch of {len(results)} results")
        return [self.evaluate_result(result) for result in results]

    def calculate_statistics(
        self,
        evaluations: list[EvaluationResult],
        execution_times: list[float],
        question_type: Optional[QuestionType] = None,
    ) -> StrategyStatistics:
        """Calculate aggregated statistics for a set of evaluations.

        Args:
            evaluations: List of evaluation results (all same strategy)
            execution_times: Corresponding execution times in ms
            question_type: Filter by question type (None for overall)

        Returns:
            Aggregated statistics
        """
        if not evaluations:
            raise ValueError("Cannot calculate statistics for empty list")

        strategy_name = evaluations[0].strategy_name

        cosine_distances = [e.cosine_distance for e in evaluations]
        semantic_similarities = [e.semantic_similarity for e in evaluations]
        exact_matches = [e.exact_match for e in evaluations]

        return StrategyStatistics(
            strategy_name=strategy_name,
            question_type=question_type,
            sample_count=len(evaluations),
            mean_cosine_distance=float(np.mean(cosine_distances)),
            std_cosine_distance=float(np.std(cosine_distances)),
            mean_semantic_similarity=float(np.mean(semantic_similarities)),
            std_semantic_similarity=float(np.std(semantic_similarities)),
            exact_match_rate=float(np.mean(exact_matches)),
            mean_execution_time_ms=float(np.mean(execution_times)),
        )

    @staticmethod
    def _check_exact_match(response: str, expected: str) -> bool:
        """Check if response exactly matches expected answer.

        Normalizes both strings for comparison.
        """
        response_normalized = response.strip().lower()
        expected_normalized = expected.strip().lower()

        if response_normalized == expected_normalized:
            return True

        if expected_normalized in response_normalized:
            return True

        return False

    @staticmethod
    def _calculate_normalized_score(semantic_sim: float, exact_match: bool) -> float:
        """Calculate a normalized score combining metrics.

        Args:
            semantic_sim: Semantic similarity score
            exact_match: Whether there was an exact match

        Returns:
            Normalized score between 0 and 1
        """
        base_score = (semantic_sim + 1) / 2

        if exact_match:
            return min(1.0, base_score * 1.2)

        return base_score
