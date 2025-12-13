"""Tests for metrics calculation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from prompt_engineering.data.models import (
    EvaluationResult,
    PromptResult,
    QuestionType,
)
from prompt_engineering.evaluation.metrics import MetricsCalculator


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    mock = MagicMock()

    mock.get_embedding.side_effect = lambda text: np.random.rand(384).astype(np.float32)

    mock.cosine_distance.return_value = 0.2
    mock.cosine_similarity.return_value = 0.8
    mock.euclidean_distance.return_value = 1.5

    return mock


@pytest.fixture
def metrics_calculator(mock_embedding_service):
    """Create metrics calculator with mock embedding service."""
    return MetricsCalculator(embedding_service=mock_embedding_service)


@pytest.fixture
def sample_prompt_result():
    """Create sample prompt result."""
    return PromptResult(
        question_id="q_001",
        strategy_name="baseline",
        prompt_used="Question: test",
        model_response="positive",
        expected_answer="positive",
        execution_time_ms=100.0,
    )


@pytest.fixture
def sample_evaluation_results():
    """Create sample evaluation results."""
    return [
        EvaluationResult(
            question_id="q_001",
            strategy_name="baseline",
            cosine_distance=0.1,
            euclidean_distance=0.5,
            semantic_similarity=0.9,
            exact_match=True,
            normalized_score=0.95,
        ),
        EvaluationResult(
            question_id="q_002",
            strategy_name="baseline",
            cosine_distance=0.3,
            euclidean_distance=1.0,
            semantic_similarity=0.7,
            exact_match=False,
            normalized_score=0.85,
        ),
        EvaluationResult(
            question_id="q_003",
            strategy_name="baseline",
            cosine_distance=0.2,
            euclidean_distance=0.8,
            semantic_similarity=0.8,
            exact_match=True,
            normalized_score=0.90,
        ),
    ]


class TestMetricsCalculatorInit:
    """Tests for MetricsCalculator initialization."""

    def test_with_custom_embedding_service(self, mock_embedding_service):
        """Test initialization with custom embedding service."""
        calc = MetricsCalculator(embedding_service=mock_embedding_service)
        assert calc._embedding_service is mock_embedding_service

    def test_default_embedding_service(self):
        """Test initialization creates default embedding service."""
        calc = MetricsCalculator()
        assert calc._embedding_service is not None


class TestExactMatchCheck:
    """Tests for exact match checking."""

    def test_exact_match_identical(self):
        """Test exact match with identical strings."""
        result = MetricsCalculator._check_exact_match("positive", "positive")
        assert result is True

    def test_exact_match_case_insensitive(self):
        """Test exact match is case insensitive."""
        result = MetricsCalculator._check_exact_match("Positive", "positive")
        assert result is True

    def test_exact_match_whitespace(self):
        """Test exact match ignores whitespace."""
        result = MetricsCalculator._check_exact_match("  positive  ", "positive")
        assert result is True

    def test_exact_match_substring(self):
        """Test exact match when expected is substring."""
        result = MetricsCalculator._check_exact_match(
            "The answer is positive because...", "positive"
        )
        assert result is True

    def test_no_match_different(self):
        """Test no match with different strings."""
        result = MetricsCalculator._check_exact_match("negative", "positive")
        assert result is False


class TestNormalizedScore:
    """Tests for normalized score calculation."""

    def test_perfect_score_with_exact_match(self):
        """Test perfect score with similarity 1.0 and exact match."""
        score = MetricsCalculator._calculate_normalized_score(1.0, True)
        assert score == 1.0

    def test_score_boosted_with_exact_match(self):
        """Test score is boosted with exact match."""
        score_with = MetricsCalculator._calculate_normalized_score(0.8, True)
        score_without = MetricsCalculator._calculate_normalized_score(0.8, False)
        assert score_with > score_without

    def test_score_capped_at_one(self):
        """Test score doesn't exceed 1.0."""
        score = MetricsCalculator._calculate_normalized_score(0.9, True)
        assert score <= 1.0

    def test_score_non_negative(self):
        """Test score is non-negative."""
        score = MetricsCalculator._calculate_normalized_score(-1.0, False)
        assert score >= 0


class TestEvaluateResult:
    """Tests for single result evaluation."""

    def test_evaluate_returns_evaluation_result(
        self, metrics_calculator, sample_prompt_result
    ):
        """Test evaluate_result returns EvaluationResult."""
        with patch.object(
            metrics_calculator._embedding_service,
            "cosine_distance",
            return_value=0.2,
        ), patch.object(
            metrics_calculator._embedding_service,
            "cosine_similarity",
            return_value=0.8,
        ), patch.object(
            metrics_calculator._embedding_service,
            "euclidean_distance",
            return_value=1.5,
        ):
            result = metrics_calculator.evaluate_result(sample_prompt_result)

        assert isinstance(result, EvaluationResult)
        assert result.question_id == "q_001"
        assert result.strategy_name == "baseline"

    def test_evaluate_computes_metrics(self, metrics_calculator, sample_prompt_result):
        """Test that metrics are computed."""
        with patch.object(
            metrics_calculator._embedding_service,
            "cosine_distance",
            return_value=0.15,
        ), patch.object(
            metrics_calculator._embedding_service,
            "cosine_similarity",
            return_value=0.85,
        ), patch.object(
            metrics_calculator._embedding_service,
            "euclidean_distance",
            return_value=1.2,
        ):
            result = metrics_calculator.evaluate_result(sample_prompt_result)

        assert result.cosine_distance == 0.15
        assert result.semantic_similarity == 0.85
        assert result.euclidean_distance == 1.2


class TestEvaluateBatch:
    """Tests for batch evaluation."""

    def test_evaluate_batch_returns_list(self, metrics_calculator):
        """Test evaluate_batch returns list of results."""
        results = [
            PromptResult(
                question_id=f"q_{i}",
                strategy_name="baseline",
                prompt_used="test",
                model_response="answer",
                expected_answer="answer",
                execution_time_ms=100.0,
            )
            for i in range(3)
        ]

        with patch.object(
            metrics_calculator._embedding_service,
            "cosine_distance",
            return_value=0.2,
        ), patch.object(
            metrics_calculator._embedding_service,
            "cosine_similarity",
            return_value=0.8,
        ), patch.object(
            metrics_calculator._embedding_service,
            "euclidean_distance",
            return_value=1.5,
        ):
            evaluations = metrics_calculator.evaluate_batch(results)

        assert len(evaluations) == 3
        assert all(isinstance(e, EvaluationResult) for e in evaluations)


class TestCalculateStatistics:
    """Tests for statistics calculation."""

    def test_calculate_statistics_basic(
        self, metrics_calculator, sample_evaluation_results
    ):
        """Test basic statistics calculation."""
        execution_times = [100.0, 150.0, 120.0]

        stats = metrics_calculator.calculate_statistics(
            sample_evaluation_results, execution_times
        )

        assert stats.strategy_name == "baseline"
        assert stats.sample_count == 3

    def test_calculate_statistics_mean_values(
        self, metrics_calculator, sample_evaluation_results
    ):
        """Test mean values are calculated correctly."""
        execution_times = [100.0, 150.0, 120.0]

        stats = metrics_calculator.calculate_statistics(
            sample_evaluation_results, execution_times
        )

        expected_mean_distance = (0.1 + 0.3 + 0.2) / 3
        assert abs(stats.mean_cosine_distance - expected_mean_distance) < 1e-6

        expected_mean_similarity = (0.9 + 0.7 + 0.8) / 3
        assert abs(stats.mean_semantic_similarity - expected_mean_similarity) < 1e-6

    def test_calculate_statistics_exact_match_rate(
        self, metrics_calculator, sample_evaluation_results
    ):
        """Test exact match rate calculation."""
        execution_times = [100.0, 150.0, 120.0]

        stats = metrics_calculator.calculate_statistics(
            sample_evaluation_results, execution_times
        )

        expected_rate = 2 / 3
        assert abs(stats.exact_match_rate - expected_rate) < 1e-6

    def test_calculate_statistics_with_question_type(
        self, metrics_calculator, sample_evaluation_results
    ):
        """Test statistics with question type filter."""
        execution_times = [100.0, 150.0, 120.0]

        stats = metrics_calculator.calculate_statistics(
            sample_evaluation_results,
            execution_times,
            question_type=QuestionType.SENTIMENT,
        )

        assert stats.question_type == QuestionType.SENTIMENT

    def test_calculate_statistics_empty_raises_error(self, metrics_calculator):
        """Test empty list raises error."""
        with pytest.raises(ValueError):
            metrics_calculator.calculate_statistics([], [])
