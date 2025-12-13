"""Tests for data models."""

import pytest
from pydantic import ValidationError

from prompt_engineering.data.models import (
    EvaluationResult,
    PromptResult,
    QuestionAnswer,
    QuestionType,
    StrategyStatistics,
)


class TestQuestionType:
    """Tests for QuestionType enum."""

    def test_sentiment_value(self):
        """Test sentiment type has correct value."""
        assert QuestionType.SENTIMENT.value == "sentiment"

    def test_math_value(self):
        """Test math type has correct value."""
        assert QuestionType.MATH.value == "math"

    def test_logic_value(self):
        """Test logic type has correct value."""
        assert QuestionType.LOGIC.value == "logic"

    def test_all_types_exist(self):
        """Test all expected types exist."""
        types = list(QuestionType)
        assert len(types) == 3


class TestQuestionAnswer:
    """Tests for QuestionAnswer model."""

    def test_create_valid_question(self):
        """Test creating a valid question-answer pair."""
        qa = QuestionAnswer(
            id="test_001",
            question_type=QuestionType.SENTIMENT,
            question="Is this positive?",
            expected_answer="yes",
        )
        assert qa.id == "test_001"
        assert qa.question_type == QuestionType.SENTIMENT
        assert qa.question == "Is this positive?"
        assert qa.expected_answer == "yes"
        assert qa.metadata is None

    def test_create_with_metadata(self):
        """Test creating question with metadata."""
        qa = QuestionAnswer(
            id="test_002",
            question_type=QuestionType.MATH,
            question="2+2?",
            expected_answer="4",
            metadata={"difficulty": "easy"},
        )
        assert qa.metadata == {"difficulty": "easy"}

    def test_missing_required_field(self):
        """Test that missing required fields raise error."""
        with pytest.raises(ValidationError):
            QuestionAnswer(
                id="test_003",
                question_type=QuestionType.LOGIC,
            )

    def test_question_is_frozen(self):
        """Test that QuestionAnswer is immutable."""
        qa = QuestionAnswer(
            id="test_004",
            question_type=QuestionType.SENTIMENT,
            question="Test?",
            expected_answer="yes",
        )
        with pytest.raises(ValidationError):
            qa.id = "new_id"

    def test_serialization(self):
        """Test model serialization to dict."""
        qa = QuestionAnswer(
            id="test_005",
            question_type=QuestionType.MATH,
            question="5+5?",
            expected_answer="10",
        )
        data = qa.model_dump()
        assert data["id"] == "test_005"
        assert data["question_type"] == "math"
        assert data["question"] == "5+5?"
        assert data["expected_answer"] == "10"


class TestPromptResult:
    """Tests for PromptResult model."""

    def test_create_valid_result(self):
        """Test creating a valid prompt result."""
        result = PromptResult(
            question_id="q_001",
            strategy_name="baseline",
            prompt_used="Question: test",
            model_response="answer",
            expected_answer="answer",
            execution_time_ms=100.5,
        )
        assert result.question_id == "q_001"
        assert result.strategy_name == "baseline"
        assert result.execution_time_ms == 100.5

    def test_with_metadata(self):
        """Test prompt result with metadata."""
        result = PromptResult(
            question_id="q_002",
            strategy_name="few_shot",
            prompt_used="Prompt",
            model_response="Response",
            expected_answer="Expected",
            execution_time_ms=50.0,
            metadata={"tokens": 100},
        )
        assert result.metadata == {"tokens": 100}


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_create_valid_evaluation(self):
        """Test creating a valid evaluation result."""
        eval_result = EvaluationResult(
            question_id="q_001",
            strategy_name="baseline",
            cosine_distance=0.1,
            euclidean_distance=0.5,
            semantic_similarity=0.9,
            exact_match=True,
            normalized_score=0.95,
        )
        assert eval_result.cosine_distance == 0.1
        assert eval_result.exact_match is True

    def test_score_bounds(self):
        """Test that scores can be at boundary values."""
        eval_result = EvaluationResult(
            question_id="q_002",
            strategy_name="cot",
            cosine_distance=0.0,
            euclidean_distance=0.0,
            semantic_similarity=1.0,
            exact_match=True,
            normalized_score=1.0,
        )
        assert eval_result.normalized_score == 1.0


class TestStrategyStatistics:
    """Tests for StrategyStatistics model."""

    def test_create_valid_statistics(self):
        """Test creating valid strategy statistics."""
        stats = StrategyStatistics(
            strategy_name="baseline",
            question_type=QuestionType.SENTIMENT,
            sample_count=10,
            mean_cosine_distance=0.15,
            std_cosine_distance=0.05,
            mean_semantic_similarity=0.85,
            std_semantic_similarity=0.03,
            exact_match_rate=0.7,
            mean_execution_time_ms=150.0,
        )
        assert stats.strategy_name == "baseline"
        assert stats.sample_count == 10
        assert stats.mean_semantic_similarity == 0.85

    def test_overall_statistics_no_type(self):
        """Test statistics without question type (overall)."""
        stats = StrategyStatistics(
            strategy_name="few_shot",
            question_type=None,
            sample_count=30,
            mean_cosine_distance=0.12,
            std_cosine_distance=0.04,
            mean_semantic_similarity=0.88,
            std_semantic_similarity=0.02,
            exact_match_rate=0.8,
            mean_execution_time_ms=200.0,
        )
        assert stats.question_type is None
