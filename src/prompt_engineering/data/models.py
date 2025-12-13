"""Data models for the prompt engineering experiment."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Types of questions in the dataset."""

    SENTIMENT = "sentiment"
    MATH = "math"
    LOGIC = "logic"


class QuestionAnswer(BaseModel):
    """A single question-answer pair with metadata.

    This is the fundamental building block for our dataset.
    Each QA pair has a unique ID, question type, the question itself,
    the expected answer, and optional metadata for additional context.
    """

    id: str = Field(..., description="Unique identifier for the QA pair")
    question_type: QuestionType = Field(..., description="Category of the question")
    question: str = Field(..., description="The question text")
    expected_answer: str = Field(..., description="The ground truth answer")
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Additional metadata for the question"
    )

    class Config:
        """Pydantic configuration."""

        frozen = True


class PromptResult(BaseModel):
    """Result from a single prompt execution.

    Captures the input, output, and timing information for analysis.
    """

    question_id: str = Field(..., description="ID of the question answered")
    strategy_name: str = Field(..., description="Name of the prompting strategy used")
    prompt_used: str = Field(..., description="The full prompt sent to the model")
    model_response: str = Field(..., description="The response from the model")
    expected_answer: str = Field(..., description="The ground truth answer")
    execution_time_ms: float = Field(..., description="Time taken in milliseconds")
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Additional execution metadata"
    )


class EvaluationResult(BaseModel):
    """Evaluation metrics for a single prompt result.

    Contains vector distance, semantic similarity, and other metrics.
    """

    question_id: str = Field(..., description="ID of the question")
    strategy_name: str = Field(..., description="Name of the prompting strategy")
    cosine_distance: float = Field(..., description="Cosine distance to expected answer")
    euclidean_distance: float = Field(..., description="Euclidean distance to expected answer")
    semantic_similarity: float = Field(..., description="Semantic similarity score (0-1)")
    exact_match: bool = Field(..., description="Whether response exactly matches expected")
    normalized_score: float = Field(..., description="Normalized evaluation score (0-1)")


class StrategyStatistics(BaseModel):
    """Aggregated statistics for a prompting strategy.

    Includes mean, variance, and distribution metrics.
    """

    strategy_name: str = Field(..., description="Name of the prompting strategy")
    question_type: Optional[QuestionType] = Field(
        default=None, description="Question type (None for overall)"
    )
    sample_count: int = Field(..., description="Number of samples evaluated")
    mean_cosine_distance: float = Field(..., description="Mean cosine distance")
    std_cosine_distance: float = Field(..., description="Standard deviation of cosine distance")
    mean_semantic_similarity: float = Field(..., description="Mean semantic similarity")
    std_semantic_similarity: float = Field(..., description="Standard deviation of similarity")
    exact_match_rate: float = Field(..., description="Percentage of exact matches")
    mean_execution_time_ms: float = Field(..., description="Mean execution time in ms")
