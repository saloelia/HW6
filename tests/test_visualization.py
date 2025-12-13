"""Tests for visualization module."""

import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

from prompt_engineering.data.models import (
    EvaluationResult,
    QuestionType,
    StrategyStatistics,
)
from prompt_engineering.visualization.plots import ResultVisualizer


@pytest.fixture
def sample_statistics() -> list[StrategyStatistics]:
    """Create sample statistics for testing."""
    return [
        StrategyStatistics(
            strategy_name="baseline",
            question_type=None,
            sample_count=10,
            mean_cosine_distance=0.25,
            std_cosine_distance=0.05,
            mean_semantic_similarity=0.75,
            std_semantic_similarity=0.08,
            exact_match_rate=0.60,
            mean_execution_time_ms=120.0,
        ),
        StrategyStatistics(
            strategy_name="few_shot",
            question_type=None,
            sample_count=10,
            mean_cosine_distance=0.18,
            std_cosine_distance=0.04,
            mean_semantic_similarity=0.82,
            std_semantic_similarity=0.06,
            exact_match_rate=0.70,
            mean_execution_time_ms=150.0,
        ),
        StrategyStatistics(
            strategy_name="chain_of_thought",
            question_type=None,
            sample_count=10,
            mean_cosine_distance=0.15,
            std_cosine_distance=0.03,
            mean_semantic_similarity=0.85,
            std_semantic_similarity=0.05,
            exact_match_rate=0.80,
            mean_execution_time_ms=200.0,
        ),
    ]


@pytest.fixture
def sample_evaluations() -> dict[str, list[EvaluationResult]]:
    """Create sample evaluations for testing."""
    return {
        "baseline": [
            EvaluationResult(
                question_id=f"q_{i}",
                strategy_name="baseline",
                cosine_distance=0.2 + i * 0.05,
                euclidean_distance=1.0 + i * 0.1,
                semantic_similarity=0.8 - i * 0.05,
                exact_match=i % 2 == 0,
                normalized_score=0.85 - i * 0.03,
            )
            for i in range(5)
        ],
        "few_shot": [
            EvaluationResult(
                question_id=f"q_{i}",
                strategy_name="few_shot",
                cosine_distance=0.15 + i * 0.04,
                euclidean_distance=0.8 + i * 0.1,
                semantic_similarity=0.85 - i * 0.04,
                exact_match=i % 3 != 0,
                normalized_score=0.88 - i * 0.02,
            )
            for i in range(5)
        ],
    }


@pytest.fixture
def visualizer():
    """Create visualizer with temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ResultVisualizer(output_dir=Path(tmpdir))


class TestResultVisualizerInit:
    """Tests for ResultVisualizer initialization."""

    def test_default_output_dir(self):
        """Test default output directory."""
        viz = ResultVisualizer()
        assert viz._output_dir == Path("results/plots")

    def test_custom_output_dir(self):
        """Test custom output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "custom"
            viz = ResultVisualizer(output_dir=path)
            assert viz._output_dir == path
            assert path.exists()

    def test_creates_output_dir(self):
        """Test output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "plots"
            viz = ResultVisualizer(output_dir=path)
            assert path.exists()


class TestPlotStrategyComparison:
    """Tests for strategy comparison plot."""

    def test_returns_figure(self, visualizer, sample_statistics):
        """Test that plot returns matplotlib figure."""
        fig = visualizer.plot_strategy_comparison(sample_statistics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_metrics(self, visualizer, sample_statistics):
        """Test plotting different metrics."""
        metrics = ["mean_semantic_similarity", "mean_cosine_distance"]

        for metric in metrics:
            fig = visualizer.plot_strategy_comparison(
                sample_statistics, metric=metric
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_custom_title(self, visualizer, sample_statistics):
        """Test custom title."""
        fig = visualizer.plot_strategy_comparison(
            sample_statistics, title="Custom Title"
        )
        assert fig.axes[0].get_title() == "Custom Title"
        plt.close(fig)

    def test_saves_to_file(self, sample_statistics):
        """Test saving plot to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = ResultVisualizer(output_dir=Path(tmpdir))
            viz.plot_strategy_comparison(
                sample_statistics, save_name="test_comparison"
            )

            assert (Path(tmpdir) / "test_comparison.png").exists()


class TestPlotDistanceHistogram:
    """Tests for distance histogram plot."""

    def test_returns_figure(self, visualizer, sample_evaluations):
        """Test that plot returns matplotlib figure."""
        fig = visualizer.plot_distance_histogram(sample_evaluations)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_metrics(self, visualizer, sample_evaluations):
        """Test plotting different distance metrics."""
        metrics = ["cosine_distance", "euclidean_distance"]

        for metric in metrics:
            fig = visualizer.plot_distance_histogram(
                sample_evaluations, metric=metric
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_saves_to_file(self, sample_evaluations):
        """Test saving histogram to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = ResultVisualizer(output_dir=Path(tmpdir))
            viz.plot_distance_histogram(
                sample_evaluations, save_name="test_histogram"
            )

            assert (Path(tmpdir) / "test_histogram.png").exists()


class TestPlotPerformanceByType:
    """Tests for performance by type plot."""

    @pytest.fixture
    def stats_by_type(self):
        """Create statistics broken down by type."""
        return {
            QuestionType.SENTIMENT: [
                StrategyStatistics(
                    strategy_name="baseline",
                    question_type=QuestionType.SENTIMENT,
                    sample_count=5,
                    mean_cosine_distance=0.20,
                    std_cosine_distance=0.04,
                    mean_semantic_similarity=0.80,
                    std_semantic_similarity=0.05,
                    exact_match_rate=0.60,
                    mean_execution_time_ms=100.0,
                ),
                StrategyStatistics(
                    strategy_name="few_shot",
                    question_type=QuestionType.SENTIMENT,
                    sample_count=5,
                    mean_cosine_distance=0.15,
                    std_cosine_distance=0.03,
                    mean_semantic_similarity=0.85,
                    std_semantic_similarity=0.04,
                    exact_match_rate=0.80,
                    mean_execution_time_ms=120.0,
                ),
            ],
            QuestionType.MATH: [
                StrategyStatistics(
                    strategy_name="baseline",
                    question_type=QuestionType.MATH,
                    sample_count=5,
                    mean_cosine_distance=0.25,
                    std_cosine_distance=0.05,
                    mean_semantic_similarity=0.75,
                    std_semantic_similarity=0.06,
                    exact_match_rate=0.40,
                    mean_execution_time_ms=110.0,
                ),
                StrategyStatistics(
                    strategy_name="few_shot",
                    question_type=QuestionType.MATH,
                    sample_count=5,
                    mean_cosine_distance=0.18,
                    std_cosine_distance=0.04,
                    mean_semantic_similarity=0.82,
                    std_semantic_similarity=0.05,
                    exact_match_rate=0.60,
                    mean_execution_time_ms=130.0,
                ),
            ],
        }

    def test_returns_figure(self, visualizer, stats_by_type):
        """Test that plot returns matplotlib figure."""
        fig = visualizer.plot_performance_by_type(stats_by_type)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, stats_by_type):
        """Test saving plot to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = ResultVisualizer(output_dir=Path(tmpdir))
            viz.plot_performance_by_type(
                stats_by_type, save_name="test_by_type"
            )

            assert (Path(tmpdir) / "test_by_type.png").exists()


class TestPlotImprovementDegradation:
    """Tests for improvement/degradation plot."""

    def test_returns_figure(self, visualizer, sample_statistics):
        """Test that plot returns matplotlib figure."""
        baseline = sample_statistics[0]
        improved = sample_statistics[1:]

        fig = visualizer.plot_improvement_degradation(baseline, improved)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_shows_improvement(self, visualizer):
        """Test that improvements are shown correctly."""
        baseline = StrategyStatistics(
            strategy_name="baseline",
            sample_count=10,
            mean_cosine_distance=0.3,
            std_cosine_distance=0.05,
            mean_semantic_similarity=0.7,
            std_semantic_similarity=0.05,
            exact_match_rate=0.5,
            mean_execution_time_ms=100.0,
        )
        improved = [
            StrategyStatistics(
                strategy_name="better",
                sample_count=10,
                mean_cosine_distance=0.2,
                std_cosine_distance=0.04,
                mean_semantic_similarity=0.84,
                std_semantic_similarity=0.04,
                exact_match_rate=0.7,
                mean_execution_time_ms=120.0,
            )
        ]

        fig = visualizer.plot_improvement_degradation(baseline, improved)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, sample_statistics):
        """Test saving plot to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = ResultVisualizer(output_dir=Path(tmpdir))
            baseline = sample_statistics[0]
            improved = sample_statistics[1:]

            viz.plot_improvement_degradation(
                baseline, improved, save_name="test_improvement"
            )

            assert (Path(tmpdir) / "test_improvement.png").exists()
