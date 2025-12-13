"""Visualization utilities for experiment results."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from prompt_engineering.data.models import EvaluationResult, QuestionType, StrategyStatistics

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """Visualizer for prompt engineering experiment results.

    Creates comparison plots, histograms, and performance charts.
    """

    def __init__(self, output_dir: Optional[Path] = None, style: str = "whitegrid") -> None:
        """Initialize the visualizer.

        Args:
            output_dir: Directory to save plots
            style: Seaborn style to use
        """
        self._output_dir = output_dir or Path("results/plots")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style(style)
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 12

    def plot_strategy_comparison(
        self,
        statistics: list[StrategyStatistics],
        metric: str = "mean_semantic_similarity",
        title: Optional[str] = None,
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """Create bar chart comparing strategies.

        Args:
            statistics: List of strategy statistics to compare
            metric: Which metric to plot
            title: Plot title
            save_name: Filename to save plot

        Returns:
            Matplotlib figure
        """
        strategies = [s.strategy_name for s in statistics]
        values = [getattr(s, metric) for s in statistics]

        std_metric = metric.replace("mean_", "std_")
        errors = [getattr(s, std_metric, 0) for s in statistics]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = sns.color_palette("husl", len(strategies))
        bars = ax.bar(strategies, values, yerr=errors, capsize=5, color=colors, edgecolor="black")

        ax.set_xlabel("Prompting Strategy", fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(title or f"Strategy Comparison: {metric.replace('_', ' ').title()}")

        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()

        if save_name:
            self._save_plot(fig, save_name)

        return fig

    def plot_distance_histogram(
        self,
        evaluations: dict[str, list[EvaluationResult]],
        metric: str = "cosine_distance",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """Create histogram of distances for each strategy.

        Args:
            evaluations: Dict mapping strategy name to evaluations
            metric: Which distance metric to plot
            save_name: Filename to save plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for strategy_name, evals in evaluations.items():
            values = [getattr(e, metric) for e in evals]
            ax.hist(values, bins=20, alpha=0.6, label=strategy_name, edgecolor="black")

        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Distribution of {metric.replace('_', ' ').title()} by Strategy")
        ax.legend()

        plt.tight_layout()

        if save_name:
            self._save_plot(fig, save_name)

        return fig

    def plot_performance_by_type(
        self,
        statistics_by_type: dict[QuestionType, list[StrategyStatistics]],
        metric: str = "mean_semantic_similarity",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """Create grouped bar chart showing performance by question type.

        Args:
            statistics_by_type: Dict mapping question type to strategy stats
            metric: Which metric to plot
            save_name: Filename to save plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        question_types = list(statistics_by_type.keys())
        strategies = [s.strategy_name for s in list(statistics_by_type.values())[0]]

        x = np.arange(len(question_types))
        width = 0.2
        colors = sns.color_palette("husl", len(strategies))

        for i, strategy in enumerate(strategies):
            values = []
            for qt in question_types:
                stat = next(s for s in statistics_by_type[qt] if s.strategy_name == strategy)
                values.append(getattr(stat, metric))

            offset = (i - len(strategies) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=strategy, color=colors[i], edgecolor="black")

        ax.set_xlabel("Question Type", fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"Performance by Question Type: {metric.replace('_', ' ').title()}")
        ax.set_xticks(x)
        ax.set_xticklabels([qt.value for qt in question_types])
        ax.legend(title="Strategy")

        plt.tight_layout()

        if save_name:
            self._save_plot(fig, save_name)

        return fig

    def plot_improvement_degradation(
        self,
        baseline_stats: StrategyStatistics,
        improved_stats: list[StrategyStatistics],
        metric: str = "mean_semantic_similarity",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """Create chart showing improvement/degradation vs baseline.

        Args:
            baseline_stats: Statistics for baseline strategy
            improved_stats: Statistics for improved strategies
            metric: Which metric to compare
            save_name: Filename to save plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        baseline_value = getattr(baseline_stats, metric)
        strategies = [s.strategy_name for s in improved_stats]
        improvements = []

        for stat in improved_stats:
            value = getattr(stat, metric)
            improvement = ((value - baseline_value) / baseline_value) * 100
            improvements.append(improvement)

        colors = ["green" if imp >= 0 else "red" for imp in improvements]
        bars = ax.bar(strategies, improvements, color=colors, edgecolor="black")

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Prompting Strategy", fontsize=12)
        ax.set_ylabel("Improvement vs Baseline (%)", fontsize=12)
        ax.set_title(f"Performance Change: {metric.replace('_', ' ').title()}")

        for bar, imp in zip(bars, improvements):
            va = "bottom" if imp >= 0 else "top"
            ax.annotate(
                f"{imp:+.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va=va,
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()

        if save_name:
            self._save_plot(fig, save_name)

        return fig

    def _save_plot(self, fig: plt.Figure, name: str) -> None:
        """Save plot to output directory."""
        path = self._output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {path}")
