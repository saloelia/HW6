"""Main entry point for the prompt engineering experiment."""

import argparse
import logging
import sys
from pathlib import Path

from prompt_engineering.data.dataset import DatasetLoader
from prompt_engineering.experiment import ExperimentRunner
from prompt_engineering.prompts.strategies import (
    BaselineStrategy,
    ChainOfThoughtStrategy,
    FewShotStrategy,
    ReActStrategy,
)
from prompt_engineering.utils.config import get_settings, setup_logging
from prompt_engineering.visualization.plots import ResultVisualizer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run prompt engineering experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/raw/dataset.json"),
        help="Path to the dataset JSON file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["baseline", "few_shot", "chain_of_thought", "react"],
        choices=["baseline", "few_shot", "chain_of_thought", "react"],
        help="Strategies to run",
    )

    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing",
    )

    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def get_strategies(strategy_names: list[str]) -> list:
    """Get strategy instances from names.

    Args:
        strategy_names: List of strategy names

    Returns:
        List of strategy instances
    """
    strategy_map = {
        "baseline": BaselineStrategy,
        "few_shot": FewShotStrategy,
        "chain_of_thought": ChainOfThoughtStrategy,
        "react": ReActStrategy,
    }

    return [strategy_map[name]() for name in strategy_names]


def generate_plots(runner: ExperimentRunner, output_dir: Path) -> None:
    """Generate all visualization plots.

    Args:
        runner: Experiment runner with results
        output_dir: Directory to save plots
    """
    visualizer = ResultVisualizer(output_dir / "plots")
    statistics = runner.get_statistics()
    evaluations = runner.get_evaluations()

    stats_list = list(statistics.values())

    visualizer.plot_strategy_comparison(
        stats_list,
        metric="mean_semantic_similarity",
        title="Semantic Similarity Comparison",
        save_name="similarity_comparison",
    )

    visualizer.plot_strategy_comparison(
        stats_list,
        metric="mean_cosine_distance",
        title="Cosine Distance Comparison",
        save_name="distance_comparison",
    )

    visualizer.plot_distance_histogram(
        evaluations,
        metric="cosine_distance",
        save_name="distance_histogram",
    )

    if "baseline" in statistics:
        baseline_stats = statistics["baseline"]
        improved_stats = [s for name, s in statistics.items() if name != "baseline"]

        if improved_stats:
            visualizer.plot_improvement_degradation(
                baseline_stats,
                improved_stats,
                metric="mean_semantic_similarity",
                save_name="improvement_vs_baseline",
            )

    stats_by_type = runner.get_statistics_by_type()
    if stats_by_type:
        formatted_stats = {
            qt: list(strats.values())
            for qt, strats in stats_by_type.items()
        }
        visualizer.plot_performance_by_type(
            formatted_stats,
            metric="mean_semantic_similarity",
            save_name="performance_by_type",
        )

    logger.info(f"Plots saved to {output_dir / 'plots'}")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_args()

    setup_logging(args.log_level)
    logger.info("Starting prompt engineering experiment")

    try:
        settings = get_settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        logger.error("Make sure OPENAI_API_KEY is set in .env file")
        return 1

    try:
        dataset = DatasetLoader.load_from_json(args.dataset)
        logger.info(f"Loaded dataset with {len(dataset)} questions")
    except FileNotFoundError:
        logger.error(f"Dataset not found: {args.dataset}")
        return 1
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1

    strategies = get_strategies(args.strategies)
    logger.info(f"Running strategies: {[s.name for s in strategies]}")

    runner = ExperimentRunner(strategies, dataset, settings)
    runner.run_experiment(use_parallel=not args.no_parallel)

    runner.save_results(args.output)

    if not args.skip_plots:
        generate_plots(runner, args.output)

    statistics = runner.get_statistics()
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    for name, stats in statistics.items():
        print(f"\n{name.upper()}")
        print(f"  Semantic Similarity: {stats.mean_semantic_similarity:.3f} ± {stats.std_semantic_similarity:.3f}")
        print(f"  Cosine Distance:     {stats.mean_cosine_distance:.3f} ± {stats.std_cosine_distance:.3f}")
        print(f"  Exact Match Rate:    {stats.exact_match_rate:.1%}")
        print(f"  Avg Execution Time:  {stats.mean_execution_time_ms:.1f} ms")

    print("\n" + "=" * 60)
    logger.info("Experiment completed successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
