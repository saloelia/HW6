"""
Prompt Engineering Experiment Package.

A comprehensive framework for comparing different prompt engineering strategies
including baseline, few-shot learning, Chain of Thought (CoT), and ReAct prompting.

This package provides tools for:
- Creating and managing datasets for prompt evaluation
- Implementing various prompting strategies
- Measuring response quality using vector embeddings
- Visualizing performance comparisons across strategies
"""

from prompt_engineering.data.dataset import Dataset, DatasetLoader
from prompt_engineering.evaluation.metrics import MetricsCalculator
from prompt_engineering.prompts.strategies import (
    BaselineStrategy,
    ChainOfThoughtStrategy,
    FewShotStrategy,
    PromptStrategy,
    ReActStrategy,
)
from prompt_engineering.visualization.plots import ResultVisualizer

__version__ = "1.0.0"
__author__ = "Student"
__all__ = [
    "Dataset",
    "DatasetLoader",
    "PromptStrategy",
    "BaselineStrategy",
    "FewShotStrategy",
    "ChainOfThoughtStrategy",
    "ReActStrategy",
    "MetricsCalculator",
    "ResultVisualizer",
]
