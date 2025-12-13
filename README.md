# Prompt Engineering Experiment

A comprehensive framework for evaluating and comparing different prompt engineering strategies including baseline prompting, few-shot learning, Chain of Thought (CoT), and ReAct.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Prompting Strategies](#prompting-strategies)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Visualization](#results-visualization)
- [Testing](#testing)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a systematic approach to evaluating prompt engineering techniques at scale. It measures the effectiveness of different prompting strategies across three question types:

- **Sentiment Analysis**: Classifying text as positive, negative, or neutral
- **Mathematical Problems**: Arithmetic, algebra, and word problems
- **Logical Reasoning**: Syllogisms, conditionals, and deductive reasoning

The experiment quantifies improvement or degradation using vector distances between model responses and expected answers.

## Features

- **Multiple Prompting Strategies**: Baseline, Few-Shot, Chain of Thought, ReAct
- **Semantic Evaluation**: Uses sentence embeddings for similarity measurement
- **Parallel Processing**: Multi-threaded execution for faster experiments
- **Comprehensive Visualization**: Bar charts, histograms, performance comparisons
- **Modular Architecture**: Building blocks design pattern for extensibility
- **Configuration Management**: Secure handling of API keys and settings
- **Type Safety**: Full type hints and Pydantic models

## Project Structure

```
prompt-engineering-experiment/
├── src/
│   └── prompt_engineering/
│       ├── __init__.py           # Package exports
│       ├── main.py               # CLI entry point
│       ├── experiment.py         # Experiment orchestration
│       ├── data/
│       │   ├── __init__.py
│       │   ├── models.py         # Pydantic data models
│       │   └── dataset.py        # Dataset management
│       ├── prompts/
│       │   ├── __init__.py
│       │   └── strategies.py     # Prompting strategies
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── embeddings.py     # Embedding service
│       │   └── metrics.py        # Metrics calculation
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── plots.py          # Visualization utilities
│       └── utils/
│           ├── __init__.py
│           ├── config.py         # Configuration management
│           └── llm_client.py     # LLM API client
├── tests/                        # Unit tests
├── data/
│   ├── raw/                      # Raw datasets
│   └── processed/                # Processed data
├── results/                      # Experiment results
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
├── config/
│   └── .env.example              # Environment template
├── pyproject.toml                # Project configuration
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or uv package manager
- OpenAI API key (or compatible API)

### Install from Source

1. Clone the repository:
```bash
git clone https://github.com/student/prompt-engineering-experiment.git
cd prompt-engineering-experiment
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. Install development dependencies (optional):
```bash
pip install -e ".[dev]"
```

## Configuration

1. Copy the example environment file:
```bash
cp config/.env.example .env
```

2. Edit `.env` and add your API key:
```env
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | Model to use |
| `OPENAI_BASE_URL` | None | Custom API endpoint |
| `MAX_TOKENS` | 500 | Maximum response tokens |
| `TEMPERATURE` | 0.0 | Sampling temperature |
| `MAX_WORKERS` | 4 | Parallel workers |
| `LOG_LEVEL` | INFO | Logging verbosity |

## Usage

### Command Line Interface

Run the full experiment:
```bash
prompt-experiment --dataset data/raw/dataset.json --output results/
```

Run specific strategies:
```bash
prompt-experiment --strategies baseline few_shot chain_of_thought
```

Disable parallel processing:
```bash
prompt-experiment --no-parallel
```

Skip plot generation:
```bash
prompt-experiment --skip-plots
```

### Python API

```python
from prompt_engineering.data.dataset import DatasetLoader
from prompt_engineering.experiment import ExperimentRunner
from prompt_engineering.prompts.strategies import (
    BaselineStrategy,
    FewShotStrategy,
    ChainOfThoughtStrategy,
)

# Load dataset
dataset = DatasetLoader.load_from_json("data/raw/dataset.json")

# Define strategies
strategies = [
    BaselineStrategy(),
    FewShotStrategy(),
    ChainOfThoughtStrategy(),
]

# Run experiment
runner = ExperimentRunner(strategies, dataset)
runner.run_experiment()

# Get results
statistics = runner.get_statistics()
for name, stats in statistics.items():
    print(f"{name}: similarity={stats.mean_semantic_similarity:.3f}")
```

## Prompting Strategies

### 1. Baseline Strategy

Minimal prompting with atomic instructions:

```
Question: [question]

Answer:
```

### 2. Few-Shot Learning

Provides 2-3 examples before the target question:

```
Here are some examples:

Example 1:
Question: [example_q1]
Answer: [example_a1]

Example 2:
Question: [example_q2]
Answer: [example_a2]

Now answer this question:
Question: [target_question]

Answer:
```

### 3. Chain of Thought (CoT)

Encourages step-by-step reasoning:

```
Question: [question]

Instructions: [type-specific reasoning instructions]

Let's think through this step by step:
1. First, I will analyze the key elements...
2. Then, I will apply relevant reasoning...
3. Finally, I will provide my answer.

Your step-by-step reasoning:
```

### 4. ReAct

Combines reasoning with action traces:

```
Question: [question]

Use the following format to solve this problem:

Thought 1: [reasoning]
Action 1: [action to take]
Observation 1: [result]

Thought 2: [next reasoning]
Action 2: [next action]
Observation 2: [result]

Final Answer: [conclusion]
```

## Evaluation Metrics

### Vector Distance Metrics

- **Cosine Distance**: Measures angle between embedding vectors (0-2)
- **Euclidean Distance**: Measures absolute distance in embedding space
- **Semantic Similarity**: Cosine similarity score (-1 to 1)

### Aggregate Statistics

- **Mean and Standard Deviation**: Central tendency and spread
- **Exact Match Rate**: Percentage of perfect matches
- **Normalized Score**: Combined metric (0-1)

## Results Visualization

The framework generates several visualization types:

1. **Strategy Comparison**: Bar charts comparing metrics across strategies
2. **Distance Histograms**: Distribution of distances for each strategy
3. **Performance by Type**: Grouped bars showing performance per question type
4. **Improvement Chart**: Percentage change vs baseline

## Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src/prompt_engineering --cov-report=html
```

Run specific tests:
```bash
pytest tests/test_strategies.py -v
```

## Architecture

### Building Blocks Pattern

The project follows a building blocks design with three key components:

1. **Input Data**: `QuestionAnswer` - Represents question-answer pairs
2. **Setup Data**: `PromptStrategy` - Configures prompting behavior
3. **Output Data**: `EvaluationResult` - Contains evaluation metrics

### Class Diagram

```
PromptStrategy (Abstract)
    ├── BaselineStrategy
    ├── FewShotStrategy
    ├── ChainOfThoughtStrategy
    └── ReActStrategy

ExperimentRunner
    ├── uses → LLMClient
    ├── uses → MetricsCalculator
    └── produces → StrategyStatistics

MetricsCalculator
    └── uses → EmbeddingService
```

### Parallel Processing

The experiment runner supports multi-threaded execution using `ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
    futures = {executor.submit(execute_single, qa): qa for qa in dataset}
    for future in as_completed(futures):
        results.append(future.result())
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the GPT API
- Sentence Transformers for embedding models
- The prompt engineering research community
