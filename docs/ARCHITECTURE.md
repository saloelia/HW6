# Architecture Documentation

## Prompt Engineering Experiment Framework

**Version**: 1.0.0
**Last Updated**: December 2024

---

## 1. Overview

This document describes the software architecture of the Prompt Engineering Experiment Framework. The system follows a modular, building-blocks design pattern optimized for extensibility and maintainability.

---

## 2. Design Principles

### 2.1 Building Blocks Pattern

The architecture follows the building blocks pattern as specified in the software submission guidelines:

```
┌─────────────────────────────────────────────────────────────┐
│                    BUILDING BLOCKS                           │
├─────────────────────────────────────────────────────────────┤
│  INPUT DATA      │  SETUP DATA       │  OUTPUT DATA         │
│  ─────────────   │  ──────────────   │  ─────────────────   │
│  QuestionAnswer  │  PromptStrategy   │  PromptResult        │
│  Dataset         │  Settings         │  EvaluationResult    │
│                  │  LLMClient        │  StrategyStatistics  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Key Design Decisions

1. **Separation of Concerns**: Each module handles a single responsibility
2. **Dependency Injection**: Components receive dependencies via constructors
3. **Abstract Interfaces**: Strategies use abstract base classes for extensibility
4. **Immutable Data Models**: Pydantic models with `frozen=True` where appropriate
5. **Lazy Loading**: Heavy resources (embeddings) loaded on first use

---

## 3. Module Architecture

### 3.1 Package Structure

```
src/prompt_engineering/
├── __init__.py           # Public API exports
├── main.py               # CLI entry point
├── experiment.py         # Experiment orchestration
│
├── data/                 # Data layer
│   ├── __init__.py
│   ├── models.py         # Pydantic data models
│   └── dataset.py        # Dataset operations
│
├── prompts/              # Prompting strategies
│   ├── __init__.py
│   └── strategies.py     # Strategy implementations
│
├── evaluation/           # Evaluation layer
│   ├── __init__.py
│   ├── embeddings.py     # Embedding service
│   └── metrics.py        # Metrics calculation
│
├── visualization/        # Presentation layer
│   ├── __init__.py
│   └── plots.py          # Plot generation
│
└── utils/                # Shared utilities
    ├── __init__.py
    ├── config.py         # Configuration management
    └── llm_client.py     # LLM API client
```

### 3.2 Module Dependencies

```
                    ┌──────────────┐
                    │    main.py   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  experiment  │
                    └──────┬───────┘
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────────┐
    │   data   │    │  prompts │    │ visualization│
    └────┬─────┘    └────┬─────┘    └──────────────┘
         │               │
         │               ▼
         │         ┌──────────┐
         └────────►│evaluation│
                   └────┬─────┘
                        │
                        ▼
                   ┌──────────┐
                   │  utils   │
                   └──────────┘
```

---

## 4. Component Details

### 4.1 Data Layer (`data/`)

#### 4.1.1 Models (`models.py`)

Defines all data structures using Pydantic:

```python
class QuestionType(Enum):
    """Question categories"""
    SENTIMENT = "sentiment"
    MATH = "math"
    LOGIC = "logic"

class QuestionAnswer(BaseModel):
    """Single Q&A pair - INPUT DATA block"""
    id: str
    question_type: QuestionType
    question: str
    expected_answer: str
    metadata: Optional[dict]

class PromptResult(BaseModel):
    """Raw execution result - OUTPUT DATA block"""
    question_id: str
    strategy_name: str
    prompt_used: str
    model_response: str
    expected_answer: str
    execution_time_ms: float

class EvaluationResult(BaseModel):
    """Computed metrics - OUTPUT DATA block"""
    question_id: str
    strategy_name: str
    cosine_distance: float
    euclidean_distance: float
    semantic_similarity: float
    exact_match: bool
    normalized_score: float

class StrategyStatistics(BaseModel):
    """Aggregated statistics - OUTPUT DATA block"""
    strategy_name: str
    mean_cosine_distance: float
    std_cosine_distance: float
    mean_semantic_similarity: float
    std_semantic_similarity: float
    exact_match_rate: float
```

#### 4.1.2 Dataset (`dataset.py`)

Manages collections of Q&A pairs:

```python
class Dataset:
    """Collection with iteration and filtering"""
    def __init__(self, items: list[QuestionAnswer])
    def add(self, item: QuestionAnswer) -> None
    def filter_by_type(self, qt: QuestionType) -> Dataset
    def get_by_id(self, id: str) -> Optional[QuestionAnswer]
    def __iter__(self) -> Iterator[QuestionAnswer]
    def __len__(self) -> int

class DatasetLoader:
    """JSON serialization utilities"""
    @staticmethod
    def load_from_json(path: Path) -> Dataset
    @staticmethod
    def save_to_json(dataset: Dataset, path: Path) -> None
```

### 4.2 Prompts Layer (`prompts/`)

#### 4.2.1 Strategy Interface

```python
class PromptStrategy(ABC):
    """Abstract base - SETUP DATA block"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier"""
        pass

    @abstractmethod
    def build_prompt(self, question: QuestionAnswer) -> str:
        """Transform question into prompt"""
        pass

    @abstractmethod
    def get_system_prompt(self) -> Optional[str]:
        """Return system prompt if any"""
        pass
```

#### 4.2.2 Concrete Strategies

| Strategy | Description | Key Features |
|----------|-------------|--------------|
| `BaselineStrategy` | Minimal atomic prompts | Simple question-answer format |
| `FewShotStrategy` | Example-based learning | 3 examples per question type |
| `ChainOfThoughtStrategy` | Step-by-step reasoning | Type-specific instructions |
| `ReActStrategy` | Reasoning + Acting | Thought-Action-Observation format |

### 4.3 Evaluation Layer (`evaluation/`)

#### 4.3.1 Embedding Service (`embeddings.py`)

```python
class EmbeddingService:
    """Vector embedding operations"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2")
    def get_embedding(self, text: str) -> NDArray[np.float32]
    def get_embeddings_batch(self, texts: list[str]) -> NDArray[np.float32]

    @staticmethod
    def cosine_similarity(v1, v2) -> float
    @staticmethod
    def cosine_distance(v1, v2) -> float
    @staticmethod
    def euclidean_distance(v1, v2) -> float
```

#### 4.3.2 Metrics Calculator (`metrics.py`)

```python
class MetricsCalculator:
    """Evaluation metrics computation"""

    def __init__(self, embedding_service: EmbeddingService)
    def evaluate_result(self, result: PromptResult) -> EvaluationResult
    def evaluate_batch(self, results: list[PromptResult]) -> list[EvaluationResult]
    def calculate_statistics(
        self,
        evaluations: list[EvaluationResult],
        execution_times: list[float],
        question_type: Optional[QuestionType]
    ) -> StrategyStatistics
```

### 4.4 Visualization Layer (`visualization/`)

```python
class ResultVisualizer:
    """Plot generation utilities"""

    def __init__(self, output_dir: Path, style: str = "whitegrid")

    def plot_strategy_comparison(
        self, statistics: list[StrategyStatistics], metric: str
    ) -> plt.Figure

    def plot_distance_histogram(
        self, evaluations: dict[str, list[EvaluationResult]], metric: str
    ) -> plt.Figure

    def plot_performance_by_type(
        self, stats_by_type: dict[QuestionType, list[StrategyStatistics]]
    ) -> plt.Figure

    def plot_improvement_degradation(
        self, baseline: StrategyStatistics, improved: list[StrategyStatistics]
    ) -> plt.Figure
```

### 4.5 Utils Layer (`utils/`)

#### 4.5.1 Configuration (`config.py`)

```python
class Settings(BaseSettings):
    """Environment-based configuration - SETUP DATA block"""

    # API Configuration
    openai_api_key: SecretStr  # Never logged
    openai_model: str = "gpt-3.5-turbo"
    openai_base_url: Optional[str] = None

    # Experiment Settings
    max_tokens: int = 500
    temperature: float = 0.0

    # Parallel Processing
    max_workers: int = 4
    batch_size: int = 10

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton"""
```

#### 4.5.2 LLM Client (`llm_client.py`)

```python
class LLMClient:
    """OpenAI API wrapper - SETUP DATA block"""

    def __init__(self, settings: Settings)

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> tuple[str, float]  # (response, time_ms)
```

### 4.6 Experiment Orchestration (`experiment.py`)

```python
class ExperimentRunner:
    """Main orchestration - coordinates all components"""

    def __init__(
        self,
        strategies: list[PromptStrategy],
        dataset: Dataset,
        settings: Settings
    )

    def run_experiment(self, use_parallel: bool = True) -> None
    def get_statistics(self) -> dict[str, StrategyStatistics]
    def get_evaluations(self) -> dict[str, list[EvaluationResult]]
    def get_statistics_by_type(self) -> dict[QuestionType, dict[str, StrategyStatistics]]
    def save_results(self, output_dir: Path) -> None
```

---

## 5. Parallel Processing

### 5.1 Threading Model

The framework uses `ThreadPoolExecutor` for parallel execution:

```python
def _run_strategy_parallel(self, strategy: PromptStrategy) -> list[PromptResult]:
    results = []

    with ThreadPoolExecutor(max_workers=self._settings.max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(self._execute_single, strategy, qa): qa
            for qa in self._dataset
        }

        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results.append(result)

    return results
```

### 5.2 Thread Safety

- **Stateless Strategies**: Strategy objects are read-only during execution
- **Independent Requests**: Each LLM call is independent
- **Thread-Local Clients**: OpenAI client handles threading internally

### 5.3 Why Threading vs Multiprocessing

- **I/O-bound workload**: API calls are network-bound, not CPU-bound
- **Lower overhead**: Threads share memory, no serialization needed
- **GIL not a bottleneck**: GIL is released during I/O operations

---

## 6. Error Handling

### 6.1 Error Categories

| Category | Handling |
|----------|----------|
| Configuration errors | Fail fast with clear message |
| Dataset errors | FileNotFoundError, ValueError |
| API errors | Propagate with context |
| Embedding errors | Lazy loading failures |

### 6.2 Logging

```python
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

---

## 7. Extension Points

### 7.1 Adding New Strategies

1. Create new class extending `PromptStrategy`
2. Implement `name`, `build_prompt()`, `get_system_prompt()`
3. Register in `main.py` strategy map

```python
class MyCustomStrategy(PromptStrategy):
    @property
    def name(self) -> str:
        return "my_custom"

    def build_prompt(self, question: QuestionAnswer) -> str:
        return f"Custom: {question.question}"

    def get_system_prompt(self) -> Optional[str]:
        return "Custom system prompt"
```

### 7.2 Adding New Question Types

1. Add to `QuestionType` enum
2. Update `FewShotStrategy._create_examples()`
3. Update `ChainOfThoughtStrategy._get_type_specific_instruction()`
4. Add questions to dataset

### 7.3 Adding New Metrics

1. Add to `EvaluationResult` model
2. Implement calculation in `MetricsCalculator.evaluate_result()`
3. Add visualization in `ResultVisualizer`

---

## 8. Security Considerations

### 8.1 Credential Management

- API keys stored in `.env` file (gitignored)
- `SecretStr` type prevents accidental logging
- No hardcoded credentials

### 8.2 Input Validation

- Pydantic models validate all inputs
- Path operations use `pathlib` for safety
- No arbitrary code execution

---

## 9. Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Dataset load | O(n) | n = number of questions |
| Single prompt | O(1) | API call latency dominates |
| Embedding | O(d) | d = embedding dimension |
| Statistics | O(n) | n = number of evaluations |
| Plotting | O(n) | n = data points |

### 9.1 Memory Usage

- Dataset: ~1KB per question
- Embeddings: 384 floats × 4 bytes = 1.5KB per text
- Results: ~2KB per evaluation

---

## 10. Testing Strategy

### 10.1 Unit Test Coverage

| Module | Coverage Target |
|--------|-----------------|
| `data/models.py` | 90% |
| `data/dataset.py` | 85% |
| `prompts/strategies.py` | 80% |
| `evaluation/metrics.py` | 80% |
| `evaluation/embeddings.py` | 75% |
| `visualization/plots.py` | 70% |
| `utils/config.py` | 75% |
| **Overall** | **≥70%** |

### 10.2 Testing Approach

- **Models**: Validation, serialization, edge cases
- **Dataset**: Loading, filtering, iteration
- **Strategies**: Prompt format, system prompts
- **Metrics**: Distance calculations (mocked embeddings)
- **Visualization**: Figure generation (don't test visual output)
