"""Dataset creation and loading utilities."""

import json
import logging
from pathlib import Path
from typing import Iterator, Optional

from prompt_engineering.data.models import QuestionAnswer, QuestionType

logger = logging.getLogger(__name__)


class Dataset:
    """A collection of question-answer pairs for prompt evaluation.

    Provides iteration, filtering, and serialization capabilities.
    """

    def __init__(self, items: Optional[list[QuestionAnswer]] = None) -> None:
        """Initialize dataset with optional items.

        Args:
            items: Initial list of QA pairs
        """
        self._items: list[QuestionAnswer] = items or []

    def add(self, item: QuestionAnswer) -> None:
        """Add a question-answer pair to the dataset.

        Args:
            item: The QA pair to add
        """
        self._items.append(item)

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self._items)

    def __iter__(self) -> Iterator[QuestionAnswer]:
        """Iterate over all items in the dataset."""
        return iter(self._items)

    def __getitem__(self, index: int) -> QuestionAnswer:
        """Get item by index."""
        return self._items[index]

    def filter_by_type(self, question_type: QuestionType) -> "Dataset":
        """Filter dataset by question type.

        Args:
            question_type: The type to filter by

        Returns:
            New Dataset containing only matching items
        """
        filtered = [item for item in self._items if item.question_type == question_type]
        return Dataset(filtered)

    def get_by_id(self, question_id: str) -> Optional[QuestionAnswer]:
        """Get a specific question by ID.

        Args:
            question_id: The ID to search for

        Returns:
            The matching QA pair or None
        """
        for item in self._items:
            if item.id == question_id:
                return item
        return None

    def to_dict(self) -> list[dict]:
        """Convert dataset to list of dictionaries."""
        return [item.model_dump() for item in self._items]

    def get_type_counts(self) -> dict[QuestionType, int]:
        """Get count of questions per type."""
        counts: dict[QuestionType, int] = {}
        for item in self._items:
            counts[item.question_type] = counts.get(item.question_type, 0) + 1
        return counts


class DatasetLoader:
    """Utility class for loading and saving datasets."""

    @staticmethod
    def load_from_json(file_path: Path) -> Dataset:
        """Load dataset from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Loaded Dataset instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON format is invalid
        """
        logger.info(f"Loading dataset from {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Dataset JSON must be a list of QA pairs")

        items = [QuestionAnswer(**item) for item in data]
        logger.info(f"Loaded {len(items)} items from dataset")

        return Dataset(items)

    @staticmethod
    def save_to_json(dataset: Dataset, file_path: Path) -> None:
        """Save dataset to a JSON file.

        Args:
            dataset: The dataset to save
            file_path: Path to save to
        """
        logger.info(f"Saving dataset to {file_path}")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dataset.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(dataset)} items to {file_path}")
