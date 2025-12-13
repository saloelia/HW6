"""Tests for dataset operations."""

import json
import tempfile
from pathlib import Path

import pytest

from prompt_engineering.data.dataset import Dataset, DatasetLoader
from prompt_engineering.data.models import QuestionAnswer, QuestionType


@pytest.fixture
def sample_questions() -> list[QuestionAnswer]:
    """Create sample questions for testing."""
    return [
        QuestionAnswer(
            id="sent_001",
            question_type=QuestionType.SENTIMENT,
            question="Is this positive?",
            expected_answer="yes",
        ),
        QuestionAnswer(
            id="sent_002",
            question_type=QuestionType.SENTIMENT,
            question="Is this negative?",
            expected_answer="no",
        ),
        QuestionAnswer(
            id="math_001",
            question_type=QuestionType.MATH,
            question="2+2?",
            expected_answer="4",
        ),
        QuestionAnswer(
            id="logic_001",
            question_type=QuestionType.LOGIC,
            question="All A are B. X is A. Is X B?",
            expected_answer="yes",
        ),
    ]


@pytest.fixture
def sample_dataset(sample_questions) -> Dataset:
    """Create sample dataset for testing."""
    return Dataset(sample_questions)


class TestDataset:
    """Tests for Dataset class."""

    def test_create_empty_dataset(self):
        """Test creating an empty dataset."""
        dataset = Dataset()
        assert len(dataset) == 0

    def test_create_with_items(self, sample_questions):
        """Test creating dataset with initial items."""
        dataset = Dataset(sample_questions)
        assert len(dataset) == 4

    def test_add_item(self):
        """Test adding item to dataset."""
        dataset = Dataset()
        qa = QuestionAnswer(
            id="test_001",
            question_type=QuestionType.MATH,
            question="1+1?",
            expected_answer="2",
        )
        dataset.add(qa)
        assert len(dataset) == 1

    def test_iteration(self, sample_dataset):
        """Test iterating over dataset."""
        items = list(sample_dataset)
        assert len(items) == 4
        assert all(isinstance(item, QuestionAnswer) for item in items)

    def test_indexing(self, sample_dataset):
        """Test accessing items by index."""
        first = sample_dataset[0]
        assert first.id == "sent_001"

        last = sample_dataset[-1]
        assert last.id == "logic_001"

    def test_filter_by_type_sentiment(self, sample_dataset):
        """Test filtering by sentiment type."""
        filtered = sample_dataset.filter_by_type(QuestionType.SENTIMENT)
        assert len(filtered) == 2
        assert all(q.question_type == QuestionType.SENTIMENT for q in filtered)

    def test_filter_by_type_math(self, sample_dataset):
        """Test filtering by math type."""
        filtered = sample_dataset.filter_by_type(QuestionType.MATH)
        assert len(filtered) == 1
        assert filtered[0].id == "math_001"

    def test_filter_by_type_logic(self, sample_dataset):
        """Test filtering by logic type."""
        filtered = sample_dataset.filter_by_type(QuestionType.LOGIC)
        assert len(filtered) == 1

    def test_filter_returns_new_dataset(self, sample_dataset):
        """Test that filter returns a new dataset."""
        filtered = sample_dataset.filter_by_type(QuestionType.MATH)
        assert filtered is not sample_dataset
        assert len(sample_dataset) == 4

    def test_get_by_id_exists(self, sample_dataset):
        """Test getting question by existing ID."""
        qa = sample_dataset.get_by_id("math_001")
        assert qa is not None
        assert qa.question == "2+2?"

    def test_get_by_id_not_exists(self, sample_dataset):
        """Test getting question by non-existing ID."""
        qa = sample_dataset.get_by_id("nonexistent")
        assert qa is None

    def test_to_dict(self, sample_dataset):
        """Test converting dataset to dict list."""
        data = sample_dataset.to_dict()
        assert isinstance(data, list)
        assert len(data) == 4
        assert all(isinstance(item, dict) for item in data)
        assert data[0]["id"] == "sent_001"

    def test_get_type_counts(self, sample_dataset):
        """Test getting question counts by type."""
        counts = sample_dataset.get_type_counts()
        assert counts[QuestionType.SENTIMENT] == 2
        assert counts[QuestionType.MATH] == 1
        assert counts[QuestionType.LOGIC] == 1


class TestDatasetLoader:
    """Tests for DatasetLoader class."""

    def test_save_and_load_json(self, sample_dataset):
        """Test saving and loading dataset from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dataset.json"

            DatasetLoader.save_to_json(sample_dataset, path)

            assert path.exists()

            loaded = DatasetLoader.load_from_json(path)

            assert len(loaded) == len(sample_dataset)
            assert loaded[0].id == sample_dataset[0].id

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            DatasetLoader.load_from_json(Path("/nonexistent/path.json"))

    def test_load_invalid_json_format(self):
        """Test loading invalid JSON format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.json"
            with open(path, "w") as f:
                json.dump({"not": "a list"}, f)

            with pytest.raises(ValueError):
                DatasetLoader.load_from_json(path)

    def test_save_creates_parent_dirs(self, sample_dataset):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "dataset.json"

            DatasetLoader.save_to_json(sample_dataset, path)

            assert path.exists()

    def test_saved_json_format(self, sample_dataset):
        """Test the format of saved JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"

            DatasetLoader.save_to_json(sample_dataset, path)

            with open(path) as f:
                data = json.load(f)

            assert isinstance(data, list)
            assert "id" in data[0]
            assert "question_type" in data[0]
            assert "question" in data[0]
            assert "expected_answer" in data[0]
