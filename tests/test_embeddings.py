"""Tests for embedding service."""

import numpy as np
import pytest

from prompt_engineering.evaluation.embeddings import EmbeddingService


class TestEmbeddingServiceStaticMethods:
    """Tests for static methods of EmbeddingService."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors is 1."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        similarity = EmbeddingService.cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors is 0."""
        vec1 = np.array([1.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0], dtype=np.float32)
        similarity = EmbeddingService.cosine_similarity(vec1, vec2)
        assert abs(similarity) < 1e-6

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors is -1."""
        vec1 = np.array([1.0, 0.0], dtype=np.float32)
        vec2 = np.array([-1.0, 0.0], dtype=np.float32)
        similarity = EmbeddingService.cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-6

    def test_cosine_similarity_with_zero_vector(self):
        """Test cosine similarity with zero vector returns 0."""
        vec1 = np.array([1.0, 2.0], dtype=np.float32)
        vec2 = np.array([0.0, 0.0], dtype=np.float32)
        similarity = EmbeddingService.cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_cosine_distance_identical(self):
        """Test cosine distance of identical vectors is 0."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        distance = EmbeddingService.cosine_distance(vec, vec)
        assert abs(distance) < 1e-6

    def test_cosine_distance_opposite(self):
        """Test cosine distance of opposite vectors is 2."""
        vec1 = np.array([1.0, 0.0], dtype=np.float32)
        vec2 = np.array([-1.0, 0.0], dtype=np.float32)
        distance = EmbeddingService.cosine_distance(vec1, vec2)
        assert abs(distance - 2.0) < 1e-6

    def test_cosine_distance_range(self):
        """Test cosine distance is in range [0, 2]."""
        np.random.seed(42)
        for _ in range(10):
            vec1 = np.random.randn(10).astype(np.float32)
            vec2 = np.random.randn(10).astype(np.float32)
            distance = EmbeddingService.cosine_distance(vec1, vec2)
            assert 0 <= distance <= 2 + 1e-6

    def test_euclidean_distance_identical(self):
        """Test Euclidean distance of identical vectors is 0."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        distance = EmbeddingService.euclidean_distance(vec, vec)
        assert abs(distance) < 1e-6

    def test_euclidean_distance_unit_vectors(self):
        """Test Euclidean distance of unit vectors."""
        vec1 = np.array([1.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0], dtype=np.float32)
        distance = EmbeddingService.euclidean_distance(vec1, vec2)
        expected = np.sqrt(2)
        assert abs(distance - expected) < 1e-6

    def test_euclidean_distance_is_positive(self):
        """Test Euclidean distance is always non-negative."""
        np.random.seed(42)
        for _ in range(10):
            vec1 = np.random.randn(10).astype(np.float32)
            vec2 = np.random.randn(10).astype(np.float32)
            distance = EmbeddingService.euclidean_distance(vec1, vec2)
            assert distance >= 0


class TestEmbeddingServiceInit:
    """Tests for EmbeddingService initialization."""

    def test_default_model_name(self):
        """Test default model name is set."""
        service = EmbeddingService()
        assert service._model_name == "all-MiniLM-L6-v2"

    def test_custom_model_name(self):
        """Test custom model name is stored."""
        service = EmbeddingService(model_name="custom-model")
        assert service._model_name == "custom-model"

    def test_model_not_loaded_initially(self):
        """Test model is not loaded on initialization (lazy loading)."""
        service = EmbeddingService()
        assert service._model is None


@pytest.mark.slow
class TestEmbeddingServiceIntegration:
    """Integration tests that load actual embedding model."""

    @pytest.fixture
    def embedding_service(self):
        """Create embedding service with actual model."""
        return EmbeddingService()

    def test_get_embedding_returns_array(self, embedding_service):
        """Test that get_embedding returns numpy array."""
        embedding = embedding_service.get_embedding("Hello world")
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32

    def test_get_embedding_dimension(self, embedding_service):
        """Test embedding has expected dimension."""
        embedding = embedding_service.get_embedding("Test text")
        assert embedding.shape == (384,)

    def test_similar_texts_close_embeddings(self, embedding_service):
        """Test that similar texts have close embeddings."""
        emb1 = embedding_service.get_embedding("The cat is sleeping")
        emb2 = embedding_service.get_embedding("The cat is resting")
        emb3 = embedding_service.get_embedding("Complex mathematical equations")

        sim_12 = EmbeddingService.cosine_similarity(emb1, emb2)
        sim_13 = EmbeddingService.cosine_similarity(emb1, emb3)

        assert sim_12 > sim_13

    def test_batch_embeddings(self, embedding_service):
        """Test batch embedding generation."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = embedding_service.get_embeddings_batch(texts)

        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    def test_batch_consistent_with_single(self, embedding_service):
        """Test batch embeddings match single embeddings."""
        text = "Test sentence"
        single = embedding_service.get_embedding(text)
        batch = embedding_service.get_embeddings_batch([text])

        np.testing.assert_array_almost_equal(single, batch[0])
