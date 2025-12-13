"""Embedding service for semantic similarity calculations."""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers.

    This is a building block that provides vector representations of text
    for semantic similarity calculations.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self._model_name = model_name
        self._model: Optional[object] = None

    def _load_model(self) -> None:
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self._model_name}")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded successfully")

    def get_embedding(self, text: str) -> NDArray[np.float32]:
        """Get embedding vector for a single text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as numpy array
        """
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)  # type: ignore
        return embedding.astype(np.float32)

    def get_embeddings_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Get embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            2D array of embeddings (num_texts x embedding_dim)
        """
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)  # type: ignore
        return embeddings.astype(np.float32)

    @staticmethod
    def cosine_similarity(vec1: NDArray[np.float32], vec2: NDArray[np.float32]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (-1 to 1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def cosine_distance(vec1: NDArray[np.float32], vec2: NDArray[np.float32]) -> float:
        """Calculate cosine distance between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine distance (0 to 2)
        """
        return 1.0 - EmbeddingService.cosine_similarity(vec1, vec2)

    @staticmethod
    def euclidean_distance(vec1: NDArray[np.float32], vec2: NDArray[np.float32]) -> float:
        """Calculate Euclidean distance between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(vec1 - vec2))
