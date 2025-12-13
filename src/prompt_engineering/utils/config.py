"""Configuration management using pydantic-settings."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    This building block manages all configuration securely.
    Sensitive values are stored as SecretStr to prevent accidental logging.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Configuration
    openai_api_key: SecretStr = Field(
        ..., description="OpenAI API key for LLM access"
    )
    openai_model: str = Field(
        default="gpt-3.5-turbo", description="OpenAI model to use"
    )
    openai_base_url: Optional[str] = Field(
        default=None, description="Custom base URL for OpenAI-compatible APIs"
    )

    # Experiment Configuration
    max_tokens: int = Field(
        default=500, description="Maximum tokens in LLM response"
    )
    temperature: float = Field(
        default=0.0, description="Temperature for LLM sampling (0 = deterministic)"
    )
    request_timeout: int = Field(
        default=60, description="Timeout for API requests in seconds"
    )

    # Parallel Processing
    max_workers: int = Field(
        default=4, description="Maximum number of parallel workers"
    )
    batch_size: int = Field(
        default=10, description="Batch size for parallel processing"
    )

    # Paths
    data_dir: Path = Field(
        default=Path("data"), description="Directory for datasets"
    )
    results_dir: Path = Field(
        default=Path("results"), description="Directory for results"
    )

    # Embedding Configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model for embeddings"
    )

    # Logging
    log_level: str = Field(
        default="INFO", description="Logging level"
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance loaded from environment
    """
    logger.info("Loading settings from environment")
    return Settings()


def setup_logging(level: Optional[str] = None) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level to use (defaults to settings)
    """
    settings = get_settings()
    log_level = level or settings.log_level

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info(f"Logging configured at {log_level} level")
