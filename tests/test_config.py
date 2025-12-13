"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from prompt_engineering.utils.config import Settings, get_settings, setup_logging


class TestSettings:
    """Tests for Settings class."""

    def test_create_settings_with_api_key(self):
        """Test creating settings with API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            assert settings.openai_api_key.get_secret_value() == "test-key"

    def test_api_key_is_secret(self):
        """Test that API key is stored as SecretStr."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret-key"}):
            settings = Settings()
            assert isinstance(settings.openai_api_key, SecretStr)
            assert "secret-key" not in str(settings.openai_api_key)

    def test_default_model(self):
        """Test default model is gpt-3.5-turbo."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            assert settings.openai_model == "gpt-3.5-turbo"

    def test_custom_model(self):
        """Test custom model configuration."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4"},
        ):
            settings = Settings()
            assert settings.openai_model == "gpt-4"

    def test_default_max_tokens(self):
        """Test default max tokens."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            assert settings.max_tokens == 500

    def test_custom_max_tokens(self):
        """Test custom max tokens."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "MAX_TOKENS": "1000"},
        ):
            settings = Settings()
            assert settings.max_tokens == 1000

    def test_default_temperature(self):
        """Test default temperature is 0."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            assert settings.temperature == 0.0

    def test_default_max_workers(self):
        """Test default max workers."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            assert settings.max_workers == 4

    def test_custom_max_workers(self):
        """Test custom max workers."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "MAX_WORKERS": "8"},
        ):
            settings = Settings()
            assert settings.max_workers == 8

    def test_default_paths(self):
        """Test default path configurations."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            assert settings.data_dir == Path("data")
            assert settings.results_dir == Path("results")

    def test_default_embedding_model(self):
        """Test default embedding model."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            assert settings.embedding_model == "all-MiniLM-L6-v2"

    def test_default_log_level(self):
        """Test default log level."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            assert settings.log_level == "INFO"

    def test_custom_log_level(self):
        """Test custom log level."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "LOG_LEVEL": "DEBUG"},
        ):
            settings = Settings()
            assert settings.log_level == "DEBUG"

    def test_base_url_optional(self):
        """Test base URL is optional."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            assert settings.openai_base_url is None

    def test_custom_base_url(self):
        """Test custom base URL."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://custom.api.com",
            },
        ):
            settings = Settings()
            assert settings.openai_base_url == "https://custom.api.com"

    def test_load_from_env_file(self):
        """Test loading from .env file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("OPENAI_API_KEY=file-key\nOPENAI_MODEL=gpt-4\n")

            with patch.dict(os.environ, {}, clear=True):
                original_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    settings = Settings(_env_file=env_path)
                    assert settings.openai_api_key.get_secret_value() == "file-key"
                    assert settings.openai_model == "gpt-4"
                finally:
                    os.chdir(original_cwd)


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings_instance(self):
        """Test that get_settings returns Settings instance."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            get_settings.cache_clear()
            settings = get_settings()
            assert isinstance(settings, Settings)

    def test_cached(self):
        """Test that get_settings is cached."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            get_settings.cache_clear()
            settings1 = get_settings()
            settings2 = get_settings()
            assert settings1 is settings2


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_with_default_level(self):
        """Test setup with default level from settings."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            get_settings.cache_clear()
            setup_logging()

    def test_setup_with_custom_level(self):
        """Test setup with custom level."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            get_settings.cache_clear()
            setup_logging(level="DEBUG")

    def test_setup_with_warning_level(self):
        """Test setup with warning level."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            get_settings.cache_clear()
            setup_logging(level="WARNING")
