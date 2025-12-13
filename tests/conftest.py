"""Pytest configuration and shared fixtures."""

import os
from unittest.mock import patch

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


@pytest.fixture(autouse=True)
def mock_env_api_key():
    """Auto-mock API key for all tests."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key-for-testing"}):
        from prompt_engineering.utils.config import get_settings
        get_settings.cache_clear()
        yield


@pytest.fixture
def clean_settings_cache():
    """Clear settings cache before and after test."""
    from prompt_engineering.utils.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
