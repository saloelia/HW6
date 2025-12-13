"""LLM client for interacting with OpenAI-compatible APIs."""

import logging
import time
from typing import Optional

from openai import OpenAI

from prompt_engineering.utils.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for LLM API interactions.

    This building block handles all communication with the LLM API,
    including retries, timing, and error handling.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the LLM client.

        Args:
            settings: Application settings (uses default if not provided)
        """
        self._settings = settings or get_settings()
        self._client = self._create_client()

    def _create_client(self) -> OpenAI:
        """Create the OpenAI client instance."""
        return OpenAI(
            api_key=self._settings.openai_api_key.get_secret_value(),
            base_url=self._settings.openai_base_url,
            timeout=self._settings.request_timeout,
        )

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> tuple[str, float]:
        """Get a completion from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Max tokens (overrides settings)
            temperature: Temperature (overrides settings)

        Returns:
            Tuple of (response text, execution time in ms)
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        start_time = time.perf_counter()

        response = self._client.chat.completions.create(
            model=self._settings.openai_model,
            messages=messages,
            max_tokens=max_tokens or self._settings.max_tokens,
            temperature=temperature if temperature is not None else self._settings.temperature,
        )

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        content = response.choices[0].message.content or ""

        logger.debug(
            f"LLM completion: {len(prompt)} chars -> {len(content)} chars in {execution_time_ms:.1f}ms"
        )

        return content, execution_time_ms

    def complete_batch(
        self,
        prompts: list[tuple[str, Optional[str]]],
    ) -> list[tuple[str, float]]:
        """Get completions for multiple prompts sequentially.

        Args:
            prompts: List of (prompt, system_prompt) tuples

        Returns:
            List of (response text, execution time in ms) tuples
        """
        results = []

        for prompt, system_prompt in prompts:
            result = self.complete(prompt, system_prompt)
            results.append(result)

        return results
