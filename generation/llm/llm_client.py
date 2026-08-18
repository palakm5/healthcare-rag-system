"""
OpenRouter LLM Client

Provides a simple interface for calling OpenRouter-hosted LLMs
through the OpenAI-compatible API using the OpenAI SDK.

OpenRouter base URL  : https://openrouter.ai/api/v1
Auth env var         : OPENROUTER_API_KEY
Default model        : mistralai/mistral-nemo:free
  (faithfulness verifier role -- distinct from RAGAS evaluator model)

One-line model swap: change the `model` default in __init__ or pass a
different model string at instantiation. Browse free models at:
  https://openrouter.ai/models?q=:free
"""

import logging
import os
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """
    Client for OpenRouter-hosted LLMs.

    Drop-in replacement for the previous NVIDIAClient -- same .generate()
    interface, different base URL and env var.

    Example:
        client = OpenRouterClient()
        answer = client.generate(prompt="Explain tuberculosis.")

        # Different model:
        client = OpenRouterClient(model="meta-llama/llama-3.1-8b-instruct:free")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nvidia/nemotron-3-super-120b-a12b:free",
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not found. "
                "Add it to your .env file."
            )

        self.model = model
        self.timeout = timeout

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.api_key,
            timeout=self.timeout,
        )

        logger.info(
            "Initialized OpenRouterClient with model: %s",
            self.model,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a response from the OpenRouter-hosted model.

        Args:
            prompt:      Prompt sent to the LLM.
            temperature: Sampling temperature.
            max_tokens:  Maximum output tokens.
            top_p:       Top-p sampling.

        Returns:
            Generated text string.

        Raises:
            RuntimeError: on API failure.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=False,
            )
            return completion.choices[0].message.content

        except Exception:
            logger.exception("Failed to call OpenRouter API (model=%s)", self.model)
            raise RuntimeError("OpenRouter API request failed")

    def close(self):
        """No-op — kept for interface compatibility with the previous client."""
        pass


# ---------------------------------------------------------------------------
# Backward-compatibility alias so any remaining imports of NVIDIAClient keep
# working without changes. Remove after all call sites are updated.
# ---------------------------------------------------------------------------
NVIDIAClient = OpenRouterClient