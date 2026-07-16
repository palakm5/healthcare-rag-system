"""
Ollama LLM Client

Provides a simple interface for calling locally hosted LLMs
through the Ollama API.

Requirements:
    - Ollama installed
    - ollama serve running
    - A model pulled locally
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for locally hosted Ollama models.

    Example:
        client = OllamaClient(model="qwen2.5:7b")

        answer = client.generate(
            prompt="Explain tuberculosis."
        )
    """

    BASE_URL = "http://localhost:11434/api/generate"

    def __init__(
        self,
        model: str = "mistral",
        timeout: int = 300,
    ):
        self.model = model
        self.timeout = timeout

        self.session = requests.Session()

        logger.info(
            "Initialized Ollama client with model: %s",
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
        Generate a response using the local Ollama model.

        Args:
            prompt: Prompt sent to the LLM.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            top_p: Top-p sampling.

        Returns:
            Generated text.
        """

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
            },
        }

        try:
            response = self.session.post(
                self.BASE_URL,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

            result = response.json()

            return result["response"]

        except requests.exceptions.RequestException as e:
            logger.exception("Failed to call Ollama API")
            raise RuntimeError("Ollama API request failed") from e

    def close(self):
        """Close the HTTP session."""
        self.session.close()