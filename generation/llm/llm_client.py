"""
NVIDIA LLM Client

Provides a simple interface for calling NVIDIA-hosted LLMs
through the OpenAI-compatible API using the OpenAI SDK.
"""

import logging
import os
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class NVIDIAClient:
    """
    Client for NVIDIA hosted LLMs.

    Example:
        client = NVIDIAClient()

        answer = client.generate(
            prompt="Explain tuberculosis."
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistralai/mistral-nemotron",
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")

        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY environment variable not found."
            )

        self.model = model
        self.timeout = timeout

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key,
            timeout=self.timeout,
        )

        logger.info(
            "Initialized NVIDIA client with model: %s",
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
        Generate a response from the NVIDIA hosted model.

        Args:
            prompt: Prompt sent to the LLM.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            top_p: Top-p sampling.

        Returns:
            Generated text.
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=False,
            )

            return completion.choices[0].message.content

        except Exception:
            logger.exception("Failed to call NVIDIA API")
            raise RuntimeError("NVIDIA API request failed")

    def close(self):
        """
        Close the client.

        (No explicit cleanup required for the OpenAI client,
        but the method is kept for interface compatibility.)
        """
        pass