"""
Ollama LLM Client

Provides a simple interface for calling locally hosted LLMs through
the Ollama API. By default this client now targets `mistral:7b` if
no explicit model is provided.

Requirements:
    - Ollama installed
    - `ollama serve` running
    - The desired model pulled locally, e.g. `ollama pull mistral:7b`
"""

import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def _strip_thinking_tokens(text: str) -> str:
    """
    Remove model-internal thinking tokens from a raw Ollama response.

    Some Ollama models (mistral:7b, qwen3, mistral-nemo) emit internal
    chain-of-thought blocks before the actual answer when thinking is not
    suppressed at the API level.  Patterns handled:

        <think>...</think>\\n\\nActual answer
        <unused94>thought\\n1. ...\\n\\nActual answer
        /think\\nActual answer

    The function strips the preamble and returns only the final answer.
    If stripping leaves nothing (the whole output was a thinking block),
    the original text is returned unchanged so callers always get *something*.
    """
    original = text

    # 1. Remove complete <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. Remove <unusedN>thought ... up to the first blank line
    text = re.sub(
        r"^<unused\d+>\s*thought\b.*?\n\n",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    # 3. Remove leading /think token left by some models
    text = re.sub(r"^/think\s*", "", text, flags=re.IGNORECASE).strip()

    # 4. Safety: if we emptied the string, return the original
    return text if text else original


# ── Model configuration ──────────────────────────────────────────────
DEFAULT_OLLAMA_MODEL = "mistral:7b"

# No-op disclaimer by default; kept for backward compatibility with the
# prompt-building flow. Set to a non-empty string if a model-specific
# disclaimer is required.
OLLAMA_DISCLAIMER = ""


class OllamaClient:
    """
    Client for locally hosted Ollama models.

    By default this client targets `mistral:7b` unless a different model
    is explicitly provided. A startup sanity check verifies the chosen
    model is pulled and available via Ollama before allowing generation
    calls. If the model is not pulled locally, a clear error is raised
    with pull instructions.

    Example:
        client = OllamaClient()  # uses mistral:7b by default

        answer = client.generate(
            prompt="Explain tuberculosis."
        )

    To override the model explicitly:
        client = OllamaClient(model="mistral:7b")
    """

    BASE_URL = "http://localhost:11434/api/generate"
    TAGS_URL = "http://localhost:11434/api/tags"

    def __init__(
        self,
        model: Optional[str] = None,
        timeout: int = 300,
    ):
        """
        Initialize the Ollama client and verify model availability.

         Args:
             model: Optional model override. If None, the client uses
                 the DEFAULT_OLLAMA_MODEL. If a model is explicitly
                 provided, it must be pulled locally or a clear error
                 is raised.
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout
        self.session = requests.Session()

        # ── Resolve which model to use ────────────────────────────────
        if model is not None:
            # Explicit override — respect it, but still verify availability.
            self.model = model
            self._verify_model_available(self.model)
        else:
            # Default to the selected Ollama model
            self.model = DEFAULT_OLLAMA_MODEL
            self._verify_model_available(self.model)

        logger.info(
            "Initialized Ollama client with model: %s",
            self.model,
        )

    # ─────────────────────────────────────────────────────────────────
    # Startup / sanity check
    # ─────────────────────────────────────────────────────────────────

    def _list_local_models(self) -> list:
        """
        Query Ollama for the list of locally pulled models.

        Returns:
            List of model name strings available locally.

        Raises:
            RuntimeError: If Ollama is not running or the tags endpoint
                         is unreachable.
        """
        try:
            response = self.session.get(self.TAGS_URL, timeout=30)
            response.raise_for_status()
            data = response.json()
            # Ollama returns {"models": [{"name": "model:tag", ...}, ...]}
            return [m.get("name", "") for m in data.get("models", [])]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                "Could not reach Ollama to verify model availability. "
                "Ensure 'ollama serve' is running "
                f"(checked {self.TAGS_URL}). Original error: {e}"
            ) from e

    def _verify_model_available(self, model: str) -> None:
        """
        Verify that the given model is pulled and available locally.

        Args:
            model: The model name (with tag) to verify.

        Raises:
            RuntimeError: If the model is not pulled locally, with clear
                         pull instructions.
        """
        local_models = self._list_local_models()

        # Ollama may return names with or without a digest suffix; do a
        # starts-with match to be robust (e.g. "mistral:7b" matches
        # "mistral:7b" exactly).
        if model not in local_models:
            raise RuntimeError(
                f"Model '{model}' is not pulled locally in Ollama. "
                f"Available models: {local_models or 'none'}. "
                f"Pull it with: ollama pull {model}"
            )

    # Note: legacy MedGemma auto-resolution is removed — the client now
    # relies on an explicit model override or the DEFAULT_OLLAMA_MODEL.

    # ─────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        image=None,  # Reserved for future multimodal use — currently ignored
    ) -> str:
        """
        Generate a response using the configured Ollama model.

        The prompt is expected to be the full RAG prompt (retrieved contexts
        + system instruction + user query) as produced by the shared
        `build_prompt()` function. Any model-specific disclaimer (empty by
        default) is appended to the prompt here.

        Args:
            prompt: Prompt sent to the LLM (RAG prompt with context).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            top_p: Top-p sampling.
            image: Reserved for future multimodal image input. Passing a
                   non-None value logs a warning and ignores it for this
                   text-only client.

        Returns:
            Generated text.
        """
        if image is not None:
            logger.warning(
                "image parameter passed to OllamaClient.generate() but "
                "image-input is not yet implemented on this text-only "
                "path. The image will be ignored."
            )

        # Append any model-specific disclaimer (empty by default).
        full_prompt = f"{prompt}{OLLAMA_DISCLAIMER}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "think": False,           # suppress <think> tokens (qwen3, mistral-nemo)
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

            return _strip_thinking_tokens(result["response"])

        except requests.exceptions.RequestException as e:
            logger.exception("Failed to call Ollama API")
            raise RuntimeError("Ollama API request failed") from e

    def close(self):
        """Close the HTTP session."""
        self.session.close()