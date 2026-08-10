"""
Ollama LLM Client (MedGemma path)

Provides a simple interface for calling locally hosted LLMs
through the Ollama API. The Ollama generation path is configured to
use MedGemma (medgemma1.5:4b, falling back to medgemma:4b if the 1.5
variant is unavailable or unstable).

Requirements:
    - Ollama installed
    - ollama serve running
    - MedGemma pulled locally:
          ollama pull medgemma1.5:4b
      (or the fallback: ollama pull medgemma:4b)

      ⚠️  Do NOT use the -q4_0 quantized tag (e.g. medgemma1.5:4b-q4_0).
          That variant has known overfitting issues and is intentionally
          not supported by this client.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ── Model configuration ──────────────────────────────────────────────
# Preferred MedGemma variant, with a fallback if 1.5 is unavailable.
MEDGEMMA_MODEL_PREFERRED = "medgemma1.5:4b"
MEDGEMMA_MODEL_FALLBACK = "medgemma:4b"

# Fixed, model-specific safety note. This is scoped to the Ollama/MedGemma
# path only (lives here, not in the shared prompt builder) because it relates
# to this model's intended use. It is separate from the broader safety
# guardrail work planned for a later phase.
MEDGEMMA_DISCLAIMER = (
    "\n\nIMPORTANT: The output below is informational only and is not a "
    "diagnostic statement. It is not a substitute for professional medical "
    "consultation, diagnosis, or treatment. Always consult a qualified "
    "healthcare professional for medical advice."
)


class OllamaClient:
    """
    Client for locally hosted Ollama models (MedGemma path).

    By default this client targets MedGemma:
        - medgemma1.5:4b  (preferred)
        - medgemma:4b    (fallback if 1.5 is unavailable/unstable)

    A startup sanity check verifies the chosen model is pulled and
    available via Ollama before allowing generation calls. If the model
    is not pulled locally, a clear error is raised with pull instructions.

    Example:
        client = OllamaClient()  # auto-selects medgemma1.5:4b or fallback

        answer = client.generate(
            prompt="Explain tuberculosis."
        )

    To override the model explicitly:
        client = OllamaClient(model="medgemma1.5:4b")
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
            model: Optional model override. If None, the client auto-selects
                   medgemma1.5:4b, falling back to medgemma:4b if 1.5 is not
                   pulled locally. If a model is explicitly provided, it must
                   be pulled locally or a clear error is raised.
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
            # Auto-select: prefer medgemma1.5:4b, fall back to medgemma:4b.
            self.model = self._resolve_medgemma_model()

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
        # starts-with match to be robust (e.g. "medgemma1.5:4b" matches
        # "medgemma1.5:4b" exactly).
        if model not in local_models:
            raise RuntimeError(
                f"Model '{model}' is not pulled locally in Ollama. "
                f"Available models: {local_models or 'none'}. "
                f"Pull it with: ollama pull {model}\n"
                "If medgemma1.5:4b is unavailable or unstable, use the "
                f"fallback: ollama pull {MEDGEMMA_MODEL_FALLBACK}\n"
                "⚠️  Do NOT use the -q4_0 quantized tag "
                "(e.g. medgemma1.5:4b-q4_0) — that variant has known "
                "overfitting issues and is not supported."
            )

    def _resolve_medgemma_model(self) -> str:
        """
        Auto-select the MedGemma variant to use.

        Prefers medgemma1.5:4b; falls back to medgemma:4b if 1.5 is not
        pulled locally. Raises a clear error if neither is available.

        Returns:
            The model name string to use.

        Raises:
            RuntimeError: If neither MedGemma variant is pulled locally.
        """
        local_models = self._list_local_models()

        if MEDGEMMA_MODEL_PREFERRED in local_models:
            return MEDGEMMA_MODEL_PREFERRED

        if MEDGEMMA_MODEL_FALLBACK in local_models:
            logger.warning(
                "%s is not pulled locally; falling back to %s. "
                "Consider pulling the preferred variant: "
                "ollama pull %s",
                MEDGEMMA_MODEL_PREFERRED,
                MEDGEMMA_MODEL_FALLBACK,
                MEDGEMMA_MODEL_PREFERRED,
            )
            return MEDGEMMA_MODEL_FALLBACK

        # Neither is available — clear, actionable error.
        raise RuntimeError(
            f"No MedGemma model found locally in Ollama. "
            f"Available models: {local_models or 'none'}. "
            f"Pull the preferred model with: ollama pull {MEDGEMMA_MODEL_PREFERRED}\n"
            f"Or the fallback with: ollama pull {MEDGEMMA_MODEL_FALLBACK}\n"
            "⚠️  Do NOT use the -q4_0 quantized tag "
            "(e.g. medgemma1.5:4b-q4_0) — that variant has known "
            "overfitting issues and is not supported."
        )

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
        Generate a response using the local Ollama model (MedGemma).

        The prompt is expected to be the full RAG prompt (retrieved chunks
        + system instruction + user query) as produced by the shared
        build_prompt() function. A fixed MedGemma-specific disclaimer is
        appended to the prompt here, scoped to this model's intended use.

        Args:
            prompt: Prompt sent to the LLM (RAG prompt with context).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            top_p: Top-p sampling.
            image: Reserved for future multimodal image input (MedGemma
                   supports image input, but this text-only path does not
                   implement it yet). Passing a non-None value logs a
                   warning and ignores it.

        Returns:
            Generated text.
        """
        if image is not None:
            logger.warning(
                "image parameter passed to OllamaClient.generate() but "
                "image-input is not yet implemented on this text-only "
                "path. The image will be ignored."
            )

        # Append the MedGemma-specific disclaimer to the prompt. This is
        # scoped to this model only and lives here rather than in the
        # shared prompt builder.
        full_prompt = f"{prompt}{MEDGEMMA_DISCLAIMER}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
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