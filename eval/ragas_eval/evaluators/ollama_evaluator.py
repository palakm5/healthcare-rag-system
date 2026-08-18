"""
Ollama RAGAS Evaluator
======================

Builds the LLM and (optional) embeddings wrappers for running RAGAS metrics
via a locally hosted Ollama model.

Design decisions:
  - LLM  : qwen3:8b via Ollama's OpenAI-compatible endpoint (localhost:11434/v1).
            We use ChatOpenAI instead of ChatOllama to avoid a known bug where
            langchain-ollama passes `temperature` as a top-level kwarg to the
            Ollama async client, which newer Ollama SDK versions reject.
  - Embed : nvidia/nv-embed-v1 via NVIDIA NIM (same key used for ingestion).
            Ollama does not expose an embeddings endpoint without --embeddings.
            Returns None if NVIDIA_API_KEY is absent; answer_relevancy is then
            skipped gracefully by run_ragas_evaluation().
  - RunConfig: timeout=600s, max_workers=1 (sequential) -- CPU-bound local model
               is slow; concurrency only causes OOM and timeout cascades.

One-line model swap:
    Change EVALUATOR_MODEL at the top of this file.
    Browse locally available models: ollama list
"""

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration -- one-line swaps
# ---------------------------------------------------------------------------

# Must be different from the answer-generation model (mistral:7b).
EVALUATOR_MODEL: str = "qwen3:8b"
OLLAMA_BASE_URL: str = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_llm():
    """
    Construct a RAGAS-compatible LLM wrapper using local Ollama (qwen3:8b).

    Uses ChatOpenAI pointed at Ollama's OpenAI-compatible endpoint to avoid
    the temperature kwarg bug in langchain-ollama.

    Returns:
        ragas.llms.LangchainLLMWrapper

    Raises:
        ImportError: if ragas or langchain-openai are not installed.
    """
    try:
        from langchain_openai import ChatOpenAI         # type: ignore
        from ragas.llms import LangchainLLMWrapper      # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies. Install with:\n"
            "  pip install ragas langchain-openai"
        ) from exc

    ollama_openai_url = f"{OLLAMA_BASE_URL}/v1"
    logger.info(
        "[ollama] Initialising evaluator LLM: %s @ %s",
        EVALUATOR_MODEL, ollama_openai_url,
    )

    langchain_llm = ChatOpenAI(
        model=EVALUATOR_MODEL,
        openai_api_key="ollama",           # dummy — Ollama ignores auth
        openai_api_base=ollama_openai_url,
        temperature=0.0,
        max_tokens=1024,
        max_retries=2,
        extra_body={"think": False},       # suppress qwen3 <think> tokens
    )

    return LangchainLLMWrapper(langchain_llm)


def build_embeddings():
    """
    Construct a RAGAS-compatible embeddings wrapper.

    Uses nvidia/nv-embed-v1 via NVIDIA NIM (requires NVIDIA_API_KEY).
    Returns None if the key is absent — answer_relevancy will be skipped.

    Returns:
        ragas.embeddings.LangchainEmbeddingsWrapper | None
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        logger.warning(
            "[ollama] NVIDIA_API_KEY not set — answer_relevancy will be skipped."
        )
        return None

    try:
        from langchain_openai import OpenAIEmbeddings             # type: ignore
        from ragas.embeddings import LangchainEmbeddingsWrapper   # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies. Install with:\n"
            "  pip install ragas langchain-openai"
        ) from exc

    logger.info("[ollama] Initialising NVIDIA NIM embeddings: nvidia/nv-embed-v1.")

    embeddings = OpenAIEmbeddings(
        model="nvidia/nv-embed-v1",
        openai_api_key=api_key,
        openai_api_base="https://integrate.api.nvidia.com/v1",
        check_embedding_ctx_length=False,
    )

    return LangchainEmbeddingsWrapper(embeddings)


def build_run_config():
    """
    Return a RAGAS RunConfig tuned for local CPU-bound Ollama inference.

    Settings:
        timeout=600     10 min per LLM call — qwen3:8b on CPU takes ~90-120s.
        max_retries=2   Retry transient Ollama errors.
        max_wait=120    Max back-off between retries.
        max_workers=1   Sequential execution — prevents OOM / thermal throttle.

    Returns:
        ragas.run_config.RunConfig
    """
    from ragas.run_config import RunConfig  # type: ignore

    return RunConfig(
        timeout=600,
        max_retries=2,
        max_wait=120,
        max_workers=1,
    )


# Descriptive name for report metadata
BACKEND_NAME = "ollama"
BACKEND_DESCRIPTION = f"Local Ollama ({EVALUATOR_MODEL})"
