"""
NVIDIA RAGAS Evaluator
======================

Builds the LLM and embeddings wrappers for running RAGAS metrics via the
NVIDIA NIM API (OpenAI-compatible endpoint).

Design decisions:
  - LLM  : meta/llama-3.1-70b-instruct via NVIDIA NIM.
            Must be a different model from the faithfulness verifier
            (qwen3:8b) used elsewhere in this project.
  - Embed : nvidia/nv-embed-v1 — same endpoint, same key.
  - Rate limiting: async asyncio.sleep between calls (--delay flag).
            NVIDIA free-tier allows ~5 RPM; default delay = 15s between calls.
            Patching _agenerate (not _generate) so the delay is non-blocking
            and RAGAS 0.2.x's asyncio executor can await it cleanly.
  - Retries: tenacity exponential back-off, up to 5 attempts, on 429/5xx.
  - RunConfig: timeout=600s, max_workers=1 (sequential, to honour rate limits).

Root cause of TimeoutError fix:
    RAGAS 0.2.x calls _agenerate (async). The old implementation patched
    _generate (sync) with time.sleep(), which blocked the event loop and
    caused asyncio's wait_for() to fire TimeoutError before any real work
    started. This version patches _agenerate with asyncio.sleep() so the
    event loop stays free and the 600s RunConfig timeout is respected.

One-line model swap:
    Change EVALUATOR_MODEL or EMBEDDING_MODEL below.
    Browse available models: https://build.nvidia.com/explore/discover
"""

import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration -- one-line swaps
# ---------------------------------------------------------------------------

# Must differ from qwen3:8b used for faithfulness verification.
# llama-3.1-70b-instruct is used because RAGAS metrics produce long JSON
# verdicts (faithfulness enumerates every claim). Smaller/lighter models
# (e.g. nemotron-lightning) hit their output limit mid-response and RAGAS
# throws LLMDidNotFinishException. llama-3.1-70b reliably finishes at 4096 tokens.
EVALUATOR_MODEL: str = "meta/llama-3.1-70b-instruct"
EMBEDDING_MODEL: str = "nvidia/nv-embed-v1"
NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

# Default minimum seconds between NVIDIA API calls (overridable via --delay).
DEFAULT_DELAY_SECONDS: float = 15.0


# ---------------------------------------------------------------------------
# Rate-limited, retry-wrapped async LLM call layer
# ---------------------------------------------------------------------------

def _make_rate_limited_llm(base_llm, delay_seconds: float):
    """
    Wrap a LangChain ChatOpenAI instance so every _agenerate() call:
      1. Awaits asyncio.sleep() for at least `delay_seconds` since last call.
         (Non-blocking — event loop stays free during the wait.)
      2. Retries on 429 / 5xx with exponential back-off (tenacity async).

    We patch _agenerate (not _generate) because RAGAS 0.2.x uses the async
    path exclusively. Patching the sync _generate with time.sleep() blocks
    the asyncio event loop, causing asyncio.wait_for() to fire TimeoutError
    before the HTTP call completes.

    Args:
        base_llm:      A ChatOpenAI instance pointed at NVIDIA NIM.
        delay_seconds: Minimum gap (seconds) between consecutive API calls.
                       Pass 0.0 to disable throttling (paid tier).

    Returns:
        The patched ChatOpenAI instance.
    """
    try:
        from tenacity import (                              # type: ignore
            retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_exception_type,
        )
        from openai import RateLimitError, APIStatusError   # type: ignore
    except ImportError as exc:
        raise ImportError(
            "tenacity and openai are required for the NVIDIA evaluator.\n"
            "Install with: pip install tenacity openai"
        ) from exc

    _last_call: list = [0.0]   # mutable container for cross-closure state

    original_agenerate = base_llm._agenerate

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=delay_seconds or 2, max=120),
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
    )
    async def _rate_limited_agenerate(messages, stop=None, run_manager=None, **kwargs):
        if delay_seconds > 0:
            now     = asyncio.get_event_loop().time()
            elapsed = now - _last_call[0]
            if elapsed < delay_seconds:
                sleep_for = delay_seconds - elapsed
                logger.debug(
                    "[nvidia] Rate-limit delay: awaiting %.1fs before next call.",
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)

        _last_call[0] = asyncio.get_event_loop().time()
        return await original_agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    base_llm._agenerate = _rate_limited_agenerate
    return base_llm


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_llm(delay_seconds: float = DEFAULT_DELAY_SECONDS):
    """
    Construct a RAGAS-compatible LLM wrapper using the NVIDIA NIM API.

    Includes:
      - Async-safe minimum delay between calls (asyncio.sleep, not time.sleep).
      - Exponential back-off retry (tenacity async) on 429 / 5xx, up to 5 attempts.

    Args:
        delay_seconds: Minimum seconds between consecutive API calls.
                       Default: 15.0 (safe for NVIDIA free tier ~4 RPM).
                       Pass 0.0 to disable throttling (paid tier).

    Returns:
        ragas.llms.LangchainLLMWrapper

    Raises:
        RuntimeError: if NVIDIA_API_KEY is not set.
        ImportError:  if ragas, langchain-openai, or tenacity are not installed.
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY environment variable is not set.\n"
            "Add it to your .env file or export it before running."
        )

    try:
        from langchain_openai import ChatOpenAI         # type: ignore
        from ragas.llms import LangchainLLMWrapper      # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies. Install with:\n"
            "  pip install ragas langchain-openai tenacity"
        ) from exc

    logger.info(
        "[nvidia] Initialising evaluator LLM: %s (delay=%.1fs between calls).",
        EVALUATOR_MODEL, delay_seconds,
    )

    langchain_llm = ChatOpenAI(
        model=EVALUATOR_MODEL,
        openai_api_key=api_key,
        openai_api_base=NVIDIA_BASE_URL,
        temperature=0.0,
        max_tokens=4096,
        max_retries=0,  # tenacity handles retries — disable OpenAI SDK retries
    )

    langchain_llm = _make_rate_limited_llm(langchain_llm, delay_seconds)

    return LangchainLLMWrapper(langchain_llm)


def build_embeddings():
    """
    Construct a RAGAS-compatible embeddings wrapper using nvidia/nv-embed-v1.

    Uses the same NVIDIA_API_KEY and NIM endpoint as the evaluator LLM.

    Returns:
        ragas.embeddings.LangchainEmbeddingsWrapper

    Raises:
        RuntimeError: if NVIDIA_API_KEY is not set.
        ImportError:  if ragas or langchain-openai are not installed.
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY environment variable is not set.\n"
            "Add it to your .env file or export it before running."
        )

    try:
        from langchain_openai import OpenAIEmbeddings             # type: ignore
        from ragas.embeddings import LangchainEmbeddingsWrapper   # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies. Install with:\n"
            "  pip install ragas langchain-openai"
        ) from exc

    logger.info(
        "[nvidia] Initialising embeddings: %s @ %s",
        EMBEDDING_MODEL, NVIDIA_BASE_URL,
    )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
        openai_api_base=NVIDIA_BASE_URL,
        check_embedding_ctx_length=False,
    )

    return LangchainEmbeddingsWrapper(embeddings)


def build_run_config():
    """
    Return a RAGAS RunConfig tuned for the NVIDIA API.

    Settings:
        timeout=600     10 min budget per job. With max_workers=1 and a
                        15s inter-call delay, RAGAS queues all jobs and the
                        timeout clock runs while jobs wait. Now that the
                        delay is async (asyncio.sleep), the event loop stays
                        free during waits and the 600s budget is not consumed
                        prematurely.
        max_retries=0   Retries handled by tenacity in build_llm(); RAGAS
                        retries would double-count and confuse back-off timing.
        max_workers=1   Sequential execution to honour per-minute rate limits
                        exactly (NVIDIA free tier ~5 RPM).

    Returns:
        ragas.run_config.RunConfig
    """
    from ragas.run_config import RunConfig  # type: ignore

    return RunConfig(
        timeout=600,
        max_retries=0,
        max_wait=0,
        max_workers=1,
    )


# Descriptive name for report metadata
BACKEND_NAME = "nvidia"
BACKEND_DESCRIPTION = f"NVIDIA NIM ({EVALUATOR_MODEL})"
