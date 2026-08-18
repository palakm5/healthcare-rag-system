"""
OpenRouter RAGAS Evaluator
==========================

Builds the LLM and embeddings wrappers for running RAGAS metrics via the
OpenRouter API (OpenAI-compatible endpoint).

Design decisions:
  - LLM  : google/gemma-2-9b-it:free (free tier, no credits needed).
            Swap to any model at https://openrouter.ai/models?q=:free
            One-line change: edit EVALUATOR_MODEL below.
  - Embed : OpenRouter does NOT provide an embeddings endpoint.
            answer_relevancy (which needs embeddings) will be skipped
            automatically. Use the NVIDIA backend if you need it.
  - Rate limiting: async asyncio.sleep between calls (--delay flag).
            OpenRouter free tier varies by model; default delay = 3s.
            Uses asyncio.sleep (non-blocking) so RAGAS 0.2.x's async
            executor can await it cleanly without triggering TimeoutError.
  - Retries: tenacity exponential back-off, up to 5 attempts, on 429/5xx.
  - RunConfig: timeout=600s, max_workers=1.

OpenRouter-specific headers:
    HTTP-Referer and X-Title are optional but recommended by OpenRouter
    for monitoring your usage on their dashboard.
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration -- one-line swaps
# ---------------------------------------------------------------------------

# Browse free models: https://openrouter.ai/models?q=:free
EVALUATOR_MODEL: str = "nvidia/nemotron-3-super-120b-a12b:free"
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# Default minimum seconds between OpenRouter API calls (overridable via --delay).
# Free tier limits vary per model. 3s is safe for most free models.
DEFAULT_DELAY_SECONDS: float = 3.0


# ---------------------------------------------------------------------------
# Rate-limited, retry-wrapped async LLM call layer
# ---------------------------------------------------------------------------

def _make_rate_limited_llm(base_llm, delay_seconds: float):
    """
    Wrap a LangChain ChatOpenAI instance so every _agenerate() call:
      1. Awaits asyncio.sleep() for at least `delay_seconds` since last call.
         (Non-blocking -- event loop stays free during the wait.)
      2. Retries on 429 / 5xx with exponential back-off (tenacity async).

    Patches _agenerate (not _generate) so RAGAS 0.2.x's asyncio executor
    can await it cleanly without blocking the event loop.

    Args:
        base_llm:      A ChatOpenAI instance pointed at OpenRouter.
        delay_seconds: Minimum gap (seconds) between consecutive API calls.
                       Pass 0.0 to disable throttling.

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
            "tenacity and openai are required for the OpenRouter evaluator.\n"
            "Install with: pip install tenacity openai"
        ) from exc

    _last_call: list = [0.0]

    original_agenerate = base_llm._agenerate

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=max(delay_seconds, 2), max=60),
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
    )
    async def _rate_limited_agenerate(messages, stop=None, run_manager=None, **kwargs):
        if delay_seconds > 0:
            now     = asyncio.get_event_loop().time()
            elapsed = now - _last_call[0]
            if elapsed < delay_seconds:
                sleep_for = delay_seconds - elapsed
                logger.debug(
                    "[openrouter] Rate-limit delay: awaiting %.1fs before next call.",
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
    Construct a RAGAS-compatible LLM wrapper using the OpenRouter API.

    Includes:
      - Async-safe minimum delay between calls (asyncio.sleep, not time.sleep).
      - Exponential back-off retry (tenacity) on 429 / 5xx, up to 5 attempts.
      - OpenRouter-recommended headers (HTTP-Referer, X-Title).

    Args:
        delay_seconds: Minimum seconds between consecutive API calls.
                       Default: 3.0 (safe for most free-tier models).
                       Pass 0.0 to disable throttling.

    Returns:
        ragas.llms.LangchainLLMWrapper

    Raises:
        RuntimeError: if OPENROUTER_API_KEY is not set.
        ImportError:  if ragas, langchain-openai, or tenacity are not installed.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set.\n"
            "Add it to your .env file:  OPENROUTER_API_KEY=sk-or-..."
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
        "[openrouter] Initialising evaluator LLM: %s (delay=%.1fs between calls).",
        EVALUATOR_MODEL, delay_seconds,
    )

    langchain_llm = ChatOpenAI(
        model=EVALUATOR_MODEL,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.0,
        max_tokens=1024,
        max_retries=0,          # tenacity handles retries
        default_headers={
            "HTTP-Referer": "https://github.com/healthcare-rag-system",
            "X-Title":      "Healthcare RAG Evaluation",
        },
    )

    langchain_llm = _make_rate_limited_llm(langchain_llm, delay_seconds)
    return LangchainLLMWrapper(langchain_llm)


def build_embeddings():
    """
    OpenRouter does not provide an embeddings endpoint.

    Returns None -- answer_relevancy will be skipped automatically in
    run_ragas_evaluation() when embeddings are unavailable.

    If you need answer_relevancy, use the --evaluator nvidia backend which
    provides nvidia/nv-embed-v1 embeddings.

    Returns:
        None
    """
    logger.warning(
        "[openrouter] No embeddings endpoint available. "
        "answer_relevancy will be SKIPPED. "
        "Use --evaluator nvidia for full metric coverage."
    )
    return None


def build_run_config():
    """
    Return a RAGAS RunConfig tuned for OpenRouter.

    Settings:
        timeout=600     10 min budget per job. Generous to handle free-tier
                        queue times without firing prematurely.
        max_retries=0   Retries handled by tenacity in build_llm().
        max_workers=1   Sequential to honour per-model rate limits.

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
BACKEND_NAME = "openrouter"
BACKEND_DESCRIPTION = f"OpenRouter ({EVALUATOR_MODEL})"
