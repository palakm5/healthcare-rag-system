"""
LLM Query Rewriter + Decomposer — Healthcare RAG System
=======================================================

Part of the Query Rewriting Layer. Handles the LLM step:
    1. Rewrites the question for clarity, completeness and search quality
       (resolves abbreviations, makes implicit context explicit).
    2. If the rewritten question is complex/compound, splits it into
       self-contained sub-questions. Otherwise returns it as a 1-item list.

This module is only invoked when the heuristic pre-filter (prefilter.py)
determines the question needs rewriting. Simple/short questions are passed
through unchanged by the pre-filter with no LLM call at all.

Design decisions:
  - One combined prompt: rewrite + decompose in a single call — avoids two
    sequential LLM calls and keeps latency low.
  - Model: DECOMPOSITION_MODEL (default qwen3:8b via Ollama).
  - Output: strict JSON {"sub_questions": ["...", ...]} with one retry on
    parse failure, then hard fallback to original question as a 1-item list.
  - Cap: at most MAX_SUB_QUESTIONS sub-questions — truncate + warn if exceeded.
  - Logging: every input/output pair appended to QUERY_REWRITING_LOG_PATH as
    JSONL for manual quality review (no automatic validation).
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DECOMPOSITION_MODEL: str = os.getenv("DECOMPOSITION_MODEL", "mistral:7b")
OLLAMA_BASE_URL: str     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Hard cap on sub-questions — prevents unbounded fan-out.
MAX_SUB_QUESTIONS: int = 4

# JSONL log file for manual quality review of query rewriting input/output pairs.
_PROJECT_ROOT        = Path(__file__).resolve().parent.parent.parent
QUERY_REWRITING_LOG_PATH: Path = _PROJECT_ROOT / "logs" / "query_rewriting_log.jsonl"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """\
You are a medical query analyst. Your job is to prepare a question for a medical RAG system.

Given the user question below, do the following in a single response:

1. REWRITE: Make the question clear, complete, and search-friendly.
   - Expand abbreviations (e.g. "T2DM" → "Type 2 diabetes mellitus").
   - Make implicit context explicit (e.g. "its side effects" → "side effects of metformin").
   - Fix grammar and ambiguity.

2. DECOMPOSE: If the rewritten question contains multiple distinct, independently-answerable
   sub-questions, split it into a list of self-contained sub-questions.
   Each sub-question must make complete sense on its own.
   If it is a single atomic question, return it as a one-item list.

Return ONLY valid JSON in this exact format — no other text, no markdown:
{{"sub_questions": ["sub-question 1", "sub-question 2"]}}

IMPORTANT:
- Maximum {max_sq} sub-questions. If more are needed, include only the most important {max_sq}.
- Every sub-question must be self-contained (no pronouns referring to other sub-questions).
- Use Indian healthcare context where relevant (Indian drug names, ICMR guidelines, etc.).

User question: {question}"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """Make a single Ollama chat completion call. Returns raw response text."""
    from openai import OpenAI  # type: ignore
    client = OpenAI(
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",
    )
    response = client.chat.completions.create(
        model=DECOMPOSITION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


def _parse_sub_questions(raw: str) -> List[str]:
    """
    Extract sub_questions list from raw LLM output.

    Tries:
      1. Direct json.loads on the full response.
      2. Regex extraction of first {...} block (handles model preamble/suffix).

    Returns empty list on all failures (caller handles fallback).
    """
    # Strip <think>...</think> blocks that qwen3 sometimes emits
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Attempt 1: direct parse
    try:
        data = json.loads(raw)
        sqs = data.get("sub_questions", [])
        if isinstance(sqs, list) and all(isinstance(s, str) for s in sqs):
            return [s.strip() for s in sqs if s.strip()]
    except (json.JSONDecodeError, AttributeError):
        pass

    # Attempt 2: extract first {...} block
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            sqs = data.get("sub_questions", [])
            if isinstance(sqs, list) and all(isinstance(s, str) for s in sqs):
                return [s.strip() for s in sqs if s.strip()]
        except (json.JSONDecodeError, AttributeError):
            pass

    return []


def _log_rewriting(entry: dict) -> None:
    """Append a query rewriting log entry to QUERY_REWRITING_LOG_PATH as JSONL."""
    try:
        QUERY_REWRITING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(QUERY_REWRITING_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Could not write query rewriting log: %s", e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decompose_question(question: str) -> List[str]:
    """
    Rewrite and decompose a question into sub-questions via a single LLM call.

    Flow:
        1. Build the combined rewrite+decompose prompt.
        2. Call Ollama once.
        3. Parse the JSON response.
        4. On parse failure: retry once.
        5. On second failure: fall back to [original_question] (never crash).
        6. Cap at MAX_SUB_QUESTIONS, log a warning if truncated.
        7. Append input/output to decomposition_log.jsonl.

    Args:
        question: The original user question (pre-filter already decided not to skip).

    Returns:
        List[str]: One or more self-contained sub-questions. Always has at least
                   one item (the original question in the worst-case fallback).
    """
    prompt = _DECOMPOSE_PROMPT.format(
        question=question,
        max_sq=MAX_SUB_QUESTIONS,
    )

    log_entry: dict = {
        "ts":            time.strftime("%Y-%m-%dT%H:%M:%S"),
        "original":      question,
        "model":         DECOMPOSITION_MODEL,
        "sub_questions": None,
        "fallback_used": False,
        "truncated":     False,
        "raw_response":  None,
        "error":         None,
    }

    sub_questions: List[str] = []

    for attempt in range(1, 3):   # max 2 attempts
        try:
            raw = _call_llm(prompt)
            log_entry["raw_response"] = raw
            sub_questions = _parse_sub_questions(raw)
            if sub_questions:
                break
            logger.warning(
                "[decompose] Attempt %d: could not parse sub_questions from response.",
                attempt,
            )
        except Exception as exc:
            log_entry["error"] = str(exc)
            logger.error("[decompose] LLM call failed (attempt %d): %s", attempt, exc)

    # Fallback: use original question unchanged
    if not sub_questions:
        logger.warning(
            "[decompose] Both attempts failed — falling back to original question: %s",
            question[:80],
        )
        sub_questions = [question]
        log_entry["fallback_used"] = True

    # Cap at MAX_SUB_QUESTIONS
    if len(sub_questions) > MAX_SUB_QUESTIONS:
        logger.warning(
            "[decompose] LLM returned %d sub-questions; truncating to %d.",
            len(sub_questions), MAX_SUB_QUESTIONS,
        )
        sub_questions = sub_questions[:MAX_SUB_QUESTIONS]
        log_entry["truncated"] = True

    log_entry["sub_questions"] = sub_questions
    _log_rewriting(log_entry)

    logger.info(
        "[decompose] '%s...' → %d sub-question(s): %s",
        question[:60],
        len(sub_questions),
        sub_questions,
    )

    return sub_questions
