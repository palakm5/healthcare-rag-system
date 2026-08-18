"""
Heuristic Pre-filter — Query Decomposition Layer
=================================================

Runs BEFORE any LLM call. Determines whether a question is simple enough
to bypass the rewrite+decompose step entirely.

Decision rule (all must hold to SKIP decomposition):
    1. Word count <= SIMPLE_WORD_THRESHOLD
    2. No compound-question indicators found (COMPOUND_INDICATORS)
    3. At most one question mark

This is intentionally conservative: when in doubt it lets the question
through to the LLM decomposer. False-positives (sending a simple question
to the decomposer) are cheap; false-negatives (skipping decomposition for
a genuinely compound question) hurt retrieval quality.

All constants are configurable at the top of this file.
"""

import re
from typing import List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Questions with more words than this are never skipped (always decomposed).
SIMPLE_WORD_THRESHOLD: int = 15

# If any of these tokens appear in the question, it is treated as compound.
# Case-insensitive, whole-word matching.
COMPOUND_INDICATORS: List[str] = [
    "and",
    "also",
    "as well as",
    "in addition",
    "additionally",
    "furthermore",
    "moreover",
    "along with",
    "both",
    "compare",
    "versus",
    "vs",
    "difference between",
    "similarities between",
]

# Questions with more than this many '?' characters are treated as compound.
MAX_QUESTION_MARKS: int = 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def should_skip_decomposition(question: str) -> bool:
    """
    Return True if the question is simple enough to bypass decomposition.

    Checks (all must hold to return True / skip):
        1. Word count is at or below SIMPLE_WORD_THRESHOLD.
        2. No compound-question indicator found.
        3. At most MAX_QUESTION_MARKS '?' characters.

    Args:
        question: The raw user question string.

    Returns:
        True  -> skip decomposition, pass question through unchanged.
        False -> proceed to LLM rewrite + decompose.

    This function is deterministic and makes no network calls.
    """
    q = question.strip()

    # 1. Word count check
    word_count = len(q.split())
    if word_count > SIMPLE_WORD_THRESHOLD:
        return False

    # 2. Compound-indicator check (case-insensitive, whole-phrase)
    q_lower = q.lower()
    for indicator in COMPOUND_INDICATORS:
        # Use word-boundary regex for single words; substring for multi-word phrases
        if " " in indicator:
            if indicator in q_lower:
                return False
        else:
            if re.search(rf"\b{re.escape(indicator)}\b", q_lower):
                return False

    # 3. Multiple question marks
    if q.count("?") > MAX_QUESTION_MARKS:
        return False

    return True


def prefilter_result(question: str) -> dict:
    """
    Run the pre-filter and return a result dict for logging/debugging.

    Returns:
        {
            "question":           original question,
            "skip_decomposition": bool,
            "reason":             human-readable reason string,
            "word_count":         int,
        }
    """
    q = question.strip()
    word_count = len(q.split())
    q_lower = q.lower()

    if word_count > SIMPLE_WORD_THRESHOLD:
        return {
            "question":           q,
            "skip_decomposition": False,
            "reason":             f"word_count={word_count} > threshold={SIMPLE_WORD_THRESHOLD}",
            "word_count":         word_count,
        }

    for indicator in COMPOUND_INDICATORS:
        if " " in indicator:
            found = indicator in q_lower
        else:
            found = bool(re.search(rf"\b{re.escape(indicator)}\b", q_lower))
        if found:
            return {
                "question":           q,
                "skip_decomposition": False,
                "reason":             f"compound indicator found: '{indicator}'",
                "word_count":         word_count,
            }

    if q.count("?") > MAX_QUESTION_MARKS:
        return {
            "question":           q,
            "skip_decomposition": False,
            "reason":             f"multiple question marks ({q.count('?')})",
            "word_count":         word_count,
        }

    return {
        "question":           q,
        "skip_decomposition": True,
        "reason":             "simple question (short, no compound indicators, single ?)",
        "word_count":         word_count,
    }
