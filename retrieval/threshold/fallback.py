"""
Fallback response builder.

When the relevance threshold check returns "insufficient_evidence",
this module builds a structured response object that matches the normal
Generator output schema, so downstream consumers (logging, evaluation,
CLI display) can handle it uniformly.

The fallback message is sourced from the single constant in
relevance_threshold.py — change it there, not here.
"""

import logging
from typing import Dict

from retrieval.threshold.relevance_threshold import FALLBACK_RESPONSE

logger = logging.getLogger(__name__)


def build_fallback_response(
    query: str,
    top_score: float,
    decision: str,
    score_gap: float,
    threshold_used: float,
) -> Dict:
    """
    Build a fallback response object matching the normal Generator schema.

    Args:
        query: The user's question (for logging context).
        top_score: The top-1 relevance score that triggered the fallback.
        decision: The threshold decision string ("insufficient_evidence").
        score_gap: The top-1 vs. avg-top-5 score gap.
        threshold_used: The threshold value that was applied.

    Returns:
        dict with keys:
            - "answer": str — the fallback message
            - "sources": list — empty (no chunks passed to LLM)
            - "llm_used": bool — False (LLM was not called)
            - "llm_metadata": dict — empty
            - "fallback_triggered": bool — True
            - "threshold_top_score": float — the top score that caused it
            - "threshold_decision": str — "insufficient_evidence"
            - "threshold_score_gap": float — the score gap
            - "threshold_used": float — the threshold value applied
    """
    logger.info(
        "Fallback triggered for query: '%s...' | top_score=%.4f | "
        "threshold=%.4f | gap=%.4f",
        query[:80],
        top_score,
        threshold_used,
        score_gap,
    )

    return {
        "answer": FALLBACK_RESPONSE,
        "sources": [],
        "llm_used": False,
        "llm_metadata": {},
        "fallback_triggered": True,
        "threshold_top_score": top_score,
        "threshold_decision": decision,
        "threshold_score_gap": score_gap,
        "threshold_used": threshold_used,
    }