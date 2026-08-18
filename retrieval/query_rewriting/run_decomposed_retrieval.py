"""
Per-Sub-Question Retrieval Runner — Healthcare RAG System
==========================================================

For each sub-question produced by the decomposer, runs the full existing
retrieval pipeline (structured lookup + vector/hybrid search + reranking +
relevance threshold) unchanged, and collects the results.

Design principles:
  - Zero modification to retrieval logic — this module only loops over it.
  - One sub-question's fallback does NOT abort the loop; processing continues
    and that sub-question is marked with fallback_triggered=True.
  - Returns a list of SubQuestionResult objects, one per sub-question, in
    input order.
  - Caller (Generator) is responsible for prompt construction and generation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SubQuestionResult:
    """
    Retrieval result for a single sub-question.

    Attributes:
        sub_question:       The sub-question string.
        chunks:             Retrieved document/guideline chunks (empty if fallback).
        structured_result:  Structured DB lookup result, or None.
        fallback_triggered: True if relevance threshold fired for this sub-question.
        error:              Non-None if retrieval raised an unexpected exception.
    """
    sub_question:       str
    chunks:             List[Dict]       = field(default_factory=list)
    structured_result:  Optional[Any]   = None
    fallback_triggered: bool             = False
    error:              Optional[str]   = None


def run_retrieval_for_subquestion(
    sub_question: str,
    config: Dict,
    retriever=None,
    structured_lookup=None,
) -> SubQuestionResult:
    """
    Run the full retrieval pipeline for a single sub-question.

    Calls exactly the same retrieval steps as Generator.generate() does today
    for a single question — vector/hybrid search, reranking, relevance threshold
    — using the same config dict. Structured lookup uses the passed-in instance
    (shared across all sub-questions to avoid repeated DB connections).

    Args:
        sub_question:     A single sub-question string.
        config:           Pipeline config dict (same as passed to Generator.generate).
        retriever:        Retriever instance (shared, reused across sub-questions).
        structured_lookup: An open StructuredLookup instance, or None.

    Returns:
        SubQuestionResult
    """
    logger.info("[decomp] Retrieving for sub-question: %s...", sub_question[:80])

    # ── Vector / hybrid retrieval ─────────────────────────────────────────────
    use_hybrid       = config.get("use_hybrid", False)
    metadata_filters = config.get("metadata_filters", None)
    chunks: List[Dict] = []

    try:
        if retriever is not None:
            chunks = retriever.retrieve(
                sub_question,
                use_hybrid=use_hybrid,
                use_rerank=config.get("use_rerank", False),
                metadata_filters=metadata_filters,
            )
        elif use_hybrid:
            from retrieval.hybrid_search import hybrid_search  # type: ignore
            chunks = hybrid_search(sub_question, metadata_filter=metadata_filters)
        else:
            from retrieval.vector_search import vector_search  # type: ignore
            chunks = vector_search(sub_question, metadata_filter=metadata_filters)
    except Exception as e:
        logger.error(
            "[decomp] Retrieval failed for sub-question '%s': %s",
            sub_question[:60], e,
        )
        return SubQuestionResult(sub_question=sub_question, error=str(e))

    # ── Relevance threshold ───────────────────────────────────────────────────
    fallback_triggered = False
    if config.get("relevance_threshold", False) and chunks:
        try:
            from retrieval.threshold.relevance_threshold import RelevanceThresholdChecker
            threshold_value = config.get("relevance_threshold_value", 0.3)
            checker = RelevanceThresholdChecker()
            threshold_result = checker.check(chunks, threshold=threshold_value)
            if threshold_result["decision"] == "insufficient_evidence":
                fallback_triggered = True
                chunks = []
                logger.info(
                    "[decomp] Sub-question fallback triggered: '%s'",
                    sub_question[:60],
                )
        except Exception as e:
            logger.warning("[decomp] Threshold check failed (non-fatal): %s", e)

    # ── Structured lookup ─────────────────────────────────────────────────────
    structured_result = None
    if config.get("structured_data_enabled", False) and structured_lookup is not None:
        try:
            structured_result = structured_lookup.lookup(sub_question)
            if structured_result.path != "no_match":
                logger.info(
                    "[decomp] Structured: path=%s rows=%d tables=%s",
                    structured_result.path,
                    structured_result.rows_returned,
                    structured_result.matched_tables,
                )
            else:
                structured_result = None
        except Exception as e:
            logger.warning("[decomp] Structured lookup failed (non-fatal): %s", e)

    return SubQuestionResult(
        sub_question=sub_question,
        chunks=chunks,
        structured_result=structured_result,
        fallback_triggered=fallback_triggered,
    )


def run_decomposed_retrieval(
    sub_questions: List[str],
    config: Dict,
    retriever=None,
    structured_lookup=None,
) -> List[SubQuestionResult]:
    """
    Run retrieval for every sub-question and collect results.

    One sub-question's fallback does NOT abort the loop — all sub-questions
    are processed and the full results list is returned. The caller decides
    what to do with fallback-triggered sub-questions at prompt-construction time.

    Args:
        sub_questions:    Ordered list of sub-question strings (>= 1).
        config:           Pipeline config dict.
        retriever:        Retriever instance (shared across sub-questions).
        structured_lookup: Open StructuredLookup instance, or None.

    Returns:
        List[SubQuestionResult] in the same order as sub_questions.
    """
    results: List[SubQuestionResult] = []
    total = len(sub_questions)

    for idx, sq in enumerate(sub_questions, start=1):
        logger.info(
            "[decomp] Processing sub-question %d/%d: %s...",
            idx, total, sq[:70],
        )
        result = run_retrieval_for_subquestion(
            sub_question=sq,
            config=config,
            retriever=retriever,
            structured_lookup=structured_lookup,
        )
        results.append(result)

    n_fallback = sum(1 for r in results if r.fallback_triggered)
    n_error    = sum(1 for r in results if r.error)
    logger.info(
        "[decomp] Retrieval complete: %d sub-questions, %d fallback, %d error.",
        total, n_fallback, n_error,
    )

    return results
