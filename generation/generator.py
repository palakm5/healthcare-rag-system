"""
RAG generator.

Takes a query, retrieves relevant chunks, builds a prompt, and calls an LLM.
Returns the answer with source citations.

Supports an optional pipeline config dict (see config/pipeline_config.py)
that controls hybrid search, reranking, metadata filtering, a
relevance-threshold quality gate before generation, faithfulness check,
and structured SQL database retrieval (additive alongside vector search).

Structured retrieval is ADDITIVE -- it never replaces or modifies existing
chunk retrieval, reranking, hybrid search, or faithfulness check logic.
"""

import logging
from typing import Callable, Dict, List, Optional

from generation.llm.ollama_client import OllamaClient
from generation.prompts.builder import build_prompt
from retrieval.search.retriever import Retriever

logger = logging.getLogger(__name__)


class Generator:
    """
    RAG generator that orchestrates retrieval and generation.

    Usage:
        # With Qwen client
        generator = Generator(llm_client=QwenClient(api_key="your-api-key"))
        result = generator.generate("What are the symptoms of tuberculosis?")

        # Dry-run mode (returns prompt + sources without calling an LLM)
        generator = Generator()
        result = generator.generate("...")

        # With pipeline config (advanced features)
        from config.pipeline_config import RAG_PLUS_PLUS_CONFIG
        result = generator.generate("...", config=RAG_PLUS_PLUS_CONFIG)
    """

    def __init__(self, llm_client: Optional[Callable] = None):
        """
        Initialize the generator.

        Args:
            llm_client: Optional callable that takes a prompt string and returns
                        the LLM's response. If None, runs in dry-run mode
                        (returns the prompt + sources without calling an LLM).
        """
        self._retriever = Retriever()
        self._llm_client = llm_client
        self._structured_lookup = None  # lazy-loaded when structured_data_enabled=True

    def _get_structured_lookup(self):
        """Lazy-load StructuredLookup to avoid import/connection cost when disabled."""
        if self._structured_lookup is None:
            try:
                from retrieval.structured.structured_lookup import StructuredLookup
                self._structured_lookup = StructuredLookup()
            except Exception as e:
                logger.warning(
                    "Could not initialise StructuredLookup: %s. "
                    "Structured retrieval disabled for this session.", e
                )
                self._structured_lookup = False  # sentinel: tried and failed
        return self._structured_lookup if self._structured_lookup is not False else None

    def generate(self, query: str, config: Optional[Dict] = None) -> Dict:
        """
        Generate an answer to the query.

        Args:
            query: The user's question.
            config: Optional pipeline config dict (e.g. STANDARD_RAG_CONFIG
                    or RAG_PLUS_PLUS_CONFIG). When provided, controls
                    hybrid search, reranking, metadata filtering, and the
                    relevance-threshold gate. When None, uses default
                    retrieval behavior (dense-only, no rerank, no threshold).

        Returns:
            dict with keys:
                - "answer": str — the generated answer (or prompt if dry-run)
                - "sources": list[dict] — the retrieved chunks used
                - "llm_used": bool — whether an LLM was actually called
                - "llm_metadata": dict — additional LLM metadata (if applicable)
                - "fallback_triggered": bool — True if threshold gate blocked
                  generation (only present when relevance_threshold is on)
                - "threshold_top_score": float — top-1 score (only when
                  fallback triggered)
                - "threshold_decision": str — "insufficient_evidence" (only
                  when fallback triggered)
                - "threshold_score_gap": float — score gap (only when
                  fallback triggered)
                - "threshold_used": float — threshold value applied (only
                  when fallback triggered)
        """
        # ── Step 1: Retrieve chunks ────────────────────────────────────
        if config:
            chunks = self._retriever.retrieve(
                query,
                use_hybrid=config.get("use_hybrid", False),
                use_rerank=config.get("use_rerank", False),
                metadata_filters=config.get("metadata_filters"),
            )
        else:
            chunks = self._retriever.retrieve(query)

        if not chunks:
            return {
                "answer": "No relevant information found in the knowledge base.",
                "sources": [],
                "llm_used": False,
                "llm_metadata": {},
            }

        # ── Step 1b: Structured SQL retrieval (additive, alongside vector search)
        structured_result = None
        if config and config.get("structured_data_enabled", False):
            sl = self._get_structured_lookup()
            if sl is not None:
                try:
                    structured_result = sl.lookup(query)
                    if structured_result.path != "no_match":
                        logger.info(
                            "Structured retrieval: path=%s, rows=%d, tables=%s",
                            structured_result.path,
                            structured_result.rows_returned,
                            structured_result.matched_tables,
                        )
                except Exception as e:
                    logger.warning("Structured lookup failed (non-fatal): %s", e)

        # ── Step 2: Relevance threshold check (pre-generation gate) ───────
        if config and config.get("relevance_threshold", False):
            from retrieval.threshold.relevance_threshold import (
                RelevanceThresholdChecker,
            )
            from retrieval.threshold.fallback import build_fallback_response

            threshold_value = config.get(
                "relevance_threshold_value", 0.3
            )
            checker = RelevanceThresholdChecker()
            threshold_result = checker.check(chunks, threshold=threshold_value)

            if threshold_result["decision"] == "insufficient_evidence":
                return build_fallback_response(
                    query=query,
                    top_score=threshold_result["top_score"],
                    decision=threshold_result["decision"],
                    score_gap=threshold_result["score_gap_top1_vs_avg_top5"],
                    threshold_used=threshold_result["threshold_used"],
                )

        # ── Step 3: Build prompt ──────────────────────────────────────────
        prompt = build_prompt(query, chunks, structured_result=structured_result)

        # ── Step 4: Call LLM or return prompt (dry-run) ───────────────────
        if self._llm_client:
            try:
                answer = self._llm_client.generate(prompt)

                llm_metadata = {
                    "model": getattr(self._llm_client, "model", "unknown"),
                }

                logger.info(f"Generated answer for query: '{query[:80]}...'")

                # ── Step 5: Faithfulness check (post-generation) ──────────
                if config and config.get("faithfulness_check", False):
                    from generation.faithfullness_check.verifier import (
                        FaithfulnessVerifier,
                        build_faithfulness_fallback_response,
                    )

                    verifier = FaithfulnessVerifier()
                    verdict = verifier.verify(answer, chunks)

                    # Log the full verdict for evaluation / spot-checking
                    logger.info(
                        "Faithfulness verdict: overall_faithful=%s | "
                        "verified=%s | claims=%d",
                        verdict.get("overall_faithful"),
                        verdict.get("verified"),
                        len(verdict.get("claims", [])),
                    )
                    for claim in verdict.get("claims", []):
                        logger.info(
                            "  Claim: '%s' | supported=%s | chunk=%s",
                            claim.get("claim", "")[:120],
                            claim.get("supported"),
                            claim.get("source_chunk_id"),
                        )

                    if not verdict.get("overall_faithful", False):
                        logger.warning(
                            "Faithfulness check FAILED for query: '%s...'",
                            query[:80],
                        )
                        return build_faithfulness_fallback_response(
                            answer=answer,
                            chunks=chunks,
                            verdict=verdict,
                        )

                    # Passed — return with verdict attached for traceability
                    result = {
                        "answer": answer,
                        "sources": chunks,
                        "llm_used": True,
                        "llm_metadata": llm_metadata,
                        "faithfulness_verdict": verdict,
                    }
                    if structured_result is not None:
                        result["structured_result"] = {
                            "path":          structured_result.path,
                            "template_id":   structured_result.template_id,
                            "tables":        structured_result.matched_tables,
                            "rows_returned": structured_result.rows_returned,
                            "provenance":    structured_result.provenance_label,
                        }
                    return result

                # ── No faithfulness check — return as-is ──────────────────
                result = {
                    "answer": answer,
                    "sources": chunks,
                    "llm_used": True,
                    "llm_metadata": llm_metadata,
                }
                if structured_result is not None:
                    result["structured_result"] = {
                        "path":          structured_result.path,
                        "template_id":   structured_result.template_id,
                        "tables":        structured_result.matched_tables,
                        "rows_returned": structured_result.rows_returned,
                        "provenance":    structured_result.provenance_label,
                    }
                return result

            except Exception as e:
                logger.error(f"LLM call failed: {e}")

                return {
                    "answer": f"Error calling LLM: {e}",
                    "sources": chunks,
                    "llm_used": False,
                    "llm_metadata": {},
                }
        else:
            logger.info(f"Dry-run mode: prompt generated for query: '{query[:80]}...'")
            return {
                "answer": prompt,
                "sources": chunks,
                "llm_used": False,
                "llm_metadata": {},
            }
