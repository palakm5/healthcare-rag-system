"""
RAG generator.

Takes a query, retrieves relevant chunks, builds a prompt, and calls an LLM.
Returns the answer with source citations.

Supports an optional pipeline config dict (see config/pipeline_config.py)
that controls hybrid search, reranking, metadata filtering, a
relevance-threshold quality gate before generation, faithfulness check,
structured SQL database retrieval (additive alongside vector search), and
query decomposition (rewrite + sub-question splitting before retrieval).

Structured retrieval is ADDITIVE -- it never replaces or modifies existing
chunk retrieval, reranking, hybrid search, or faithfulness check logic.

Query decomposition is PRE-RETRIEVAL -- it splits a question into
sub-questions, runs the full existing retrieval pipeline once per
sub-question, then merges results for prompt construction. When
decomposition is disabled (STANDARD_RAG_CONFIG) the pipeline is identical
to the original single-question flow.
"""

import logging
from typing import Callable, Dict, List, Optional

from generation.llm.ollama_client import OllamaClient
from generation.prompts.builder import build_prompt, build_prompt_from_sub_results
from retrieval.search.retriever import Retriever

logger = logging.getLogger(__name__)


class Generator:
    """
    RAG generator that orchestrates retrieval and generation.

    Usage:
        generator = Generator(llm_client=OllamaClient())
        result = generator.generate("What are the symptoms of tuberculosis?")

        # With pipeline config (advanced features):
        from config.pipeline_config import RAG_PLUS_PLUS_CONFIG
        result = generator.generate("...", config=RAG_PLUS_PLUS_CONFIG)
    """

    def __init__(self, llm_client: Optional[Callable] = None):
        """
        Initialize the generator.

        Args:
            llm_client: Optional callable with a generate(prompt) -> str
                        interface. If None, runs in dry-run mode
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

    # ──────────────────────────────────────────────────────────────────────
    # Internal: single-question retrieval (original flow, used both directly
    # and as the per-sub-question building block in decomposed mode).
    # ──────────────────────────────────────────────────────────────────────

    def _retrieve_single(
        self,
        query: str,
        config: Dict,
        structured_lookup=None,
    ) -> Dict:
        """
        Run the full retrieval pipeline for a single query.

        Returns a dict with keys:
            chunks              -- retrieved doc chunks (may be empty on fallback)
            structured_result   -- StructuredResult or None
            fallback_triggered  -- bool
            fallback_response   -- pre-built fallback dict if triggered, else None
        """
        # ── Vector / hybrid retrieval ────────────────────────────────────────
        chunks = self._retriever.retrieve(
            query,
            use_hybrid=config.get("use_hybrid", False),
            use_rerank=config.get("use_rerank", False),
            metadata_filters=config.get("metadata_filters"),
        )

        # ── Structured SQL retrieval (additive) ──────────────────────────────
        structured_result = None
        if config.get("structured_data_enabled", False) and structured_lookup is not None:
            try:
                structured_result = structured_lookup.lookup(query)
                if structured_result.path != "no_match":
                    logger.info(
                        "Structured retrieval: path=%s, rows=%d, tables=%s",
                        structured_result.path,
                        structured_result.rows_returned,
                        structured_result.matched_tables,
                    )
                else:
                    structured_result = None
            except Exception as e:
                logger.warning("Structured lookup failed (non-fatal): %s", e)

        # ── Relevance threshold check ────────────────────────────────────────
        if config.get("relevance_threshold", False) and chunks:
            from retrieval.threshold.relevance_threshold import RelevanceThresholdChecker
            from retrieval.threshold.fallback import build_fallback_response

            threshold_value = config.get("relevance_threshold_value", 0.3)
            checker = RelevanceThresholdChecker()
            threshold_result = checker.check(chunks, threshold=threshold_value)

            if threshold_result["decision"] == "insufficient_evidence":
                fallback = build_fallback_response(
                    query=query,
                    top_score=threshold_result["top_score"],
                    decision=threshold_result["decision"],
                    score_gap=threshold_result["score_gap_top1_vs_avg_top5"],
                    threshold_used=threshold_result["threshold_used"],
                )
                return {
                    "chunks": [],
                    "structured_result": structured_result,
                    "fallback_triggered": True,
                    "fallback_response": fallback,
                }

        return {
            "chunks": chunks,
            "structured_result": structured_result,
            "fallback_triggered": False,
            "fallback_response": None,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Internal: decomposed retrieval path
    # ──────────────────────────────────────────────────────────────────────

    def _run_decomposed(
        self,
        query: str,
        config: Dict,
        structured_lookup=None,
    ) -> Optional[Dict]:
        """
        Run the query decomposition path.

        1. Heuristic pre-filter: if simple, skip decomposition entirely.
        2. LLM rewrite + decompose → sub-questions list.
        3. Per-sub-question retrieval loop (calls _retrieve_single each time).
        4. Build decomposed prompt.

        Returns a partial result dict (without answer/faithfulness) or None
        if decomposition should be aborted (shouldn't happen — always produces
        at least 1 sub-question).
        """
        from retrieval.query_rewriting.prefilter import prefilter_result
        from retrieval.query_rewriting.decompose import decompose_question
        from retrieval.query_rewriting.run_decomposed_retrieval import (
            run_decomposed_retrieval,
        )

        # Step 1: pre-filter
        pf = prefilter_result(query)
        logger.info(
            "[decomp] Pre-filter: skip=%s reason='%s' words=%d",
            pf["skip_decomposition"], pf["reason"], pf["word_count"],
        )

        if pf["skip_decomposition"]:
            sub_questions = [query]
            logger.info("[decomp] Pre-filter: passing question through unchanged.")
        else:
            # Step 2: LLM decompose
            sub_questions = decompose_question(query)

        # Step 3: per-sub-question retrieval
        sub_results = run_decomposed_retrieval(
            sub_questions=sub_questions,
            config=config,
            retriever=self._retriever,
            structured_lookup=structured_lookup,
        )

        # Step 4: build decomposed prompt
        prompt = build_prompt_from_sub_results(
            original_query=query,
            sub_results=sub_results,
        )

        # Flatten chunks and structured_result for logging/storage
        all_chunks: List[Dict] = []
        for r in sub_results:
            all_chunks.extend(r.chunks)

        any_fallback = any(r.fallback_triggered for r in sub_results)
        all_fallback = all(r.fallback_triggered for r in sub_results)

        return {
            "prompt":          prompt,
            "all_chunks":      all_chunks,
            "sub_results":     sub_results,
            "sub_questions":   sub_questions,
            "any_fallback":    any_fallback,
            "all_fallback":    all_fallback,
            "skipped_decomp":  pf["skip_decomposition"],
        }

    # ──────────────────────────────────────────────────────────────────────
    # Internal: per-sub-question faithfulness check
    # ──────────────────────────────────────────────────────────────────────

    def _check_faithfulness_decomposed(
        self,
        answer: str,
        sub_results,
        query: str,
    ) -> Dict:
        """
        Run faithfulness check per sub-question.

        For each sub-question that has chunks, verify the corresponding
        section of the answer against only that sub-question's chunks.
        Overall result is False if ANY sub-question fails.

        Because we can't cleanly split a single generated answer into
        per-sub-question parts, we check the full answer against each
        sub-question's chunks separately — this is strict (any sub-
        question's chunks not supporting the answer fails the whole thing)
        but consistent with the existing "strict" faithfulness policy.

        Returns a merged verdict dict compatible with the standard schema.
        """
        from generation.faithfullness_check.verifier import FaithfulnessVerifier

        verifier = FaithfulnessVerifier()
        all_claims: List[Dict] = []
        overall_faithful = True
        verified = True

        for r in sub_results:
            if r.fallback_triggered or not r.chunks:
                # No evidence for this sub-question — skip verification for it
                continue

            verdict = verifier.verify(answer, r.chunks)
            all_claims.extend(verdict.get("claims", []))

            if not verdict.get("overall_faithful", True):
                overall_faithful = False
                logger.warning(
                    "[decomp] Faithfulness check FAILED for sub-question: %s",
                    r.sub_question[:70],
                )
            if not verdict.get("verified", True):
                verified = False

        return {
            "claims":           all_claims,
            "overall_faithful": overall_faithful,
            "verified":         verified,
            "verifier_error":   None,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Public: main generate()
    # ──────────────────────────────────────────────────────────────────────

    def generate(self, query: str, config: Optional[Dict] = None) -> Dict:
        """
        Generate an answer to the query.

        When query_decomposition_enabled=True (RAG_PLUS_PLUS_CONFIG):
            1. Heuristic pre-filter (no LLM, instant).
            2. LLM rewrite + decompose (if not skipped).
            3. Per-sub-question retrieval (full existing pipeline × N).
            4. Decomposed prompt construction.
            5. Single LLM generation call.
            6. Per-sub-question faithfulness check (if enabled).

        When query_decomposition_enabled=False (STANDARD_RAG_CONFIG or None):
            Exactly the original single-question pipeline — no behaviour change.

        Args:
            query:  The user's question.
            config: Optional pipeline config dict.

        Returns:
            dict with keys: answer, sources, llm_used, llm_metadata,
            fallback_triggered (optional), faithfulness_verdict (optional),
            faithfulness_check_failed (optional), sub_questions (optional),
            structured_result (optional).
        """
        config = config or {}

        use_decomposition = config.get("query_rewriting_enabled", False)
        structured_lookup = (
            self._get_structured_lookup()
            if config.get("structured_data_enabled", False)
            else None
        )

        # ══════════════════════════════════════════════════════════════
        # DECOMPOSED PATH
        # ══════════════════════════════════════════════════════════════
        if use_decomposition:
            decomp = self._run_decomposed(query, config, structured_lookup)

            # If ALL sub-questions triggered fallback, return early
            if decomp["all_fallback"]:
                return {
                    "answer": (
                        "I could not find sufficient relevant information in the "
                        "knowledge base to answer this question confidently."
                    ),
                    "sources":              [],
                    "fallback_triggered":   True,
                    "sub_questions":        decomp["sub_questions"],
                    "faithfulness_verdict": None,
                    "faithfulness_check_failed": False,
                    "llm_used":             False,
                    "llm_metadata":         {},
                }

            prompt      = decomp["prompt"]
            all_chunks  = decomp["all_chunks"]
            sub_results = decomp["sub_results"]

            # ── Dry-run mode ─────────────────────────────────────────
            if not self._llm_client:
                return {
                    "answer":           prompt,
                    "sources":          all_chunks,
                    "llm_used":         False,
                    "llm_metadata":     {},
                    "sub_questions":    decomp["sub_questions"],
                    "fallback_triggered": decomp["any_fallback"],
                }

            # ── LLM generation (single call over merged prompt) ──────
            try:
                answer = self._llm_client.generate(prompt)
            except Exception as e:
                logger.error("LLM generation failed: %s", e)
                return {
                    "answer":        f"Generation failed: {e}",
                    "sources":       all_chunks,
                    "llm_used":      False,
                    "llm_metadata":  {},
                    "sub_questions": decomp["sub_questions"],
                }

            # ── Faithfulness check (per sub-question) ────────────────
            faithfulness_verdict      = None
            faithfulness_check_failed = False

            if config.get("faithfulness_check", False):
                try:
                    faithfulness_verdict = self._check_faithfulness_decomposed(
                        answer=answer,
                        sub_results=sub_results,
                        query=query,
                    )
                    if not faithfulness_verdict.get("overall_faithful", True):
                        faithfulness_check_failed = True
                        logger.warning(
                            "Faithfulness check FAILED for decomposed query: %s",
                            query[:80],
                        )
                except Exception as e:
                    logger.error("Faithfulness check raised exception: %s", e)
                    faithfulness_check_failed = True

            return {
                "answer":                    answer,
                "sources":                   all_chunks,
                "llm_used":                  True,
                "llm_metadata":              {"model": getattr(self._llm_client, "model", "unknown")},
                "sub_questions":             decomp["sub_questions"],
                "fallback_triggered":        decomp["any_fallback"],
                "faithfulness_verdict":      faithfulness_verdict,
                "faithfulness_check_failed": faithfulness_check_failed,
            }

        # ══════════════════════════════════════════════════════════════
        # SINGLE-QUESTION PATH (original, unchanged)
        # ══════════════════════════════════════════════════════════════

        # ── Step 1: Retrieve chunks ───────────────────────────────────
        chunks = self._retriever.retrieve(
            query,
            use_hybrid=config.get("use_hybrid", False),
            use_rerank=config.get("use_rerank", False),
            metadata_filters=config.get("metadata_filters"),
        )

        if not chunks:
            return {
                "answer":        "No relevant information found in the knowledge base.",
                "sources":       [],
                "llm_used":      False,
                "llm_metadata":  {},
            }

        # ── Step 1b: Structured SQL retrieval (additive) ─────────────
        structured_result = None
        if structured_lookup is not None:
            try:
                structured_result = structured_lookup.lookup(query)
                if structured_result.path != "no_match":
                    logger.info(
                        "Structured retrieval: path=%s, rows=%d, tables=%s",
                        structured_result.path,
                        structured_result.rows_returned,
                        structured_result.matched_tables,
                    )
                else:
                    structured_result = None
            except Exception as e:
                logger.warning("Structured lookup failed (non-fatal): %s", e)

        # ── Step 2: Relevance threshold check ─────────────────────────
        if config.get("relevance_threshold", False):
            from retrieval.threshold.relevance_threshold import RelevanceThresholdChecker
            from retrieval.threshold.fallback import build_fallback_response

            threshold_value = config.get("relevance_threshold_value", 0.3)
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

        # ── Step 3: Build prompt ──────────────────────────────────────
        prompt = build_prompt(query, chunks, structured_result=structured_result)

        # ── Step 4: Call LLM or return prompt (dry-run) ───────────────
        if not self._llm_client:
            logger.info("Dry-run mode: prompt generated for query: '%s...'", query[:80])
            return {
                "answer":       prompt,
                "sources":      chunks,
                "llm_used":     False,
                "llm_metadata": {},
            }

        try:
            answer = self._llm_client.generate(prompt)
            llm_metadata = {"model": getattr(self._llm_client, "model", "unknown")}
            logger.info("Generated answer for query: '%s...'", query[:80])
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return {
                "answer":       f"Error calling LLM: {e}",
                "sources":      chunks,
                "llm_used":     False,
                "llm_metadata": {},
            }

        # ── Step 5: Faithfulness check (post-generation) ─────────────
        faithfulness_verdict      = None
        faithfulness_check_failed = False

        if config.get("faithfulness_check", False):
            from generation.faithfullness_check.verifier import (
                FaithfulnessVerifier,
                build_faithfulness_fallback_response,
            )
            verifier = FaithfulnessVerifier()
            verdict  = verifier.verify(answer, chunks)

            logger.info(
                "Faithfulness verdict: overall_faithful=%s | verified=%s | claims=%d",
                verdict.get("overall_faithful"),
                verdict.get("verified"),
                len(verdict.get("claims", [])),
            )

            if not verdict.get("overall_faithful", False):
                logger.warning(
                    "Faithfulness check FAILED for query: '%s...'", query[:80]
                )
                return build_faithfulness_fallback_response(
                    answer=answer, chunks=chunks, verdict=verdict
                )

            faithfulness_verdict = verdict

        result: Dict = {
            "answer":                    answer,
            "sources":                   chunks,
            "llm_used":                  True,
            "llm_metadata":              llm_metadata,
            "faithfulness_verdict":      faithfulness_verdict,
            "faithfulness_check_failed": faithfulness_check_failed,
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
