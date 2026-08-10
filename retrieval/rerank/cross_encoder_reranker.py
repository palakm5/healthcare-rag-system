"""
Cross-encoder reranker for retrieval results.

Uses a sentence-transformers cross-encoder model to rescore candidate
chunks against the query. Cross-encoders process (query, document) pairs
jointly, producing more accurate relevance scores than bi-encoder cosine
similarity alone.

This module is designed to be called by Retriever after initial retrieval.
"""

import logging
from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Default cross-encoder model — small, fast, good for medical text.
# Alternatives: "cross-encoder/ms-marco-MiniLM-L-6-v2" (faster, smaller)
#              "cross-encoder/ms-marco-MiniLM-L-12-v2" (larger, slower)
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Rerank a candidate list of chunks using a cross-encoder model.

    Usage:
        reranker = CrossEncoderReranker()
        candidates = [{"text": "...", "score": 0.8}, ...]
        reranked = reranker.rerank("query", candidates, top_k=5)
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model name.
                        Defaults to ms-marco-MiniLM-L-6-v2.
        """
        self.model_name = model_name or DEFAULT_CROSS_ENCODER_MODEL
        logger.info("Loading cross-encoder model: %s", self.model_name)
        self._model = CrossEncoder(self.model_name)
        logger.info("Cross-encoder model loaded successfully.")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Rerank candidate chunks using the cross-encoder.

        Args:
            query: The user's question.
            candidates: List of candidate dicts, each must have a "text" key.
                        Typically the output of Retriever.retrieve() or
                        HybridSearcher.search().
            top_k: Number of top results to return after reranking.
                   If None, returns all candidates in reranked order.

        Returns:
            List of dicts with the same keys as input, plus:
                - "rerank_score": float — cross-encoder relevance score
                - "score": updated to the rerank_score (original preserved
                  as "retrieval_score" if it existed)
        """
        if not candidates:
            return []

        # Build (query, document) pairs
        pairs = [(query, c["text"]) for c in candidates]

        # Cross-encoder predict — returns numpy array of scores
        scores = self._model.predict(
            pairs,
            show_progress_bar=False,
        ).tolist()

        # Attach scores
        reranked = []
        for i, candidate in enumerate(candidates):
            entry = dict(candidate)
            # Preserve original retrieval score
            if "score" in entry:
                entry["retrieval_score"] = entry["score"]
            entry["rerank_score"] = round(float(scores[i]), 4)
            entry["score"] = entry["rerank_score"]  # update primary score
            reranked.append(entry)

        # Sort by rerank score descending
        reranked.sort(key=lambda r: r["rerank_score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        logger.info(
            "Reranked %d candidates → %d results (model=%s)",
            len(candidates),
            len(reranked),
            self.model_name,
        )
        return reranked