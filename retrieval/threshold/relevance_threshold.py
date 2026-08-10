"""
Relevance threshold checker for post-reranking quality gate.

After cross-encoder reranking, each candidate chunk carries a `rerank_score`
(and `score` is set to it). This module checks whether the top-ranked chunk
meets a minimum relevance threshold before allowing generation to proceed.

If the top score is below the threshold, the pipeline should skip the LLM
call entirely and return a fallback response — do NOT send weak context to
the LLM and hope it declines to answer.

Also computes a secondary signal: the score gap between the top-1 chunk and
the average of the top-5. A single high-scoring chunk surrounded by low
scores is weaker evidence than several chunks all scoring similarly well.
This gap is logged for later analysis but is NOT used as a primary decision
driver in this version.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# PLACEHOLDER DEFAULT — PENDING EMPIRICAL CALIBRATION
#
# This value (0.3) is a placeholder. It has NOT been calibrated against
# real query distributions. Use the eval/calibrate_threshold.py script to
# collect score distributions for answerable vs. unanswerable questions,
# then choose a threshold based on where the two distributions diverge.
#
# DO NOT treat this as a final value for production use.
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_RELEVANCE_THRESHOLD = 0.3

# Configurable fallback message. Change this constant to adjust wording
# across all fallback responses without hunting through multiple files.
FALLBACK_RESPONSE = (
    "I don't have sufficient evidence in the knowledge base to answer "
    "this question confidently. Please consult a healthcare professional "
    "or refer to primary medical literature."
)


class RelevanceThresholdChecker:
    """
    Check whether reranked candidates meet a minimum relevance threshold.

    Usage:
        checker = RelevanceThresholdChecker()
        result = checker.check(candidates, threshold=0.3)
        if result["decision"] == "proceed":
            # safe to call LLM
        else:
            # return fallback response
    """

    def __init__(self):
        pass

    def check(
        self,
        candidates: List[Dict],
        threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> Dict:
        """
        Evaluate the top-ranked candidate against the relevance threshold.

        Args:
            candidates: List of chunk dicts from reranking. Each must have
                        a "rerank_score" key (preferred) or "score" key.
            threshold: Minimum score for the top-1 chunk to proceed.

        Returns:
            dict with keys:
                - "decision": str — "proceed" or "insufficient_evidence"
                - "top_score": float — the top-1 relevance score
                - "score_gap_top1_vs_avg_top5": float — gap between top-1
                  and the mean of the top-5 scores (secondary signal)
                - "threshold_used": float — the threshold value applied
                - "num_candidates": int — how many candidates were checked
        """
        if not candidates:
            logger.warning("Threshold check: no candidates provided.")
            return {
                "decision": "insufficient_evidence",
                "top_score": 0.0,
                "score_gap_top1_vs_avg_top5": 0.0,
                "threshold_used": threshold,
                "num_candidates": 0,
            }

        # Extract scores — prefer rerank_score, fall back to score
        scores = []
        for c in candidates:
            s = c.get("rerank_score", c.get("score", 0.0))
            scores.append(float(s))

        top_score = scores[0]

        # Compute top-1 vs. average-of-top-5 gap (secondary signal)
        top_n = min(5, len(scores))
        avg_top_n = sum(scores[:top_n]) / top_n
        score_gap = round(top_score - avg_top_n, 4)

        # Primary decision: does top-1 meet the threshold?
        if top_score >= threshold:
            decision = "proceed"
        else:
            decision = "insufficient_evidence"

        logger.info(
            "Threshold check: decision=%s | top_score=%.4f | "
            "threshold=%.4f | gap(top1-avgTop%d)=%.4f | candidates=%d",
            decision,
            top_score,
            threshold,
            top_n,
            score_gap,
            len(candidates),
        )

        return {
            "decision": decision,
            "top_score": round(top_score, 4),
            "score_gap_top1_vs_avg_top5": score_gap,
            "threshold_used": threshold,
            "num_candidates": len(candidates),
        }