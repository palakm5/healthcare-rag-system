#!/usr/bin/env python3
"""
Calibration script for the relevance threshold.

Reads a labeled question set (answerable vs. unanswerable), runs each
question through retrieval + reranking, and logs the top-1 reranked score
for every question. Outputs a simple report with score distributions
(min/max/mean/median) for each category so you can visually inspect the
separation and manually choose a threshold value.

Usage:
    python -m eval.calibrate_threshold --questions eval/labeled_questions.json

Expected JSON format:
    [
        {"question": "...", "category": "answerable"},
        {"question": "...", "category": "unanswerable"}
    ]

This script does NOT auto-select a threshold — it only provides the data
you need to make that decision.
"""

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_questions(path: str) -> List[Dict]:
    """Load the labeled question set from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")

    for i, item in enumerate(data):
        if "question" not in item or "category" not in item:
            raise ValueError(
                f"Item {i} missing required keys 'question' and/or 'category'"
            )
        if item["category"] not in ("answerable", "unanswerable"):
            raise ValueError(
                f"Item {i} has invalid category '{item['category']}'. "
                "Expected 'answerable' or 'unanswerable'."
            )

    return data


def compute_stats(scores: List[float]) -> Dict:
    """Compute min, max, mean, median for a list of scores."""
    if not scores:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}

    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    if n % 2 == 1:
        median = sorted_scores[n // 2]
    else:
        median = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2

    return {
        "count": n,
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "mean": round(sum(scores) / n, 4),
        "median": round(median, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate relevance threshold from labeled questions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="eval/labeled_questions.json",
        help="Path to the labeled question set JSON file.",
    )
    args = parser.parse_args()

    # ── Load questions ──────────────────────────────────────────────
    questions_path = Path(args.questions)
    if not questions_path.exists():
        logger.error("Question file not found: %s", questions_path)
        sys.exit(1)

    questions = load_questions(str(questions_path))
    logger.info(
        "Loaded %d questions from %s", len(questions), questions_path
    )

    # ── Initialize retriever (lazy-loads reranker on first use) ─────
    from retrieval.search.retriever import Retriever

    retriever = Retriever()

    # ── Run each question through retrieval + reranking ─────────────
    answerable_scores = []
    unanswerable_scores = []
    per_question_results = []

    for i, item in enumerate(questions):
        q = item["question"]
        category = item["category"]
        logger.info(
            "[%d/%d] Processing %s: '%s...'",
            i + 1,
            len(questions),
            category,
            q[:80],
        )

        # Use hybrid + rerank to match RAG++ retrieval path
        results = retriever.retrieve(
            q, top_k=5, use_hybrid=True, use_rerank=True
        )

        if not results:
            top_score = 0.0
        else:
            # Prefer rerank_score, fall back to score
            top_score = results[0].get("rerank_score", results[0].get("score", 0.0))

        per_question_results.append({
            "question": q,
            "category": category,
            "top_score": round(float(top_score), 4),
        })

        if category == "answerable":
            answerable_scores.append(float(top_score))
        else:
            unanswerable_scores.append(float(top_score))

    # ── Print per-question results ──────────────────────────────────
    print("\n" + "=" * 80)
    print("  PER-QUESTION TOP-1 RERANK SCORES")
    print("=" * 80)
    for r in per_question_results:
        print(f"  [{r['category']:>12}] score={r['top_score']:.4f}  |  {r['question'][:100]}")

    # ── Print distribution report ───────────────────────────────────
    answerable_stats = compute_stats(answerable_scores)
    unanswerable_stats = compute_stats(unanswerable_scores)

    print("\n" + "=" * 80)
    print("  SCORE DISTRIBUTION REPORT")
    print("=" * 80)

    print(f"\n  {'Category':<15} {'Count':>6} {'Min':>8} {'Max':>8} {'Mean':>8} {'Median':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for label, stats in [
        ("answerable", answerable_stats),
        ("unanswerable", unanswerable_stats),
    ]:
        print(
            f"  {label:<15} "
            f"{stats['count']:>6} "
            f"{stats['min'] or 'N/A':>8} "
            f"{stats['max'] or 'N/A':>8} "
            f"{stats['mean'] or 'N/A':>8} "
            f"{stats['median'] or 'N/A':>8}"
        )

    # ── Raw score lists for copy-paste analysis ─────────────────────
    print(f"\n  Answerable scores:   {[round(s, 4) for s in sorted(answerable_scores)]}")
    print(f"  Unanswerable scores: {[round(s, 4) for s in sorted(unanswerable_scores)]}")

    print("\n" + "=" * 80)
    print(
        "  Use the distributions above to manually choose a threshold.\n"
        "  Look for the point where the two distributions diverge —\n"
        "  typically between the max unanswerable score and the min\n"
        "  answerable score. Update DEFAULT_RELEVANCE_THRESHOLD in\n"
        "  retrieval/threshold/relevance_threshold.py and\n"
        "  config/pipeline_config.py with your chosen value."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()