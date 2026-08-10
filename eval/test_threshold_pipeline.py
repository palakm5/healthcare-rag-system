#!/usr/bin/env python3
"""
Smoke test for the relevance threshold + fallback mechanism.

Runs a small set of sample questions (mix of clearly answerable and
clearly out-of-scope) through the full pipeline with relevance_threshold
enabled, and prints the decision, top score, and final response for each.

This is a manual verification tool — run it and inspect the output to
confirm the mechanism works correctly before moving to the next guardrail.

Usage:
    python -m eval.test_threshold_pipeline
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Sample questions: mix of answerable (in-corpus) and out-of-scope
SAMPLE_QUESTIONS = [
    {
        "question": "What are the symptoms of tuberculosis?",
        "expected": "answerable",
    },
    {
        "question": "What are the ICMR guidelines for antimicrobial use?",
        "expected": "answerable",
    },
    {
        "question": "How is dengue fever managed clinically?",
        "expected": "answerable",
    },
    {
        "question": "What is the capital of France?",
        "expected": "unanswerable",
    },
    {
        "question": "Who won the 2024 Super Bowl?",
        "expected": "unanswerable",
    },
]


def main():
    from config.pipeline_config import RAG_PLUS_PLUS_CONFIG
    from generation.generator import Generator

    print("=" * 80)
    print("  RELEVANCE THRESHOLD SMOKE TEST")
    print("=" * 80)
    print(f"  Config: RAG_PLUS_PLUS_CONFIG")
    print(f"  Threshold: {RAG_PLUS_PLUS_CONFIG['relevance_threshold_value']}")
    print(f"  Questions: {len(SAMPLE_QUESTIONS)}")
    print("=" * 80)

    # Use dry-run mode (no LLM) — we only care about the threshold gate
    generator = Generator()

    for i, item in enumerate(SAMPLE_QUESTIONS):
        q = item["question"]
        expected = item["expected"]

        print(f"\n{'─' * 80}")
        print(f"  [{i+1}/{len(SAMPLE_QUESTIONS)}] Expected: {expected}")
        print(f"  Query: {q}")
        print(f"{'─' * 80}")

        result = generator.generate(q, config=RAG_PLUS_PLUS_CONFIG)

        fallback = result.get("fallback_triggered", False)
        decision = "insufficient_evidence" if fallback else "proceed"
        top_score = result.get("threshold_top_score", "N/A")
        score_gap = result.get("threshold_score_gap", "N/A")

        print(f"  Decision:        {decision}")
        print(f"  Top Score:       {top_score}")
        print(f"  Score Gap:       {score_gap}")
        print(f"  Fallback:        {fallback}")
        print(f"  Answer preview:  {result['answer'][:120]}...")

        # Quick sanity check
        if expected == "unanswerable" and not fallback:
            print(f"  ⚠️  WARNING: Expected unanswerable but got 'proceed'")
        elif expected == "answerable" and fallback:
            print(f"  ⚠️  WARNING: Expected answerable but got fallback")

    print("\n" + "=" * 80)
    print("  SMOKE TEST COMPLETE")
    print("  Review the output above. If answerable questions are being")
    print("  blocked or unanswerable questions are passing through, adjust")
    print("  the threshold value in config/pipeline_config.py and re-run.")
    print("=" * 80)


if __name__ == "__main__":
    main()