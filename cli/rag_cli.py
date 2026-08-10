#!/usr/bin/env python3
"""
Command-line interface for the Healthcare RAG System.

This script provides a simple CLI to interact with the RAG system,
allowing users to input queries and get answers from the system.
"""

import argparse
import logging
import sys

from config.pipeline_config import STANDARD_RAG_CONFIG, RAG_PLUS_PLUS_CONFIG
from generation.generator import Generator
from generation.llm.ollama_client import OllamaClient


def setup_logging():
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def main():
    """Main CLI entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Healthcare RAG System CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Ollama model to use. Defaults to medgemma1.5:4b, falling back "
            "to medgemma:4b if 1.5 is unavailable. To override, pass a "
            "pulled model name (e.g. medgemma1.5:4b). "
            "⚠️  Do NOT use the -q4_0 quantized tag — it has known "
            "overfitting issues."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (retrieval + prompt only, no LLM call)",
    )

    parser.add_argument(
        "--config",
        type=str,
        choices=["standard", "rag++"],
        default=None,
        help=(
            "Pipeline config preset to use. 'standard' = baseline RAG "
            "(dense-only, no rerank, no threshold). 'rag++' = all advanced "
            "features (hybrid, rerank, relevance threshold). "
            "If not provided, uses default retrieval behavior."
        ),
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query to process (if not provided, interactive mode starts)",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Resolve pipeline config
    # ---------------------------------------------------------
    if args.config == "standard":
        config = STANDARD_RAG_CONFIG
    elif args.config == "rag++":
        config = RAG_PLUS_PLUS_CONFIG
    else:
        config = None

    # ---------------------------------------------------------
    # Initialize Generator
    # ---------------------------------------------------------
    if args.dry_run:
        generator = Generator()

    else:
        if args.model is not None:
            client = OllamaClient(model=args.model)
        else:
            client = OllamaClient()   # Auto-selects medgemma1.5:4b or fallback

        generator = Generator(llm_client=client)

    # ---------------------------------------------------------
    # Get user query
    # ---------------------------------------------------------
    query = args.query

    if not query:
        query = input("Enter your medical question: ").strip()

        if not query:
            print("Error: No query provided.")
            sys.exit(1)

    # ---------------------------------------------------------
    # Generate answer
    # ---------------------------------------------------------
    result = generator.generate(query, config=config)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["answer"])

    # ── Fallback info ──────────────────────────────────────────
    if result.get("fallback_triggered"):
        print("\n" + "=" * 80)
        print("FALLBACK INFO")
        print("=" * 80)
        print(f"  Decision:       {result.get('threshold_decision', 'N/A')}")
        print(f"  Top Score:      {result.get('threshold_top_score', 'N/A')}")
        print(f"  Score Gap:      {result.get('threshold_score_gap', 'N/A')}")
        print(f"  Threshold Used: {result.get('threshold_used', 'N/A')}")

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for i, source in enumerate(result["sources"], start=1):
        metadata = source["metadata"]

        print(
            f"{i}. {metadata.get('source_type', 'Unknown')} | "
            f"{metadata.get('title', 'Unknown')}"
        )

        if metadata.get("section"):
            print(f"   Section: {metadata['section']}")

        print(f"   Similarity Score: {source['score']:.4f}")

    if result["llm_used"]:
        print("\n" + "=" * 80)
        print("LLM METADATA")
        print("=" * 80)
        print(result["llm_metadata"])


if __name__ == "__main__":
    main()