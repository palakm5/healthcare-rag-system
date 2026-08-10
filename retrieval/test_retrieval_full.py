"""
Full retrieval test: exercises all retrieval features across all source collections.

Tests:
1. Basic retrieval (dense only, all sources)
2. Source-type filtering
3. Hybrid search (dense + BM25)
4. Cross-encoder reranking
5. Metadata filtering
6. Combined: hybrid + rerank + filter

Run with:
    python -m retrieval.test_retrieval_full
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from retrieval.search.retriever import Retriever


def print_results(label: str, results: list, max_chars: int = 200):
    """Pretty-print retrieval results."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    if not results:
        print("  (no results)")
        return
    for i, r in enumerate(results):
        print(f"\n  ── Rank {i+1} ──")
        print(f"  Score:        {r.get('score', 'N/A')}")
        if "dense_score" in r:
            print(f"  Dense:        {r['dense_score']}")
        if "sparse_score" in r:
            print(f"  Sparse:       {r['sparse_score']}")
        if "rerank_score" in r:
            print(f"  Rerank:       {r['rerank_score']}")
        if "retrieval_score" in r:
            print(f"  Retrieval:    {r['retrieval_score']}")
        meta = r.get("metadata", {})
        print(f"  Source:       {meta.get('source_type', '?')}")
        print(f"  Document:     {meta.get('document_title', meta.get('title', '?'))}")
        print(f"  Section:      {meta.get('section', '-')}")
        print(f"  Page:         {meta.get('page_number', '-')}")
        text = r.get("text", "")
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "..."
        print(f"  Text:         {text}")


def main():
    print("=" * 80)
    print("  RETRIEVAL FULL TEST SUITE")
    print("=" * 80)

    retriever = Retriever()

    queries = [
        "What are the symptoms of tuberculosis?",
        "What are the treatment guidelines for antimicrobial use?",
        "How is dengue fever managed clinically?",
    ]

    # ── Test 1: Basic dense retrieval (all sources) ──
    print("\n" + "─" * 80)
    print("  TEST 1: Basic dense retrieval (all sources)")
    print("─" * 80)
    for q in queries:
        results = retriever.retrieve(q, top_k=5)
        print_results(f"Query: {q}", results, max_chars=200)

    # ── Test 2: Source-type filtering ──
    print("\n" + "─" * 80)
    print("  TEST 2: Source-type filtering")
    print("─" * 80)
    for src_filter in [{"NHP"}, {"ICMR"}, {"PubMed", "MOHFW"}]:
        results = retriever.retrieve(
            queries[0], top_k=5, source_types=src_filter
        )
        print_results(f"Filter: {src_filter}", results, max_chars=200)

    # ── Test 3: Hybrid search ──
    print("\n" + "─" * 80)
    print("  TEST 3: Hybrid search (dense + BM25)")
    print("─" * 80)
    for q in queries[:2]:
        results = retriever.retrieve(q, top_k=5, use_hybrid=True)
        print_results(f"Hybrid: '{q}'", results, max_chars=200)

    # ── Test 4: Cross-encoder reranking ──
    print("\n" + "─" * 80)
    print("  TEST 4: Cross-encoder reranking")
    print("─" * 80)
    for q in queries[:2]:
        results = retriever.retrieve(q, top_k=5, use_rerank=True)
        print_results(f"Reranked: '{q}'", results, max_chars=200)

    # ── Test 5: Metadata filtering ──
    print("\n" + "─" * 80)
    print("  TEST 5: Metadata filtering")
    print("─" * 80)
    # Filter to only ICMR sources
    results = retriever.retrieve(
        queries[1], top_k=10,
        metadata_filters={"source_types": {"ICMR"}},
    )
    print_results("Filter: source_type=ICMR", results, max_chars=200)

    # Exclude NHP
    results = retriever.retrieve(
        queries[0], top_k=10,
        metadata_filters={"exclude_source_types": {"NHP"}},
    )
    print_results("Filter: exclude NHP", results, max_chars=200)

    # ── Test 6: Combined (hybrid + rerank + filter) ──
    print("\n" + "─" * 80)
    print("  TEST 6: Combined (hybrid + rerank + metadata filter)")
    print("─" * 80)
    results = retriever.retrieve(
        queries[1],
        top_k=5,
        use_hybrid=True,
        use_rerank=True,
        metadata_filters={"source_types": {"ICMR", "MOHFW"}},
    )
    print_results("Hybrid + Rerank + ICMR/MOHFW only", results, max_chars=200)

    # ── Test 7: Edge cases ──
    print("\n" + "─" * 80)
    print("  TEST 7: Edge cases")
    print("─" * 80)

    # Empty query
    results = retriever.retrieve("", top_k=3)
    print_results("Empty query", results, max_chars=200)

    # Very long query
    long_query = (
        "I am a medical professional looking for detailed information about "
        "the standard treatment protocols for tuberculosis in India, including "
        "first-line drug regimens, duration of therapy, and management of "
        "drug-resistant cases as per the latest ICMR and WHO guidelines."
    )
    results = retriever.retrieve(long_query, top_k=5)
    print_results("Long query", results, max_chars=200)

    # Non-existent source type
    results = retriever.retrieve(
        queries[0], top_k=5, source_types={"NONEXISTENT"}
    )
    print_results("Non-existent source filter", results, max_chars=200)

    print("\n" + "=" * 80)
    print("  ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()