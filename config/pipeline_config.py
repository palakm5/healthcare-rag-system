"""
Pipeline configuration presets for the Healthcare RAG system.

Two named configs are provided:

- STANDARD_RAG_CONFIG: baseline RAG with all advanced features off.
  Used for comparison / ablation against the enhanced pipeline.

- RAG_PLUS_PLUS_CONFIG: all advanced features enabled -- hybrid search,
  cross-encoder reranking, metadata filtering, relevance threshold
  gating before generation, faithfulness check, and structured SQL
  retrieval alongside the existing vector search.

These configs are consumed by Generator.generate() and the CLI.
"""

# ──────────────────────────────────────────────────────────────────────────────
# PLACEHOLDER DEFAULT -- PENDING EMPIRICAL CALIBRATION
#
# The relevance_threshold_value (0.3) is a placeholder. It has NOT been
# calibrated against real query distributions. Use the
# eval/calibrate_threshold.py script to collect score distributions for
# answerable vs. unanswerable questions, then choose a threshold based on
# where the two distributions diverge.
#
# DO NOT treat this as a final value for production use.
# ──────────────────────────────────────────────────────────────────────────────
_PLACEHOLDER_THRESHOLD = 0.3

STANDARD_RAG_CONFIG = {
    "use_hybrid":                False,
    "use_rerank":                False,
    "metadata_filters":          None,
    "relevance_threshold":       False,
    "relevance_threshold_value": _PLACEHOLDER_THRESHOLD,
    "faithfulness_check":        False,
    # Structured SQL retrieval disabled for standard RAG baseline
    "structured_data_enabled":   False,
}

RAG_PLUS_PLUS_CONFIG = {
    "use_hybrid":                True,
    "use_rerank":                True,
    "metadata_filters":          None,
    "relevance_threshold":       True,
    "relevance_threshold_value": _PLACEHOLDER_THRESHOLD,
    "faithfulness_check":        True,
    # Structured SQL retrieval enabled for RAG++ -- runs alongside
    # vector search and is merged into the prompt as a separate evidence block.
    # Requires DATABASE_URL_READONLY to be set in .env and
    # python -m retrieval.structured.build_entity_cache to have been run.
    "structured_data_enabled":   True,
}
