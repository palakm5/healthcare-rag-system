"""
Prompt Builder — Healthcare RAG System
=======================================

Constructs the system + user prompt sent to the LLM for answer generation.

Supports two calling modes:
  1. Single-question mode (original behaviour, unchanged):
         build_prompt(query, chunks, structured_result=...)
     Formats evidence as a flat labelled block, identical to the original.

  2. Decomposed mode (new, for query decomposition layer):
         build_prompt_from_sub_results(query, sub_results)
     Accepts a list of SubQuestionResult objects and formats each as a
     clearly separated, labelled evidence block so the LLM can answer
     each part distinctly.

Design principles:
  - Source labelling: each evidence block is labelled by source type (guideline
    chunk vs. structured DB row) so the LLM can cite appropriately.
  - Token budget: structured rows are rendered compactly; guideline chunks include
    their full text but are capped (configurable) to prevent context overflow.
  - Safety: the prompt always instructs the LLM to stay within retrieved evidence
    and clearly acknowledge when evidence is absent.
  - Backward compatibility: build_prompt() signature is unchanged; all existing
    call sites continue to work without modification.
"""

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.decomposition.run_decomposed_retrieval import SubQuestionResult

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_CHUNKS      = 5          # max guideline chunks included in prompt
MAX_CHUNK_CHARS = 800        # max characters per chunk (truncated if longer)
MAX_STRUCT_ROWS = 10         # max structured DB rows included

# ── System prompt (single-question mode) ─────────────────────────────────────
SYSTEM_PROMPT = """You are a medical information assistant specialising in Indian healthcare guidelines.

Your task: answer the user's medical question using ONLY the evidence provided below.

Rules:
- Cite the source label (e.g. [Guideline 1], [DB Row 1]) when referencing evidence.
- If the evidence is insufficient or absent, say so clearly — do NOT guess or hallucinate.
- Keep answers concise, factual, and relevant to the Indian healthcare context.
- Do not provide general medical advice beyond what the evidence explicitly states."""

# ── System prompt (decomposed mode) ──────────────────────────────────────────
SYSTEM_PROMPT_DECOMPOSED = """You are a medical information assistant specialising in Indian healthcare guidelines.

Your task: answer each part of the user's question using ONLY the evidence provided below.
The evidence is organised by sub-question — use only the relevant section for each part.

Rules:
- Answer each sub-question separately and clearly.
- Cite the source label (e.g. [Guideline 1], [DB Row 1]) when referencing evidence.
- If evidence was insufficient for any part, say so explicitly for that part — do NOT guess.
- Keep answers concise, factual, and relevant to the Indian healthcare context.
- Do not provide general medical advice beyond what the evidence explicitly states."""


# ── Shared rendering helpers ──────────────────────────────────────────────────

def _render_structured_rows(structured_result, prefix: str = "") -> str:
    """Render structured DB rows as a compact evidence block."""
    if structured_result is None or not structured_result.rows:
        return ""
    label = f"{prefix}Structured Database Evidence [{structured_result.provenance_label}]"
    lines = [f"\n--- {label} ---"]
    for i, row in enumerate(structured_result.rows[:MAX_STRUCT_ROWS], start=1):
        row_text = ", ".join(f"{k}: {v}" for k, v in row.items() if v is not None)
        lines.append(f"[DB Row {i}] {row_text}")
    lines.append("---")
    return "\n".join(lines)


def _render_chunks(chunks: List[Dict], start_index: int = 1) -> str:
    """Render guideline/document chunks as labelled evidence blocks."""
    if not chunks:
        return ""
    lines = ["\n--- Guideline / Document Evidence ---"]
    for i, chunk in enumerate(chunks[:MAX_CHUNKS], start=start_index):
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        text = text[:MAX_CHUNK_CHARS]
        meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        source = meta.get("source", "unknown")
        lines.append(f"[Guideline {i}] (source: {source})\n{text}")
    lines.append("---")
    return "\n".join(lines)


# ── Public API: single-question mode (original, unchanged) ───────────────────

def build_prompt(
    query: str,
    chunks: List[Dict],
    structured_result=None,
) -> str:
    """
    Build the full LLM prompt from query + retrieved evidence.

    This is the original single-question prompt builder. Signature and
    behaviour are unchanged — all existing call sites work without modification.

    Args:
        query:            The original user question.
        chunks:           Retrieved document/guideline chunks (list of dicts).
        structured_result: StructuredResult object or None.

    Returns:
        str: The complete prompt string to send to the LLM.
    """
    parts = [SYSTEM_PROMPT, "\n\n=== RETRIEVED EVIDENCE ==="]

    struct_block = _render_structured_rows(structured_result)
    chunk_block  = _render_chunks(chunks)

    if struct_block:
        parts.append(struct_block)
    if chunk_block:
        parts.append(chunk_block)

    if not struct_block and not chunk_block:
        parts.append(
            "\n[No relevant evidence was retrieved for this question.]"
        )

    parts.append(f"\n\n=== QUESTION ===\n{query}")
    parts.append("\n\n=== ANSWER ===")

    return "\n".join(parts)


# ── Public API: decomposed mode (new) ────────────────────────────────────────

def build_prompt_from_sub_results(
    original_query: str,
    sub_results: "List[SubQuestionResult]",
) -> str:
    """
    Build a prompt from a list of sub-question retrieval results.

    Each sub-question gets its own clearly labelled evidence block.
    Sub-questions where retrieval fell back (no sufficient evidence) are
    explicitly marked so the LLM acknowledges the gap rather than guessing.

    If sub_results contains exactly one item (the common case after
    pre-filtering), this produces output semantically equivalent to
    build_prompt() — one labelled block, same structure.

    Args:
        original_query: The original user question (shown at the end for context).
        sub_results:    List of SubQuestionResult from run_decomposed_retrieval().

    Returns:
        str: Complete prompt string to send to the LLM.
    """
    parts = [SYSTEM_PROMPT_DECOMPOSED, "\n\n=== RETRIEVED EVIDENCE ==="]

    chunk_counter = 1  # global chunk index across all sub-questions

    for idx, result in enumerate(sub_results, start=1):
        sq_header = f"\n\n[Sub-question {idx}: {result.sub_question}]"
        parts.append(sq_header)

        if result.error:
            parts.append(
                f"  ⚠ Retrieval error for this part: {result.error}\n"
                "  No evidence available."
            )
            continue

        if result.fallback_triggered:
            parts.append(
                "  No sufficient evidence found for this part.\n"
                "  (Relevance threshold not met — answer explicitly that evidence is absent.)"
            )
            continue

        has_evidence = False

        # Structured DB evidence for this sub-question
        struct_block = _render_structured_rows(
            result.structured_result,
            prefix=f"Sub-question {idx} — ",
        )
        if struct_block:
            parts.append(struct_block)
            has_evidence = True

        # Guideline chunks for this sub-question
        if result.chunks:
            chunk_block = _render_chunks(result.chunks, start_index=chunk_counter)
            parts.append(chunk_block)
            chunk_counter += min(len(result.chunks), MAX_CHUNKS)
            has_evidence = True

        if not has_evidence:
            parts.append(
                "  [No relevant evidence retrieved for this sub-question.]"
            )

    parts.append(f"\n\n=== ORIGINAL QUESTION ===\n{original_query}")
    parts.append(
        "\n\nAnswer all parts clearly and separately, using only the evidence above.\n"
        "If evidence was insufficient for any part, say so explicitly for that part "
        "rather than guessing or inferring from other parts."
    )
    parts.append("\n\n=== ANSWER ===")

    return "\n".join(parts)