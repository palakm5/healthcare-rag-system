"""
RAG prompt builder.

Constructs a prompt from the user query and retrieved chunks,
with source attribution for each chunk.

Now also supports an optional structured evidence block (from SQL database
retrieval) presented as a clearly separate section from guideline chunks,
so the generation model does not conflate database facts with retrieved text.
"""

from typing import Any, Dict, List, Optional


def _format_structured_rows(
    rows: List[Dict],
    columns: List[str],
    provenance_label: str,
    max_rows: int = 10,
) -> str:
    """
    Format structured SQL result rows into a readable evidence block.

    Each row is rendered as a key: value list. Provenance is stated explicitly.
    """
    if not rows:
        return ""

    display_rows = rows[:max_rows]
    lines = []
    for i, row in enumerate(display_rows, start=1):
        pairs = ""
        for col in columns:
            val = row.get(col)
            if val is not None and str(val).strip():
                pairs += f"  {col}: {val}\n"
        lines.append(f"Record {i} [{provenance_label}]:\n{pairs}")

    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows not shown)")

    return "\n".join(lines)


def build_prompt(
    query: str,
    chunks: List[Dict],
    structured_result: Optional[Any] = None,
) -> str:
    """
    Build a RAG prompt from the query, retrieved chunks, and optional
    structured database results.

    The structured evidence block (if present) is clearly labelled as coming
    from a structured database rather than guideline text, so the LLM treats
    them as factual records rather than narrative context.

    Args:
        query:            The user's question.
        chunks:           List of retrieved chunk dicts (vector search results).
                          Each must have "text" and "metadata" keys.
        structured_result: Optional StructuredResult from StructuredLookup.
                          When provided and has rows, a second evidence block
                          is added before the unstructured chunks.

    Returns:
        Formatted prompt string ready to send to an LLM.
    """
    # ── Check if structured result has usable data ──────────────────────────
    has_structured = (
        structured_result is not None
        and getattr(structured_result, "rows_returned", 0) > 0
        and getattr(structured_result, "error", None) is None
    )

    # ── System instruction ──────────────────────────────────────────────────
    if has_structured:
        system = (
            "You are a medical information assistant. "
            "You have been given two types of evidence below:\n"
            "  1. STRUCTURED DATABASE RECORDS: exact, factual records from a "
            "clinical/pharmaceutical database. Treat these as ground truth -- "
            "do not contradict or rephrase them.\n"
            "  2. GUIDELINE TEXT CHUNKS: excerpts from medical guidelines and "
            "literature. Use these for context and clinical reasoning.\n\n"
            "Answer using ONLY the provided evidence. "
            "If the evidence does not contain enough information, say so clearly. "
            "Do not make up information. "
            "Cite your sources using the [Source] or [DB:table] labels."
        )
    else:
        system = (
            "You are a medical information assistant. "
            "Answer the question using ONLY the provided context chunks below. "
            "If the context does not contain enough information to answer the question, "
            "say so clearly. Do not make up information.\n"
            "Cite the source of each piece of information you use by referring to "
            "the [Source] label in brackets."
        )

    # ── Structured evidence block (SQL database results) ────────────────────
    structured_block = ""
    if has_structured:
        prov_label = getattr(structured_result, "provenance_label", "DB")
        rows       = getattr(structured_result, "rows", [])
        columns    = getattr(structured_result, "columns", [])
        tmpl_id    = getattr(structured_result, "template_id", None)

        source_note = (
            f"[Template: {tmpl_id}]" if tmpl_id
            else f"[LLM-generated SQL | tables: {prov_label}]"
        )
        structured_block = (
            f"\nSTRUCTURED DATABASE EVIDENCE {source_note}:\n"
            + _format_structured_rows(rows, columns, prov_label)
            + "\n"
        )

    # ── Unstructured chunk blocks ────────────────────────────────────────────
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta        = chunk.get("metadata", {})
        source_type = meta.get("source_type", "Unknown")
        title       = meta.get("title", "Unknown")
        section     = meta.get("section", "N/A")

        header = f"[Source {i}: {source_type} | {title}"
        if section:
            header += f" | Section: {section}"
        header += "]"

        context_blocks.append(f"{header}\n{chunk['text']}")

    context_text = (
        "\n\n---\n\n".join(context_blocks) if context_blocks
        else "(No guideline chunks retrieved.)"
    )

    # ── Final prompt ────────────────────────────────────────────────────────
    if has_structured:
        prompt = (
            f"{system}"
            f"{structured_block}"
            f"\nGUIDELINE TEXT CHUNKS:\n{context_text}"
            f"\n\nQUESTION: {query}"
            f"\n\nANSWER:"
        )
    else:
        prompt = (
            f"{system}"
            f"\n\nCONTEXT:\n{context_text}"
            f"\n\nQUESTION: {query}"
            f"\n\nANSWER:"
        )

    return prompt