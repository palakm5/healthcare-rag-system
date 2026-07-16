"""
RAG prompt builder.

Constructs a prompt from the user query and retrieved chunks,
with source attribution for each chunk.
"""

from typing import Dict, List


def build_prompt(query: str, chunks: List[Dict]) -> str:
    """
    Build a RAG prompt from the query and retrieved chunks.

    Args:
        query: The user's question.
        chunks: List of retrieved chunk dicts, each with "text" and "metadata".

    Returns:
        Formatted prompt string ready to send to an LLM.
    """
    # ── System instruction ─────────────────────────────────────────────
    system = (
        "You are a medical information assistant. "
        "Answer the question using ONLY the provided context chunks below. "
        "If the context does not contain enough information to answer the question, "
        "say so clearly. Do not make up information.\n"
        "Cite the source of each piece of information you use by referring to "
        "the [Source] label in brackets."
    )

    # ── Context blocks ─────────────────────────────────────────────────
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source_type = meta.get("source_type", "Unknown")
        title = meta.get("title", "Unknown")
        section = meta.get("section", "N/A")

        header = f"[Source {i}: {source_type} | {title}"
        if section:
            header += f" | Section: {section}"
        header += "]"

        context_blocks.append(f"{header}\n{chunk['text']}")

    context_text = "\n\n---\n\n".join(context_blocks)

    # ── Final prompt ───────────────────────────────────────────────────
    prompt = f"""{system}

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""

    return prompt