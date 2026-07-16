"""
Fixed-size sliding-window chunker for NHP (National Health Portal) text.

NHP content is semi-structured (sections like Introduction, Symptoms, Causes, etc.)
but for v1 we use simple fixed-size token-aware chunking with overlap.
Section-aware chunking for NHP can be added later.
"""

import logging
from typing import Dict, List

import tiktoken

from config.settings import settings

logger = logging.getLogger(__name__)

# Use cl100k_base encoding (good general-purpose tokenizer)
_ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_nhp_text(clean_text: str, metadata: Dict[str, str]) -> List[Dict]:
    """
    Chunk NHP text using fixed-size sliding window with overlap.

    Args:
        clean_text: The cleaned full text from nhp_parser.
        metadata: Base metadata dict with at least "title".
                  e.g. {"title": "NHP_Japanese_Encephalitis"}

    Returns:
        List of dicts, each with:
            - "text": str — the chunk content
            - "metadata": dict — enriched with source_type, chunk_index, etc.
    """
    tokens = _ENCODING.encode(clean_text)
    total_tokens = len(tokens)
    chunk_size = settings.NHP_CHUNK_SIZE
    overlap = settings.NHP_CHUNK_OVERLAP
    step = chunk_size - overlap

    if step <= 0:
        raise ValueError(f"Chunk overlap ({overlap}) must be less than chunk size ({chunk_size})")

    chunks = []
    chunk_idx = 0

    for start in range(0, total_tokens, step):
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = _ENCODING.decode(chunk_tokens)

        if not chunk_text.strip():
            continue

        chunk_meta = {
            **metadata,
            "source_type": "NHP",
            "chunk_index": chunk_idx,
            "token_start": start,
            "token_end": end,
        }

        chunks.append({"text": chunk_text.strip(), "metadata": chunk_meta})
        chunk_idx += 1

        # If we've covered the entire text, stop
        if end >= total_tokens:
            break

    logger.info(f"NHP chunking: {total_tokens} tokens -> {len(chunks)} chunks "
                f"(size={chunk_size}, overlap={overlap})")

    return chunks