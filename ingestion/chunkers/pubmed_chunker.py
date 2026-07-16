"""
Section-aware chunker for PubMed research papers.

Each section (Abstract, Introduction, Methods, Results, Discussion, Conclusions)
becomes one chunk where possible. If a section exceeds PUBMED_MAX_SECTION_TOKENS,
it is split into sub-chunks with overlap.

Metadata attached to every chunk includes the section heading, which enables
section-filtered retrieval later.
"""

import logging
from typing import Dict, List

import tiktoken

from config.settings import settings

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_pubmed_sections(
    sections: List[Dict[str, str]],
    metadata: Dict[str, str],
) -> List[Dict]:
    """
    Chunk PubMed paper sections — one chunk per section where possible.

    Args:
        sections: List of {"heading": str, "content": str} from pubmed_parser.
        metadata: Base metadata dict with at least "title".

    Returns:
        List of dicts, each with:
            - "text": str — the chunk content
            - "metadata": dict — enriched with source_type, section, chunk_index
    """
    max_tokens = settings.PUBMED_MAX_SECTION_TOKENS
    overlap = settings.NHP_CHUNK_OVERLAP  # reuse overlap setting
    all_chunks = []
    global_chunk_idx = 0

    for section in sections:
        heading = section["heading"]
        content = section["content"]

        if not content.strip():
            continue

        # Skip Preamble if it's just title/author metadata (very short)
        if heading == "Preamble" and len(content) < 200:
            logger.debug(f"Skipping short Preamble ({len(content)} chars)")
            continue

        tokens = _ENCODING.encode(content)
        section_token_count = len(tokens)

        # If section fits within max_tokens, keep it as one chunk
        if section_token_count <= max_tokens:
            chunk_meta = {
                **metadata,
                "source_type": "PubMed",
                "section": heading,
                "chunk_index": global_chunk_idx,
                "token_count": section_token_count,
            }
            all_chunks.append({"text": content.strip(), "metadata": chunk_meta})
            global_chunk_idx += 1
        else:
            # Split long section into sub-chunks with overlap
            step = max_tokens - overlap
            sub_idx = 0
            for start in range(0, section_token_count, step):
                end = min(start + max_tokens, section_token_count)
                chunk_tokens = tokens[start:end]
                chunk_text = _ENCODING.decode(chunk_tokens)

                if not chunk_text.strip():
                    continue

                chunk_meta = {
                    **metadata,
                    "source_type": "PubMed",
                    "section": heading,
                    "chunk_index": global_chunk_idx,
                    "token_count": end - start,
                    "sub_chunk": f"{sub_idx}/{((section_token_count - 1) // step) + 1}",
                }
                all_chunks.append({"text": chunk_text.strip(), "metadata": chunk_meta})
                global_chunk_idx += 1
                sub_idx += 1

                if end >= section_token_count:
                    break

    logger.info(
        f"PubMed chunking: {len(sections)} sections -> {len(all_chunks)} chunks "
        f"(max_section_tokens={max_tokens})"
    )

    return all_chunks