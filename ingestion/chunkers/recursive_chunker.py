"""
Recursive chunker for NHP, ICMR, and MOHFW content.

This chunker uses a hierarchy of text separators to split text into
semantically coherent chunks. It is applied to:
  - NHP (token-aware recursive chunking),
  - ICMR (character-aware recursive chunking),
  - MOHFW (character-aware recursive chunking).

Algorithm
---------
1. Try to split on the first separator (paragraph break by default).
2. For any piece still larger than chunk_size, recurse with the next
   separator in the hierarchy.
3. Merge consecutive small pieces greedily until the running buffer would
   exceed chunk_size, then emit a chunk. Carry the last `overlap` chars of
   each emitted chunk into the next one.

Metadata preserved on every page-based chunk
--------------------------------------------
  source_type     : "ICMR" or "MOHFW"
  source_file     : original PDF filename
  document_title  : human-readable title (passed by the caller)
  page_number     : 1-based page number where the chunk originates
  chunk_index     : zero-based chunk index within the document
"""

import logging
from typing import Dict, List, Optional

import tiktoken

from config.settings import settings

logger = logging.getLogger(__name__)
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Default separator hierarchy — coarser to finer.
# Can be overridden at call time.
DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def chunk_icmr_pages(
    pages: List[Dict],
    metadata: Dict,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separators: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Chunk ICMR parsed pages using recursive splitting.

    Args:
        pages: List of {"page_number": int, "text": str} from the parser.
        metadata: Base metadata dict. Must include at least "document_title".
        chunk_size: Target chunk size in characters. Defaults to
                    settings.ICMR_CHUNK_SIZE.
        chunk_overlap: Overlap between consecutive chunks in characters.
                       Defaults to settings.ICMR_CHUNK_OVERLAP.
        separators: Separator hierarchy. Defaults to DEFAULT_SEPARATORS.

    Returns:
        List of {"text": str, "metadata": dict} chunks.
    """
    return _chunk_pages(
        pages=pages,
        metadata=metadata,
        source_type="ICMR",
        chunk_size=chunk_size or settings.ICMR_CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.ICMR_CHUNK_OVERLAP,
        separators=separators or DEFAULT_SEPARATORS,
    )


def chunk_mohfw_pages(
    pages: List[Dict],
    metadata: Dict,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separators: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Chunk MOHFW parsed pages using recursive splitting.

    Args:
        pages: List of {"page_number": int, "text": str} from the parser.
        metadata: Base metadata dict. Must include at least "document_title".
        chunk_size: Target chunk size in characters. Defaults to
                    settings.MOHFW_CHUNK_SIZE.
        chunk_overlap: Overlap between consecutive chunks in characters.
                       Defaults to settings.MOHFW_CHUNK_OVERLAP.
        separators: Separator hierarchy. Defaults to DEFAULT_SEPARATORS.

    Returns:
        List of {"text": str, "metadata": dict} chunks.
    """
    return _chunk_pages(
        pages=pages,
        metadata=metadata,
        source_type="MOHFW",
        chunk_size=chunk_size or settings.MOHFW_CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.MOHFW_CHUNK_OVERLAP,
        separators=separators or DEFAULT_SEPARATORS,
    )


def chunk_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: Optional[List[str]] = None,
    token_aware: bool = False,
) -> List[str]:
    """
    Split a single text string recursively into chunks.

    This is the core algorithm — callable directly for testing or
    use cases outside the page-based pipeline.

    Args:
        text: Input text to split.
        chunk_size: Target maximum chunk size (chars or tokens).
        chunk_overlap: Overlap to carry over between chunks (chars or tokens).
        separators: Separator hierarchy (coarser first). Defaults to
                    DEFAULT_SEPARATORS.
        token_aware: If True, chunk_size/overlap are measured in tokens
                     using cl100k_base. If False, character-aware.

    Returns:
        List of text strings split recursively and merged with overlap.
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS
    return _split_recursive(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        token_aware=token_aware,
    )


def chunk_nhp_text_recursive(
    clean_text: str,
    metadata: Dict[str, str],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separators: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Chunk NHP text recursively using token-aware splitting/overlap.

    Metadata added to each chunk:
      - source_type="NHP"
      - chunk_index
      - token_start
      - token_end
      - caller-provided metadata (e.g., title)
    """
    size = chunk_size or settings.NHP_CHUNK_SIZE
    overlap = chunk_overlap or settings.NHP_CHUNK_OVERLAP
    seps = separators or DEFAULT_SEPARATORS

    chunks = chunk_recursive(
        text=clean_text,
        chunk_size=size,
        chunk_overlap=overlap,
        separators=seps,
        token_aware=True,
    )

    total_tokens = len(_ENCODING.encode(clean_text))
    chunk_dicts: List[Dict] = []
    prev_end = 0
    for idx, chunk_text in enumerate(chunks):
        chunk_tokens = len(_ENCODING.encode(chunk_text))
        start = 0 if idx == 0 else max(0, prev_end - overlap)
        end = min(total_tokens, start + chunk_tokens)
        prev_end = end

        chunk_meta = {
            **metadata,
            "source_type": "NHP",
            "chunk_index": idx,
            "token_start": start,
            "token_end": end,
        }
        chunk_dicts.append({"text": chunk_text.strip(), "metadata": chunk_meta})

    return chunk_dicts


# ═══════════════════════════════════════════════════════════════════════
# Internal — page-level orchestration
# ═══════════════════════════════════════════════════════════════════════

def _chunk_pages(
    pages: List[Dict],
    metadata: Dict,
    source_type: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
) -> List[Dict]:
    """
    Iterate over pages, recursively chunk each, and attach metadata.
    """
    all_chunks: List[Dict] = []
    chunk_idx = 0

    for page in pages:
        page_number = page["page_number"]
        text = page.get("text", "").strip()
        if not text:
            continue

        pieces = _split_recursive(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            token_aware=False,
        )

        for piece in pieces:
            if not piece.strip():
                continue

            chunk_meta = {
                **metadata,
                "source_type": source_type,
                "page_number": page_number,
                "chunk_index": chunk_idx,
            }
            # Ensure document_title is present (use source_file as fallback)
            if "document_title" not in chunk_meta:
                chunk_meta["document_title"] = metadata.get(
                    "source_file", "Unknown"
                )

            all_chunks.append({"text": piece.strip(), "metadata": chunk_meta})
            chunk_idx += 1

    logger.info(
        "%s chunking: %d pages → %d chunks (chunk_size=%d, overlap=%d)",
        source_type, len(pages), len(all_chunks), chunk_size, chunk_overlap,
    )
    return all_chunks


# ═══════════════════════════════════════════════════════════════════════
# Internal — recursive splitting core
# ═══════════════════════════════════════════════════════════════════════

def _split_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
    token_aware: bool,
) -> List[str]:
    """
    Recursively split text using the separator hierarchy.

    If text <= chunk_size: return as-is.
    Otherwise: split on separators[0], recurse on each piece that is
    still too large (using separators[1:]), then merge short pieces
    greedily while respecting chunk_size and carrying overlap.
    """
    # Base case: text fits in one chunk
    if _measure_len(text, token_aware) <= chunk_size:
        return [text]

    # No more separators: hard split (last resort)
    if not separators:
        return _hard_split(text, chunk_size, chunk_overlap, token_aware)

    sep = separators[0]
    remaining_seps = separators[1:]

    # Split on current separator
    if sep == "":
        # Character-level split — split into individual characters then merge
        raw_pieces = list(text)
    elif sep in text:
        raw_pieces = text.split(sep)
    else:
        # Separator not found at this level; try the next
        return _split_recursive(
            text, chunk_size, chunk_overlap, remaining_seps, token_aware
        )

    if sep == "" and token_aware:
        # For token-aware mode, last-resort split should be token hard split.
        return _hard_split(text, chunk_size, chunk_overlap, token_aware)

    # Recursively split any piece that is still too large, then merge
    fine_pieces: List[str] = []
    for piece in raw_pieces:
        piece = piece.strip()
        if not piece:
            continue
        if _measure_len(piece, token_aware) > chunk_size:
            fine_pieces.extend(
                _split_recursive(
                    piece, chunk_size, chunk_overlap, remaining_seps, token_aware
                )
            )
        else:
            fine_pieces.append(piece)

    # Merge fine pieces greedily up to chunk_size, with overlap
    return _merge_pieces(
        pieces=fine_pieces,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=sep,
        token_aware=token_aware,
    )


def _merge_pieces(
    pieces: List[str],
    chunk_size: int,
    chunk_overlap: int,
    separator: str,
    token_aware: bool,
) -> List[str]:
    """
    Greedily merge short pieces into chunks of up to chunk_size characters.
    Adds `chunk_overlap` characters of the previous chunk's tail to the start
    of each new chunk.
    """
    chunks: List[str] = []
    current_parts: List[str] = []
    current_len: int = 0
    # Separator to join pieces back together (use space if sep was empty)
    join_sep = separator if separator not in ("", " ") else " "

    def _flush(parts: List[str]) -> str:
        return join_sep.join(parts).strip()

    for piece in pieces:
        piece_len = _measure_len(piece, token_aware)
        sep_overhead = _measure_len(join_sep, token_aware) if current_parts else 0

        if current_len + sep_overhead + piece_len > chunk_size and current_parts:
            # Emit current buffer as a chunk
            chunk_text = _flush(current_parts)
            if chunk_text:
                chunks.append(chunk_text)

            # Seed the next chunk with overlap from the tail of the current
            overlap_seed = (
                _tail_overlap(chunk_text, chunk_overlap, token_aware)
                if chunk_overlap > 0
                else ""
            )
            if overlap_seed:
                current_parts = [overlap_seed, piece]
                current_len = (
                    _measure_len(overlap_seed, token_aware)
                    + _measure_len(join_sep, token_aware)
                    + piece_len
                )
            else:
                current_parts = [piece]
                current_len = piece_len
        else:
            current_parts.append(piece)
            current_len += sep_overhead + piece_len

    # Flush remaining
    if current_parts:
        chunk_text = _flush(current_parts)
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _hard_split(
    text: str, chunk_size: int, chunk_overlap: int, token_aware: bool
) -> List[str]:
    """Last-resort hard split, character-aware or token-aware."""
    if not token_aware:
        return _hard_split_chars(text, chunk_size, chunk_overlap)

    tokens = _ENCODING.encode(text)
    chunks: List[str] = []
    step = max(chunk_size - chunk_overlap, 1)
    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        chunks.append(_ENCODING.decode(tokens[start:end]))
        if end >= len(tokens):
            break
    return chunks


def _hard_split_chars(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Character-aware hard split."""
    chunks: List[str] = []
    step = max(chunk_size - chunk_overlap, 1)
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
    return chunks


def _measure_len(text: str, token_aware: bool) -> int:
    """Measure text length in tokens (cl100k) or characters."""
    if token_aware:
        return len(_ENCODING.encode(text))
    return len(text)


def _tail_overlap(text: str, overlap: int, token_aware: bool) -> str:
    """Return tail overlap in tokens/chars for seeding the next chunk."""
    if overlap <= 0:
        return ""
    if not token_aware:
        return text[-overlap:]

    tokens = _ENCODING.encode(text)
    return _ENCODING.decode(tokens[-overlap:])
