"""
Parser for ICMR and MOHFW PDF documents using PyMuPDF (fitz).

These are government health PDFs that contain primarily narrative text.
This parser does plain text extraction only — no table parsing, no image
handling. Those can be layered on later.

Key design decisions:
  - Uses PyMuPDF (fitz), NOT Unstructured.io, keeping it separate from
    the NHP/PubMed pipeline.
  - Page numbers in output are always 1-based (human-readable), regardless
    of which pages were skipped.
  - skip_pages is per-document, not global, because front-matter length
    varies across MOHFW documents.

Output schema (JSON file saved to disk):
    {
        "source_file": "<filename>.pdf",
        "source_type": "ICMR" | "MOHFW",
        "total_pages_in_pdf": <int>,
        "pages_extracted": <int>,
        "pages": [
            {"page_number": <int>, "text": "<cleaned text>"},
            ...
        ]
    }
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise ImportError(
        "PyMuPDF is required for ICMR/MOHFW parsing. "
        "Install it with: pip install pymupdf>=1.24.0"
    ) from exc

from config.settings import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def parse_icmr_pdf(file_path: Path) -> Dict:
    """
    Parse a single ICMR PDF and return structured page content.

    All pages are extracted (no skipping). Use parse_mohfw_pdf if you
    need to skip front-matter pages.

    Args:
        file_path: Path to the .pdf file.

    Returns:
        Dict with keys: source_file, source_type, total_pages_in_pdf,
        pages_extracted, pages (list of {page_number, text}).
    """
    return _parse_pdf(
        file_path=file_path,
        source_type="ICMR",
        skip_pages=[],
    )


def parse_mohfw_pdf(file_path: Path, skip_pages: Optional[List[int]] = None) -> Dict:
    """
    Parse a single MOHFW PDF and return structured page content.

    Args:
        file_path: Path to the .pdf file.
        skip_pages: Zero-indexed page numbers to skip (e.g. [0, 1, 2]
                    skips the first three pages). Defaults to [] (no
                    skipping). This is per-document so callers control
                    it — front-matter length varies across MOHFW docs.

    Returns:
        Dict with keys: source_file, source_type, total_pages_in_pdf,
        pages_extracted, pages (list of {page_number, text}).
    """
    if skip_pages is None:
        skip_pages = []
    return _parse_pdf(
        file_path=file_path,
        source_type="MOHFW",
        skip_pages=skip_pages,
    )


def save_icmr_json(parsed: Dict, output_dir: Optional[Path] = None) -> Path:
    """
    Save ICMR parsed output as a JSON file.

    Args:
        parsed: Dict returned by parse_icmr_pdf.
        output_dir: Directory to save to. Defaults to settings.ICMR_PROCESSED_DIR.

    Returns:
        Path to the saved JSON file.
    """
    out_dir = output_dir or settings.ICMR_PROCESSED_DIR
    return _save_json(parsed, out_dir)


def save_mohfw_json(parsed: Dict, output_dir: Optional[Path] = None) -> Path:
    """
    Save MOHFW parsed output as a JSON file.

    Args:
        parsed: Dict returned by parse_mohfw_pdf.
        output_dir: Directory to save to. Defaults to settings.MOHFW_PROCESSED_DIR.

    Returns:
        Path to the saved JSON file.
    """
    out_dir = output_dir or settings.MOHFW_PROCESSED_DIR
    return _save_json(parsed, out_dir)


# ═══════════════════════════════════════════════════════════════════════
# Internal implementation
# ═══════════════════════════════════════════════════════════════════════

def _parse_pdf(
    file_path: Path,
    source_type: str,
    skip_pages: List[int],
) -> Dict:
    """
    Core parsing logic shared by ICMR and MOHFW.

    Args:
        file_path: Path to the PDF.
        source_type: "ICMR" or "MOHFW".
        skip_pages: Zero-indexed page numbers to skip entirely.

    Returns:
        Structured dict with page content.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info(
        "Parsing %s PDF: %s (skip_pages=%s)",
        source_type, file_path.name, skip_pages,
    )

    skip_set = set(skip_pages)
    pages_output = []

    with fitz.open(str(file_path)) as doc:
        total_pages = len(doc)

        for zero_idx in range(total_pages):
            if zero_idx in skip_set:
                logger.debug(
                    "  Skipping page %d (zero-indexed) of %s",
                    zero_idx, file_path.name,
                )
                continue

            page = doc[zero_idx]

            # Plain text extraction only — no table/image handling yet.
            raw_text = page.get_text("text")
            cleaned = _clean_page_text(raw_text)

            # Skip effectively blank pages (less than 30 non-whitespace chars)
            if len(cleaned.replace(" ", "")) < 30:
                logger.debug(
                    "  Skipping near-blank page %d of %s",
                    zero_idx + 1, file_path.name,
                )
                continue

            pages_output.append({
                # page_number is 1-based for human readability
                "page_number": zero_idx + 1,
                "text": cleaned,
            })

    result = {
        "source_file": file_path.name,
        "source_type": source_type,
        "total_pages_in_pdf": total_pages,
        "pages_extracted": len(pages_output),
        "pages": pages_output,
    }

    logger.info(
        "Parsed %s '%s': %d/%d pages extracted",
        source_type, file_path.name, len(pages_output), total_pages,
    )
    return result


def _clean_page_text(raw: str) -> str:
    """
    Light-touch cleaning of text extracted from a single PDF page.

    - Normalises line endings
    - Collapses runs of 3+ blank lines to 2
    - Strips leading/trailing whitespace
    - Removes soft-hyphen line-break artefacts (word-\ncontinuation)
    - Does NOT strip page headers/footers (patterns differ by document)
    """
    # Normalise Windows line endings
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Remove soft-hyphen line breaks: "treat-\nment" → "treatment"
    text = re.sub(r"-\n(\S)", r"\1", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces / tabs to a single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def _save_json(parsed: Dict, output_dir: Path) -> Path:
    """
    Save parsed dict as <source_file_stem>.json inside output_dir.

    Args:
        parsed: Dict with at least "source_file" key.
        output_dir: Directory to write into (created if missing).

    Returns:
        Path to the saved JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(parsed["source_file"]).stem
    out_path = output_dir / f"{stem}.json"

    out_path.write_text(
        json.dumps(parsed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Saved parsed JSON to %s", out_path)
    return out_path
