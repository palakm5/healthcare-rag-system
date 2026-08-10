"""
Sanity check for ICMR/MOHFW ingestion.

Parses 2-3 ICMR and 2-3 MOHFW PDFs, chunks them, and prints 5-10 sample
chunks each so you can confirm:
  1. Text isn't garbled
  2. MOHFW front-matter is actually excluded (via skip_pages)

This script does NOT embed or store anything — it's a fast, no-API-call
verification step to run before the full ingestion pipeline.

Usage:
    python -m ingestion.sanity_check_icmr_mohfw
"""

import logging
import sys
from pathlib import Path

from config.settings import settings
from ingestion.parsers.icmr_mohfw_parser import (
    parse_icmr_pdf,
    parse_mohfw_pdf,
    save_icmr_json,
    save_mohfw_json,
)
from ingestion.chunkers.recursive_chunker import chunk_icmr_pages, chunk_mohfw_pages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Number of PDFs to sample from each source
NUM_ICMR_SAMPLES = 2
NUM_MOHFW_SAMPLES = 2
# Number of chunks to print per source
NUM_CHUNKS_TO_PRINT = 7


def run_sanity_check():
    """Parse, chunk, and print sample chunks for ICMR and MOHFW."""
    print("=" * 80)
    print("ICMR / MOHFW INGESTION SANITY CHECK")
    print("=" * 80)

    # ── ICMR ───────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("ICMR SAMPLES")
    print("─" * 80)

    icmr_pdfs = sorted(settings.ICMR_DIR.glob("*.pdf"))[:NUM_ICMR_SAMPLES]
    if not icmr_pdfs:
        print(f"  ⚠ No ICMR PDFs found in {settings.ICMR_DIR}")
    else:
        _check_icmr(icmr_pdfs)

    # ── MOHFW ──────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("MOHFW SAMPLES")
    print("─" * 80)

    mohfw_pdfs = sorted(settings.MOHFW_DIR.glob("*.pdf"))[:NUM_MOHFW_SAMPLES]
    if not mohfw_pdfs:
        print(f"  ⚠ No MOHFW PDFs found in {settings.MOHFW_DIR}")
    else:
        _check_mohfw(mohfw_pdfs)

    print("\n" + "=" * 80)
    print("SANITY CHECK COMPLETE")
    print("=" * 80)
    print("\nReview the chunks above to confirm:")
    print("  1. Text is readable and not garbled")
    print("  2. MOHFW front-matter pages are excluded (check page numbers)")
    print("  3. Metadata fields are correct (source_type, page_number, etc.)")


def _check_icmr(pdf_files):
    """Parse and chunk ICMR PDFs, print sample chunks."""
    all_chunks = []

    for file_path in pdf_files:
        print(f"\n📄 Parsing: {file_path.name}")
        parsed = parse_icmr_pdf(file_path)
        save_icmr_json(parsed)

        print(f"   Pages: {parsed['pages_extracted']}/{parsed['total_pages_in_pdf']} extracted")
        print(f"   First page number: {parsed['pages'][0]['page_number'] if parsed['pages'] else 'N/A'}")

        # Show first 200 chars of first page to verify text quality
        if parsed["pages"]:
            first_text = parsed["pages"][0]["text"][:200]
            print(f"   First page preview: {first_text!r}")

        metadata = {
            "document_title": file_path.stem,
            "source_file": parsed["source_file"],
            "title": file_path.stem,
        }
        chunks = chunk_icmr_pages(parsed["pages"], metadata)
        all_chunks.extend(chunks)
        print(f"   Chunks produced: {len(chunks)}")

    # Print sample chunks
    if all_chunks:
        print(f"\n📋 Sample ICMR chunks (showing {min(NUM_CHUNKS_TO_PRINT, len(all_chunks))} of {len(all_chunks)}):")
        for i, chunk in enumerate(all_chunks[:NUM_CHUNKS_TO_PRINT]):
            _print_chunk(i, chunk)


def _check_mohfw(pdf_files):
    """Parse and chunk MOHFW PDFs, print sample chunks."""
    all_chunks = []

    for file_path in pdf_files:
        skip_pages = settings.MOHFW_SKIP_PAGES.get(file_path.name, [])
        print(f"\n📄 Parsing: {file_path.name} (skip_pages={skip_pages})")
        parsed = parse_mohfw_pdf(file_path, skip_pages=skip_pages)
        save_mohfw_json(parsed)

        print(f"   Pages: {parsed['pages_extracted']}/{parsed['total_pages_in_pdf']} extracted")
        print(f"   Skipped pages (zero-indexed): {skip_pages}")

        if parsed["pages"]:
            first_page = parsed["pages"][0]
            print(f"   First extracted page number: {first_page['page_number']}")
            print(f"   First page preview: {first_page['text'][:200]!r}")

            # Verify front-matter was skipped
            if skip_pages:
                expected_first = min(skip_pages) + len(skip_pages)
                if first_page["page_number"] <= max(skip_pages) + 1:
                    print(f"   ⚠ WARNING: First page number {first_page['page_number']} "
                          f"suggests front-matter may not be fully excluded")

        metadata = {
            "document_title": file_path.stem,
            "source_file": parsed["source_file"],
            "title": file_path.stem,
        }
        chunks = chunk_mohfw_pages(parsed["pages"], metadata)
        all_chunks.extend(chunks)
        print(f"   Chunks produced: {len(chunks)}")

    # Print sample chunks
    if all_chunks:
        print(f"\n📋 Sample MOHFW chunks (showing {min(NUM_CHUNKS_TO_PRINT, len(all_chunks))} of {len(all_chunks)}):")
        for i, chunk in enumerate(all_chunks[:NUM_CHUNKS_TO_PRINT]):
            _print_chunk(i, chunk)


def _print_chunk(index, chunk):
    """Print a single chunk with its metadata."""
    text = chunk["text"]
    meta = chunk["metadata"]

    # Truncate text for display
    display_text = text[:300] + ("..." if len(text) > 300 else "")

    print(f"\n  ── Chunk {index + 1} ─────────────────────────────────")
    print(f"  source_type:  {meta.get('source_type', '?')}")
    print(f"  document_title: {meta.get('document_title', '?')}")
    print(f"  source_file:  {meta.get('source_file', '?')}")
    print(f"  page_number:  {meta.get('page_number', '?')}")
    print(f"  chunk_index:  {meta.get('chunk_index', '?')}")
    print(f"  text ({len(text)} chars): {display_text!r}")


if __name__ == "__main__":
    run_sanity_check()