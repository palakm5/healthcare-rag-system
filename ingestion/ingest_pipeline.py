"""
Ingestion pipeline orchestrator.

Walks the raw data directories, parses files, chunks them, embeds the chunks,
and stores them in ChromaDB collections. Also saves processed chunks as JSON
for traceability.

Usage:
    python -m ingestion.ingest_pipeline          # ingest all
    python -m ingestion.ingest_pipeline --nhp-only
    python -m ingestion.ingest_pipeline --pubmed-only
    python -m ingestion.ingest_pipeline --icmr-only
    python -m ingestion.ingest_pipeline --mohfw-only
"""

import json
import logging
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from ingestion.parsers.nhp_parser import parse_nhp_txt
from ingestion.chunkers.recursive_chunker import (
    chunk_icmr_pages,
    chunk_mohfw_pages,
    chunk_nhp_text_recursive,
)
from ingestion.embedders.embedder import Embedder

# Lazy imports for PubMed (unstructured.io requires Python 3.10+ and is
# only needed when actually ingesting PubMed PDFs)
def _get_pubmed_parser():
    from ingestion.parsers.pubmed_parser import parse_pubmed_pdf
    return parse_pubmed_pdf

def _get_pubmed_chunker():
    from ingestion.chunkers.pubmed_chunker import chunk_pubmed_sections
    return chunk_pubmed_sections

def _get_icmr_mohfw_parsers():
    from ingestion.parsers.icmr_mohfw_parser import (
        parse_icmr_pdf,
        parse_mohfw_pdf,
        save_icmr_json,
        save_mohfw_json,
    )
    return parse_icmr_pdf, parse_mohfw_pdf, save_icmr_json, save_mohfw_json

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def run_ingestion(
    nhp_only: bool = False,
    pubmed_only: bool = False,
    icmr_only: bool = False,
    mohfw_only: bool = False,
) -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        nhp_only: If True, only ingest NHP .txt files.
        pubmed_only: If True, only ingest PubMed PDFs.
        icmr_only: If True, only ingest ICMR PDFs.
        mohfw_only: If True, only ingest MOHFW PDFs.

    Returns:
        dict with summary counts per source type.
    """
    do_all = not (nhp_only or pubmed_only or icmr_only or mohfw_only)

    # ── Initialize ChromaDB client ─────────────────────────────────────
    client = chromadb.PersistentClient(
        path=str(settings.VECTOR_STORE_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # ── Initialize Embedder (loads model once) ─────────────────────────
    embedder = Embedder()
    logger.info(f"Embedder ready. Dimension: {embedder.dimension}")

    summary = {
        "nhp_docs": 0, "nhp_chunks": 0,
        "pubmed_docs": 0, "pubmed_chunks": 0,
        "icmr_docs": 0, "icmr_chunks": 0,
        "mohfw_docs": 0, "mohfw_chunks": 0,
    }

    # ── Ingest NHP ─────────────────────────────────────────────────────
    if do_all or nhp_only:
        nhp_docs, nhp_chunks = _ingest_nhp(client, embedder)
        summary["nhp_docs"] = nhp_docs
        summary["nhp_chunks"] = nhp_chunks

    # ── Ingest PubMed ──────────────────────────────────────────────────
    if do_all or pubmed_only:
        pubmed_docs, pubmed_chunks = _ingest_pubmed(client, embedder)
        summary["pubmed_docs"] = pubmed_docs
        summary["pubmed_chunks"] = pubmed_chunks

    # ── Ingest ICMR ────────────────────────────────────────────────────
    if do_all or icmr_only:
        icmr_docs, icmr_chunks = _ingest_icmr(client, embedder)
        summary["icmr_docs"] = icmr_docs
        summary["icmr_chunks"] = icmr_chunks

    # ── Ingest MOHFW ───────────────────────────────────────────────────
    if do_all or mohfw_only:
        mohfw_docs, mohfw_chunks = _ingest_mohfw(client, embedder)
        summary["mohfw_docs"] = mohfw_docs
        summary["mohfw_chunks"] = mohfw_chunks

    logger.info(f"Ingestion complete. Summary: {summary}")
    return summary

# ═══════════════════════════════════════════════════════════════════════
# NHP ingestion
# ═══════════════════════════════════════════════════════════════════════

def _ingest_nhp(client: chromadb.PersistentClient, embedder: Embedder) -> tuple:
    """Ingest all NHP .txt files. Returns (doc_count, chunk_count)."""
    nhp_dir = settings.NHP_DIR
    if not nhp_dir.exists():
        logger.warning(f"NHP directory not found: {nhp_dir}")
        return 0, 0

    txt_files = sorted(nhp_dir.glob("*.txt"))
    if not txt_files:
        logger.warning(f"No .txt files found in {nhp_dir}")
        return 0, 0

    collection = _get_or_create_collection(client, settings.CHROMA_NHP_COLLECTION)
    deleted_vectors = _delete_nhp_vectors(collection)
    logger.info("NHP cleanup: deleted %d old vectors", deleted_vectors)

    all_chunks = []
    parsed_files = 0

    for file_path in txt_files:
        try:
            parsed = parse_nhp_txt(file_path)
        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}: {e}")
            continue

        parsed_files += 1
        title = parsed["title"]
        chunks = chunk_nhp_text_recursive(parsed["full_text"], {"title": title})
        all_chunks.extend(chunks)
        logger.info(f"  {file_path.name}: {len(chunks)} chunks")

    logger.info("NHP parsing: parsed %d files", parsed_files)

    if not all_chunks:
        logger.info("No NHP chunks generated after parsing.")
        return parsed_files, 0

    chunk_token_sizes = [
        max(0, int(c["metadata"]["token_end"]) - int(c["metadata"]["token_start"]))
        for c in all_chunks
    ]
    avg_chunk_tokens = (
        sum(chunk_token_sizes) / len(chunk_token_sizes) if chunk_token_sizes else 0.0
    )
    logger.info(
        "NHP chunking: generated %d recursive chunks (avg %.2f tokens/chunk)",
        len(all_chunks),
        avg_chunk_tokens,
    )

    inserted = _store_chunks(collection, embedder, all_chunks)
    logger.info("NHP storage: inserted %d vectors", inserted)

    # Save processed chunks to disk
    _save_processed(all_chunks, "nhp")

    logger.info(
        "NHP ingestion done: %d parsed files, %d chunks stored.",
        parsed_files,
        len(all_chunks),
    )
    return parsed_files, len(all_chunks)

# ═══════════════════════════════════════════════════════════════════════
# PubMed ingestion
# ═══════════════════════════════════════════════════════════════════════

def _ingest_pubmed(client: chromadb.PersistentClient, embedder: Embedder) -> tuple:
    """Ingest all PubMed PDFs. Returns (doc_count, chunk_count)."""
    pubmed_dir = settings.PUBMED_DIR
    if not pubmed_dir.exists():
        logger.warning(f"PubMed directory not found: {pubmed_dir}")
        return 0, 0

    pdf_files = sorted(pubmed_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No .pdf files found in {pubmed_dir}")
        return 0, 0

    collection = _get_or_create_collection(client, settings.CHROMA_PUBMED_COLLECTION)
    existing_titles = _get_existing_titles(collection)

    all_chunks = []
    doc_count = 0

    # Lazy-load PubMed parser/chunker (unstructured.io requires Python 3.10+)
    parse_pubmed_pdf = _get_pubmed_parser()
    chunk_pubmed_sections = _get_pubmed_chunker()

    for file_path in pdf_files:
        try:
            parsed = parse_pubmed_pdf(file_path)
        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}: {e}")
            continue

        title = parsed["title"]

        if title in existing_titles:
            logger.info(f"Skipping already-ingested PubMed doc: {title}")
            continue

        chunks = chunk_pubmed_sections(parsed["sections"], {"title": title})
        all_chunks.extend(chunks)
        doc_count += 1
        logger.info(f"  {file_path.name}: {len(parsed['sections'])} sections -> {len(chunks)} chunks")

    if not all_chunks:
        logger.info("No new PubMed documents to ingest.")
        return doc_count, 0

    _store_chunks(collection, embedder, all_chunks)
    _save_processed(all_chunks, "pubmed")

    logger.info(f"PubMed ingestion done: {doc_count} docs, {len(all_chunks)} chunks stored.")
    return doc_count, len(all_chunks)

# ═══════════════════════════════════════════════════════════════════════
# ICMR ingestion
# ═══════════════════════════════════════════════════════════════════════

def _ingest_icmr(client: chromadb.PersistentClient, embedder: Embedder) -> tuple:
    """
    Ingest all ICMR PDFs using PyMuPDF parser + recursive chunker.

    Returns (doc_count, chunk_count).
    """
    icmr_dir = settings.ICMR_DIR
    if not icmr_dir.exists():
        logger.warning(f"ICMR directory not found: {icmr_dir}")
        return 0, 0

    pdf_files = sorted(icmr_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No .pdf files found in {icmr_dir}")
        return 0, 0

    collection = _get_or_create_collection(client, settings.CHROMA_ICMR_COLLECTION)
    existing_titles = _get_existing_titles(collection)

    all_chunks = []
    doc_count = 0
    parse_icmr_pdf, _, save_icmr_json, _ = _get_icmr_mohfw_parsers()

    for file_path in pdf_files:
        try:
            parsed = parse_icmr_pdf(file_path)
        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}: {e}")
            continue

        # Save parsed JSON to ICMR processed dir
        save_icmr_json(parsed)

        title = file_path.stem  # use filename as document title

        # Idempotency check
        if title in existing_titles:
            logger.info(f"Skipping already-ingested ICMR doc: {title}")
            continue

        # Chunk using recursive chunker
        metadata = {
            "document_title": title,
            "source_file": parsed["source_file"],
            "title": title,  # for compatibility with _store_chunks ID generation
        }
        chunks = chunk_icmr_pages(parsed["pages"], metadata)
        all_chunks.extend(chunks)
        doc_count += 1
        logger.info(
            f"  {file_path.name}: {parsed['pages_extracted']}/{parsed['total_pages_in_pdf']} pages -> {len(chunks)} chunks"
        )

    if not all_chunks:
        logger.info("No new ICMR documents to ingest.")
        return doc_count, 0

    _store_chunks(collection, embedder, all_chunks)
    _save_processed(all_chunks, "icmr")

    logger.info(f"ICMR ingestion done: {doc_count} docs, {len(all_chunks)} chunks stored.")
    return doc_count, len(all_chunks)

# ═══════════════════════════════════════════════════════════════════════
# MOHFW ingestion
# ═══════════════════════════════════════════════════════════════════════

def _ingest_mohfw(client: chromadb.PersistentClient, embedder: Embedder) -> tuple:
    """
    Ingest all MOHFW PDFs using PyMuPDF parser + recursive chunker.

    Per-document front-matter page skipping is configured via
    settings.MOHFW_SKIP_PAGES (dict of filename -> zero-indexed page list).

    Returns (doc_count, chunk_count).
    """
    mohfw_dir = settings.MOHFW_DIR
    if not mohfw_dir.exists():
        logger.warning(f"MOHFW directory not found: {mohfw_dir}")
        return 0, 0

    pdf_files = sorted(mohfw_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No .pdf files found in {mohfw_dir}")
        return 0, 0

    collection = _get_or_create_collection(client, settings.CHROMA_MOHFW_COLLECTION)
    existing_titles = _get_existing_titles(collection)

    all_chunks = []
    doc_count = 0
    _, parse_mohfw_pdf, _, save_mohfw_json = _get_icmr_mohfw_parsers()

    for file_path in pdf_files:
        # Look up per-document skip_pages from settings (default: no skip)
        skip_pages = settings.MOHFW_SKIP_PAGES.get(file_path.name, [])

        try:
            parsed = parse_mohfw_pdf(file_path, skip_pages=skip_pages)
        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}: {e}")
            continue

        # Save parsed JSON to MOHFW processed dir
        save_mohfw_json(parsed)

        title = file_path.stem  # use filename as document title

        # Idempotency check
        if title in existing_titles:
            logger.info(f"Skipping already-ingested MOHFW doc: {title}")
            continue

        # Chunk using recursive chunker
        metadata = {
            "document_title": title,
            "source_file": parsed["source_file"],
            "title": title,  # for compatibility with _store_chunks ID generation
        }
        chunks = chunk_mohfw_pages(parsed["pages"], metadata)
        all_chunks.extend(chunks)
        doc_count += 1
        logger.info(
            f"  {file_path.name}: {parsed['pages_extracted']}/{parsed['total_pages_in_pdf']} pages "
            f"(skipped {skip_pages}) -> {len(chunks)} chunks"
        )

    if not all_chunks:
        logger.info("No new MOHFW documents to ingest.")
        return doc_count, 0

    _store_chunks(collection, embedder, all_chunks)
    _save_processed(all_chunks, "mohfw")

    logger.info(f"MOHFW ingestion done: {doc_count} docs, {len(all_chunks)} chunks stored.")
    return doc_count, len(all_chunks)

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_or_create_collection(
    client: chromadb.PersistentClient, name: str
):
    """Get or create a ChromaDB collection with cosine distance."""
    try:
        collection = client.get_collection(name)
        logger.info(f"Using existing collection: {name} ({collection.count()} docs)")
    except Exception:
        collection = client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Created new collection: {name}")
    return collection

def _get_existing_titles(collection) -> set:
    """Get set of document titles already in the collection (for idempotency)."""
    try:
        results = collection.get(include=["metadatas"])
        if results["metadatas"]:
            return {m["title"] for m in results["metadatas"] if m and "title" in m}
    except Exception:
        pass
    return set()

def _store_chunks(
    collection,
    embedder: Embedder,
    chunks: List[dict],
) -> int:
    """Embed chunks in batches and add to ChromaDB collection."""
    texts = [c["text"] for c in chunks]
    metadatas = []

    for chunk in chunks:
        cleaned = {}

        for key, value in chunk["metadata"].items():

            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)

        metadatas.append(cleaned)

    ids = []

    for m in metadatas:
        unique_string = (
            f"{m['source_type']}"
            f"{m['title']}"
            f"{m['chunk_index']}"
            f"{m.get('section', '')}"
        )

        chunk_id = hashlib.md5(
            unique_string.encode()
        ).hexdigest()

        ids.append(chunk_id)

    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.embed_passages(texts)

    # Add in batches to avoid overwhelming Chroma
    batch_size = 100

    for i in range(0, len(chunks), batch_size):

        batch_end = min(i + batch_size, len(chunks))

        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end].tolist(),
            documents=texts[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )
    logger.debug(f"  Stored batch {i // batch_size + 1}: {i}-{batch_end}")
    return len(ids)


def _delete_nhp_vectors(collection) -> int:
    """
    Delete only NHP vectors from the NHP collection before re-ingestion.
    Falls back to ID-based deletion if metadata filtering is unavailable.
    """
    before_count = collection.count()
    if before_count == 0:
        return 0

    try:
        collection.delete(where={"source_type": "NHP"})
        after_count = collection.count()
        deleted = max(0, before_count - after_count)
        if deleted > 0:
            return deleted
    except Exception as exc:
        logger.warning("NHP cleanup by metadata filter failed: %s", exc)

    try:
        results = collection.get(include=["metadatas"])
        ids = results.get("ids") or []
        metadatas = results.get("metadatas") or []
        nhp_ids = []
        for idx, chunk_id in enumerate(ids):
            md = metadatas[idx] if idx < len(metadatas) else None
            if not md or md.get("source_type") == "NHP":
                nhp_ids.append(chunk_id)
        if nhp_ids:
            collection.delete(ids=nhp_ids)
        return len(nhp_ids)
    except Exception as exc:
        logger.warning("NHP cleanup by IDs failed (%s); deleting full NHP collection", exc)
        results = collection.get()
        all_ids = results.get("ids") or []
        if all_ids:
            collection.delete(ids=all_ids)
        return len(all_ids)

def _save_processed(chunks: List[dict], source_type: str):
    """Save processed chunks as JSON for traceability."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = settings.PROCESSED_DATA_DIR / f"{source_type}_chunks_{timestamp}.json"

    # Serialize: convert any non-serializable metadata values
    serializable = []
    for c in chunks:
        serializable.append({
            "text": c["text"],
            "metadata": {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                         for k, v in c["metadata"].items()},
        })

    output_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved processed chunks to {output_path}")

# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    nhp_only = "--nhp-only" in sys.argv
    pubmed_only = "--pubmed-only" in sys.argv
    icmr_only = "--icmr-only" in sys.argv
    mohfw_only = "--mohfw-only" in sys.argv
    run_ingestion(
        nhp_only=nhp_only,
        pubmed_only=pubmed_only,
        icmr_only=icmr_only,
        mohfw_only=mohfw_only,
    )