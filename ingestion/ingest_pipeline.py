"""
Ingestion pipeline orchestrator.

Walks the raw data directories, parses files, chunks them, embeds the chunks,
and stores them in ChromaDB collections. Also saves processed chunks as JSON
for traceability.

Usage:
    python -m ingestion.ingest_pipeline          # ingest all
    python -m ingestion.ingest_pipeline --nhp-only
    python -m ingestion.ingest_pipeline --pubmed-only
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
from ingestion.parsers.pubmed_parser import parse_pubmed_pdf
from ingestion.chunkers.nhp_chunker import chunk_nhp_text
from ingestion.chunkers.pubmed_chunker import chunk_pubmed_sections
from ingestion.embedders.embedder import Embedder

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def run_ingestion(nhp_only: bool = False, pubmed_only: bool = False) -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        nhp_only: If True, only ingest NHP .txt files.
        pubmed_only: If True, only ingest PubMed PDFs.

    Returns:
        dict with summary counts: {"nhp_docs": int, "nhp_chunks": int,
                                    "pubmed_docs": int, "pubmed_chunks": int}
    """
    do_all = not nhp_only and not pubmed_only

    # ── Initialize ChromaDB client ─────────────────────────────────────
    client = chromadb.PersistentClient(
        path=str(settings.VECTOR_STORE_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # ── Initialize Embedder (loads model once) ─────────────────────────
    embedder = Embedder()
    logger.info(f"Embedder ready. Dimension: {embedder.dimension}")

    summary = {"nhp_docs": 0, "nhp_chunks": 0, "pubmed_docs": 0, "pubmed_chunks": 0}

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

    # Get or create collection
    collection = _get_or_create_collection(client, settings.CHROMA_NHP_COLLECTION)

    # Track existing documents for idempotency
    existing_titles = _get_existing_titles(collection)

    all_chunks = []
    doc_count = 0

    for file_path in txt_files:
        try:
            parsed = parse_nhp_txt(file_path)
        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}: {e}")
            continue

        title = parsed["title"]

        # Idempotency check
        if title in existing_titles:
            logger.info(f"Skipping already-ingested NHP doc: {title}")
            continue

        chunks = chunk_nhp_text(parsed["full_text"], {"title": title})
        all_chunks.extend(chunks)
        doc_count += 1
        logger.info(f"  {file_path.name}: {len(chunks)} chunks")

    if not all_chunks:
        logger.info("No new NHP documents to ingest.")
        return doc_count, 0

    # Embed and store
    _store_chunks(collection, embedder, all_chunks)

    # Save processed chunks to disk
    _save_processed(all_chunks, "nhp")

    logger.info(f"NHP ingestion done: {doc_count} docs, {len(all_chunks)} chunks stored.")
    return doc_count, len(all_chunks)

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
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_or_create_collection(
    client: chromadb.PersistentClient, name: str
) :
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

def _get_existing_titles(collection) -> set[str]:
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
):
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

        print("\n========== DEBUG BATCH ==========")
        print("Batch:", i, "-", batch_end)

        for idx, metadata in enumerate(metadatas[i:batch_end]):

            print("\nChunk:", i + idx)

            for key, value in metadata.items():

                print(
                    key,
                    "=>",
                    value,
                    "| TYPE:",
                    type(value)
                )

                if not isinstance(value, (str, int, float, bool)) and value is not None:
                    print(
                        "❌ INVALID METADATA FOUND:",
                        key,
                        value,
                        type(value)
                    )

        print("=================================\n")

        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end].tolist(),
            documents=texts[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )
    logger.debug(f"  Stored batch {i // batch_size + 1}: {i}-{batch_end}")

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
    run_ingestion(nhp_only=nhp_only, pubmed_only=pubmed_only)