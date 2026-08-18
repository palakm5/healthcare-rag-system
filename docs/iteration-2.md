# Healthcare RAG System — Iteration 2

## Overview

Iteration 2 expanded the data sources from NHP + PubMed to the full Indian
healthcare corpus (ICMR, MoHFW, NHP) and upgraded retrieval from simple cosine
similarity to a full hybrid search + reranking pipeline.

---

## 1. New Data Sources

### ICMR Standard Treatment Workflows (STW)
- **Location**: `unstructured-data/raw/icmr/`
- **Documents added**: ICMR STW Volume 1, 2, 3; STW modules for TB (pulmonary + extrapulmonary), Diabetes Type 2, Diabetic Foot, Cardiology, Dermatology/Psoriasis, Endocrinology, Paediatrics, Cataract/Ophthalmology, Urology, Lung Cancer, Acute Respiratory Infections, AMR Guidelines 2019.

### MoHFW Guidelines
- **Location**: `unstructured-data/raw/mohfw/`
- **Documents added**: Dengue Clinical Management Guidelines 2023, NPCDCS Operational Guidelines, TB Preventive Treatment Guidelines 2021, WHO Global TB Report 2023.

### Parser — `ingestion/parsers/icmr_mohfw_parser.py`
- Handles scanned and native-text PDFs via `unstructured.io`
- Section detection for ICMR STW format (Diagnosis, Treatment, Monitoring, Drug Dosage)
- Strips header/footer boilerplate, page numbers, and table-of-contents pages

### Chunker — `ingestion/chunkers/recursive_chunker.py`
- Recursive character splitter with semantic section boundaries
- Configurable `chunk_size` (default 800 tokens) and `chunk_overlap` (default 100)
- Falls back to fixed-size split when section boundaries are absent

---

## 2. Ingestion Pipeline Upgrade

**File**: `ingestion/ingest_pipeline.py`

- Unified pipeline now handles NHP, PubMed, ICMR, and MoHFW sources in a single run
- Source-type metadata tag added to every chunk: `source_type ∈ {nhp, pubmed, icmr, mohfw}`
- Document-level metadata per chunk: `document_title`, `section`, `page_number`
- Deduplication: chunks with identical `(source_file, page, chunk_index)` are skipped on re-ingestion

---

## 3. Hybrid Search

**File**: `retrieval/search/hybrid_search.py`

Combines dense (vector) and sparse (BM25) retrieval into a single ranked list.

| Component | Detail |
|---|---|
| Dense retrieval | ChromaDB cosine similarity with BGE-M3 embeddings |
| Sparse retrieval | BM25 via `rank_bm25` over the chunk corpus |
| Fusion | Reciprocal Rank Fusion (RRF) with `k=60` |
| Top-k | Default 20 candidates passed to reranker |

---

## 4. Cross-Encoder Reranking

**File**: `retrieval/rerank/cross_encoder_reranker.py`

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Scores every (query, chunk) pair directly — no embedding approximation
- Input: top-20 hybrid search candidates; Output: top-k (default 5) re-ordered chunks
- Adds `rerank_score` to each chunk's metadata for downstream inspection

---

## 5. Metadata Filtering

**File**: `retrieval/filters/metadata_filter.py`

Rule-based pre-filter applied before vector search via ChromaDB `where` clause:

| Filter | Effect |
|---|---|
| `source_type` | Restrict to specific source(s), e.g. `icmr` only |
| `document_title` | Restrict to a specific guideline document |
| `section` | Restrict to a specific clinical section |

---

## 6. Retriever Orchestrator

**File**: `retrieval/search/retriever.py`

Single entry point composing the full pipeline:
```
Query → Metadata filter → Hybrid search (BM25+dense RRF) → Cross-encoder rerank → top-k chunks
```

---

## 7. Updated File Structure

```
retrieval/
├── filters/
│   └── metadata_filter.py
├── rerank/
│   └── cross_encoder_reranker.py
├── search/
│   ├── hybrid_search.py
│   └── retriever.py
ingestion/
├── chunkers/
│   └── recursive_chunker.py
└── parsers/
    └── icmr_mohfw_parser.py
```
