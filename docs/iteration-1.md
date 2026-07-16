# Healthcare RAG System - Iteration 1

## Overview

This document captures the key decisions made during the first iteration of building a modular retrieval-augmented generation (RAG) system for medical information.

## Key Decisions

### 1. Source Types
- **National Health Portal (NHP) .txt files**: Plain text files from nhp.gov.in
- **PubMed research paper PDFs**: Native-text PDFs with standard IMRaD structure

### 2. Parsing
- **NHP Parser**:
  - Strips metadata headers (SOURCE, TITLE, SOURCE_TAG, SCRAPED)
  - Removes MENU navigation blocks
  - Strips footer boilerplate (Related Pages, CREATED/VALIDATED BY, Discussion)
  - Extracts document title from header

- **PubMed Parser**:
  - Uses Unstructured.io for PDF parsing
  - Detects standard sections: Abstract, Introduction, Methods, Results, Discussion, Conclusions
  - Strips non-clinical sections: References, Acknowledgments, Author Contributions
  - Extracts title from first page

### 3. Chunking
- **NHP Chunking**:
  - Fixed-size sliding window (500 tokens with 50-token overlap)
  - Uses tiktoken for token-aware splitting
  - No section detection in v1

- **PubMed Chunking**:
  - Section-aware: one chunk per section where possible
  - Splits long sections (>1000 tokens) into sub-chunks with overlap
  - Preserves section headings in metadata

### 4. Embedding
- **Model**: BGE-M3 (`BAAI/bge-m3`, 1024-dim)
  - Asymmetric instructions:
    - Passages: "Represent this document for retrieval: <text>"
    - Queries: "Represent this question for retrieving supporting documents: <query>"
  - Normalized embeddings for cosine similarity

### 5. Vector Store
- **ChromaDB**:
  - Two collections: `nhp_collection` and `pubmed_collection`
  - Cosine distance metric
  - Persistent storage in `vector_store/` directory

### 6. Retrieval
- **Simple cosine similarity search**:
  - No reranking
  - No hybrid search
  - Optional source filtering (NHP or PubMed)
  - Returns top-k chunks with metadata and similarity scores

### 7. Generation
- **Prompt Construction**:
  - System instruction: Answer only from provided context, cite sources
  - Context blocks with source attribution: [Source X: type | title | section]
  - Question-answer format

- **LLM Integration**:
  - Pluggable LLM client (OpenAI, Ollama, Groq, etc.)
  - Dry-run mode: returns prompt + sources without calling LLM

### 8. File Structure
```
healthcare-rag-system/
├── config/
│   └── settings.py          # Central configuration
├── docs/
│   └── iteration-1.md       # This document
├── generation/
│   ├── prompts/
│   │   └── builder.py       # RAG prompt construction
│   └── generator.py         # Orchestrates retrieval and generation
├── ingestion/
│   ├── chunkers/
│   │   ├── nhp_chunker.py   # Fixed-size chunking for NHP
│   │   └── pubmed_chunker.py # Section-aware chunking for PubMed
│   ├── embedders/
│   │   └── embedder.py      # BGE-M3 embedding wrapper
│   ├── parsers/
│   │   ├── nhp_parser.py    # NHP .txt file parser
│   │   └── pubmed_parser.py # PubMed PDF parser
│   └── ingest_pipeline.py   # Ingestion orchestrator
├── retrieval/
│   └── search/
│       └── retriever.py     # Cosine similarity search
├── requirements.txt         # Dependencies
└── vector_store/            # ChromaDB collections
```

### 9. Dependencies
- `unstructured` (PDF parsing)
- `chromadb` (vector store)
- `sentence-transformers` (embedding)
- `tiktoken` (tokenization)
- `pypdf` (fallback PDF parsing)
- `python-dotenv` (environment variables)

## Next Steps

1. Run the ingestion pipeline end-to-end and verify
2. Test the retrieval and generation with sample queries
3. Add evaluation metrics (e.g., faithfulness, relevance)
4. Implement safety guardrails
5. Add section filtering for PubMed retrieval
6. Implement hybrid search (sparse + dense)
7. Add reranking
8. Extend to more source types