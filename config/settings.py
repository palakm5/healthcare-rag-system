"""
Central configuration for the Healthcare RAG system.
All paths, model names, chunking parameters, and collection settings
are defined here so individual modules can import them cleanly.
"""
from dotenv import load_dotenv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

load_dotenv()

@dataclass
class Settings:
    # ── Paths ──────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR: Path = PROJECT_ROOT / "unstructured-data" / "raw"
    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "unstructured-data" / "processed"
    VECTOR_STORE_DIR: Path = PROJECT_ROOT / "vector_store"

    # Source sub-directories
    NHP_DIR: Path = RAW_DATA_DIR / "nhp"
    PUBMED_DIR: Path = RAW_DATA_DIR / "pubmed"

    # ── Chunking ───────────────────────────────────────────────────────
    NHP_CHUNK_SIZE: int = 500       # tokens (approximate, via tiktoken)
    NHP_CHUNK_OVERLAP: int = 50     # tokens
    PUBMED_MAX_SECTION_TOKENS: int = 1000  # split sections longer than this

    # ── Embedding ─────────────────────────────────────────

    EMBEDDING_PROVIDER = "nvidia"
    EMBEDDING_MODEL_NAME = "baai/bge-m3"
    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    NVIDIA_API_KEY = None
    EMBEDDING_DIM = 1024
    EMBEDDING_BATCH_SIZE = 32
    # ── ChromaDB ───────────────────────────────────────────────────────
    CHROMA_NHP_COLLECTION: str = "nhp_collection"
    CHROMA_PUBMED_COLLECTION: str = "pubmed_collection"

    # ── Retrieval ──────────────────────────────────────────────────────
    TOP_K: int = 5

    # ── Generation (pluggable) ─────────────────────────────────────────
    # Set to "openai", "ollama", "groq", or None (dry-run / prompt-only mode)
    LLM_PROVIDER: Optional[str] = None
    LLM_MODEL_NAME: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GROQ_API_KEY: Optional[str] = None

    # ── Logging ────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    def __post_init__(self):
        """Ensure directories exist."""
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


# Singleton instance for easy importing
settings = Settings()