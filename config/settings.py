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
    ICMR_DIR: Path = RAW_DATA_DIR / "icmr"
    MOHFW_DIR: Path = RAW_DATA_DIR / "mohfw"

    # Processed output sub-directories (JSON per document)
    ICMR_PROCESSED_DIR: Path = PROCESSED_DATA_DIR / "icmr"
    MOHFW_PROCESSED_DIR: Path = PROCESSED_DATA_DIR / "mohfw"

    # ── Chunking ───────────────────────────────────────────────────────
    NHP_CHUNK_SIZE: int = 500       # tokens (approximate, via tiktoken)
    NHP_CHUNK_OVERLAP: int = 50     # tokens
    PUBMED_MAX_SECTION_TOKENS: int = 1000  # split sections longer than this

    # ICMR / MOHFW recursive chunking (character-based, not token-based)
    ICMR_CHUNK_SIZE: int = 1000     # characters
    ICMR_CHUNK_OVERLAP: int = 200  # characters
    MOHFW_CHUNK_SIZE: int = 1000   # characters
    MOHFW_CHUNK_OVERLAP: int = 200 # characters

    # MOHFW front-matter page skip config — per-document (zero-indexed pages).
    # Keys are PDF filenames; values are lists of zero-indexed pages to skip.
    # Add entries here as you identify front-matter for each MOHFW document.
    MOHFW_SKIP_PAGES: dict = field(default_factory=lambda: {
        # Example: "Dengue_Clinical_Management_Guidelines_2023.pdf": [0, 1, 2],
    })

    # ── Embedding (still uses NVIDIA hosted API for nv-embed-v1) ─────────
    EMBEDDING_PROVIDER = "nvidia"
    EMBEDDING_MODEL_NAME = "nvidia/nv-embed-v1"
    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    NVIDIA_API_KEY = None
    EMBEDDING_DIM = 4096

    EMBEDDING_BATCH_SIZE = 16
    # ── ChromaDB ───────────────────────────────────────────────────────
    CHROMA_NHP_COLLECTION: str = "nhp_collection"
    CHROMA_PUBMED_COLLECTION: str = "pubmed_collection"
    CHROMA_ICMR_COLLECTION: str = "icmr_collection"
    CHROMA_MOHFW_COLLECTION: str = "mohfw_collection"

    # ── Retrieval ──────────────────────────────────────────────────────
    TOP_K: int = 5

    # ── Generation (pluggable) ─────────────────────────────────────────
    # Set to "openai", "ollama", "groq", or None (dry-run / prompt-only mode)
    LLM_PROVIDER: Optional[str] = None
    LLM_MODEL_NAME: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GROQ_API_KEY: Optional[str] = None

    # ── OpenRouter (faithfulness verifier + RAGAS evaluator) ───────────
    # LLM calls (faithfulness check, RAGAS eval) use OpenRouter.
    # Browse free models: https://openrouter.ai/models?q=:free
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # ── Logging ────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    def __post_init__(self):
        """Ensure directories exist."""
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


# Singleton instance for easy importing
settings = Settings()