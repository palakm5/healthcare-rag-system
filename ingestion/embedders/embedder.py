"""
Embedder wrapper around NVIDIA's hosted BAAI/bge-m3 embedding model.

This module sends text to NVIDIA's Embeddings API and returns
normalized embedding vectors as NumPy arrays.

The interface matches the previous local embedder so the rest of
the RAG pipeline does not need to change.
"""

import logging
import os
from typing import List

import numpy as np
import requests

from config.settings import settings

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wrapper around NVIDIA Embeddings API.

    Usage:
        embedder = Embedder()

        passage_embeddings = embedder.embed_passages([
            "chunk 1",
            "chunk 2"
        ])

        query_embedding = embedder.embed_query(
            "What are the symptoms of tuberculosis?"
        )
    """

    def __init__(self):
        self.api_key = settings.NVIDIA_API_KEY or os.getenv("NVIDIA_API_KEY")

        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY not found. "
                "Set it in your environment or settings.py."
            )

        self.base_url = settings.NVIDIA_BASE_URL.rstrip("/")
        self.model = settings.EMBEDDING_MODEL_NAME
        self.batch_size = settings.EMBEDDING_BATCH_SIZE

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "Initialized NVIDIA Embedder (model=%s)",
            self.model,
        )

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Internal helper for embedding a batch of texts.
        """

        if not texts:
            return np.array([])

        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }

        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        result = response.json()

        embeddings = [
            item["embedding"]
            for item in result["data"]
        ]

        return np.array(embeddings, dtype=np.float32)

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        """
        Embed document chunks.
        """

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed(batch)
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        """

        return self._embed([query])[0]

    @property
    def dimension(self) -> int:
        """
        Return embedding dimension.
        """

        return settings.EMBEDDING_DIM