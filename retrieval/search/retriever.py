"""
Simple retriever: embed query → cosine similarity search in ChromaDB.

No reranking, no hybrid search, no metadata filtering beyond optional source_type.
Returns top-k chunks with their metadata and similarity scores.
"""

import logging
from typing import Dict, List, Optional

import chromadb

from config.settings import settings
from ingestion.embedders.embedder import Embedder

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant chunks from ChromaDB using cosine similarity.

    Usage:
        retriever = Retriever()
        results = retriever.retrieve("What are the symptoms of tuberculosis?")
        # Optionally filter by source type:
        results = retriever.retrieve("...", source_type="PubMed")
    """

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(settings.VECTOR_STORE_DIR),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        self._embedder = Embedder()

        # Load both collections
        self._nhp_collection = self._get_collection(settings.CHROMA_NHP_COLLECTION)
        self._pubmed_collection = self._get_collection(settings.CHROMA_PUBMED_COLLECTION)

        nhp_count = self._nhp_collection.count() if self._nhp_collection else 0
        pubmed_count = self._pubmed_collection.count() if self._pubmed_collection else 0
        logger.info(f"Retriever ready. NHP: {nhp_count} chunks, PubMed: {pubmed_count} chunks")

    def _get_collection(self, name: str) -> Optional[chromadb.Collection]:
        """Get a collection by name, returning None if it doesn't exist."""
        try:
            return self._client.get_collection(name)
        except Exception:
            logger.warning(f"Collection '{name}' not found. Run ingestion first.")
            return None

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        source_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve top-k chunks for a query.

        Args:
            query: The user's question.
            top_k: Number of chunks to return (default: settings.TOP_K).
            source_type: If "NHP" or "PubMed", search only that collection.
                         If None, searches both and merges results.

        Returns:
            List of dicts, each with:
                - "text": str — chunk content
                - "metadata": dict — source_type, title, section, chunk_index, etc.
                - "score": float — cosine similarity (higher = more relevant)
        """
        if top_k is None:
            top_k = settings.TOP_K

        query_embedding = self._embedder.embed_query(query)

        all_results = []

        # Determine which collections to search
        if source_type and source_type.lower() == "nhp":
            collections = [("NHP", self._nhp_collection)]
        elif source_type and source_type.lower() == "pubmed":
            collections = [("PubMed", self._pubmed_collection)]
        else:
            collections = [
                ("NHP", self._nhp_collection),
                ("PubMed", self._pubmed_collection),
            ]

        for src_label, collection in collections:
            if collection is None or collection.count() == 0:
                continue

            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Chroma returns distances (lower = more similar for cosine).
            # Convert to similarity scores: similarity = 1 - distance
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                text = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i]
                similarity = 1.0 - distance  # cosine similarity

                all_results.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata,
                    "score": round(similarity, 4),
                })

        # Sort by similarity descending, take top_k
        all_results.sort(key=lambda r: r["score"], reverse=True)
        top_results = all_results[:top_k]

        logger.info(
            f"Retrieved {len(top_results)} chunks for query: '{query[:80]}...' "
            f"(source_filter={source_type or 'all'})"
        )
        for i, r in enumerate(top_results):
            logger.debug(
                f"  [{i+1}] score={r['score']:.4f} | "
                f"source={r['metadata'].get('source_type', '?')} | "
                f"title={r['metadata'].get('title', '?')[:50]} | "
                f"section={r['metadata'].get('section', '-')}"
            )

        return top_results