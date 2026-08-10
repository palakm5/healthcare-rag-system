"""
Hybrid search: combines dense (embedding) similarity with sparse (keyword) search.

Dense search uses the existing ChromaDB embedding collection.
Sparse search uses a simple BM25-like keyword scoring over chunk texts.
The two score streams are normalized and fused via a weighted combination
(default: 0.7 dense, 0.3 sparse).

This module is designed to be called by Retriever, not used standalone.
"""

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import chromadb

from config.settings import settings
from ingestion.embedders.embedder import Embedder

logger = logging.getLogger(__name__)

# Simple tokenizer: lowercase, split on non-alphanumeric
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase and tokenize on word boundaries."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """
    Minimal in-memory BM25 index over a set of documents.

    This is intentionally lightweight — no external dependency (no rank-bm25).
    Suitable for the corpus sizes in this project (thousands of chunks).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.df: Counter = Counter()  # document frequency per term
        self.idf: Dict[str, float] = {}
        self.tf: List[Counter] = []  # term frequency per doc
        self.n_docs: int = 0

    def add_documents(self, doc_ids: List[str], texts: List[str]) -> None:
        """Build the BM25 index from a set of documents."""
        self.doc_ids = doc_ids
        self.doc_tokens = [_tokenize(t) for t in texts]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.n_docs = len(self.doc_ids)
        self.avgdl = sum(self.doc_len) / self.n_docs if self.n_docs else 0.0

        self.tf = [Counter(tokens) for tokens in self.doc_tokens]
        for tf_counter in self.tf:
            for term in tf_counter:
                self.df[term] += 1

        # Precompute IDF (BM25 variant)
        for term, df in self.df.items():
            self.idf[term] = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Score documents against the query using BM25.

        Returns list of (doc_id, score) sorted descending, limited to top_k.
        """
        if self.n_docs == 0:
            return []

        query_terms = _tokenize(query)
        scores = [0.0] * self.n_docs

        for term in query_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i in range(self.n_docs):
                tf = self.tf[i].get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * (self.doc_len[i] / max(self.avgdl, 1.0))
                )
                scores[i] += idf * (tf * (self.k1 + 1)) / denom

        ranked = sorted(
            zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]


class HybridSearcher:
    """
    Hybrid search combining dense (ChromaDB embedding) and sparse (BM25) retrieval.

    Usage:
        searcher = HybridSearcher(collection, embedder)
        results = searcher.search("query text", top_k=10)
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        embedder: Embedder,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        self.collection = collection
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Build BM25 index from all documents in the collection
        self.bm25 = BM25Index()
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Fetch all documents from the collection and build a BM25 index."""
        if self.collection is None or self.collection.count() == 0:
            logger.warning("Cannot build BM25 index: collection empty or missing.")
            return

        # Fetch all docs (Chroma supports get() with include)
        all_data = self.collection.get(include=["documents", "metadatas"])
        doc_ids = all_data["ids"]
        texts = all_data["documents"] or [""] * len(doc_ids)

        self.bm25.add_documents(doc_ids, texts)
        logger.info(
            "BM25 index built for collection '%s' with %d documents.",
            getattr(self.collection, "name", "?"),
            self.bm25.n_docs,
        )

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        """Min-max normalize scores to [0, 1]."""
        if not scores:
            return []
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            return [1.0] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]

    def search(
        self,
        query: str,
        top_k: int = 10,
        prefilter_top_k: int = 50,
    ) -> List[Dict]:
        """
        Run hybrid search and return fused results.

        Args:
            query: User query text.
            top_k: Final number of results to return.
            prefilter_top_k: How many candidates to fetch from each retriever
                             before fusion (should be >= top_k).

        Returns:
            List of dicts with keys: id, text, metadata, score (fused), 
            dense_score, sparse_score.
        """
        if self.collection is None or self.collection.count() == 0:
            return []

        # --- Dense search ---
        query_embedding = self.embedder.embed_query(query)
        dense_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=prefilter_top_k,
            include=["documents", "metadatas", "distances"],
        )

        dense_map: Dict[str, Dict] = {}
        for i in range(len(dense_results["ids"][0])):
            doc_id = dense_results["ids"][0][i]
            text = dense_results["documents"][0][i] if dense_results["documents"] else ""
            metadata = dense_results["metadatas"][0][i] if dense_results["metadatas"] else {}
            distance = dense_results["distances"][0][i]
            similarity = 1.0 - distance
            dense_map[doc_id] = {
                "id": doc_id,
                "text": text,
                "metadata": metadata,
                "dense_score": similarity,
            }

        # --- Sparse search ---
        sparse_results = self.bm25.search(query, top_k=prefilter_top_k)
        sparse_map: Dict[str, float] = {doc_id: score for doc_id, score in sparse_results}

        # --- Fuse ---
        all_ids = set(dense_map.keys()) | set(sparse_map.keys())
        if not all_ids:
            return []

        dense_scores = [dense_map.get(did, {}).get("dense_score", 0.0) for did in all_ids]
        sparse_scores = [sparse_map.get(did, 0.0) for did in all_ids]

        norm_dense = self._normalize_scores(dense_scores)
        norm_sparse = self._normalize_scores(sparse_scores)

        id_list = list(all_ids)
        fused_results = []
        for idx, doc_id in enumerate(id_list):
            d_score = norm_dense[idx]
            s_score = norm_sparse[idx]
            fused = self.dense_weight * d_score + self.sparse_weight * s_score

            entry = dense_map.get(doc_id, {"id": doc_id, "text": "", "metadata": {}})
            fused_results.append({
                "id": doc_id,
                "text": entry.get("text", ""),
                "metadata": entry.get("metadata", {}),
                "score": round(fused, 4),
                "dense_score": round(d_score, 4),
                "sparse_score": round(s_score, 4),
            })

        fused_results.sort(key=lambda r: r["score"], reverse=True)
        return fused_results[:top_k]