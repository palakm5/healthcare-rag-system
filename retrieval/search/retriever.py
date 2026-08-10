"""
Retriever: embed query → cosine similarity search in ChromaDB across all sources.

Supports:
- Multi-collection search (NHP, PubMed, ICMR, MOHFW)
- Optional hybrid search (dense + sparse BM25 fusion)
- Optional cross-encoder reranking
- Optional metadata filtering (post-retrieval)
- Source-type filtering (search only specific collections)

No reranking, no hybrid search, no metadata filtering beyond optional source_type.
Returns top-k chunks with their metadata and similarity scores.
"""

import logging
from typing import Dict, List, Optional, Set

import chromadb

from config.settings import settings
from ingestion.embedders.embedder import Embedder

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant chunks from ChromaDB using cosine similarity.

    Supports all four source collections: NHP, PubMed, ICMR, MOHFW.

    Usage:
        retriever = Retriever()
        results = retriever.retrieve("What are the symptoms of tuberculosis?")
        # Optionally filter by source type:
        results = retriever.retrieve("...", source_types={"PubMed", "ICMR"})
        # Enable hybrid search:
        results = retriever.retrieve("...", use_hybrid=True)
        # Enable reranking:
        results = retriever.retrieve("...", use_rerank=True)
    """

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(settings.VECTOR_STORE_DIR),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        self._embedder = Embedder()

        # Load all collections
        self._collections = {}
        for name, coll_name in [
            ("NHP", settings.CHROMA_NHP_COLLECTION),
            ("PubMed", settings.CHROMA_PUBMED_COLLECTION),
            ("ICMR", settings.CHROMA_ICMR_COLLECTION),
            ("MOHFW", settings.CHROMA_MOHFW_COLLECTION),
        ]:
            coll = self._get_collection(coll_name)
            if coll is not None:
                self._collections[name] = coll

        # Log collection stats
        counts = {name: c.count() for name, c in self._collections.items()}
        logger.info(
            "Retriever ready. Collections: %s (total: %d chunks)",
            ", ".join(f"{k}={v}" for k, v in counts.items()),
            sum(counts.values()),
        )

        # Lazy-loaded components
        self._hybrid_searchers = {}
        self._reranker = None
        self._metadata_filter = None

    def _get_collection(self, name: str) -> Optional[chromadb.Collection]:
        """Get a collection by name, returning None if it doesn't exist."""
        try:
            return self._client.get_collection(name)
        except Exception:
            logger.warning(f"Collection '{name}' not found. Run ingestion first.")
            return None

    def _get_hybrid_searcher(self, source_name: str):
        """Lazy-load a HybridSearcher for a given source collection."""
        if source_name not in self._hybrid_searchers:
            from retrieval.search.hybrid_search import HybridSearcher
            coll = self._collections.get(source_name)
            if coll is None:
                return None
            self._hybrid_searchers[source_name] = HybridSearcher(coll, self._embedder)
        return self._hybrid_searchers[source_name]

    def _get_reranker(self):
        """Lazy-load the CrossEncoderReranker."""
        if self._reranker is None:
            from retrieval.rerank.cross_encoder_reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def _get_metadata_filter(self):
        """Lazy-load the MetadataFilter."""
        if self._metadata_filter is None:
            from retrieval.filters.metadata_filter import MetadataFilter
            self._metadata_filter = MetadataFilter()
        return self._metadata_filter

    def _get_candidate_limit(self, top_k: int) -> int:
        """
        Candidate pool size used before optional reranking.

        Falls back to 80 if settings.RERANK_CANDIDATES is not defined.
        """
        return max(top_k, int(getattr(settings, "RERANK_CANDIDATES", 80)))

    def _query_dense_collection(
        self,
        collection: chromadb.Collection,
        query_embedding,
        n_results: int,
    ) -> List[Dict]:
        """Query one collection using dense vector similarity search."""
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        rows: List[Dict] = []
        ids = results["ids"][0] if results.get("ids") else []
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i, doc_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            metadata = metas[i] if i < len(metas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            similarity = 1.0 - distance  # cosine similarity
            rows.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata or {},
                "score": round(similarity, 4),
            })
        return rows

    def _query_collection_candidates(
        self,
        src_label: str,
        collection: chromadb.Collection,
        query: str,
        query_embedding,
        candidate_limit: int,
        use_hybrid: bool,
    ) -> List[Dict]:
        """Query one collection and return candidate chunks."""
        if use_hybrid:
            searcher = self._get_hybrid_searcher(src_label)
            if searcher is None:
                return []
            return searcher.search(query, top_k=candidate_limit)

        return self._query_dense_collection(
            collection=collection,
            query_embedding=query_embedding,
            n_results=candidate_limit,
        )

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Drop duplicate chunks while keeping highest-scored instance."""
        best_by_key: Dict[str, Dict] = {}
        for row in results:
            metadata = row.get("metadata") or {}
            key = row.get("id") or (
                f"{metadata.get('source_type','')}|{metadata.get('title','')}|"
                f"{metadata.get('chunk_index','')}|{row.get('text','')}"
            )
            existing = best_by_key.get(key)
            if existing is None or row.get("score", 0.0) > existing.get("score", 0.0):
                best_by_key[key] = row
        return list(best_by_key.values())

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        source_types: Optional[Set[str]] = None,
        use_hybrid: bool = False,
        use_rerank: bool = False,
        metadata_filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Retrieve top-k chunks for a query across all source collections.

        Args:
            query: The user's question.
            top_k: Number of chunks to return (default: settings.TOP_K).
            source_types: If provided, only search these source types
                          (e.g. {"NHP", "ICMR"}). If None, searches all.
            use_hybrid: If True, use hybrid (dense + BM25) search per collection.
            use_rerank: If True, apply cross-encoder reranking after retrieval.
            metadata_filters: Optional dict of metadata filter criteria.
                              See MetadataFilter.filter() for supported keys.

        Returns:
            List of dicts, each with:
                - "id": str — chunk ID
                - "text": str — chunk content
                - "metadata": dict — source_type, title, section, chunk_index, etc.
                - "score": float — relevance score (higher = more relevant)
                - Additional score fields if hybrid/rerank enabled.
        """
        if top_k is None:
            top_k = settings.TOP_K
        candidate_limit = self._get_candidate_limit(top_k)

        # Determine which collections to search
        if source_types:
            collections_to_search = [
                (name, self._collections[name])
                for name in source_types
                if name in self._collections
            ]
        else:
            collections_to_search = list(self._collections.items())

        # Embed query once for dense retrieval. (Hybrid searchers may use query text.)
        query_embedding = self._embedder.embed_query(query)
        all_results: List[Dict] = []

        for src_label, collection in collections_to_search:
            if collection is None or collection.count() == 0:
                logger.info("Stage1 candidates from %s: 0 (empty/missing)", src_label)
                continue

            source_candidates = self._query_collection_candidates(
                src_label=src_label,
                collection=collection,
                query=query,
                query_embedding=query_embedding,
                candidate_limit=candidate_limit,
                use_hybrid=use_hybrid,
            )
            logger.info(
                "Stage1 candidates from %s: %d (requested up to %d)",
                src_label,
                len(source_candidates),
                candidate_limit,
            )
            all_results.extend(source_candidates)

        deduped_candidates = self._deduplicate_results(all_results)
        deduped_candidates.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        stage1_candidates = deduped_candidates[:candidate_limit]

        logger.info(
            "Stage1 merged candidates: %d raw -> %d deduped -> %d kept",
            len(all_results),
            len(deduped_candidates),
            len(stage1_candidates),
        )

        # Optional: metadata filtering
        if metadata_filters:
            mf = self._get_metadata_filter()
            stage1_candidates = mf.filter(stage1_candidates, **metadata_filters)

        # Optional: cross-encoder reranking
        if use_rerank and stage1_candidates:
            reranker = self._get_reranker()
            logger.info(
                "Stage2 reranker candidates: %d",
                len(stage1_candidates),
            )
            top_results = reranker.rerank(query, stage1_candidates, top_k=top_k)
        else:
            top_results = stage1_candidates[:top_k]

        logger.info(
            f"Retrieved {len(top_results)} chunks for query: '{query[:80]}...' "
            f"(sources={source_types or 'all'}, hybrid={use_hybrid}, "
            f"rerank={use_rerank}, top_k={top_k}, candidates={candidate_limit})"
        )
        for i, r in enumerate(top_results):
            logger.debug(
                f"  [{i+1}] score={r['score']:.4f} | "
                f"source={r['metadata'].get('source_type', '?')} | "
                f"title={r['metadata'].get('title', '?')[:50]} | "
                f"section={r['metadata'].get('section', '-')}"
            )

        return top_results