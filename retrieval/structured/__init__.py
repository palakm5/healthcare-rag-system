"""
Structured SQL retrieval package for the Healthcare RAG system.

This package adds structured database retrieval alongside existing
unstructured chunk retrieval (Chroma vector store). The two paths
are always additive -- existing retrieval logic is never modified.

Public API:
    from retrieval.structured.structured_lookup import StructuredLookup

Setup (one-time, requires DATABASE_URL in .env):
    python -m retrieval.structured.build_entity_cache
"""
