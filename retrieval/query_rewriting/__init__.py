"""
Query Rewriting Layer
=====================

Pre-retrieval layer for the RAG++ pipeline.

For simple questions: rewrites the query for clarity and passes it through
directly (heuristic pre-filter, no LLM call).

For complex / compound questions: rewrites + decomposes into self-contained
sub-questions, then runs the full retrieval pipeline once per sub-question.

Modules:
    prefilter                 -- heuristic check: skip decomposition for simple questions
    decompose                 -- LLM rewrite + decompose via Ollama (single combined call)
    run_decomposed_retrieval  -- per-sub-question retrieval loop
"""
