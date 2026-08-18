# Healthcare RAG System — Documentation Index

This directory contains the full changelog and design decisions for each
iteration of the Healthcare RAG system.

---

## Documents

| File | Covers |
|---|---|
| [setup_and_usage.md](setup_and_usage.md) | Installation, environment setup, running the system |
| [structured_db_schema_generated.md](structured_db_schema_generated.md) | PostgreSQL schema for the structured data layer |
| [iteration-1.md](iteration-1.md) | Baseline: NHP + PubMed ingestion, ChromaDB, simple cosine retrieval |
| [iteration-2.md](iteration-2.md) | ICMR/MoHFW ingestion, hybrid search (BM25+dense), cross-encoder reranking, metadata filtering |
| [iteration-3.md](iteration-3.md) | Structured data layer (PostgreSQL), faithfulness verification, relevance threshold, pipeline configs |
| [iteration-4.md](iteration-4.md) | Query rewriting layer, OpenRouter evaluator, all 4 RAGAS metrics, `--metrics` + `--pipeline` CLI flags, metric bug fixes |

---

## Quick-start Commands

### Run the pipeline (generate logged results)
```bash
# Standard RAG
python3.11 -m eval.eval_runner.run_pipeline --pipeline standard_rag

# RAG++
python3.11 -m eval.eval_runner.run_pipeline --pipeline rag_plus_plus
```

### RAGAS Evaluation
```bash
# All metrics, both pipelines, NVIDIA backend
python3.11 -m eval.ragas_eval.run_ragas --evaluator nvidia --sample 10 --delay 15

# Standard RAG only
python3.11 -m eval.ragas_eval.run_ragas --evaluator nvidia --pipeline standard_rag --sample 10 --delay 15

# RAG++ only
python3.11 -m eval.ragas_eval.run_ragas --evaluator nvidia --pipeline rag_plus_plus --sample 10 --delay 15

# Specific metrics only
python3.11 -m eval.ragas_eval.run_ragas --evaluator nvidia --sample 10 --delay 15 \
  --metrics faithfulness context_recall

# Custom metrics only — zero API calls
python3.11 -m eval.ragas_eval.run_ragas --skip-ragas \
  --metrics correct_refusal_rate faithfulness_check_pass_rate

# OpenRouter backend
python3.11 -m eval.ragas_eval.run_ragas --evaluator openrouter --delay 3 --sample 5
```

### Query Rewriting Smoke Test
```bash
python3.11 -m eval.query_rewriting.test_query_rewriting
```

### Structured Retrieval Test
```bash
python3.11 -m eval.structured.test_structured_retrieval
```

### Build Entity Cache (run once after DB is loaded)
```bash
python3.11 -m retrieval.structured.build_entity_cache
```

---

## Architecture Overview

```
Question
  │
  ├─ [Query Rewriting Layer]  (RAG++ only)
  │     ├─ Heuristic pre-filter (no LLM)
  │     └─ LLM rewrite + decompose → sub-questions
  │
  ├─ [Structured Lookup]  (RAG++ only)
  │     ├─ Entity cache heuristic
  │     └─ LLM SQL → PostgreSQL (janaushadhi / herb / labtesttype)
  │
  ├─ [Unstructured Retrieval]
  │     ├─ Metadata filter (optional)
  │     ├─ Hybrid search: BM25 + dense (BGE-M3) → RRF fusion
  │     └─ Cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
  │
  ├─ [Relevance Threshold Gate]  (RAG++ only)
  │     └─ fallback if max_score < 0.30
  │
  ├─ [Prompt Construction]
  │     ├─ build_prompt()                  (standard)
  │     └─ build_prompt_from_sub_results() (decomposed)
  │
  └─ [LLM Generation]
        └─ medgemma / configurable model


Offline Evaluation (RAGAS)
  ├─ Metrics: faithfulness, context_precision, context_recall, answer_relevancy
  ├─ Custom:  correct_refusal_rate, faithfulness_check_pass_rate
  ├─ Backends: ollama | nvidia (llama-3.1-70b) | openrouter (gemma-2-9b)
  ├─ --pipeline: standard_rag | rag_plus_plus | both
  └─ --metrics:  pick any subset of the 6 metrics
```

---

## Model Role Separation

> No model serves two roles. This is enforced by design.

| Role | Model |
|---|---|
| Answer generation | `mistral:7b` (or configurable LLM) |
| Faithfulness verification | `qwen3:8b` via Ollama |
| Query rewriting / SQL generation | `qwen3:8b` via Ollama |
| RAGAS evaluation (NVIDIA) | `meta/llama-3.1-70b-instruct` |
| RAGAS evaluation (OpenRouter) | `google/gemma-2-9b-it:free` |
| RAGAS evaluation (Ollama) | local model via `OLLAMA_EVAL_MODEL` |
| Embeddings (ingestion + retrieval) | `BAAI/bge-m3` (1024-dim) |
| Embeddings (answer_relevancy) | `nvidia/nv-embed-v1` (NVIDIA backend only) |
