# Healthcare RAG System — Iteration 3

## Overview

Iteration 3 added a structured data layer (PostgreSQL), a faithfulness verification
step in generation, a relevance-threshold fallback gate, and defined two distinct
pipeline configurations: Standard RAG and RAG++.

---

## 1. Structured Data Layer

### PostgreSQL Database
Three tables loaded from curated CSVs:

| Table | Content |
|---|---|
| `janaushadhi` | Jan Aushadhi drug catalogue — generic names, MRP, pack size |
| `herb` | Ayurvedic herb database — Latin name, Rasa, Virya, Vipaka, Guna, actions |
| `labtesttype` | Lab test reference — test name, unit, category, normal range |

### Entity Cache — `retrieval/structured/build_entity_cache.py`
- Pre-builds in-memory lookup sets of known drug names, herb names, and lab test names
- Run once: `python -m retrieval.structured.build_entity_cache`
- Output stored in `data/entity_cache.json`
- Used by the structured lookup to decide cheaply whether a query mentions a known entity

### Structured Lookup — `retrieval/structured/structured_lookup.py`
Two-stage lookup:
1. **Heuristic match**: checks entity cache; if no known entity found, returns immediately (no DB hit)
2. **LLM SQL generation** (`retrieval/structured/llm_sql_generator.py`): `qwen3:8b` converts the question to a parameterised SQL query against the appropriate table
3. Results are returned as structured context blocks, formatted identically to unstructured chunks for the prompt builder

---

## 2. Faithfulness Verification

**File**: `generation/faithfullness_check/verifier.py`

Post-generation step that checks whether each claim in the LLM answer is
supported by the retrieved context.

| Property | Detail |
|---|---|
| Model | `qwen3:8b` via Ollama (separate role from answer generator) |
| Method | Enumerates all factual claims in the answer; verifies each against context |
| Output | `FaithfulnessVerdict`: `overall_faithful: bool`, per-claim breakdown |
| On failure | Answer is discarded; pipeline returns a "cannot verify" response |
| Configurable | Enabled/disabled per pipeline via `faithfulness_check` in `pipeline_config.py` |

---

## 3. Relevance Threshold Gate

**Files**: `retrieval/threshold/relevance_threshold.py`, `retrieval/threshold/fallback.py`

Applied after retrieval, before generation:

- If the **maximum rerank score** across all retrieved chunks falls below `relevance_threshold` (default `0.30`), the pipeline triggers a fallback instead of generating an answer
- Fallback response is a templated "out of scope" message logged with `fallback_triggered: True`
- Prevents hallucination on questions the knowledge base cannot answer
- Calibration script: `eval/calibrate_threshold.py`

---

## 4. Pipeline Configurations

**File**: `config/pipeline_config.py`

Two named configs; both passed to `Generator.generate()` at call time:

### `STANDARD_RAG_CONFIG`
```python
{
    "faithfulness_check":        False,
    "structured_data_enabled":   False,
    "query_rewriting_enabled":   False,
}
```
Baseline — vector retrieval + generation only. No structured lookup, no faithfulness check, no query rewriting.

### `RAG_PLUS_PLUS_CONFIG`
```python
{
    "faithfulness_check":        False,   # disabled after iteration 4
    "structured_data_enabled":   True,
    "query_rewriting_enabled":   True,
    "relevance_threshold":       0.30,
}
```
Full pipeline — all layers enabled.

> **Model role separation rule** (enforced in code):
> - Answer generation: `medgemma` or configurable LLM via `generation/llm/`
> - Faithfulness verification: `qwen3:8b` (Ollama)
> - Query rewriting / SQL generation: `qwen3:8b` (Ollama)
> - RAGAS evaluation: separate evaluator model (NVIDIA / OpenRouter / Ollama)
> No model serves two roles.

---

## 5. Generator Upgrade

**File**: `generation/generator.py`

The `Generator.generate()` method now:
1. Optionally calls `structured_lookup` (if `structured_data_enabled`)
2. Calls `Retriever` for unstructured chunks
3. Applies relevance threshold gate
4. Builds prompt via `generation/prompts/builder.py`
5. Calls LLM
6. Optionally runs faithfulness check (if `faithfulness_check`)
7. Returns `GenerationResult` dataclass with full provenance

---

## 6. Updated File Structure

```
config/
└── pipeline_config.py          # STANDARD_RAG_CONFIG, RAG_PLUS_PLUS_CONFIG
generation/
└── faithfullness_check/
    └── verifier.py             # Post-generation faithfulness checker
retrieval/
├── structured/
│   ├── build_entity_cache.py
│   ├── llm_sql_generator.py
│   ├── query_templates.py
│   └── structured_lookup.py
└── threshold/
    ├── fallback.py
    └── relevance_threshold.py
eval/
└── calibrate_threshold.py      # Threshold calibration script
```
