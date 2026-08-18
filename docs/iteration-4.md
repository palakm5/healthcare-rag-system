# Healthcare RAG System — Iteration 4

## Overview

Iteration 4 added a **Query Rewriting Layer** to the RAG++ pipeline,
integrated an **OpenRouter evaluator** backend, **disabled faithfulness_check**
in RAG++ to reduce latency, fixed several evaluation metric bugs, and introduced
a **per-metric CLI flag** and a **per-pipeline CLI flag** for the RAGAS script.

---

## 1. Query Rewriting Layer

A new pre-retrieval layer added exclusively to the RAG++ pipeline. The primary
concept is **query rewriting**: every question is rewritten for clarity and
search quality. For complex or compound questions, the rewriter additionally
**decomposes** the question into self-contained sub-questions, each of which
runs through the full retrieval pipeline independently.

### Architecture

```
Original question
  → [Heuristic pre-filter]  ← cheap, no LLM
       │
       ├─ Simple question → pass through unchanged (1 sub-question)
       │
       └─ Complex question → [LLM rewrite + decompose]
                                  │
                            [sub-question 1] → full retrieval pipeline
                            [sub-question 2] → full retrieval pipeline
                            [sub-question N] → full retrieval pipeline
                                  │
                            Grouped prompt construction
                                  │
                            Single LLM generation call
```

### Files Created

#### `retrieval/query_rewriting/prefilter.py`
Heuristic pre-filter — **zero LLM calls**.

A question is classified as *simple* (skip rewriting) if ALL hold:
1. Word count ≤ `SIMPLE_WORD_THRESHOLD` (default 15)
2. No compound-question indicators found (`and`, `also`, `compare`, `versus`, `both`, `difference between`, etc.)
3. At most 1 question mark

Public API:
- `should_skip_decomposition(question: str) → bool`
- `prefilter_result(question: str) → dict` (for logging/debugging)

#### `retrieval/query_rewriting/decompose.py`
Single LLM call that rewrites and optionally decomposes the question.

| Property | Detail |
|---|---|
| Model | `qwen3:8b` via Ollama (`DECOMPOSITION_MODEL` env var) |
| Output format | Strict JSON `{"sub_questions": ["...", ...]}` |
| Parse retry | 1 retry on JSON parse failure; hard fallback to original question |
| Cap | Max `MAX_SUB_QUESTIONS = 4` (truncated with warning if exceeded) |
| Logging | Every input/output pair appended to `logs/query_rewriting_log.jsonl` |
| `<think>` stripping | Removes qwen3 chain-of-thought blocks before JSON parsing |

#### `retrieval/query_rewriting/run_decomposed_retrieval.py`
Per-sub-question retrieval loop.

- Reuses the same `Retriever` instance across all sub-questions (avoids reconnection overhead)
- Returns a list of `SubQuestionResult(sub_question, chunks, fallback_triggered)`
- Non-aborting: if retrieval fails for one sub-question, logs a warning and continues

### Prompt Construction — `generation/prompts/builder.py`

Added `build_prompt_from_sub_results(sub_results, question)` alongside the
existing `build_prompt()`. Evidence blocks are grouped and labelled by sub-question:

```
[Sub-question 1: What is the mechanism of action of Metformin?]
[Source 1: icmr | STW Diabetes Type 2 | Drug Dosage]
...

[Sub-question 2: What are the major adverse drug reactions of Metformin?]
[Source 3: icmr | STW Diabetes Type 2 | Adverse Effects]
...
```

The original `build_prompt()` is unchanged — the standard RAG path is unaffected.

### Generator Integration — `generation/generator.py`

When `query_rewriting_enabled: True` in the pipeline config:
1. Pre-filter runs (`prefilter_result`)
2. If simple → single-question path (original code, unchanged)
3. If complex → `_run_decomposed()` private method:
   - Calls `decompose_question()`
   - Calls `run_decomposed_retrieval()`
   - Calls `build_prompt_from_sub_results()`
   - Single LLM generation

### Config Changes — `config/pipeline_config.py`

```python
STANDARD_RAG_CONFIG = {
    ...
    "query_rewriting_enabled": False,   # baseline unchanged
}

RAG_PLUS_PLUS_CONFIG = {
    ...
    "query_rewriting_enabled": True,    # new layer enabled
    "faithfulness_check":      False,   # disabled (see §2)
}
```

---

## 2. Faithfulness Check Disabled in RAG++

`faithfulness_check` was set to `False` in `RAG_PLUS_PLUS_CONFIG`.

**Reason**: the faithfulness verifier (`qwen3:8b`) ran after every generation
call, roughly doubling end-to-end latency. Its signal is better captured by
the RAGAS `faithfulness` metric during offline evaluation. Removing it from the
live pipeline makes logged results faster to generate without sacrificing
offline quality measurement.

The verifier code (`generation/faithfullness_check/verifier.py`) is intact and
can be re-enabled by flipping the config flag.

---

## 3. RAGAS Evaluation — OpenRouter Backend

**File**: `eval/ragas_eval/evaluators/openrouter_evaluator.py`

A third pluggable backend alongside `ollama` and `nvidia`.

| Property | Detail |
|---|---|
| API | OpenRouter (`https://openrouter.ai/api/v1`) — OpenAI-compatible |
| LLM | `google/gemma-2-9b-it:free` (configurable via `OPENROUTER_MODEL`) |
| Auth | `OPENROUTER_API_KEY` in `.env` |
| Embeddings | Not available — `answer_relevancy` skipped automatically |
| Rate limiting | Same async `asyncio.sleep` pattern as the NVIDIA backend |

```bash
python3.11 -m eval.ragas_eval.run_ragas --evaluator openrouter --delay 3 --sample 5
```

---

## 4. RAGAS Evaluation — NVIDIA Backend Fix

**File**: `eval/ragas_eval/evaluators/nvidia_evaluator.py`

**Problem**: `nvidia/nemotron-3.5-lightning-30b-a3b` was silently truncating
RAGAS JSON verdict responses at its model-side output limit, causing
`LLMDidNotFinishException` on ~70% of records.

**Fix**: Switched to `meta/llama-3.1-70b-instruct`, which reliably produces
complete 4096-token responses needed for RAGAS faithfulness verdicts (which
enumerate every factual claim as a JSON array).

---

## 5. All 4 RAGAS Metrics Restored

**File**: `eval/ragas_eval/run_ragas.py`

Previously only `faithfulness` was active. All 4 metrics are now enabled:

| Metric | LLM needed | Embeddings needed |
|---|---|---|
| `faithfulness` | ✅ | ❌ |
| `context_precision` | ✅ | ❌ |
| `context_recall` | ✅ | ❌ |
| `answer_relevancy` | ✅ | ✅ (nvidia only) |

`answer_relevancy` is conditionally added only when an embeddings wrapper is
available (NVIDIA backend provides `nvidia/nv-embed-v1`; OpenRouter and Ollama
do not).

---

## 6. Custom Metric Bug Fixes

**File**: `eval/ragas_eval/run_ragas.py` — `compute_custom_metrics()`

### Bug 1 — `correct_refusal_rate` always 0%
**Cause**: metric only counted records where `category == "unanswerable"`. After
the test question update (§8), there are no longer any `"unanswerable"` records —
only `"answerable"` and `"complex_answerable"`.

**Fix**: metric now counts **any record where `category != "answerable"`** as a
non-answerable record and checks whether `fallback_triggered == True`. Returns
`N/A` when no non-answerable records exist in the test set.

### Bug 2 — `faithfulness_check_pass_rate` always 0%
**Cause**: metric read `faithfulness_check_passed` — a field that was never
written by the pipeline. The pipeline writes `faithfulness_check_failed`.

**Fix**: metric now reads `faithfulness_check_failed` and inverts it. Records
where `faithfulness_verdict is None` (check disabled) are excluded from the
denominator. Returns `N/A -- faithfulness_check disabled` when the check is off.

---

## 7. Per-Metric CLI Flag — `--metrics`

**File**: `eval/ragas_eval/run_ragas.py`

New `--metrics` flag selects any subset of the 6 metrics per run.
Only selected metrics are built, sent to the API, and appear in the report.

```bash
# All metrics (default)
python3.11 -m eval.ragas_eval.run_ragas --evaluator nvidia --sample 10 --delay 15

# Single metric — cheapest run
python3.11 -m eval.ragas_eval.run_ragas --evaluator nvidia --sample 10 --delay 15 \
  --metrics faithfulness

# Custom metrics only — zero API calls
python3.11 -m eval.ragas_eval.run_ragas --skip-ragas \
  --metrics correct_refusal_rate faithfulness_check_pass_rate

# Mixed
python3.11 -m eval.ragas_eval.run_ragas --evaluator nvidia --sample 10 --delay 15 \
  --metrics faithfulness context_recall correct_refusal_rate
```

All 6 selectable names:
```
RAGAS:  faithfulness  context_precision  context_recall  answer_relevancy
Custom: correct_refusal_rate  faithfulness_check_pass_rate
```

---

## 8. Per-Pipeline CLI Flag — `--pipeline`

**File**: `eval/ragas_eval/run_ragas.py`

New `--pipeline` flag controls which pipeline(s) are loaded and evaluated.

| Value | Effect |
|---|---|
| `both` *(default)* | Load and evaluate both; show side-by-side comparison |
| `standard_rag` | Load and evaluate standard RAG only; rag_plus_plus column shows `--` |
| `rag_plus_plus` | Load and evaluate RAG++ only; standard_rag column shows `--` |

```bash
# Standard RAG only
python3.11 -m eval.ragas_eval.run_ragas \
  --evaluator nvidia --pipeline standard_rag --sample 10 --delay 15

# RAG++ only
python3.11 -m eval.ragas_eval.run_ragas \
  --evaluator nvidia --pipeline rag_plus_plus --sample 10 --delay 15
```

Skipped pipelines are not loaded from disk at all — no file-not-found errors
if only one result file exists.

---

## 9. Test Question Set Update

**File**: `eval/eval_runner/test_questions.json`

Two off-topic unanswerable questions replaced with complex clinical queries:

| # | New Question | Source Documents |
|---|---|---|
| 14 | TB-HIV co-treatment: DOTS modifications + ART drug interactions | `ICMR_STW_Volume2_TB_2021.pdf`, `TB_Preventive_Treatment_Guidelines_2021.pdf` |
| 15 | Psoriasis second-line therapy + methotrexate dosing differences | `STW_Psoriasis.pdf`, `STW_Dermatology.pdf` |

New `category` values:
- `"complex_answerable"` — answerable but requires rewriting + multi-step retrieval
- `"requires_rewriting": true` — explicit flag for downstream metric grouping

---

## 10. Query Rewriting Smoke Test

**File**: `eval/query_rewriting/test_query_rewriting.py`

6 test cases covering all pre-filter and decomposer code paths:

| Case | Question | Expected |
|---|---|---|
| SIMPLE-1 | Jan Aushadhi price of Paracetamol 500mg? | Pre-filter skips, 1 sub-question |
| SIMPLE-2 | What does an HbA1c test measure? | Pre-filter skips, 1 sub-question |
| COMPOUND-1 | Metformin mechanism **and** major ADRs? | 2–3 sub-questions |
| COMPOUND-2 | Metformin price **also** Ashwagandha Virya/Vipaka? | 2–3 sub-questions |
| AMBIGUOUS | "What is **its** dose in **CKD** patients?" | Rewritten, 1–3 sub-questions |
| OVER-DECOMPOSE | 6-facet NTEP/TB protocol query | Capped at 4 sub-questions |

```bash
python3.11 -m eval.query_rewriting.test_query_rewriting
```

---

## 11. Updated File Structure

```
retrieval/
└── query_rewriting/
    ├── __init__.py
    ├── prefilter.py                  # Heuristic skip (no LLM)
    ├── decompose.py                  # qwen3:8b rewrite + split
    └── run_decomposed_retrieval.py   # Per-sub-question retrieval loop
generation/
└── prompts/
    └── builder.py                   # + build_prompt_from_sub_results()
eval/
├── query_rewriting/
│   └── test_query_rewriting.py      # Smoke test
└── ragas_eval/
    ├── run_ragas.py                  # --metrics, --pipeline flags; 4 metrics; bug fixes
    └── evaluators/
        ├── nvidia_evaluator.py       # Model → llama-3.1-70b-instruct
        └── openrouter_evaluator.py   # New backend
config/
└── pipeline_config.py               # query_rewriting_enabled toggle
logs/
└── query_rewriting_log.jsonl         # Auto-created on first rewriting run
```
