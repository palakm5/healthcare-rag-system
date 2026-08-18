#!/usr/bin/env python3
"""
Query Rewriting Layer — Test Script
=====================================

Tests the query rewriting layer in isolation (no full LLM generation).

For simple questions the pre-filter skips the LLM entirely and passes
the question through unchanged. For complex/compound questions it calls
qwen3:8b to rewrite + decompose into sub-questions before retrieval.

Tests 6 categories of questions:
  1. Atomic/simple   (x2) — expect: pre-filter skips, 1 sub-question
  2. Compound        (x2) — expect: decomposed into 2-3 sub-questions
  3. Ambiguous       (x1) — expect: meaningfully rewritten
  4. Over-decompose  (x1) — expect: capped at MAX_SUB_QUESTIONS (4)

For each question prints:
  - Pre-filter decision + reason
  - Rewritten sub-questions (if rewriter ran)
  - Per-sub-question retrieval result summary (chunks returned, fallback?)
  - The final combined prompt (so you can verify context grouping visually)

Run from project root:
    python3.11 -m eval.query_rewriting.test_query_rewriting

No LLM generation is performed — retrieval + rewriting smoke test only.
"""

import logging
import os
import sys
import textwrap
from pathlib import Path

# ── Setup ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

for line in open(".env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(
    level=logging.WARNING,   # keep noisy retrieval logs quiet
    format="%(name)s [%(levelname)s] %(message)s",
)
# Show decomposition logs at INFO
logging.getLogger("retrieval.decomposition").setLevel(logging.INFO)

from retrieval.query_rewriting.prefilter   import prefilter_result
from retrieval.query_rewriting.decompose   import decompose_question, MAX_SUB_QUESTIONS
from retrieval.query_rewriting.run_decomposed_retrieval import run_decomposed_retrieval
from generation.prompts.builder          import build_prompt_from_sub_results
from retrieval.search.retriever          import Retriever
from config.pipeline_config              import RAG_PLUS_PLUS_CONFIG

# ── Test questions ─────────────────────────────────────────────────────────────

TEST_CASES = [
    # ── SIMPLE-1: Short, single-concept, no compound indicators ───────────────
    # Should be caught by pre-filter (≤15 words, no "and"/"also"/etc.)
    {
        "label":       "SIMPLE-1 — single drug lookup",
        "question":    "What is the Jan Aushadhi price of Paracetamol 500mg?",
        "expect_skip": True,
        "expect_min_sq": 1,
        "expect_max_sq": 1,
        "notes": "Short, single entity, no compound indicators — pre-filter must skip.",
    },

    # ── SIMPLE-2: Short clinical term, no ambiguity ────────────────────────────
    {
        "label":       "SIMPLE-2 — single lab test",
        "question":    "What does an HbA1c test measure?",
        "expect_skip": True,
        "expect_min_sq": 1,
        "expect_max_sq": 1,
        "notes": "Four words, single clinical concept — trivially atomic.",
    },

    # ── COMPOUND-1: Two clearly separable clinical sub-questions ───────────────
    # Mechanism + side-effects are distinct retrieval targets
    {
        "label":       "COMPOUND-1 — mechanism + ADR",
        "question":    (
            "What is the mechanism of action of Metformin for Type 2 diabetes "
            "and what are its major adverse drug reactions?"
        ),
        "expect_skip": False,
        "expect_min_sq": 2,
        "expect_max_sq": 3,
        "notes": "Two independent clinical topics joined by 'and' — should split.",
    },

    # ── COMPOUND-2: Structured DB + unstructured guideline in one question ─────
    # Price is in janaushadhi DB; Ayurvedic properties are in herb DB
    {
        "label":       "COMPOUND-2 — DB price + herb profile",
        "question":    (
            "What is the Jan Aushadhi price of Metformin 500mg tablets "
            "and also what are the Ayurvedic properties of Ashwagandha "
            "including its Virya and Vipaka?"
        ),
        "expect_skip": False,
        "expect_min_sq": 2,
        "expect_max_sq": 3,
        "notes": "Structured DB sub-question + herb knowledge base sub-question.",
    },

    # ── AMBIGUOUS: Pronoun + abbreviation, needs rewriting to be searchable ────
    {
        "label":       "AMBIGUOUS — pronoun + abbreviation",
        "question":    (
            "What is its recommended dose in CKD patients and are there any "
            "contraindications?"
        ),
        "expect_skip": False,
        "expect_min_sq": 1,
        "expect_max_sq": 3,
        "notes": (
            "Dangling 'its' with no referent + unexpanded 'CKD'. "
            "Rewriter should expand CKD → chronic kidney disease and flag "
            "the missing drug context."
        ),
    },

    # ── OVER-DECOMPOSE: Many facets — must cap at MAX_SUB_QUESTIONS (4) ────────
    {
        "label":       "OVER-DECOMPOSE — multi-facet TB query",
        "question":    (
            "For a newly diagnosed sputum-positive pulmonary TB patient "
            "under the RNTCP/NTEP programme in India, what are the "
            "first-line drug regimen and dosages, the common and serious "
            "side effects of each drug, the recommended monitoring "
            "investigations during treatment, contraindications in "
            "HIV-positive patients, and the expected treatment duration "
            "and success criteria under DOTS?"
        ),
        "expect_skip": False,
        "expect_min_sq": 2,
        "expect_max_sq": 4,   # hard cap must hold
        "notes": (
            "Six distinct clinical facets — LLM will want to produce 5-6 "
            "sub-questions. Cap must truncate to MAX_SUB_QUESTIONS=4."
        ),
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _banner(label: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {label}")
    print(f"{'═' * 70}")


def _check(condition: bool, msg: str) -> str:
    return f"  {'✅' if condition else '❌'}  {msg}"


def _wrap(text: str, width: int = 65, indent: str = "      ") -> str:
    return textwrap.fill(text, width=width, subsequent_indent=indent)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 70)
    print("  QUERY REWRITING LAYER — SMOKE TEST")
    print(f"  Config: RAG_PLUS_PLUS_CONFIG")
    print(f"  MAX_SUB_QUESTIONS cap: {MAX_SUB_QUESTIONS}")
    print("═" * 70)

    retriever = Retriever()
    config    = RAG_PLUS_PLUS_CONFIG

    pass_count = 0
    fail_count = 0

    for tc in TEST_CASES:
        q     = tc["question"]
        label = tc["label"]
        _banner(label)
        print(f"  Question: {_wrap(q)}\n")

        # ── Step 1: Pre-filter ────────────────────────────────────────────────
        pf = prefilter_result(q)
        skip = pf["skip_decomposition"]
        print(f"  PRE-FILTER:")
        print(f"    skip={skip}  words={pf['word_count']}  reason={pf['reason']}")

        skip_ok = (skip == tc["expect_skip"])
        print(_check(skip_ok, f"expect skip={tc['expect_skip']}"))
        if skip_ok:
            pass_count += 1
        else:
            fail_count += 1

        # ── Step 2: Decompose (if not skipped) ───────────────────────────────
        if skip:
            sub_questions = [q]
            print(f"\n  QUERY REWRITING: skipped by pre-filter — question passed through unchanged")
        else:
            print(f"\n  QUERY REWRITING (calling qwen3:8b to rewrite + decompose if needed)...")
            sub_questions = decompose_question(q)
            print(f"    → {len(sub_questions)} sub-question(s):")
            for i, sq in enumerate(sub_questions, 1):
                print(f"      [{i}] {_wrap(sq)}")

        sq_count = len(sub_questions)
        sq_ok = tc["expect_min_sq"] <= sq_count <= tc["expect_max_sq"]
        print(_check(sq_ok,
            f"sub-question count={sq_count} "
            f"(expect {tc['expect_min_sq']}–{tc['expect_max_sq']})"
        ))
        if sq_ok:
            pass_count += 1
        else:
            fail_count += 1

        cap_ok = sq_count <= MAX_SUB_QUESTIONS
        print(_check(cap_ok, f"cap enforced ({sq_count} ≤ {MAX_SUB_QUESTIONS})"))
        if cap_ok:
            pass_count += 1
        else:
            fail_count += 1

        # ── Step 3: Per-sub-question retrieval ────────────────────────────────
        print(f"\n  RETRIEVAL ({len(sub_questions)} sub-question(s)):")
        sub_results = run_decomposed_retrieval(
            sub_questions=sub_questions,
            config=config,
            retriever=retriever,
        )

        for r in sub_results:
            status = "FALLBACK" if r.fallback_triggered else (
                "ERROR" if r.error else f"{len(r.chunks)} chunks"
            )
            print(f"    [{status}] {r.sub_question[:60]}...")
            if r.structured_result:
                sr = r.structured_result
                print(f"             + structured: {sr.path} rows={sr.rows_returned}")

        retrieval_ok = any(not r.fallback_triggered and not r.error for r in sub_results)
        print(_check(retrieval_ok, "at least one sub-question returned chunks"))
        if retrieval_ok:
            pass_count += 1
        else:
            fail_count += 1

        # ── Step 4: Prompt construction ───────────────────────────────────────
        prompt = build_prompt_from_sub_results(
            original_query=q,
            sub_results=sub_results,
        )
        prompt_lines = prompt.split("\n")
        print(f"\n  PROMPT ({len(prompt_lines)} lines, {len(prompt)} chars):")
        # Print first 30 lines to show structure
        for line in prompt_lines[:30]:
            print(f"    {line}")
        if len(prompt_lines) > 30:
            print(f"    ... ({len(prompt_lines) - 30} more lines)")

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    total = pass_count + fail_count
    print(f"\n{'═' * 70}")
    print(f"  TEST SUMMARY: {pass_count}/{total} checks passed")
    if fail_count:
        print(f"  ❌ {fail_count} checks failed — review output above")
    else:
        print(f"  ✅ All checks passed")
    print(f"{'═' * 70}\n")

    print(f"  Query rewriting log: logs/query_rewriting_log.jsonl")
    print(f"  Review it to manually verify rewrite quality.\n")


if __name__ == "__main__":
    main()
