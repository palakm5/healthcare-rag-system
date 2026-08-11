#!/usr/bin/env python3
"""
Structured Retrieval Test Script
==================================

Tests the full structured lookup pipeline with 10 question cases covering:
    - Fast-path template matches (drug prices, herb profiles, formulations)
    - LLM-generated SQL matches (complex/aggregate queries)
    - No-match cases (questions with no structured entity)
    - TRAP tests (injection attempts)

For each question, prints:
    - Entity match result (matched entity, tables, match type, score)
    - Which path was used (template / llm_sql / no_match)
    - The SQL executed (if applicable)
    - The query result rows (first 3)
    - The final merged context summary

Usage:
    python -m eval.structured.test_structured_retrieval
    python -m eval.structured.test_structured_retrieval --no-llm
    python -m eval.structured.test_structured_retrieval --question "custom question"

Requirements:
    - DATABASE_URL_READONLY set in .env
    - Entity cache built: python -m retrieval.structured.build_entity_cache
    - Ollama running with Mistral (for LLM SQL tests only)
"""

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
# Show INFO only for our structured retrieval modules
for _mod in [
    "retrieval.structured.structured_lookup",
    "retrieval.structured.llm_sql_generator",
    "retrieval.structured.query_templates",
]:
    logging.getLogger(_mod).setLevel(logging.INFO)

logger = logging.getLogger(__name__)


# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    # ── Fast-path template tests ────────────────────────────────────────────
    {
        "id":                "TC-01",
        "label":             "Fast-path: drug price lookup",
        "expected_path":     "template",
        "expected_template": "drug_price_lookup",
        "question":          "What is the MRP of Paracetamol 500mg?",
    },
    {
        "id":                "TC-02",
        "label":             "Fast-path: medicine uses lookup",
        "expected_path":     "template",
        "expected_template": "drug_uses_lookup",
        "question":          "What are the uses of Metformin 500mg tablet?",
    },
    {
        "id":                "TC-03",
        "label":             "Fast-path: drug side effects lookup",
        "expected_path":     "template",
        "expected_template": "drug_sideeffects_lookup",
        "question":          "What are the side effects and adverse reactions of Omeprazole?",
    },
    {
        "id":                "TC-04",
        "label":             "Fast-path: herb profile lookup (by common name)",
        "expected_path":     "template",
        "expected_template": "herb_profile_lookup",
        "question":          "Tell me about Tulsi herb -- its botanical name and Ayurvedic properties.",
    },
    {
        "id":                "TC-05",
        "label":             "Fast-path: formulation dosage lookup",
        "expected_path":     "template",
        "expected_template": "formulation_dose_lookup",
        "question":          "What is the dose and precaution for Abhayarishta?",
    },
    {
        "id":                "TC-06",
        "label":             "Fast-path: formulations by classical indication",
        "expected_path":     "template",
        "expected_template": "formulation_by_indication",
        "question":          "Which Ayurveda formulations are indicated for Abhishyanda?",
    },
    # ── LLM-generated SQL tests ─────────────────────────────────────────────
    {
        "id":                "TC-07",
        "label":             "LLM SQL: aggregate -- diabetes patient count",
        "expected_path":     "llm_sql",
        "expected_template": None,
        "question":          "How many patients in the dataset have a diabetes diagnosis?",
    },
    {
        "id":                "TC-08",
        "label":             "LLM SQL: ambiguous multi-table -- herb dosha join",
        "expected_path":     "llm_sql",
        "expected_template": None,
        "question":          "List all herbs and their effect on Kapha dosha from the database.",
    },
    # ── No-match tests ──────────────────────────────────────────────────────
    {
        "id":                "TC-09",
        "label":             "No match: completely out-of-scope",
        "expected_path":     "no_match",
        "expected_template": None,
        "question":          "What is the current stock price of Sun Pharma today?",
    },
    # ── TRAP test ───────────────────────────────────────────────────────────
    {
        "id":                "TC-10",
        "label":             "TRAP: SQL injection attempt via question phrasing",
        "expected_path":     "template_or_llm_sql_or_no_match",
        "expected_template": None,
        "question":          "'; DELETE FROM janaushadhi; SELECT * FROM janaushadhi WHERE '1'='1",
        "_note": (
            "Entity matching may or may not fire. If it does, the template path uses "
            "parameterised %s so injection cannot reach the DB. "
            "If LLM SQL path is taken, the validator must reject any non-SELECT output. "
            "Either way, the DB must NOT be modified."
        ),
    },
]


def print_sep(label: str = "", width: int = 72) -> None:
    if label:
        pad = max(2, (width - len(label) - 2) // 2)
        print("=" * pad + f" {label} " + "=" * pad)
    else:
        print("=" * width)


def run_test(case: dict, lookup) -> dict:
    """Run a single test case and return a result summary dict."""
    print_sep(f"{case['id']}: {case['label']}")
    print(f"Question : {case['question']}")
    if case.get("_note"):
        print(f"Note     : {case['_note']}")
    print()

    try:
        result = lookup.lookup(case["question"])
    except Exception as e:
        print(f"ERROR: lookup() raised: {e}")
        return {"id": case["id"], "status": "ERROR", "path": "error",
                "rows": 0, "error": str(e)}

    # ── Entity matches ───────────────────────────────────────────────────────
    print(f"Entity matches  : {len(result.entity_matches)}")
    for m in result.entity_matches[:5]:
        print(
            f"  [{m.match_type:5s} {m.score:.2f}] '{m.original_token}' "
            f"-> '{m.matched_value[:30]}' (tables: {m.tables})"
        )

    # ── Path ─────────────────────────────────────────────────────────────────
    print(f"\nPath            : {result.path}")
    if result.template_id:
        print(f"Template ID     : {result.template_id}")
    print(f"Matched tables  : {result.matched_tables}")

    # ── SQL ──────────────────────────────────────────────────────────────────
    if result.sql:
        print("\nSQL executed:")
        for line in result.sql.strip().splitlines():
            stripped = line.strip()
            if stripped:
                print(f"    {stripped}")

    # ── Rows ─────────────────────────────────────────────────────────────────
    print(f"\nRows returned   : {result.rows_returned}")
    if result.rows:
        print(f"Columns         : {result.columns}")
        print("Rows (first 3):")
        for row in result.rows[:3]:
            safe = {k: str(v)[:40] for k, v in row.items()}
            print(f"  {json.dumps(safe, ensure_ascii=False)}")

    if result.error:
        print(f"\nERROR           : {result.error}")

    # ── Provenance + merged context ──────────────────────────────────────────
    print(f"\nProvenance label: {result.provenance_label}")
    fake_chunks = [{"text": "Placeholder chunk", "metadata": {"source_type": "TEST", "title": "Test"}}]
    from retrieval.structured.structured_lookup import StructuredLookup
    ctx = StructuredLookup.merge_with_chunks(result, fake_chunks)
    print(f"Merged context  : has_structured={ctx.has_structured}, chunks={len(ctx.chunks)}")

    # ── Pass / Fail ───────────────────────────────────────────────────────────
    ep = case["expected_path"]
    if ep == "template_or_llm_sql_or_no_match":
        path_ok = result.path in ("template", "llm_sql", "no_match")
    else:
        path_ok = result.path == ep

    tmpl_ok = True
    if case.get("expected_template") and result.template_id:
        tmpl_ok = result.template_id == case["expected_template"]

    status = "PASS" if (path_ok and tmpl_ok) else "FAIL"
    print(f"\nResult          : {status}")
    if not path_ok:
        print(f"  Expected path: {ep}, got: {result.path}")
    if not tmpl_ok:
        print(f"  Expected template: {case['expected_template']}, got: {result.template_id}")

    print_sep()
    print()

    return {
        "id": case["id"], "status": status,
        "path": result.path, "rows": result.rows_returned,
        "error": result.error,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test the structured retrieval pipeline.")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip TC-07 and TC-08 (require Ollama/Mistral running)")
    p.add_argument("--question", default=None,
                   help="Run a single custom question instead of the test suite")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from retrieval.structured.structured_lookup import StructuredLookup

    print_sep("STRUCTURED RETRIEVAL TEST SUITE")
    print("Initialising StructuredLookup (loads entity cache + opens DB connection)...")
    try:
        lookup = StructuredLookup()
    except RuntimeError as e:
        print(f"\nFATAL: {e}")
        print("\nSetup checklist:")
        print("  1. python -m retrieval.structured.build_entity_cache")
        print("  2. Add DATABASE_URL_READONLY=... to .env")
        sys.exit(1)

    # ── Single question mode ─────────────────────────────────────────────────
    if args.question:
        result = lookup.lookup(args.question)
        print(f"Path         : {result.path}")
        print(f"Template     : {result.template_id}")
        print(f"Tables       : {result.matched_tables}")
        print(f"Rows returned: {result.rows_returned}")
        if result.sql:
            print(f"SQL:\n{result.sql}")
        for row in result.rows[:3]:
            print(f"  {json.dumps({k: str(v)[:40] for k, v in row.items()}, ensure_ascii=False)}")
        if result.error:
            print(f"Error: {result.error}")
        lookup.close()
        return

    # ── Full test suite ───────────────────────────────────────────────────────
    cases = TEST_CASES
    if args.no_llm:
        cases = [c for c in TEST_CASES if c["id"] not in ("TC-07", "TC-08")]
        print("--no-llm: skipping TC-07 and TC-08 (LLM SQL tests)\n")

    results = []
    for case in cases:
        results.append(run_test(case, lookup))

    lookup.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print_sep("SUMMARY")
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    print(f"Total: {len(results)} | PASS: {passed} | FAIL: {failed} | ERROR: {errors}\n")
    for r in results:
        icon = "✅" if r["status"] == "PASS" else ("❌" if r["status"] == "FAIL" else "⚠️ ")
        print(f"  {icon} {r['id']}: {r['status']:4s} | path={r['path']:<12s} | rows={r['rows']}")
        if r.get("error"):
            print(f"       error: {r['error']}")
    print_sep()


if __name__ == "__main__":
    main()
