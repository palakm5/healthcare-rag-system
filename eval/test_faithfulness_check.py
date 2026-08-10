#!/usr/bin/env python3
"""
Smoke test for the faithfulness / hallucination check.

Runs 3 test cases against the FaithfulnessVerifier:
  1. A clearly faithful answer (all claims supported by source chunks).
  2. An answer with a deliberately injected false claim (made-up dosage).
  3. A very short/simple answer (edge case — verifier shouldn't break).

Each test prints the verdict and claims breakdown for manual inspection.

The test uses a STUB verifier client by default (deterministic, no API
calls needed) so it validates the wiring and logic without depending on
network or API keys. To run against the real NVIDIA verifier, pass --live.

Usage:
    # Stub mode (no API calls):
    python -m eval.test_faithfulness_check

    # Live mode (calls NVIDIA API — requires NVIDIA_API_KEY):
    python -m eval.test_faithfulness_check --live
"""

import argparse
import json
import logging
import sys
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Stub verifier client — returns deterministic JSON for testing.
# ──────────────────────────────────────────────────────────────────────────

class _StubVerifierClient:
    """
    A deterministic stub that returns canned JSON verdicts.

    The stub inspects the answer text to decide which verdict to return:
    - If the answer contains "500mg" (our injected false claim), return FAIL.
    - Otherwise, return PASS.
    """

    def __init__(self):
        self.model = "stub-verifier"

    def generate(self, prompt: str) -> str:
        """
        Return a canned JSON verdict based on the answer content.

        The prompt contains the answer after "GENERATED ANSWER TO VERIFY:"
        and before "VERIFICATION JSON:". We extract it and check for the
        injected false claim marker.
        """
        # Extract the answer from the prompt
        marker = "GENERATED ANSWER TO VERIFY:\n"
        end_marker = "\n\nVERIFICATION JSON:"
        answer_start = prompt.find(marker)
        answer_end = prompt.find(end_marker)

        if answer_start != -1 and answer_end != -1:
            answer = prompt[answer_start + len(marker):answer_end].strip()
        else:
            answer = ""

        # ── Decide verdict based on answer content ─────────────────────
        if "500 mg" in answer or "500mg" in answer.lower():
            # Injected false claim — return FAIL verdict
            return json.dumps({
                "claims": [
                    {
                        "claim": "Tuberculosis is caused by Mycobacterium tuberculosis.",
                        "supported": True,
                        "source_chunk_id": "chunk_001",
                    },
                    {
                        "claim": "The recommended dose is 500 mg three times daily for 14 days.",
                        "supported": False,
                        "source_chunk_id": None,
                    },
                ],
                "overall_faithful": False,
            })

        # Short answer edge case
        if len(answer) < 100:
            return json.dumps({
                "claims": [
                    {
                        "claim": answer.strip(),
                        "supported": True,
                        "source_chunk_id": "chunk_001",
                    },
                ],
                "overall_faithful": True,
            })

        # Default: PASS verdict
        return json.dumps({
            "claims": [
                {
                    "claim": "Tuberculosis is caused by Mycobacterium tuberculosis.",
                    "supported": True,
                    "source_chunk_id": "chunk_001",
                },
                {
                    "claim": "Common symptoms include persistent cough, fever, and weight loss.",
                    "supported": True,
                    "source_chunk_id": "chunk_002",
                },
            ],
            "overall_faithful": True,
        })


# ──────────────────────────────────────────────────────────────────────────
# Test data — source chunks
# ──────────────────────────────────────────────────────────────────────────

_SOURCE_CHUNKS = [
    {
        "id": "chunk_001",
        "text": (
            "Tuberculosis (TB) is an infectious disease caused by the bacterium "
            "Mycobacterium tuberculosis. It primarily affects the lungs but can "
            "also affect other parts of the body. TB is spread through the air "
            "when an infected person coughs or sneezes."
        ),
        "metadata": {
            "source_type": "NHP",
            "title": "Tuberculosis Overview",
            "section": "Introduction",
        },
    },
    {
        "id": "chunk_002",
        "text": (
            "Common symptoms of pulmonary tuberculosis include a persistent "
            "cough lasting more than three weeks, chest pain, coughing up blood "
            "or sputum, fatigue, fever, night sweats, and unexplained weight loss."
        ),
        "metadata": {
            "source_type": "NHP",
            "title": "Tuberculosis Symptoms",
            "section": "Clinical Presentation",
        },
    },
    {
        "id": "chunk_003",
        "text": (
            "The standard treatment for drug-susceptible TB consists of a 6-month "
            "regimen of four first-line drugs: isoniazid, rifampicin, ethambutol, "
            "and pyrazinamide. The intensive phase lasts 2 months with all four "
            "drugs, followed by a 4-month continuation phase with isoniazid and "
            "rifampicin."
        ),
        "metadata": {
            "source_type": "ICMR",
            "title": "TB Treatment Guidelines",
            "section": "Standard Regimen",
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────────────────────────────────

def run_test_case(
    label: str,
    answer: str,
    chunks: List[Dict],
    verifier_client,
    expected_faithful: bool,
):
    """
    Run a single faithfulness check test case and print results.

    Args:
        label: Human-readable test case label.
        answer: The generated answer to verify.
        chunks: Source chunks used to generate the answer.
        verifier_client: The verifier LLM client (or stub).
        expected_faithful: Whether we expect the verdict to be faithful.
    """
    from generation.faithfullness_check.verifier import FaithfulnessVerifier

    print(f"\n{'=' * 80}")
    print(f"  TEST: {label}")
    print(f"  Expected: {'FAITHFUL' if expected_faithful else 'UNFAITHFUL'}")
    print(f"{'=' * 80}")

    verifier = FaithfulnessVerifier(verifier_client=verifier_client)
    verdict = verifier.verify(answer, chunks)

    overall = verdict.get("overall_faithful", False)
    verified = verdict.get("verified", False)
    error = verdict.get("verifier_error")
    claims = verdict.get("claims", [])

    print(f"\n  Verdict:")
    print(f"    overall_faithful: {overall}")
    print(f"    verified:         {verified}")
    print(f"    verifier_error:   {error}")
    print(f"    claims count:     {len(claims)}")

    if claims:
        print(f"\n  Claims breakdown:")
        for i, claim in enumerate(claims, start=1):
            status = "✓ SUPPORTED" if claim["supported"] else "✗ UNSUPPORTED"
            chunk_id = claim.get("source_chunk_id", "N/A")
            print(f"    [{i}] {status}")
            print(f"        Claim:  {claim['claim'][:120]}")
            print(f"        Chunk:  {chunk_id}")

    # ── Assertion ──────────────────────────────────────────────────────
    if overall == expected_faithful:
        print(f"\n  ✅ PASS: Verdict matches expected ({'faithful' if overall else 'unfaithful'}).")
    else:
        print(f"\n  ❌ FAIL: Expected {'faithful' if expected_faithful else 'unfaithful'} "
              f"but got {'faithful' if overall else 'unfaithful'}.")


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for faithfulness / hallucination check."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the real NVIDIA verifier client instead of the stub.",
    )
    args = parser.parse_args()

    # ── Choose verifier client ─────────────────────────────────────────
    if args.live:
        print("=" * 80)
        print("  LIVE MODE — using NVIDIA API verifier")
        print("  (requires NVIDIA_API_KEY environment variable)")
        print("=" * 80)
        verifier_client = None  # FaithfulnessVerifier will auto-init NVIDIA
    else:
        print("=" * 80)
        print("  STUB MODE — using deterministic stub verifier")
        print("  (no LLM calls, fast and reproducible)")
        print("  Pass --live to use the real NVIDIA API verifier.")
        print("=" * 80)
        verifier_client = _StubVerifierClient()

    # ──────────────────────────────────────────────────────────────────
    # Test 1: Faithful answer (should PASS)
    # ──────────────────────────────────────────────────────────────────
    run_test_case(
        label="Test 1 — Faithful answer (should PASS)",
        answer=(
            "Tuberculosis is caused by the bacterium Mycobacterium tuberculosis. "
            "Common symptoms include a persistent cough lasting more than three "
            "weeks, fever, night sweats, and unexplained weight loss. Standard "
            "treatment involves a 6-month regimen of four first-line drugs: "
            "isoniazid, rifampicin, ethambutol, and pyrazinamide."
        ),
        chunks=_SOURCE_CHUNKS,
        verifier_client=verifier_client,
        expected_faithful=True,
    )

    # ──────────────────────────────────────────────────────────────────
    # Test 2: Injected false claim (should FAIL)
    # ──────────────────────────────────────────────────────────────────
    run_test_case(
        label="Test 2 — Injected false dosage claim (should FAIL)",
        answer=(
            "Tuberculosis is caused by the bacterium Mycobacterium tuberculosis. "
            "Common symptoms include a persistent cough, fever, and weight loss. "
            "The recommended dose is 500 mg three times daily for 14 days. "
            "Standard treatment involves a 6-month regimen of four first-line drugs."
        ),
        chunks=_SOURCE_CHUNKS,
        verifier_client=verifier_client,
        expected_faithful=False,
    )

    # ──────────────────────────────────────────────────────────────────
    # Test 3: Short/simple answer (edge case)
    # ──────────────────────────────────────────────────────────────────
    run_test_case(
        label="Test 3 — Short answer (edge case, should PASS)",
        answer="Tuberculosis is caused by the bacterium Mycobacterium tuberculosis.",
        chunks=_SOURCE_CHUNKS,
        verifier_client=verifier_client,
        expected_faithful=True,
    )

    print("\n" + "=" * 80)
    print("  FAITHFULNESS CHECK SMOKE TEST COMPLETE")
    print("  Review the output above. If any test shows unexpected results,")
    print("  inspect the verifier prompt and response parsing logic.")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    main()