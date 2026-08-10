"""
Faithfulness verifier — LLM-as-verifier for hallucination detection.

After generation, this module takes the generated answer and the retrieved
source chunks, and asks a separate verifier LLM to check whether each
factual claim in the answer is supported by the source chunks.

The verifier returns structured JSON. If any claim is unsupported, the
answer is considered unfaithful and a fallback response is returned instead.

Verifier LLM choice: NVIDIA API (NVIDIAClient) by default.
Rationale: The verifier task is narrow (structured claim-checking) and
NVIDIA's hosted mistralai/mistral-nemotron is a strong instruction-follower
for JSON output — more reliable for structured responses than a small 4B
local model. It also avoids local GPU contention with the generation model.
The verifier client is injectable so it can be swapped to Ollama if needed.
"""

import json
import logging
import re
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Fallback message — returned when the faithfulness check fails.
# ──────────────────────────────────────────────────────────────────────────
FAITHFULNESS_FALLBACK_RESPONSE = (
    "I was unable to fully verify this answer against the available sources. "
    "Please consult a healthcare professional or refer to primary medical "
    "literature."
)

# ──────────────────────────────────────────────────────────────────────────
# Verifier prompt template
# ──────────────────────────────────────────────────────────────────────────
_VERIFIER_SYSTEM = (
    "You are a strict medical fact-checker. Your job is to verify whether "
    "every factual claim in a generated answer is supported by the provided "
    "source chunks.\n\n"
    "Instructions:\n"
    "1. Break the answer into individual factual claims.\n"
    "2. For each claim, determine whether it is supported by at least one "
    "source chunk. A claim is supported ONLY if the chunk explicitly states "
    "the same fact — do NOT infer or assume support.\n"
    "3. If a claim is supported, provide the chunk ID of the best supporting "
    "chunk. If unsupported, set source_chunk_id to null.\n"
    "4. Be strict: if a claim contains a specific number, dosage, percentage, "
    "or statistic that does not appear verbatim in the sources, mark it as "
    "unsupported.\n"
    "5. Return ONLY a valid JSON object — no markdown, no code fences, no "
    "explanatory text before or after the JSON.\n\n"
    "The JSON must have this exact structure:\n"
    '{"claims": [{"claim": "...", "supported": true/false, '
    '"source_chunk_id": "..." or null}], "overall_faithful": true/false}\n\n'
    "Set overall_faithful to false if ANY claim is unsupported."
)


def _build_verifier_prompt(answer: str, chunks: List[Dict]) -> str:
    """
    Build the verifier prompt from the generated answer and source chunks.

    Args:
        answer: The generated answer to verify.
        chunks: List of retrieved chunk dicts, each with "id" and "text".

    Returns:
        Formatted prompt string for the verifier LLM.
    """
    # ── Source chunks section ──────────────────────────────────────────
    chunk_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("id", f"chunk_{i}")
        meta = chunk.get("metadata", {})
        source_type = meta.get("source_type", "Unknown")
        title = meta.get("title", "Unknown")

        chunk_blocks.append(
            f"[Chunk ID: {chunk_id} | Source: {source_type} | Title: {title}]\n"
            f"{chunk['text']}"
        )

    chunks_text = "\n\n---\n\n".join(chunk_blocks)

    # ── Full prompt ────────────────────────────────────────────────────
    prompt = f"""{_VERIFIER_SYSTEM}

SOURCE CHUNKS:
{chunks_text}

GENERATED ANSWER TO VERIFY:
{answer}

VERIFICATION JSON:"""

    return prompt


def _parse_verifier_response(raw_response: str) -> Optional[Dict]:
    """
    Parse the verifier LLM's response as JSON.

    Handles common formatting issues: markdown code fences, leading/trailing
    whitespace, and attempts to extract a JSON object from surrounding text.

    Args:
        raw_response: Raw text response from the verifier LLM.

    Returns:
        Parsed dict, or None if parsing fails.
    """
    if not raw_response or not raw_response.strip():
        return None

    text = raw_response.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try stripping markdown code fences: ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find a JSON object in the text (first { to last })
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _validate_verdict(verdict: Dict) -> Dict:
    """
    Validate and normalise the parsed verdict dict.

    Ensures required keys exist and have correct types. If the verdict
    is malformed (missing keys, wrong types), returns a safe fallback
    verdict that treats the answer as unverified.

    Args:
        verdict: Parsed verdict dict from the verifier.

    Returns:
        Normalised verdict dict with guaranteed keys.
    """
    if not isinstance(verdict, dict):
        return {
            "claims": [],
            "overall_faithful": False,
            "_parse_error": "verdict is not a dict",
        }

    claims = verdict.get("claims", [])
    if not isinstance(claims, list):
        claims = []

    # Normalise each claim
    normalised_claims = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        normalised_claims.append({
            "claim": str(c.get("claim", "")),
            "supported": bool(c.get("supported", False)),
            "source_chunk_id": c.get("source_chunk_id", None),
        })

    overall = verdict.get("overall_faithful", False)
    if not isinstance(overall, bool):
        overall = False

    return {
        "claims": normalised_claims,
        "overall_faithful": overall,
    }


class FaithfulnessVerifier:
    """
    Post-generation faithfulness verifier using an LLM-as-verifier.

    Takes the generated answer and the retrieved source chunks, calls a
    separate verifier LLM, and returns a structured verdict indicating
    whether each factual claim is supported.

    Usage:
        verifier = FaithfulnessVerifier()  # defaults to NVIDIA client
        verdict = verifier.verify(answer, chunks)

        # Or inject a different client:
        from generation.llm.ollama_client import OllamaClient
        verifier = FaithfulnessVerifier(verifier_client=OllamaClient())
    """

    def __init__(self, verifier_client: Optional[Callable] = None):
        """
        Initialize the faithfulness verifier.

        Args:
            verifier_client: Optional callable with a generate(prompt) -> str
                             interface. If None, lazily constructs an
                             NVIDIAClient (requires NVIDIA_API_KEY env var).
                             Pass None for dry-run/test mode (verify() will
                             return an unverified fallback).
        """
        self._verifier_client = verifier_client
        self._verifier_client_initialized = verifier_client is not None

    def _get_verifier_client(self) -> Optional[Callable]:
        """
        Lazily initialise the verifier LLM client.

        Returns:
            The verifier client callable, or None if unavailable.
        """
        if self._verifier_client_initialized:
            return self._verifier_client

        # Try NVIDIA first (cheaper/faster for this narrow task)
        try:
            from generation.llm.llm_client import NVIDIAClient
            self._verifier_client = NVIDIAClient()
            self._verifier_client_initialized = True
            logger.info("Faithfulness verifier using NVIDIA API client.")
            return self._verifier_client
        except Exception as e:
            logger.warning(
                "Could not initialize NVIDIA verifier client: %s. "
                "Trying Ollama fallback...",
                e,
            )

        # Fall back to Ollama
        try:
            from generation.llm.ollama_client import OllamaClient
            self._verifier_client = OllamaClient()
            self._verifier_client_initialized = True
            logger.info("Faithfulness verifier using Ollama client.")
            return self._verifier_client
        except Exception as e:
            logger.error(
                "Could not initialize any verifier client: %s. "
                "Faithfulness check will treat all answers as unverified.",
                e,
            )
            self._verifier_client_initialized = True
            return None

    def verify(self, answer: str, chunks: List[Dict]) -> Dict:
        """
        Verify the faithfulness of a generated answer against source chunks.

        Args:
            answer: The generated answer text to verify.
            chunks: List of retrieved chunk dicts used to generate the answer.
                    Each must have "id" and "text" keys.

        Returns:
            dict with keys:
                - "claims": list of claim dicts (claim, supported, source_chunk_id)
                - "overall_faithful": bool — False if any claim unsupported
                - "verified": bool — True if verification completed successfully
                - "verifier_error": str or None — error message if verification failed
        """
        if not answer or not answer.strip():
            return {
                "claims": [],
                "overall_faithful": False,
                "verified": False,
                "verifier_error": "Empty answer — nothing to verify.",
            }

        if not chunks:
            return {
                "claims": [],
                "overall_faithful": False,
                "verified": False,
                "verifier_error": "No source chunks provided for verification.",
            }

        client = self._get_verifier_client()
        if client is None:
            return {
                "claims": [],
                "overall_faithful": False,
                "verified": False,
                "verifier_error": "No verifier LLM client available.",
            }

        prompt = _build_verifier_prompt(answer, chunks)

        # ── Attempt 1 ──────────────────────────────────────────────────
        try:
            raw_response = client.generate(prompt)
            verdict = _parse_verifier_response(raw_response)

            if verdict is not None:
                validated = _validate_verdict(verdict)
                validated["verified"] = True
                validated["verifier_error"] = None
                return validated

            logger.warning(
                "Verifier returned non-JSON response. Retrying once. "
                "Raw (first 200 chars): %s",
                raw_response[:200] if raw_response else "(empty)",
            )
        except Exception as e:
            logger.warning(
                "Verifier call failed on attempt 1: %s. Retrying once...", e
            )

        # ── Attempt 2 (retry) ──────────────────────────────────────────
        try:
            raw_response = client.generate(prompt)
            verdict = _parse_verifier_response(raw_response)

            if verdict is not None:
                validated = _validate_verdict(verdict)
                validated["verified"] = True
                validated["verifier_error"] = None
                return validated

            logger.error(
                "Verifier returned non-JSON response on retry. "
                "Treating answer as unverified. "
                "Raw (first 200 chars): %s",
                raw_response[:200] if raw_response else "(empty)",
            )
        except Exception as e:
            logger.error(
                "Verifier call failed on retry: %s. "
                "Treating answer as unverified.",
                e,
            )

        # ── Fallback: unverified ───────────────────────────────────────
        return {
            "claims": [],
            "overall_faithful": False,
            "verified": False,
            "verifier_error": (
                "Verifier LLM returned non-JSON response after retry."
            ),
        }


def build_faithfulness_fallback_response(
    answer: str,
    chunks: List[Dict],
    verdict: Dict,
) -> Dict:
    """
    Build a fallback response when the faithfulness check fails.

    Matches the normal Generator output schema so downstream consumers
    (logging, evaluation, CLI display) can handle it uniformly.

    Args:
        answer: The original generated answer (logged, not returned to user).
        chunks: The retrieved source chunks.
        verdict: The faithfulness verdict dict from FaithfulnessVerifier.verify().

    Returns:
        dict matching the Generator response schema, with:
            - "answer": str — the fallback message
            - "sources": list — the retrieved chunks (for traceability)
            - "llm_used": bool — True (generation did run)
            - "llm_metadata": dict — empty
            - "faithfulness_check_failed": bool — True
            - "faithfulness_verdict": dict — the full verdict
            - "unsupported_claims": list — claims that were unsupported
            - "original_answer": str — the original answer (for logging)
    """
    unsupported = [
        c for c in verdict.get("claims", []) if not c.get("supported", True)
    ]

    logger.info(
        "Faithfulness check FAILED. "
        "Total claims: %d | Unsupported: %d | Verified: %s",
        len(verdict.get("claims", [])),
        len(unsupported),
        verdict.get("verified", False),
    )

    for claim in unsupported:
        logger.info(
            "  Unsupported claim: '%s'",
            claim.get("claim", "")[:120],
        )

    return {
        "answer": FAITHFULNESS_FALLBACK_RESPONSE,
        "sources": chunks,
        "llm_used": True,
        "llm_metadata": {},
        "faithfulness_check_failed": True,
        "faithfulness_verdict": verdict,
        "unsupported_claims": unsupported,
        "original_answer": answer,
    }