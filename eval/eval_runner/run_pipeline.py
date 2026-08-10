#!/usr/bin/env python3
"""
Pipeline Evaluation Runner -- Healthcare RAG System
====================================================

Reads a test question set, runs each question through both pipeline
variants (Standard RAG and RAG++), and writes structured JSON result logs
that eval/ragas_eval/run_ragas.py loads for metric computation.

Model roles (strictly separate):
  - Answer generation : OllamaClient (MedGemma via Ollama)
  - Faithfulness check (RAG++ only) : NVIDIA API (mistralai/mistral-nemotron)
  - RAGAS evaluation  : run_ragas.py (separate script -- not called here)

Input (test question set):
    JSON file at eval/eval_runner/test_questions.json.
    Each record must have:
        "question"     : str
        "ground_truth" : str
        "category"     : "answerable" | "unanswerable"

Output:
    eval/eval_runner/logged_results/standard_rag_results.json
    eval/eval_runner/logged_results/rag_plus_plus_results.json

Usage:
    # Full run:
    python -m eval.eval_runner.run_pipeline

    # Sample first N questions (to validate before full run):
    python -m eval.eval_runner.run_pipeline --sample 5

    # Dry-run (no Ollama calls -- uses prompt text as answer):
    python -m eval.eval_runner.run_pipeline --dry-run --sample 3

    # Custom question file:
    python -m eval.eval_runner.run_pipeline --questions path/to/questions.json
"""

# -- Standard library ----------------------------------------------------------
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# -- Third-party ---------------------------------------------------------------
from dotenv import load_dotenv

# -- Project -------------------------------------------------------------------
from config.pipeline_config import RAG_PLUS_PLUS_CONFIG, STANDARD_RAG_CONFIG
from generation.generator import Generator

# -- Paths ---------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_QUESTIONS_PATH = _HERE / "test_questions.json"
_RESULTS_DIR    = _HERE / "logged_results"

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Question loading
# ==============================================================================

def load_questions(path: Path) -> List[Dict[str, Any]]:
    """
    Load the test question set from a JSON file.

    Expected schema per record:
        "question"     : str  -- the evaluation question
        "ground_truth" : str  -- the reference / ideal answer
        "category"     : str  -- "answerable" or "unanswerable"

    Args:
        path: Path to the questions JSON file.

    Returns:
        List of question dicts.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:        if the file is empty or malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Test questions file not found: {path}\n"
            "Create eval/eval_runner/test_questions.json with your question set.\n"
            "Each record needs: question (str), ground_truth (str), category (str)."
        )

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list) or not data:
        raise ValueError(
            f"Expected a non-empty JSON list in {path}, got: {type(data).__name__}"
        )

    # Basic schema validation
    required_keys = {"question", "ground_truth", "category"}
    invalid = [
        i for i, rec in enumerate(data)
        if not required_keys.issubset(rec.keys())
    ]
    if invalid:
        missing_info = [
            f"record {i}: missing {required_keys - set(data[i].keys())}"
            for i in invalid[:5]
        ]
        raise ValueError(
            f"Records missing required keys: {'; '.join(missing_info)}"
        )

    logger.info("Loaded %d questions from %s", len(data), path)
    return data


def maybe_sample(questions: List[Dict], n: Optional[int]) -> List[Dict]:
    """Return a random sample of n questions, or all if n is None."""
    if n is None or n >= len(questions):
        return questions
    sampled = random.sample(questions, n)
    logger.info("Sampled %d / %d questions.", n, len(questions))
    return sampled


# ==============================================================================
# 2. Single-record runner
# ==============================================================================

def _extract_context_texts(sources: List[Dict]) -> List[str]:
    """
    Extract plain text strings from source chunk dicts.

    Handles both:
        {"text": "...", "metadata": {...}}  -- full chunk dicts
        "raw string"                        -- already plain text
    """
    texts = []
    for s in sources:
        if isinstance(s, dict):
            t = s.get("text", "")
            if t:
                texts.append(t)
        elif isinstance(s, str) and s:
            texts.append(s)
    return texts


def run_one(
    question: Dict[str, Any],
    generator: Generator,
    config: Dict,
    pipeline_label: str,
    idx: int,
    total: int,
) -> Dict[str, Any]:
    """
    Run a single question through one pipeline variant and build a result record.

    Logs timing and maps the Generator output to the schema expected by
    run_ragas.py:
        question                  : str
        ground_truth              : str
        answer                    : str
        contexts                  : List[str]
        category                  : str
        fallback_triggered        : bool  (RAG++ only, else False)
        faithfulness_check_passed : bool  (RAG++ only, else None)

    Args:
        question:       Question dict (question, ground_truth, category).
        generator:      Initialised Generator instance.
        config:         Pipeline config dict (STANDARD_RAG_CONFIG or RAG_PLUS_PLUS_CONFIG).
        pipeline_label: Human-readable label for logging.
        idx:            1-indexed question number (for progress logging).
        total:          Total number of questions being run.

    Returns:
        Result dict ready for JSON serialisation.
    """
    q       = question["question"]
    gt      = question["ground_truth"]
    cat     = question.get("category", "answerable")

    logger.info(
        "[%s] (%d/%d) Running: %s...",
        pipeline_label, idx, total, q[:80]
    )

    t0 = time.perf_counter()
    try:
        result = generator.generate(q, config=config)
    except Exception as exc:
        logger.error(
            "[%s] (%d/%d) Generator raised an exception: %s",
            pipeline_label, idx, total, exc,
        )
        result = {
            "answer":   f"ERROR: {exc}",
            "sources":  [],
            "llm_used": False,
            "llm_metadata": {},
        }
    elapsed = time.perf_counter() - t0

    answer   = result.get("answer", "")
    sources  = result.get("sources", [])
    contexts = _extract_context_texts(sources)

    # -- Fallback / faithfulness fields ------------------------------------
    fallback_triggered = result.get("fallback_triggered", False)

    # faithfulness_check_passed:
    #   True  = faithfulness check ran and passed
    #   False = faithfulness check ran and failed (or fallback triggered)
    #   None  = faithfulness check not run (standard RAG)
    uses_fc = config.get("faithfulness_check", False)
    if not uses_fc:
        faithfulness_check_passed = None
    elif fallback_triggered:
        # Fallback means generation never ran -- faithfulness check wasn't run
        faithfulness_check_passed = False
    else:
        verdict = result.get("faithfulness_verdict", {})
        if result.get("faithfulness_check_failed"):
            faithfulness_check_passed = False
        elif verdict:
            faithfulness_check_passed = bool(verdict.get("overall_faithful", False))
        else:
            # faithfulness_check=True but no verdict -- generation failed
            faithfulness_check_passed = False

    record = {
        "question":                  q,
        "ground_truth":              gt,
        "answer":                    answer,
        "contexts":                  contexts,
        "category":                  cat,
        "fallback_triggered":        fallback_triggered,
        "faithfulness_check_passed": faithfulness_check_passed,
        "elapsed_seconds":           round(elapsed, 3),
        "llm_used":                  result.get("llm_used", False),
    }

    status = (
        "FALLBACK" if fallback_triggered
        else ("FC-FAIL" if faithfulness_check_passed is False and uses_fc
              else "OK")
    )
    logger.info(
        "[%s] (%d/%d) Done in %.1fs | status=%s | contexts=%d",
        pipeline_label, idx, total, elapsed, status, len(contexts)
    )

    return record


# ==============================================================================
# 3. Pipeline runner
# ==============================================================================

def run_pipeline(
    questions: List[Dict],
    config: Dict,
    pipeline_label: str,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    """
    Run all questions through one pipeline variant and return result records.

    Args:
        questions:      List of question dicts.
        config:         Pipeline config (STANDARD_RAG_CONFIG or RAG_PLUS_PLUS_CONFIG).
        pipeline_label: Label for logging.
        dry_run:        If True, no Ollama calls -- Generator runs without LLM.

    Returns:
        List of result dicts.
    """
    if dry_run:
        logger.info("[%s] DRY-RUN mode -- no Ollama calls.", pipeline_label)
        generator = Generator(llm_client=None)
    else:
        from generation.llm.ollama_client import OllamaClient  # type: ignore
        logger.info(
            "[%s] Initialising OllamaClient (MedGemma)...", pipeline_label
        )
        try:
            llm_client = OllamaClient()
        except RuntimeError as exc:
            logger.error(
                "[%s] Failed to initialise OllamaClient: %s\n"
                "Is Ollama running? Try: ollama serve",
                pipeline_label, exc,
            )
            sys.exit(1)
        generator = Generator(llm_client=llm_client)

    results = []
    total = len(questions)

    for idx, question in enumerate(questions, start=1):
        record = run_one(
            question=question,
            generator=generator,
            config=config,
            pipeline_label=pipeline_label,
            idx=idx,
            total=total,
        )
        results.append(record)

    logger.info(
        "[%s] Finished %d questions. "
        "Fallbacks: %d | FC-failed: %d | OK: %d",
        pipeline_label,
        total,
        sum(1 for r in results if r.get("fallback_triggered")),
        sum(1 for r in results if r.get("faithfulness_check_passed") is False
            and config.get("faithfulness_check")),
        sum(1 for r in results if not r.get("fallback_triggered")
            and r.get("faithfulness_check_passed") is not False),
    )

    return results


# ==============================================================================
# 4. Saving results
# ==============================================================================

def save_results(records: List[Dict], path: Path) -> None:
    """
    Save result records as a JSON list.

    Args:
        records: List of result dicts.
        path:    Output file path. Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    logger.info("Saved %d records to %s", len(records), path)


# ==============================================================================
# 5. CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Standard RAG and RAG++ pipelines on a test question set\n"
            "and log the results for RAGAS evaluation (run_ragas.py)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=_QUESTIONS_PATH,
        metavar="PATH",
        help=(
            "Path to the test question set JSON file. "
            f"Default: {_QUESTIONS_PATH}"
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Run only a random sample of N questions per pipeline. "
            "Useful for a quick sanity check before committing to the full set."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip Ollama/LLM calls and use the RAG prompt text as the answer. "
            "Useful for testing the pipeline wiring and logging without "
            "needing Ollama running."
        ),
    )
    parser.add_argument(
        "--standard-only",
        action="store_true",
        help="Run Standard RAG pipeline only (skip RAG++).",
    )
    parser.add_argument(
        "--rag-plus-only",
        action="store_true",
        help="Run RAG++ pipeline only (skip Standard RAG).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_RESULTS_DIR,
        metavar="DIR",
        help=(
            "Directory for output JSON files. "
            f"Default: {_RESULTS_DIR}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --sample reproducibility (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    random.seed(args.seed)

    # ---- Load questions ------------------------------------------------------
    try:
        all_questions = load_questions(args.questions)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    questions = maybe_sample(all_questions, args.sample)

    # ---- Standard RAG -------------------------------------------------------
    if not args.rag_plus_only:
        logger.info("=" * 60)
        logger.info("PIPELINE: Standard RAG")
        logger.info("=" * 60)

        standard_results = run_pipeline(
            questions=questions,
            config=STANDARD_RAG_CONFIG,
            pipeline_label="standard_rag",
            dry_run=args.dry_run,
        )

        out_path = args.output_dir / "standard_rag_results.json"
        save_results(standard_results, out_path)

    # ---- RAG++ ---------------------------------------------------------------
    if not args.standard_only:
        logger.info("=" * 60)
        logger.info("PIPELINE: RAG++")
        logger.info("=" * 60)

        rag_plus_results = run_pipeline(
            questions=questions,
            config=RAG_PLUS_PLUS_CONFIG,
            pipeline_label="rag_plus_plus",
            dry_run=args.dry_run,
        )

        out_path = args.output_dir / "rag_plus_plus_results.json"
        save_results(rag_plus_results, out_path)

    logger.info("Pipeline run complete. Results written to: %s", args.output_dir)
    logger.info(
        "Next step: python -m eval.ragas_eval.run_ragas --sample 10"
    )


if __name__ == "__main__":
    main()
