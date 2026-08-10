#!/usr/bin/env python3
"""
RAGAS Evaluation Script -- Healthcare RAG System
================================================

Loads pre-logged pipeline results (standard RAG and RAG++) from disk,
converts them into RAGAS EvaluationDataset format, runs four RAGAS metrics
via an NVIDIA-hosted evaluator LLM, computes custom metrics from logged
fields, and writes a final comparison table + JSON report.

Model-role separation (strictly enforced -- do NOT collapse these):
  - Test-question / ground-truth generation : NVIDIA API (separate script)
  - Answer generation (evaluated system)    : Mistral via Ollama (run_pipeline.py)
  - Faithfulness verifier in pipeline       : NVIDIA API -- mistralai/mistral-nemotron
  - RAGAS evaluator (this script)           : NVIDIA API -- see EVALUATOR_MODEL below

Usage:
    # Run on all records:
    python -m eval.ragas_eval.run_ragas

    # Validate on a small sample first (recommended before committing to full run):
    python -m eval.ragas_eval.run_ragas --sample 10

    # Explicitly specify result files if paths differ:
    python -m eval.ragas_eval.run_ragas \
        --standard-results path/to/standard_rag_results.json \
        --rag-plus-results  path/to/rag_plus_plus_results.json
"""

# -- Standard library -----------------------------------------------------------
import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# -- Third-party ----------------------------------------------------------------
from dotenv import load_dotenv

# ==============================================================================
# CONFIG -- change these one-liners if a model is deprecated; nothing else
# in this file needs touching.
# ==============================================================================

# RAGAS evaluator model (NVIDIA API).
# MUST be a different model family from the faithfulness-verifier model
# (mistralai/mistral-nemotron) used elsewhere in this project.
# Current choice: Meta Llama-3.1 70B Instruct (hosted on NVIDIA NIM).
# One-line swap: replace the string below with any other model available
# at https://build.nvidia.com/explore/discover when this one is deprecated.
EVALUATOR_MODEL: str = "meta/llama-3.1-70b-instruct"

NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

# -- Paths ---------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOGGED_RESULTS_DIR = _PROJECT_ROOT / "eval" / "eval_runner" / "logged_results"
_REPORT_PATH = Path(__file__).resolve().parent / "comparison_report.json"

DEFAULT_STANDARD_RESULTS: Path = _LOGGED_RESULTS_DIR / "standard_rag_results.json"
DEFAULT_RAG_PLUS_RESULTS: Path = _LOGGED_RESULTS_DIR / "rag_plus_plus_results.json"

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# -- Sentinel for metrics that don't apply to a dataset variant ----------------
_NA = "N/A -- no fallback mechanism"


# ==============================================================================
# 1. Data loading
# ==============================================================================

def load_results(path: Path) -> List[Dict[str, Any]]:
    """
    Load a logged-results JSON file produced by eval_runner/run_pipeline.py.

    Expected schema per record (minimum required keys):
        question        : str   -- the evaluation question
        ground_truth    : str   -- reference answer
        answer          : str   -- pipeline-generated answer
        contexts        : list  -- list of retrieved chunk text strings
        category        : str   -- e.g. "answerable" / "unanswerable"
        fallback_triggered       : bool (RAG++ only)
        faithfulness_check_passed: bool (RAG++ only)

    Args:
        path: Absolute or relative path to the JSON results file.

    Returns:
        List of result dicts.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is empty or not a JSON list.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Logged results file not found: {path}\n"
            "Run eval/eval_runner/run_pipeline.py first to generate results."
        )

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            f"Expected a non-empty JSON list in {path}, got: {type(data).__name__}"
        )

    logger.info("Loaded %d records from %s", len(data), path)
    return data


def maybe_sample(records: List[Dict], n: Optional[int], label: str) -> List[Dict]:
    """
    Optionally subsample *n* records at random (without replacement).

    Args:
        records: Full list of result dicts.
        n:       Number of records to sample; None = use all.
        label:   Dataset label for logging.

    Returns:
        Sampled (or original) list.
    """
    if n is None or n >= len(records):
        logger.info("[%s] Using all %d records.", label, len(records))
        return records

    sampled = random.sample(records, n)
    logger.info(
        "[%s] Sampled %d / %d records for RAGAS evaluation.",
        label, n, len(records)
    )
    return sampled


# ==============================================================================
# 2. RAGAS dataset construction
# ==============================================================================

def build_ragas_dataset(records: List[Dict], label: str):
    """
    Convert logged pipeline results to a RAGAS EvaluationDataset.

    Uses the RAGAS v0.2+ API:
        SingleTurnSample(user_input, retrieved_contexts, response, reference)
        EvaluationDataset(samples=[...])

    Field mapping from logged results to RAGAS schema:
        question     -> user_input
        contexts     -> retrieved_contexts  (must be List[str])
        answer       -> response
        ground_truth -> reference

    Args:
        records: List of result dicts from load_results().
        label:   Dataset label for error context.

    Returns:
        ragas.EvaluationDataset
    """
    from ragas import EvaluationDataset, SingleTurnSample  # type: ignore

    samples = []
    skipped = 0

    for i, rec in enumerate(records):
        q = rec.get("question", "").strip()
        gt = rec.get("ground_truth", "").strip()
        ans = rec.get("answer", "").strip()
        raw_contexts = rec.get("contexts", [])

        # Normalise contexts -- RAGAS needs List[str]
        if isinstance(raw_contexts, list):
            contexts = [
                c["text"] if isinstance(c, dict) else str(c)
                for c in raw_contexts
                if c
            ]
        else:
            contexts = []

        if not q or not ans:
            logger.warning(
                "[%s] Record %d missing question or answer -- skipping.", label, i
            )
            skipped += 1
            continue

        if not contexts:
            logger.warning(
                "[%s] Record %d has no contexts -- including with empty context list.",
                label, i
            )

        samples.append(
            SingleTurnSample(
                user_input=q,
                retrieved_contexts=contexts,
                response=ans,
                reference=gt or None,
            )
        )

    if skipped:
        logger.warning("[%s] Skipped %d malformed records.", label, skipped)

    logger.info("[%s] Built RAGAS dataset with %d samples.", label, len(samples))
    return EvaluationDataset(samples=samples)


# ==============================================================================
# 3. NVIDIA evaluator LLM wrapper
# ==============================================================================

def build_evaluator_llm():
    """
    Construct the RAGAS evaluator LLM using the NVIDIA API.

    RAGAS v0.2+ expects the LLM to be wrapped in a LangchainLLMWrapper.
    We use ChatOpenAI (OpenAI-compatible) pointed at the NVIDIA NIM endpoint.

    Model: EVALUATOR_MODEL (defined at the top of this file).
    This must remain distinct from mistralai/mistral-nemotron used elsewhere.

    Returns:
        ragas.llms.LangchainLLMWrapper

    Raises:
        RuntimeError: if NVIDIA_API_KEY is not set.
        ImportError:  if ragas or langchain-openai are not installed.
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY environment variable is not set.\n"
            "Add it to your .env file or export it before running."
        )

    try:
        from langchain_openai import ChatOpenAI        # type: ignore
        from ragas.llms import LangchainLLMWrapper     # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies. Install with:\n"
            "  pip install ragas langchain-openai"
        ) from exc

    logger.info(
        "Initialising NVIDIA evaluator LLM: %s (via %s)",
        EVALUATOR_MODEL,
        NVIDIA_BASE_URL,
    )

    langchain_llm = ChatOpenAI(
        model=EVALUATOR_MODEL,
        openai_api_key=api_key,
        openai_api_base=NVIDIA_BASE_URL,
        temperature=0.0,
        max_tokens=1024,
        max_retries=1,
    )

    return LangchainLLMWrapper(langchain_llm)


def build_evaluator_embeddings():
    """
    Construct the RAGAS embeddings wrapper for answer_relevancy.

    Reuses the NVIDIA NIM endpoint with nv-embed-v1 (already used for
    ingestion in this project), so no additional API key is needed.

    Returns:
        ragas.embeddings.LangchainEmbeddingsWrapper
    """
    api_key = os.getenv("NVIDIA_API_KEY")

    try:
        from langchain_openai import OpenAIEmbeddings             # type: ignore
        from ragas.embeddings import LangchainEmbeddingsWrapper   # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies. Install with:\n"
            "  pip install ragas langchain-openai"
        ) from exc

    embeddings = OpenAIEmbeddings(
        model="nvidia/nv-embed-v1",
        openai_api_key=api_key,
        openai_api_base=NVIDIA_BASE_URL,
        check_embedding_ctx_length=False,
    )

    return LangchainEmbeddingsWrapper(embeddings)


# ==============================================================================
# 4. RAGAS metric evaluation (with per-record error tracking)
# ==============================================================================

def run_ragas_evaluation(
    dataset,
    evaluator_llm,
    evaluator_embeddings,
    label: str,
) -> Dict[str, Any]:
    """
    Run RAGAS metrics on the given EvaluationDataset.

    Metrics computed:
        faithfulness, context_precision, context_recall, answer_relevancy

    Error handling strategy:
        - Dataset-level failures (auth, connectivity) are caught and returned
          as error strings so one broken dataset does not abort the other.
        - Per-record API failures are surfaced by RAGAS as NaN values in the
          result DataFrame. We log a warning per affected metric, exclude NaN
          records from the aggregate mean, and note the exclusion count in
          the output so the aggregate is never silently corrupted.

    Args:
        dataset:              ragas.EvaluationDataset
        evaluator_llm:        LangchainLLMWrapper
        evaluator_embeddings: LangchainEmbeddingsWrapper
        label:                Dataset label for logging.

    Returns:
        Dict mapping metric name to float score, or error string on failure.
    """
    from ragas import evaluate                   # type: ignore
    from ragas.metrics import (                  # type: ignore
        Faithfulness,
        ContextPrecision,
        ContextRecall,
        AnswerRelevancy,
    )

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    logger.info("[%s] Starting RAGAS evaluation (%d metrics)...", label, len(metrics))

    try:
        result = evaluate(dataset=dataset, metrics=metrics)
    except Exception as exc:
        logger.error(
            "[%s] RAGAS evaluation failed: %s\n"
            "  Check: NVIDIA API key, rate limits, model availability at %s.",
            label, exc, NVIDIA_BASE_URL,
        )
        error_str = f"ERROR: {exc}"
        return {
            "faithfulness":      error_str,
            "context_precision": error_str,
            "context_recall":    error_str,
            "answer_relevancy":  error_str,
        }

    # Aggregate per-sample scores; NaN = that record's API call failed.
    df = result.to_pandas()

    metric_col_map = {
        "faithfulness":      "faithfulness",
        "context_precision": "context_precision",
        "context_recall":    "context_recall",
        "answer_relevancy":  "answer_relevancy",
    }

    scores: Dict[str, Any] = {}

    for metric_name, col in metric_col_map.items():
        if col not in df.columns:
            scores[metric_name] = "ERROR: column missing from RAGAS output"
            logger.error(
                "[%s] Metric column '%s' not found. Available: %s",
                label, col, list(df.columns),
            )
            continue

        series = df[col]
        nan_count = int(series.isna().sum())

        if nan_count > 0:
            logger.warning(
                "[%s] %s: %d/%d records returned NaN (NVIDIA API failure or "
                "malformed response). Excluded from mean. "
                "Review RAGAS log output above for per-record details.",
                label, metric_name, nan_count, len(series),
            )

        valid = series.dropna()
        if valid.empty:
            msg = f"ERROR: all {len(series)} records failed -- no valid scores"
            scores[metric_name] = msg
            logger.error(
                "[%s] %s: ALL records returned NaN. "
                "This indicates a systemic API or model issue.",
                label, metric_name,
            )
        else:
            scores[metric_name] = round(float(valid.mean()), 4)

    logger.info("[%s] RAGAS scores: %s", label, scores)
    return scores


# ==============================================================================
# 5. Custom metrics (computed directly from logged fields, no LLM needed)
# ==============================================================================

def compute_custom_metrics(
    records: List[Dict], has_fallback: bool
) -> Dict[str, Any]:
    """
    Compute custom metrics from logged extra fields without calling any LLM.

    correct_refusal_rate:
        Percentage of records where category == "unanswerable" AND
        fallback_triggered == True.
        Measures whether the RAG++ fallback activates correctly on
        out-of-scope questions.
        Returns the _NA sentinel for standard RAG (no fallback mechanism).

    faithfulness_check_pass_rate:
        Percentage of records where faithfulness_check_passed == True.
        Returns the _NA sentinel for standard RAG (no faithfulness check).

    Args:
        records:      Full list of result dicts (always the full set, not sampled).
        has_fallback: True for RAG++, False for standard RAG.

    Returns:
        Dict with keys: correct_refusal_rate, faithfulness_check_pass_rate.
    """
    if not has_fallback:
        return {
            "correct_refusal_rate":         _NA,
            "faithfulness_check_pass_rate": _NA,
        }

    total = len(records)
    if total == 0:
        return {
            "correct_refusal_rate":         "0.00%",
            "faithfulness_check_pass_rate": "0.00%",
        }

    unanswerable_and_fallback = sum(
        1 for r in records
        if str(r.get("category", "")).lower() == "unanswerable"
        and r.get("fallback_triggered") is True
    )
    correct_refusal_rate = round(unanswerable_and_fallback / total * 100, 2)

    fc_passed = sum(
        1 for r in records
        if r.get("faithfulness_check_passed") is True
    )
    faithfulness_check_pass_rate = round(fc_passed / total * 100, 2)

    logger.info(
        "Custom metrics -- correct_refusal_rate: %.2f%% (%d/%d)  "
        "faithfulness_check_pass_rate: %.2f%% (%d/%d)",
        correct_refusal_rate, unanswerable_and_fallback, total,
        faithfulness_check_pass_rate, fc_passed, total,
    )

    return {
        "correct_refusal_rate":         f"{correct_refusal_rate}%",
        "faithfulness_check_pass_rate": f"{faithfulness_check_pass_rate}%",
    }


# ==============================================================================
# 6. Output -- console table + JSON report
# ==============================================================================

def _fmt(value: Any) -> str:
    """Format a metric value for console display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_comparison_table(
    standard_scores: Dict, rag_plus_scores: Dict
) -> None:
    """
    Print a clean side-by-side comparison table to stdout.

    Rows:    metric names
    Columns: standard_rag | rag_plus_plus

    Args:
        standard_scores: Merged RAGAS + custom scores for standard RAG.
        rag_plus_scores: Merged RAGAS + custom scores for RAG++.
    """
    all_metrics = list(dict.fromkeys(
        list(standard_scores.keys()) + list(rag_plus_scores.keys())
    ))

    col_m = 36
    col_v = 28
    width  = col_m + col_v * 2 + 4

    print(f"\n{'=' * width}")
    print(f"  RAGAS EVALUATION -- MEDICAL RAG COMPARISON REPORT")
    print(f"  Evaluator model : {EVALUATOR_MODEL}")
    print(f"{'=' * width}")
    print(f"  {'Metric':<{col_m}}  {'standard_rag':<{col_v}}  {'rag_plus_plus':<{col_v}}")
    print(f"  {'-' * col_m}  {'-' * col_v}  {'-' * col_v}")

    for metric in all_metrics:
        s_val = _fmt(standard_scores.get(metric, "--"))
        r_val = _fmt(rag_plus_scores.get(metric, "--"))
        print(f"  {metric:<{col_m}}  {s_val:<{col_v}}  {r_val:<{col_v}}")

    print(f"{'=' * width}\n")


def save_report(
    standard_scores: Dict,
    rag_plus_scores: Dict,
    standard_n: int,
    rag_plus_n: int,
    sampled_n: Optional[int],
    output_path: Path,
) -> None:
    """
    Save the comparison report as structured JSON.

    Args:
        standard_scores: Merged scores dict for standard RAG.
        rag_plus_scores: Merged scores dict for RAG++.
        standard_n:      Total records in standard RAG dataset.
        rag_plus_n:      Total records in RAG++ dataset.
        sampled_n:       --sample N value (None = all records used).
        output_path:     File path to write the JSON report to.
    """
    report = {
        "evaluator_model": EVALUATOR_MODEL,
        "sample_size": sampled_n,
        "note": (
            f"RAGAS metrics computed on sample of {sampled_n} records per dataset."
            if sampled_n
            else "RAGAS metrics computed on full dataset."
        ),
        "datasets": {
            "standard_rag": {
                "total_records": standard_n,
                "ragas_records_evaluated": min(sampled_n or standard_n, standard_n),
                "scores": standard_scores,
            },
            "rag_plus_plus": {
                "total_records": rag_plus_n,
                "ragas_records_evaluated": min(sampled_n or rag_plus_n, rag_plus_n),
                "scores": rag_plus_scores,
            },
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info("Comparison report saved to: %s", output_path)


# ==============================================================================
# 7. CLI entry point
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RAGAS evaluation for the Healthcare RAG system.\n"
            "Loads pre-logged pipeline results, evaluates via NVIDIA API,\n"
            "computes custom metrics, and saves a comparison report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Run RAGAS metrics on a random subset of N records per dataset. "
            "Custom metrics are always computed on the FULL dataset. "
            "Omit to run on all records."
        ),
    )
    parser.add_argument(
        "--standard-results",
        type=Path,
        default=DEFAULT_STANDARD_RESULTS,
        metavar="PATH",
        help=(
            "Path to the standard RAG logged results JSON. "
            f"Default: {DEFAULT_STANDARD_RESULTS}"
        ),
    )
    parser.add_argument(
        "--rag-plus-results",
        type=Path,
        default=DEFAULT_RAG_PLUS_RESULTS,
        metavar="PATH",
        help=(
            "Path to the RAG++ logged results JSON. "
            f"Default: {DEFAULT_RAG_PLUS_RESULTS}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPORT_PATH,
        metavar="PATH",
        help=(
            "Output path for the JSON comparison report. "
            f"Default: {_REPORT_PATH}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --sample reproducibility (default: 42).",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help=(
            "Skip RAGAS LLM-based metrics; only compute custom metrics. "
            "Useful for testing the data pipeline without using API credits."
        ),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    random.seed(args.seed)

    # ---- 1. Load logged results ---------------------------------------------
    logger.info("Loading logged pipeline results...")

    try:
        standard_records = load_results(args.standard_results)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    try:
        rag_plus_records = load_results(args.rag_plus_results)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # ---- 2. Sample for RAGAS (custom metrics always use full set) -----------
    standard_sample = maybe_sample(standard_records, args.sample, "standard_rag")
    rag_plus_sample = maybe_sample(rag_plus_records, args.sample, "rag_plus_plus")

    # ---- 3. Build RAGAS datasets --------------------------------------------
    standard_dataset = build_ragas_dataset(standard_sample, "standard_rag")
    rag_plus_dataset  = build_ragas_dataset(rag_plus_sample, "rag_plus_plus")

    # ---- 4. RAGAS evaluation ------------------------------------------------
    if args.skip_ragas:
        logger.info("--skip-ragas set: skipping RAGAS LLM evaluation.")
        _skipped = {
            "faithfulness":      "SKIPPED",
            "context_precision": "SKIPPED",
            "context_recall":    "SKIPPED",
            "answer_relevancy":  "SKIPPED",
        }
        ragas_standard = dict(_skipped)
        ragas_rag_plus  = dict(_skipped)
    else:
        logger.info("Building NVIDIA evaluator LLM (%s)...", EVALUATOR_MODEL)
        try:
            evaluator_llm        = build_evaluator_llm()
            evaluator_embeddings = build_evaluator_embeddings()
        except (RuntimeError, ImportError) as exc:
            logger.error("Failed to initialise evaluator: %s", exc)
            sys.exit(1)

        ragas_standard = run_ragas_evaluation(
            standard_dataset, evaluator_llm, evaluator_embeddings, "standard_rag"
        )
        ragas_rag_plus = run_ragas_evaluation(
            rag_plus_dataset, evaluator_llm, evaluator_embeddings, "rag_plus_plus"
        )

    # ---- 5. Custom metrics (always on FULL dataset) -------------------------
    logger.info("Computing custom metrics on full datasets...")
    custom_standard = compute_custom_metrics(standard_records, has_fallback=False)
    custom_rag_plus  = compute_custom_metrics(rag_plus_records, has_fallback=True)

    # ---- 6. Merge -----------------------------------------------------------
    standard_scores = {**ragas_standard, **custom_standard}
    rag_plus_scores  = {**ragas_rag_plus,  **custom_rag_plus}

    # ---- 7. Output ----------------------------------------------------------
    print_comparison_table(standard_scores, rag_plus_scores)

    save_report(
        standard_scores=standard_scores,
        rag_plus_scores=rag_plus_scores,
        standard_n=len(standard_records),
        rag_plus_n=len(rag_plus_records),
        sampled_n=args.sample,
        output_path=args.output,
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
