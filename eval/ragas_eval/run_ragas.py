#!/usr/bin/env python3
"""
RAGAS Evaluation Script -- Healthcare RAG System
================================================

Loads pre-logged pipeline results (standard RAG and RAG++) from disk,
converts them into RAGAS EvaluationDataset format, runs four RAGAS metrics
via a pluggable evaluator backend, computes custom metrics from logged
fields, and writes a final comparison table + JSON report.

Model-role separation (strictly enforced -- do NOT collapse these):
  - Test-question / ground-truth generation : separate script
  - Answer generation (evaluated system)    : MedGemma via Ollama (run_pipeline.py)
  - Faithfulness verifier in pipeline       : qwen3:8b via Ollama
  - RAGAS evaluator (this script)           : pluggable -- see --evaluator flag

Usage:
    # Local Ollama backend (default -- no API key needed):
    python -m eval.ragas_eval.run_ragas --sample 5

    # NVIDIA NIM backend:
    python -m eval.ragas_eval.run_ragas --evaluator nvidia --delay 15 --sample 5

    # Skip RAGAS metrics, only compute custom metrics:
    python -m eval.ragas_eval.run_ragas --skip-ragas

    # Explicitly specify result files if paths differ:
    python -m eval.ragas_eval.run_ragas \\
        --standard-results path/to/standard_rag_results.json \\
        --rag-plus-results  path/to/rag_plus_plus_results.json
"""

# -- Standard library -----------------------------------------------------------
import argparse
import json
import logging
import os
import random
import re as _re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# -- Third-party ----------------------------------------------------------------
from dotenv import load_dotenv

# ==============================================================================
# CONFIG
# ==============================================================================
# Evaluator backend settings live in their respective modules:
#   eval/ragas_eval/evaluators/ollama_evaluator.py  (default)
#   eval/ragas_eval/evaluators/nvidia_evaluator.py
# Select the backend at runtime with --evaluator ollama|nvidia.

# -- Paths ---------------------------------------------------------------------
_PROJECT_ROOT       = Path(__file__).resolve().parent.parent.parent
_LOGGED_RESULTS_DIR = _PROJECT_ROOT / "eval" / "eval_runner" / "logged_results"
_REPORT_PATH        = Path(__file__).resolve().parent / "comparison_report.json"

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

# Sentinel for metrics that don't apply to a dataset variant
_NA = "N/A -- no fallback mechanism"

# All selectable metric names (used for --metrics validation)
ALL_RAGAS_METRICS   = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
ALL_CUSTOM_METRICS  = ["correct_refusal_rate", "faithfulness_check_pass_rate"]
ALL_METRICS         = ALL_RAGAS_METRICS + ALL_CUSTOM_METRICS


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

def _clean_answer(text: str) -> str:
    """
    Strip thinking-token artifacts injected by Ollama models before RAGAS sees them.

    Some Ollama models (e.g. qwen3, mistral) occasionally prepend internal
    reasoning tokens to their output:
        <unused94>thought 1. Analyze...\\n\\nActual answer here.
        <think>...reasoning...</think>\\n\\nActual answer here.

    These corrupt RAGAS metrics because the 'answer' seen by the evaluator
    contains internal chain-of-thought text instead of the final response.

    Strategy:
        1. Strip <think>...</think> blocks entirely.
        2. Strip any leading <unusedN>... token up to the first blank line.
        3. Strip any remaining <...> tags from the start of the string.
        4. Strip surrounding whitespace.
    """
    # Remove <think>...</think> blocks (possibly multiline)
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)

    # Remove <unusedN>... prefix up to first blank line
    text = _re.sub(r"^<unused\d+>\s*thought.*?\n\n", "", text,
                   flags=_re.DOTALL | _re.IGNORECASE)

    # Remove any remaining leading XML-style tags at the very start
    text = _re.sub(r"^\s*<[^>]+>\s*", "", text)

    return text.strip()


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
    cleaned = 0

    for i, rec in enumerate(records):
        q   = rec.get("question", "").strip()
        gt  = rec.get("ground_truth", "").strip()
        ans = _clean_answer(rec.get("answer", "").strip())
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
            logger.info(
                "[%s] Record %d has no contexts (unanswerable question).", label, i
            )

        orig = rec.get("answer", "").strip()
        if orig != ans:
            cleaned += 1

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
    if cleaned:
        logger.info(
            "[%s] Stripped thinking-token prefixes from %d/%d answers.",
            label, cleaned, len(samples) + skipped,
        )

    logger.info("[%s] Built RAGAS dataset with %d samples.", label, len(samples))
    return EvaluationDataset(samples=samples)


# ==============================================================================
# 3. Evaluator backend dispatcher
# ==============================================================================

def load_evaluator_backend(name: str):
    """
    Import and return the evaluator backend module by name.

    Each backend module exposes:
        build_llm(delay_seconds=...)  -> LangchainLLMWrapper
        build_embeddings()            -> LangchainEmbeddingsWrapper | None
        build_run_config()            -> ragas.run_config.RunConfig
        BACKEND_NAME                  -> str  ("ollama" | "nvidia")
        BACKEND_DESCRIPTION           -> str  (human-readable label for reports)

    Args:
        name: "ollama" or "nvidia"

    Returns:
        The backend module.

    Raises:
        ValueError: if name is not a known backend.
    """
    if name == "ollama":
        from eval.ragas_eval.evaluators import ollama_evaluator as backend  # type: ignore
    elif name == "nvidia":
        from eval.ragas_eval.evaluators import nvidia_evaluator as backend  # type: ignore
    elif name == "openrouter":
        from eval.ragas_eval.evaluators import openrouter_evaluator as backend  # type: ignore
    else:
        raise ValueError(
            f"Unknown evaluator backend: '{name}'. "
            "Choose 'ollama', 'nvidia', or 'openrouter'."
        )
    logger.info("Evaluator backend loaded: %s", backend.BACKEND_DESCRIPTION)
    return backend


# ==============================================================================
# 4. RAGAS metric evaluation (with per-record error tracking)
# ==============================================================================

def run_ragas_evaluation(
    dataset,
    evaluator_llm,
    evaluator_embeddings,
    label: str,
    run_config=None,
    enabled_metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run RAGAS metrics on the given EvaluationDataset.

    Only the metrics listed in `enabled_metrics` are built and sent to the
    API. Omitting a metric entirely saves API calls and latency.

    Args:
        dataset:              ragas.EvaluationDataset
        evaluator_llm:        LangchainLLMWrapper
        evaluator_embeddings: LangchainEmbeddingsWrapper or None
        label:                Dataset label for logging.
        run_config:           ragas.run_config.RunConfig (backend-specific).
        enabled_metrics:      List of metric names to run. None = all.

    Returns:
        Dict mapping metric name to float score, or error string on failure.
        Only enabled metrics appear as keys.
    """
    enabled = set(enabled_metrics) if enabled_metrics is not None else set(ALL_RAGAS_METRICS)
    from ragas import evaluate                   # type: ignore
    from ragas.metrics import (                  # type: ignore
        Faithfulness,
        ContextPrecision,
        ContextRecall,
        AnswerRelevancy,
    )

    metrics = []
    if "faithfulness"      in enabled: metrics.append(Faithfulness(llm=evaluator_llm))
    if "context_precision" in enabled: metrics.append(ContextPrecision(llm=evaluator_llm))
    if "context_recall"    in enabled: metrics.append(ContextRecall(llm=evaluator_llm))

    # answer_relevancy needs an embeddings model; skip gracefully if unavailable
    if "answer_relevancy" in enabled:
        if evaluator_embeddings is not None:
            metrics.append(AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings))
            logger.info("[%s] answer_relevancy enabled (embeddings available).", label)
        else:
            logger.warning(
                "[%s] answer_relevancy SKIPPED -- no embeddings wrapper available. "
                "Use --evaluator nvidia to enable it (nvidia/nv-embed-v1).",
                label,
            )

    if not metrics:
        logger.warning("[%s] No RAGAS metrics enabled — skipping evaluation.", label)
        return {}

    logger.info("[%s] Starting RAGAS evaluation (%d metrics)...", label, len(metrics))

    try:
        result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
    except Exception as exc:
        logger.error(
            "[%s] RAGAS evaluation failed: %s",
            label, exc,
        )
        error_str = f"ERROR: {exc}"
        return {m: error_str for m in enabled}

    # Aggregate per-sample scores; NaN = that record's API call failed.
    df = result.to_pandas()

    metric_col_map = {
        m: m for m in ALL_RAGAS_METRICS if m in enabled
    }

    scores: Dict[str, Any] = {}

    for metric_name, col in metric_col_map.items():
        if col not in df.columns:
            if metric_name == "answer_relevancy" and evaluator_embeddings is None:
                scores[metric_name] = "SKIPPED -- no embeddings available (set NVIDIA_API_KEY)"
                logger.info("[%s] answer_relevancy: skipped (no embeddings).", label)
            else:
                scores[metric_name] = "ERROR: column missing from RAGAS output"
                logger.error(
                    "[%s] Metric column '%s' not found. Available: %s",
                    label, col, list(df.columns),
                )
            continue

        series    = df[col]
        nan_count = int(series.isna().sum())

        if nan_count > 0:
            logger.warning(
                "[%s] %s: %d/%d records returned NaN (model or API failure, "
                "or malformed response). Excluded from mean.",
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
    records: List[Dict],
    has_fallback: bool,
    enabled_metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute custom metrics from logged extra fields without calling any LLM.

    Only metrics listed in `enabled_metrics` are computed and returned.
    Omitting a metric means it won't appear in the report at all.

    correct_refusal_rate:
        Percentage of non-answerable records (category != "answerable") where
        fallback_triggered == True.
        Returns N/A for standard RAG (no fallback mechanism).
        Returns N/A when the test set contains no non-answerable records.

    faithfulness_check_pass_rate:
        Percentage of records where faithfulness_check_failed == False AND
        faithfulness_verdict is not None (check actually ran).
        Returns N/A for standard RAG (no faithfulness check).
        Returns N/A when faithfulness_check is disabled in the pipeline config.

    Args:
        records:         Full list of result dicts (always the full set, not sampled).
        has_fallback:    True for RAG++, False for standard RAG.
        enabled_metrics: List of metric names to include. None = all custom metrics.
    """
    enabled = set(enabled_metrics) if enabled_metrics is not None else set(ALL_CUSTOM_METRICS)
    result: Dict[str, Any] = {}

    if not has_fallback:
        if "correct_refusal_rate"         in enabled: result["correct_refusal_rate"]         = _NA
        if "faithfulness_check_pass_rate" in enabled: result["faithfulness_check_pass_rate"] = _NA
        return result

    total = len(records)
    if total == 0:
        if "correct_refusal_rate"         in enabled: result["correct_refusal_rate"]         = _NA
        if "faithfulness_check_pass_rate" in enabled: result["faithfulness_check_pass_rate"] = _NA
        return result

    # ── correct_refusal_rate ─────────────────────────────────────────────────
    if "correct_refusal_rate" in enabled:
        non_answerable = [
            r for r in records
            if str(r.get("category", "")).lower() != "answerable"
        ]
        if not non_answerable:
            result["correct_refusal_rate"] = "N/A -- no non-answerable questions in test set"
            log_crr = "N/A (no non-answerable questions)"
        else:
            triggered = sum(
                1 for r in non_answerable
                if r.get("fallback_triggered") is True
            )
            rate = round(triggered / len(non_answerable) * 100, 2)
            result["correct_refusal_rate"] = f"{rate}% ({triggered}/{len(non_answerable)})"
            log_crr = result["correct_refusal_rate"]
        logger.info("Custom metric -- correct_refusal_rate: %s", log_crr)

    # ── faithfulness_check_pass_rate ─────────────────────────────────────────
    if "faithfulness_check_pass_rate" in enabled:
        checked = [
            r for r in records
            if r.get("faithfulness_verdict") is not None
        ]
        if not checked:
            result["faithfulness_check_pass_rate"] = "N/A -- faithfulness_check disabled in pipeline config"
            log_fcp = "N/A (disabled)"
        else:
            fc_passed = sum(
                1 for r in checked
                if r.get("faithfulness_check_failed") is False
            )
            rate = round(fc_passed / len(checked) * 100, 2)
            result["faithfulness_check_pass_rate"] = f"{rate}% ({fc_passed}/{len(checked)} checked)"
            log_fcp = result["faithfulness_check_pass_rate"]
        logger.info("Custom metric -- faithfulness_check_pass_rate: %s", log_fcp)

    return result


# ==============================================================================
# 6. Output -- console table + JSON report
# ==============================================================================

def _fmt(value: Any) -> str:
    """Format a metric value for console display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_comparison_table(
    standard_scores: Dict,
    rag_plus_scores: Dict,
    evaluator_label: str = "unknown",
) -> None:
    """
    Print a clean side-by-side comparison table to stdout.

    Args:
        standard_scores: Merged RAGAS + custom scores for standard RAG.
        rag_plus_scores: Merged RAGAS + custom scores for RAG++.
        evaluator_label: Human-readable evaluator description for the header.
    """
    all_metrics = list(dict.fromkeys(
        list(standard_scores.keys()) + list(rag_plus_scores.keys())
    ))

    col_m = 36
    col_v = 28
    width  = col_m + col_v * 2 + 4

    print(f"\n{'=' * width}")
    print(f"  RAGAS EVALUATION -- MEDICAL RAG COMPARISON REPORT")
    print(f"  Evaluator : {evaluator_label}")
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
    evaluator_name: str = "unknown",
    evaluator_description: str = "unknown",
) -> None:
    """
    Save the comparison report as structured JSON.

    Args:
        standard_scores:       Merged scores dict for standard RAG.
        rag_plus_scores:       Merged scores dict for RAG++.
        standard_n:            Total records in standard RAG dataset.
        rag_plus_n:            Total records in RAG++ dataset.
        sampled_n:             --sample N value (None = all records used).
        output_path:           File path to write the JSON report to.
        evaluator_name:        Short backend name ("ollama" | "nvidia").
        evaluator_description: Human-readable label.
    """
    report = {
        "evaluator_used":        evaluator_name,
        "evaluator_description": evaluator_description,
        "sample_size":           sampled_n,
        "note": (
            f"RAGAS metrics computed on sample of {sampled_n} records per dataset."
            if sampled_n
            else "RAGAS metrics computed on full dataset."
        ),
        "datasets": {
            "standard_rag": {
                "total_records":          standard_n,
                "ragas_records_evaluated": min(sampled_n or standard_n, standard_n),
                "scores":                 standard_scores,
            },
            "rag_plus_plus": {
                "total_records":          rag_plus_n,
                "ragas_records_evaluated": min(sampled_n or rag_plus_n, rag_plus_n),
                "scores":                 rag_plus_scores,
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
            "Loads pre-logged pipeline results, evaluates via a pluggable\n"
            "backend (ollama or nvidia), computes custom metrics, and saves\n"
            "a comparison report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--evaluator",
        choices=["ollama", "nvidia", "openrouter"],
        default="ollama",
        help=(
            "Evaluator backend to use. "
            "'ollama' (default): local qwen3:8b via Ollama -- no API key needed. "
            "'nvidia': NVIDIA NIM API -- requires NVIDIA_API_KEY in .env. "
            "'openrouter': OpenRouter API -- requires OPENROUTER_API_KEY in .env."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=15.0,
        metavar="SECONDS",
        help=(
            "Minimum seconds between consecutive NVIDIA API calls "
            "(only used when --evaluator nvidia). Default: 15.0. "
            "Set to 0 to disable throttling (paid tier)."
        ),
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
    parser.add_argument(
        "--metrics",
        nargs="+",
        metavar="METRIC",
        default=["all"],
        help=(
            "Which metrics to compute. Pass one or more names, or 'all' for everything. "
            f"RAGAS metrics: {', '.join(ALL_RAGAS_METRICS)}. "
            f"Custom metrics: {', '.join(ALL_CUSTOM_METRICS)}. "
            "Example: --metrics faithfulness context_recall correct_refusal_rate"
        ),
    )
    parser.add_argument(
        "--pipeline",
        choices=["standard_rag", "rag_plus_plus", "both"],
        default="both",
        help=(
            "Which pipeline to evaluate. "
            "'standard_rag': evaluate the baseline pipeline only. "
            "'rag_plus_plus': evaluate the RAG++ pipeline only. "
            "'both' (default): evaluate both and show a side-by-side comparison."
        ),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    random.seed(args.seed)

    # ── Resolve enabled metrics ───────────────────────────────────────────────
    if "all" in args.metrics:
        enabled_metrics = list(ALL_METRICS)
    else:
        unknown = [m for m in args.metrics if m not in ALL_METRICS]
        if unknown:
            logger.error(
                "Unknown metric(s): %s. Valid choices: %s",
                unknown, ALL_METRICS,
            )
            sys.exit(1)
        enabled_metrics = args.metrics

    enabled_ragas  = [m for m in enabled_metrics if m in ALL_RAGAS_METRICS]
    enabled_custom = [m for m in enabled_metrics if m in ALL_CUSTOM_METRICS]

    logger.info(
        "Enabled metrics: RAGAS=%s  Custom=%s",
        enabled_ragas or "(none)", enabled_custom or "(none)",
    )

    # ---- 1. Load logged results ---------------------------------------------
    logger.info("Loading logged pipeline results...")

    run_standard = args.pipeline in ("standard_rag", "both")
    run_rag_plus  = args.pipeline in ("rag_plus_plus", "both")

    if run_standard:
        try:
            standard_records = load_results(args.standard_results)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            sys.exit(1)
    else:
        standard_records = []
        logger.info("--pipeline=%s: skipping standard_rag.", args.pipeline)

    if run_rag_plus:
        try:
            rag_plus_records = load_results(args.rag_plus_results)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            sys.exit(1)
    else:
        rag_plus_records = []
        logger.info("--pipeline=%s: skipping rag_plus_plus.", args.pipeline)

    # ---- 2. Sample for RAGAS (custom metrics always use full set) -----------
    standard_sample = maybe_sample(standard_records, args.sample, "standard_rag") if run_standard else []
    rag_plus_sample = maybe_sample(rag_plus_records, args.sample, "rag_plus_plus") if run_rag_plus else []

    # ---- 3. Build RAGAS datasets --------------------------------------------
    standard_dataset = build_ragas_dataset(standard_sample, "standard_rag") if run_standard else None
    rag_plus_dataset  = build_ragas_dataset(rag_plus_sample, "rag_plus_plus") if run_rag_plus else None

    # ---- 4. RAGAS evaluation ------------------------------------------------
    if args.skip_ragas or not enabled_ragas:
        if args.skip_ragas:
            logger.info("--skip-ragas set: skipping RAGAS LLM evaluation.")
        else:
            logger.info("No RAGAS metrics selected -- skipping LLM evaluation.")
        _skipped = {m: "SKIPPED" for m in enabled_ragas}
        ragas_standard = dict(_skipped)
        ragas_rag_plus  = dict(_skipped)
        evaluator_name  = "skipped"
        evaluator_desc  = "SKIPPED (--skip-ragas)" if args.skip_ragas else "SKIPPED (no RAGAS metrics selected)"
    else:
        # Load the chosen backend module
        try:
            backend = load_evaluator_backend(args.evaluator)
        except (ValueError, ImportError) as exc:
            logger.error("Failed to load evaluator backend: %s", exc)
            sys.exit(1)

        logger.info("Building evaluator: %s ...", backend.BACKEND_DESCRIPTION)
        try:
            import inspect
            if "delay_seconds" in inspect.signature(backend.build_llm).parameters:
                evaluator_llm = backend.build_llm(delay_seconds=args.delay)
            else:
                evaluator_llm = backend.build_llm()
            evaluator_embeddings = backend.build_embeddings()
            run_config           = backend.build_run_config()
        except (RuntimeError, ImportError) as exc:
            logger.error("Failed to initialise evaluator: %s", exc)
            sys.exit(1)

        evaluator_name = backend.BACKEND_NAME
        evaluator_desc = backend.BACKEND_DESCRIPTION

        ragas_standard = run_ragas_evaluation(
            standard_dataset, evaluator_llm, evaluator_embeddings,
            "standard_rag", run_config=run_config,
            enabled_metrics=enabled_ragas,
        ) if run_standard else {}
        ragas_rag_plus = run_ragas_evaluation(
            rag_plus_dataset, evaluator_llm, evaluator_embeddings,
            "rag_plus_plus", run_config=run_config,
            enabled_metrics=enabled_ragas,
        ) if run_rag_plus else {}

    # ---- 5. Custom metrics (always on FULL dataset) -------------------------
    if run_standard or run_rag_plus:
        logger.info("Computing custom metrics on full datasets...")
    custom_standard = compute_custom_metrics(
        standard_records, has_fallback=False, enabled_metrics=enabled_custom,
    ) if run_standard else {}
    custom_rag_plus  = compute_custom_metrics(
        rag_plus_records, has_fallback=True, enabled_metrics=enabled_custom,
    ) if run_rag_plus else {}

    # ---- 6. Merge -----------------------------------------------------------
    standard_scores = {**ragas_standard, **custom_standard}
    rag_plus_scores  = {**ragas_rag_plus,  **custom_rag_plus}

    # ---- 7. Output ----------------------------------------------------------
    # For single-pipeline runs collapse to a single-column table
    if not run_standard:
        standard_scores = {m: "--" for m in rag_plus_scores}
    if not run_rag_plus:
        rag_plus_scores = {m: "--" for m in standard_scores}

    print_comparison_table(
        standard_scores, rag_plus_scores,
        evaluator_label=evaluator_desc,
    )

    save_report(
        standard_scores=standard_scores,
        rag_plus_scores=rag_plus_scores,
        standard_n=len(standard_records),
        rag_plus_n=len(rag_plus_records),
        sampled_n=args.sample,
        output_path=args.output,
        evaluator_name=evaluator_name,
        evaluator_description=evaluator_desc,
    )

    logger.info("Evaluation complete. Evaluator used: %s", evaluator_desc)


if __name__ == "__main__":
    main()
