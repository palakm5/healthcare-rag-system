"""
LLM-Generated SQL for Structured Retrieval
==========================================

Falls back to this path when a matched entity does not map to a fast-path
template. Uses Mistral via Ollama (consistent with the rest of the pipeline)
to generate a single, validated SELECT query scoped to only the 2-4 tables
that are actually relevant to the user's question.

Safety guarantees (in order):
    1. Only tables from the entity-matched subset are passed to the LLM --
       never the full 48-table schema.
    2. Generated SQL is validated before any execution:
           - Must be a single statement starting with SELECT
           - No forbidden DML/DDL keywords anywhere (INSERT, UPDATE, DELETE,
             DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE, COPY, EXECUTE)
           - No semicolons embedded mid-query
           - LIMIT clause added if missing
    3. Executed with a hard statement_timeout (default 5 s).
    4. Every generated query -- successful or rejected -- is appended to
       logs/structured_sql_log.jsonl for traceability.
    5. Uses DATABASE_URL_READONLY exclusively -- never DATABASE_URL.

If validation fails: logs reason, returns a generation_failed result,
  does not execute.
If execution fails: logs error, returns execution_failed result.
One retry on Ollama call failure before giving up.
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SQL_LOG_PATH  = _PROJECT_ROOT / "logs" / "structured_sql_log.jsonl"

# ── Config ─────────────────────────────────────────────────────────────────
DEFAULT_ROW_LIMIT         = 20      # added if LLM omits LIMIT
QUERY_TIMEOUT_SECONDS     = 5       # statement_timeout passed to Postgres
MAX_SCHEMA_ROWS_IN_PROMPT = 2       # sample rows shown to LLM per table

# DML/DDL keywords that must NEVER appear in a generated query
FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "GRANT", "REVOKE", "COPY", "EXECUTE", "EXEC",
    "CALL", "DO", "VACUUM", "ANALYZE", "CLUSTER", "REINDEX",
]


# ── SQL validation ──────────────────────────────────────────────────────────

def validate_and_clean_sql(raw_sql: str) -> Tuple[bool, str, str]:
    """
    Validate and clean an LLM-generated SQL string.

    Returns:
        (is_valid, cleaned_sql, rejection_reason)

    Checks (in order):
        1. Not empty.
        2. Strip markdown code fences if present.
        3. Strip trailing semicolon.
        4. No embedded semicolons (multiple statement detection).
        5. Must start with SELECT (case-insensitive).
        6. No forbidden DML/DDL keywords as whole words.
        7. If LIMIT missing, append LIMIT {DEFAULT_ROW_LIMIT}.
    """
    if not raw_sql or not raw_sql.strip():
        return False, "", "LLM returned empty response"

    sql = raw_sql.strip()

    # Strip markdown code fences  ```sql ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", sql, re.DOTALL | re.IGNORECASE)
    if fence_match:
        sql = fence_match.group(1).strip()

    # Strip trailing semicolon
    if sql.endswith(";"):
        sql = sql[:-1].strip()

    # Multiple statement detection
    if ";" in sql:
        return False, sql, "Multiple statements detected (embedded semicolon)"

    # Normalise whitespace for keyword checks
    sql_upper = " " + sql.upper() + " "

    # Must start with SELECT
    parts = sql.split()
    first_word = parts[0].upper() if parts else ""
    if first_word != "SELECT":
        return False, sql, (
            f"Query does not start with SELECT (starts with: '{first_word}'). "
            "Only SELECT statements are permitted."
        )

    # Forbidden keyword check (whole-word boundary)
    for kw in FORBIDDEN_KEYWORDS:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, sql_upper):
            return False, sql, f"Forbidden keyword detected: '{kw}'"

    # Add LIMIT if missing
    if not re.search(r"\bLIMIT\b", sql_upper):
        sql = f"{sql} LIMIT {DEFAULT_ROW_LIMIT}"
        logger.info("No LIMIT in generated SQL -- appended LIMIT %d.", DEFAULT_ROW_LIMIT)

    return True, sql, ""


# ── Schema prompt builder ───────────────────────────────────────────────────

def build_narrowed_schema_prompt(
    matched_tables: List[str],
    full_schema: Dict[str, Any],
) -> str:
    """
    Build a compact schema description for the LLM prompt.

    Only includes the 2-4 matched tables -- never the full 48-table schema.
    Includes column names/types and 2 sample rows per table.
    """
    blocks = []
    for tbl in matched_tables:
        info = full_schema.get(tbl, {})
        cols = info.get("columns", [])
        samples = info.get("sample_rows", [])[:MAX_SCHEMA_ROWS_IN_PROMPT]

        col_lines = ", ".join(
            c["name"] + " (" + c["type"] + ")" for c in cols
        )
        block = f"Table: {tbl}\nColumns: {col_lines}"

        if samples:
            sample_lines = []
            for row in samples:
                items = list(row.items())[:5]
                row_str = ", ".join(k + "=" + str(v)[:40] for k, v in items)
                sample_lines.append(f"  row: {row_str}")
            block += "\nSample rows:\n" + "\n".join(sample_lines)

        blocks.append(block)

    return "\n\n".join(blocks)


def build_sql_generation_prompt(
    question: str,
    schema_snippet: str,
    matched_tables: List[str],
) -> str:
    """Build the full prompt sent to Mistral for SQL generation."""
    tables_list = ", ".join(matched_tables)
    return (
        "You are a PostgreSQL expert generating safe, read-only SQL queries.\n\n"
        f"SCHEMA (use ONLY these tables and columns -- do NOT invent names):\n{schema_snippet}\n\n"
        "STRICT RULES -- violating any rule makes the query unusable:\n"
        "1. Output EXACTLY ONE SQL query. Nothing else -- no explanation, no markdown.\n"
        "2. The query MUST start with SELECT.\n"
        f"3. Only use tables: {tables_list}\n"
        "4. Only use columns that appear in the SCHEMA above.\n"
        "5. No INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE.\n"
        "6. No semicolons anywhere in the query.\n"
        "7. Include a LIMIT clause (max 20 rows).\n"
        "8. Use ILIKE for case-insensitive text matching where appropriate.\n"
        "9. Use proper PostgreSQL syntax.\n\n"
        f"QUESTION: {question}\n\n"
        "SQL QUERY (single SELECT statement only):"
    )


# ── SQL log ─────────────────────────────────────────────────────────────────

def log_sql_attempt(
    question: str,
    matched_tables: List[str],
    raw_sql: str,
    cleaned_sql: str,
    is_valid: bool,
    rejection_reason: str,
    execution_result: Optional[str],
    rows_returned: int,
    elapsed_ms: float,
    error: Optional[str] = None,
) -> None:
    """
    Append a structured log entry to logs/structured_sql_log.jsonl.

    Every generated SQL attempt is logged -- successful or rejected --
    for traceability, debugging, and project write-up demonstration.
    """
    entry = {
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "question":         question,
        "matched_tables":   matched_tables,
        "raw_sql":          raw_sql,
        "cleaned_sql":      cleaned_sql if is_valid else "",
        "is_valid":         is_valid,
        "rejection_reason": rejection_reason if not is_valid else None,
        "execution_result": execution_result,
        "rows_returned":    rows_returned,
        "elapsed_ms":       round(elapsed_ms, 2),
        "error":            error,
    }
    try:
        SQL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SQL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("Failed to write SQL log entry: %s", e)


# ── Ollama / Mistral call ───────────────────────────────────────────────────

def call_mistral(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """
    Call Mistral via OllamaClient and return the raw text response.

    Uses temperature=0.0 for deterministic SQL generation.
    Retries once on failure before raising.

    Raises:
        RuntimeError: if both attempts fail.
    """
    from generation.llm.ollama_client import OllamaClient  # type: ignore

    last_exc: Optional[Exception] = None
    for attempt in (1, 2):
        try:
            client = OllamaClient()
            return client.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RuntimeError as e:
            last_exc = e
            if attempt == 1:
                logger.warning(
                    "Mistral call failed (attempt 1): %s -- retrying...", e
                )
    raise RuntimeError(
        f"Mistral SQL generation failed after 2 attempts: {last_exc}"
    ) from last_exc


# ── SQL execution ───────────────────────────────────────────────────────────

def execute_generated_sql(
    sql: str,
    conn: psycopg2.extensions.connection,
    timeout_seconds: int = QUERY_TIMEOUT_SECONDS,
) -> Tuple[List[Dict], List[str]]:
    """
    Execute a validated SQL query with a hard statement_timeout.

    Args:
        sql:             Validated, cleaned SQL string (no semicolons).
        conn:            Open read-only psycopg2 connection.
        timeout_seconds: Hard execution limit.

    Returns:
        (rows, column_names)

    Raises:
        Exception: on execution failure (caller logs and handles).
    """
    cur = conn.cursor()
    try:
        timeout_ms = timeout_seconds * 1000
        cur.execute(f"SET LOCAL statement_timeout = {timeout_ms};")
        cur.execute(sql)
        col_names = [desc[0] for desc in cur.description]
        rows = [dict(zip(col_names, row)) for row in cur.fetchall()]
        return rows, col_names
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


# ── Main entry point ────────────────────────────────────────────────────────

def run_llm_sql(
    question: str,
    matched_tables: List[str],
    full_schema: Dict[str, Any],
    conn: psycopg2.extensions.connection,
) -> Dict[str, Any]:
    """
    Generate, validate, and execute an LLM SQL query for the given question.

    Args:
        question:       The user's natural-language question.
        matched_tables: Tables identified by the entity matcher (2-4 max).
        full_schema:    Schema dict mapping table names to column/sample info.
        conn:           Open read-only psycopg2 connection.

    Returns:
        Dict with keys: path, question, matched_tables, raw_sql, cleaned_sql,
        is_valid, rejection_reason, rows, columns, rows_returned, error, elapsed_ms.
    """
    t0 = time.perf_counter()

    schema_snippet = build_narrowed_schema_prompt(matched_tables, full_schema)
    prompt         = build_sql_generation_prompt(question, schema_snippet, matched_tables)

    raw_sql = ""
    try:
        raw_sql = call_mistral(prompt)
        logger.info(
            "Mistral generated SQL for '%s...': %s",
            question[:60], raw_sql[:120]
        )
    except RuntimeError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log_sql_attempt(
            question=question, matched_tables=matched_tables,
            raw_sql="", cleaned_sql="", is_valid=False,
            rejection_reason=f"LLM call failed: {e}",
            execution_result="generation_failed", rows_returned=0,
            elapsed_ms=elapsed_ms, error=str(e),
        )
        return {
            "path": "llm_sql", "question": question,
            "matched_tables": matched_tables, "raw_sql": "",
            "cleaned_sql": "", "is_valid": False,
            "rejection_reason": f"LLM call failed: {e}",
            "rows": [], "columns": [], "rows_returned": 0,
            "error": str(e), "elapsed_ms": elapsed_ms,
        }

    # ── Validation ───────────────────────────────────────────────────────────
    is_valid, cleaned_sql, rejection_reason = validate_and_clean_sql(raw_sql)

    if not is_valid:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "SQL validation REJECTED for '%s...': %s | SQL: %s",
            question[:60], rejection_reason, raw_sql[:200]
        )
        log_sql_attempt(
            question=question, matched_tables=matched_tables,
            raw_sql=raw_sql, cleaned_sql="", is_valid=False,
            rejection_reason=rejection_reason,
            execution_result="validation_failed", rows_returned=0,
            elapsed_ms=elapsed_ms,
        )
        return {
            "path": "llm_sql", "question": question,
            "matched_tables": matched_tables, "raw_sql": raw_sql,
            "cleaned_sql": "", "is_valid": False,
            "rejection_reason": rejection_reason,
            "rows": [], "columns": [], "rows_returned": 0,
            "error": None, "elapsed_ms": elapsed_ms,
        }

    # ── Execution ─────────────────────────────────────────────────────────────
    try:
        rows, columns = execute_generated_sql(cleaned_sql, conn)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("LLM SQL executed: %d rows in %.0fms.", len(rows), elapsed_ms)
        log_sql_attempt(
            question=question, matched_tables=matched_tables,
            raw_sql=raw_sql, cleaned_sql=cleaned_sql, is_valid=True,
            rejection_reason="", execution_result="success",
            rows_returned=len(rows), elapsed_ms=elapsed_ms,
        )
        return {
            "path": "llm_sql", "question": question,
            "matched_tables": matched_tables, "raw_sql": raw_sql,
            "cleaned_sql": cleaned_sql, "is_valid": True,
            "rejection_reason": None, "rows": rows, "columns": columns,
            "rows_returned": len(rows), "error": None, "elapsed_ms": elapsed_ms,
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error("LLM SQL execution FAILED for '%s...': %s", question[:60], e)
        log_sql_attempt(
            question=question, matched_tables=matched_tables,
            raw_sql=raw_sql, cleaned_sql=cleaned_sql, is_valid=True,
            rejection_reason="", execution_result="execution_failed",
            rows_returned=0, elapsed_ms=elapsed_ms, error=str(e),
        )
        return {
            "path": "llm_sql", "question": question,
            "matched_tables": matched_tables, "raw_sql": raw_sql,
            "cleaned_sql": cleaned_sql, "is_valid": True,
            "rejection_reason": None, "rows": [], "columns": [],
            "rows_returned": 0, "error": str(e), "elapsed_ms": elapsed_ms,
        }
