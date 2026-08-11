#!/usr/bin/env python3
"""
Schema Introspection + Entity Cache Builder
==========================================

One-time setup script. Re-run manually whenever the database schema or
data changes significantly.

What it does:
    1. Connects to Supabase Postgres (DATABASE_URL -- read-only SELECTs only;
       session is set to read-only at connect time as a safety layer).
    2. Introspects all tables via information_schema.
    3. Writes docs/structured_db_schema_generated.md -- full schema reference
       with column types, row counts, and 2-3 sample rows per table.
    4. Extracts distinct values from ENTITY_COLUMNS (name-like lookup fields).
    5. Saves data/structured_cache/entity_cache.json -- maps every known
       entity name (lowercased) to the table(s) and column(s) it lives in.

Usage:
    python -m retrieval.structured.build_entity_cache
    python -m retrieval.structured.build_entity_cache --db-url postgresql://...
    python -m retrieval.structured.build_entity_cache --skip-samples
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2
from dotenv import load_dotenv

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_PATH     = _PROJECT_ROOT / "docs" / "structured_db_schema_generated.md"
CACHE_DIR     = _PROJECT_ROOT / "data" / "structured_cache"
CACHE_PATH    = CACHE_DIR / "entity_cache.json"

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Entity columns config ────────────────────────────────────────────────────
# Maps table_name -> list of column names to index as entity lookup values.
# Only name-like text columns included.
# Pure FK/junction tables and numeric measurement tables are excluded.
ENTITY_COLUMNS: Dict[str, List[str]] = {
    # Drug / medicine catalogs
    "janaushadhi":                   ["genericname", "groupname"],
    "medicinedetails":               ["medicinename"],

    # Ayurveda herbs (NHP corpus)
    "herb":                          ["name", "botanicalname", "englishname"],
    "herbsynonym":                   ["synonym"],

    # AyurKosh herb catalog (name may be Devanagari)
    "ayurkoshherb":                  ["name", "latinname"],

    # Remedies, compounds, diseases, symptoms (AyurKosh)
    "ayurkoshremedy":                ["name"],
    "ayurkoshcompound":              ["name"],
    "ayurkoshdisease":               ["name"],
    "ayurkoshlakshan":               ["name"],
    "ayurkoshlakshanprakritidhatu":  ["lakshaninenglish", "name"],

    # Classical indications (English)
    "classicalindication":           ["name"],

    # Ayurveda formulations
    "ayushformulation":              ["name", "system", "category"],

    # Clinical lookup tables (small, English)
    "comorbiditytype":               ["name"],
    "diagnosistype":                 ["name", "category"],
    "labtesttype":                   ["testname", "category"],
    "vitalsigntype":                 ["name"],
    "dosha":                         ["name"],
    "rasagunaattribute":             ["value", "attributetype"],
    "partused":                      ["name"],
    "ayurkoshgroup":                 ["name"],
    "ayurkoshquality":               ["name"],
    "dataset":                       ["datasetname", "domain"],
}

SKIPPED_TABLES_REASON: Dict[str, str] = {
    "herbpartused":            "junction table -- integer FKs only",
    "herbrasaguna":            "junction table -- integer FKs only",
    "herbindication":          "junction table -- integer FKs only",
    "herbdoshaeffect":         "effect column too generic for entity matching",
    "formulationindication":   "junction table -- integer FKs only",
    "formulationpotency":      "junction table -- integer FKs only",
    "ayurkoshherbgroup":       "junction table -- integer FKs only",
    "ayurkoshherbquality":     "junction table -- integer FKs only",
    "ayurkoshherbcompound":    "junction table -- integer FKs only",
    "ayurkoshcompounddisease": "junction table -- integer FKs only",
    "ayurkoshcompoundlakshan": "junction table -- integer FKs only",
    "ayurkoshdiseaselakshan":  "junction table -- integer FKs only",
    "ayurkoshdiseaseremedy":   "junction table -- integer FKs only",
    "ayurkoshherbalternate":   "junction table -- integer FKs only",
    "potency":                 "small lookup (7 rows) -- not a primary entity",
    "patient":                 "patient-level numeric/ID data",
    "admission":               "patient-level dates/stay data",
    "labresult":               "patient-level measurement data",
    "vitalsign":               "patient-level measurement data",
    "patientcomorbidity":      "patient-level boolean flags",
    "patientdiagnosis":        "patient-level boolean flags",
    "diabetesprofile":         "patient-level clinical profile",
    "cardiovascularprofile":   "patient-level clinical profile",
    "renalprofile":            "patient-level clinical profile",
    "liverprofile":            "patient-level clinical profile",
    "reproductiveprofile":     "patient-level clinical profile",
}

# Tables where entity values are compound strings (e.g. "Paracetamol 500mg Tablet").
# For these tables, we ALSO index each individual word from the compound name
# (min length 4, alpha-only) pointing back to the full name. This allows the
# entity matcher to fire on "Paracetamol" in a question even though the cache
# key is "paracetamol 500mg tablet".
COMPOUND_NAME_TABLES = {"janaushadhi", "medicinedetails"}
MIN_WORD_LEN_FOR_PREFIX_INDEX = 4   # skip short words like "and", "for", "mg"


def get_connection(db_url: str) -> psycopg2.extensions.connection:
    """Open a Postgres connection forced to read-only at the session level."""
    conn = psycopg2.connect(db_url, connect_timeout=15)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SET default_transaction_read_only = on;")
    cur.execute("SET statement_timeout = '30s';")
    conn.commit()
    cur.close()
    return conn


def fetch_all_tables(cur) -> List[str]:
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    return [r[0] for r in cur.fetchall()]


def fetch_columns(cur, table: str) -> List[Dict[str, Any]]:
    cur.execute("""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position;
    """, (table,))
    return [
        {"name": r[0], "type": r[1], "nullable": r[2], "default": r[3]}
        for r in cur.fetchall()
    ]


def fetch_row_count(cur, table: str) -> int:
    cur.execute(f'SELECT COUNT(*) FROM "{table}";')
    return cur.fetchone()[0]


def fetch_sample_rows(cur, table: str, columns: List[Dict], n: int = 3) -> List[Dict]:
    col_names = [c["name"] for c in columns]
    col_select = ", ".join(f'CAST("{c}" AS TEXT) AS "{c}"' for c in col_names)
    try:
        cur.execute(f'SELECT {col_select} FROM "{table}" LIMIT {n};')
        return [dict(zip(col_names, row)) for row in cur.fetchall()]
    except Exception as e:
        logger.warning("Could not fetch samples for %s: %s", table, e)
        return []


def fetch_distinct_entity_values(
    cur, table: str, column: str, limit: int = 50_000
) -> List[str]:
    try:
        cur.execute(
            f'SELECT DISTINCT CAST("{column}" AS TEXT) FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL AND CAST("{column}" AS TEXT) != \'\' '
            f'ORDER BY 1 LIMIT %s;',
            (limit,)
        )
        return [r[0].strip() for r in cur.fetchall() if r[0] and r[0].strip()]
    except Exception as e:
        logger.warning("Could not fetch distinct values for %s.%s: %s", table, column, e)
        return []


def write_schema_markdown(
    schema_data: Dict[str, Any],
    tables: List[str],
    generated_at: str,
) -> None:
    """Write docs/structured_db_schema_generated.md."""
    lines = [
        "# Structured Database Schema Reference",
        "",
        "> **Auto-generated** by `retrieval/structured/build_entity_cache.py`  ",
        f"> Last updated: {generated_at}  ",
        f"> Total tables: {len(tables)}  ",
        "> Do not edit manually -- re-run the script to refresh.",
        "",
        "---",
        "",
    ]

    for tbl in sorted(tables):
        info      = schema_data.get(tbl, {})
        row_count = info.get("row_count", "?")
        columns   = info.get("columns", [])
        samples   = info.get("sample_rows", [])
        reason    = SKIPPED_TABLES_REASON.get(tbl, "")
        is_entity = tbl in ENTITY_COLUMNS

        tag = " `[entity-indexed]`" if is_entity else (f" *(skipped: {reason})*" if reason else "")
        lines.append(f"## `{tbl}` ({row_count} rows){tag}")
        lines.append("")

        if columns:
            lines.append("| Column | Type | Nullable |")
            lines.append("|--------|------|----------|")
            for col in columns:
                lines.append(f"| `{col['name']}` | {col['type']} | {col['nullable']} |")
            lines.append("")

        if samples:
            lines.append("**Sample rows:**")
            lines.append("")
            if columns:
                col_names = [c["name"] for c in columns]
                lines.append("| " + " | ".join(col_names) + " |")
                lines.append("|" + "---|" * len(col_names))
                for row in samples:
                    cells = [str(row.get(c, "") or "")[:40] for c in col_names]
                    lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

        lines.append("---")
        lines.append("")

    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Schema markdown written to %s", DOCS_PATH)


def build_entity_cache(cur, schema_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the entity cache.

    Returns:
        {"by_value": {entity_lower: [{table, column, original}]},
         "by_table": {table: [list_of_lowercase_keys]}}

    For COMPOUND_NAME_TABLES (janaushadhi, medicinedetails), individual words
    from compound drug names are also indexed pointing back to the full name,
    so "paracetamol" matches "paracetamol 500mg tablet" at query time.
    """
    import re as _re

    by_value: Dict[str, List[Dict]] = {}
    by_table: Dict[str, List[str]]  = {}

    for table, columns in ENTITY_COLUMNS.items():
        info = schema_data.get(table, {})
        if not info:
            logger.warning("Table %s not found in schema -- skipping.", table)
            continue

        table_keys: List[str] = []
        col_names = [c["name"] for c in info.get("columns", [])]
        is_compound = table in COMPOUND_NAME_TABLES

        for col in columns:
            if col not in col_names:
                logger.warning("Column %s.%s not in schema -- skipping.", table, col)
                continue

            logger.info("Indexing %s.%s ...", table, col)
            values = fetch_distinct_entity_values(cur, table, col)
            logger.info("  -> %d distinct values", len(values))

            for val in values:
                key = val.lower().strip()
                if not key:
                    continue
                entry = {"table": table, "column": col, "original": val}

                # Index the full compound name
                if key not in by_value:
                    by_value[key] = []
                if entry not in by_value[key]:
                    by_value[key].append(entry)
                if key not in table_keys:
                    table_keys.append(key)

                # For compound-name tables, also index individual words
                # (alpha-only, min length) pointing back to the full name.
                # This lets "paracetamol" match "paracetamol 500mg tablet".
                if is_compound:
                    words = _re.findall(r'[a-z]+', key)
                    for word in words:
                        if len(word) >= MIN_WORD_LEN_FOR_PREFIX_INDEX:
                            if word not in by_value:
                                by_value[word] = []
                            if entry not in by_value[word]:
                                by_value[word].append(entry)
                            if word not in table_keys:
                                table_keys.append(word)

        by_table[table] = table_keys
        logger.info("Table %s: %d entity keys indexed.", table, len(table_keys))

    return {"by_value": by_value, "by_table": by_table}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build entity cache for structured DB retrieval.")
    p.add_argument("--db-url",        default=None,           help="Override DATABASE_URL")
    p.add_argument("--skip-samples",  action="store_true",    help="Skip sample rows (faster)")
    p.add_argument("--output-cache",  default=str(CACHE_PATH))
    p.add_argument("--output-docs",   default=str(DOCS_PATH))
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set. Add it to .env or pass --db-url.")
        sys.exit(1)

    logger.info("Connecting to Postgres (read-only session)...")
    try:
        conn = get_connection(db_url)
    except Exception as e:
        logger.error("Connection failed: %s", e)
        sys.exit(1)

    cur          = conn.cursor()
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # 1. Introspect
    logger.info("Introspecting schema...")
    tables = fetch_all_tables(cur)
    logger.info("Found %d tables.", len(tables))

    schema_data: Dict[str, Any] = {}
    for tbl in tables:
        cols      = fetch_columns(cur, tbl)
        row_count = fetch_row_count(cur, tbl)
        samples   = [] if args.skip_samples else fetch_sample_rows(cur, tbl, cols)
        schema_data[tbl] = {"row_count": row_count, "columns": cols, "sample_rows": samples}
        logger.info("  %s: %d cols, %d rows", tbl, len(cols), row_count)

    # 2. Write markdown
    global DOCS_PATH, CACHE_PATH
    DOCS_PATH  = Path(args.output_docs)
    CACHE_PATH = Path(args.output_cache)
    write_schema_markdown(schema_data, tables, generated_at)

    # 3. Build entity cache
    logger.info("Building entity cache...")
    cache = build_entity_cache(cur, schema_data)
    total = len(cache["by_value"])
    logger.info("Entity cache: %d unique values across %d tables.", total, len(cache["by_table"]))

    # 4. Save
    output = {
        "_meta": {
            "generated_at":         generated_at,
            "total_entity_values":  total,
            "tables_indexed":       sorted(ENTITY_COLUMNS.keys()),
            "tables_skipped":       SKIPPED_TABLES_REASON,
        },
        "by_value": cache["by_value"],
        "by_table": cache["by_table"],
    }

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Entity cache saved to %s -- %d values indexed.", CACHE_PATH, total)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
