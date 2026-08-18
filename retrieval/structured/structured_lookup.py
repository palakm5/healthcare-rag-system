"""
Structured Database Lookup -- Main Integration Module
======================================================

Public interface consumed by Generator.generate().

Flow for each question:
    1. EntityMatcher scans the question for known entity values using
       the pre-built entity cache (exact match + fuzzy difflib match).
       No LLM call here -- pure string matching.

    2. If matched entities map to a fast-path template:
           -> execute_template() (query_templates.py)
           Template takes priority for correctness on safety-critical lookups.

    3. If entities match tables but no template applies:
           -> run_llm_sql() (llm_sql_generator.py)
           Scoped to only the matched tables' schemas.

    4. If no entity match at all:
           -> return no_match (no DB query executed)

Connection:
    Uses DATABASE_URL_READONLY exclusively.
    Session is forced to read-only at connect time as a second safety layer.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import psycopg2
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_PATH    = _PROJECT_ROOT / "data" / "structured_cache" / "entity_cache.json"
SCHEMA_PATH   = _PROJECT_ROOT / "data" / "structured_cache" / "schema_snapshot.json"

# ── Config ────────────────────────────────────────────────────────────────────
FUZZY_THRESHOLD   = 0.85   # minimum SequenceMatcher ratio (raised from 0.82 to reduce false positives)
MIN_TOKEN_LENGTH  = 3      # ignore shorter tokens in entity matching
MAX_ENTITY_TABLES = 4      # cap tables passed to LLM


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class EntityMatch:
    """A single entity found in the question."""
    original_token: str
    matched_value:  str
    matched_key:    str
    tables:         List[str]
    match_type:     str   # "exact" or "fuzzy"
    score:          float


@dataclass
class StructuredResult:
    """
    Output of a structured lookup for one question.

    Provenance is preserved per row (table + columns) so the prompt
    builder can cite structured facts separately from guideline chunks.
    """
    path:              str
    template_id:       Optional[str]
    question:          str
    entity_matches:    List[EntityMatch]
    matched_tables:    List[str]
    sql:               Optional[str]
    rows:              List[Dict]
    columns:           List[str]
    rows_returned:     int
    provenance_label:  str
    error:             Optional[str]


@dataclass
class LabeledContext:
    """
    Merged context combining chunk results and structured results.
    Passed to the prompt builder to render as separate evidence blocks.
    """
    chunks:         List[Dict]
    structured:     Optional[StructuredResult]
    has_structured: bool


# ── Entity cache ──────────────────────────────────────────────────────────────

class EntityCache:
    """Loads and provides access to the entity cache."""

    def __init__(self, cache_path: Path = CACHE_PATH):
        self._path     = cache_path
        self._by_value: Dict[str, List[Dict]] = {}
        self._by_table: Dict[str, List[str]]  = {}
        self._loaded   = False
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            logger.warning(
                "Entity cache not found at %s. "
                "Run: python -m retrieval.structured.build_entity_cache",
                self._path,
            )
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            self._by_value = data.get("by_value", {})
            self._by_table = data.get("by_table", {})
            self._loaded   = True
            logger.info(
                "Entity cache loaded: %d values across %d tables.",
                len(self._by_value), len(self._by_table)
            )
        except Exception as e:
            logger.error("Failed to load entity cache: %s", e)

    def reload(self) -> None:
        self._load()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_entries(self, key: str) -> List[Dict]:
        return self._by_value.get(key.lower().strip(), [])

    def all_keys(self) -> List[str]:
        return list(self._by_value.keys())


# ── Schema snapshot ───────────────────────────────────────────────────────────

def _load_schema_snapshot() -> Dict[str, Any]:
    if SCHEMA_PATH.exists():
        try:
            with open(SCHEMA_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not load schema snapshot: %s", e)
    return {}


# ── Connection ────────────────────────────────────────────────────────────────

def open_readonly_connection() -> psycopg2.extensions.connection:
    """
    Open a Postgres connection using DATABASE_URL_READONLY.

    Forces session to read-only at the server level immediately after
    connecting -- second layer of protection beyond role-level permissions.

    Raises:
        RuntimeError: if DATABASE_URL_READONLY is not set or connection fails.
    """
    db_url = os.getenv("DATABASE_URL_READONLY")
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL_READONLY is not set in .env.\n"
            "Create a read-only Postgres role and add:\n"
            "  DATABASE_URL_READONLY=postgresql://rag_readonly:...@host/postgres\n"
            "Setup SQL:\n"
            "  CREATE ROLE rag_readonly WITH LOGIN PASSWORD '...';\n"
            "  GRANT CONNECT ON DATABASE postgres TO rag_readonly;\n"
            "  GRANT USAGE ON SCHEMA public TO rag_readonly;\n"
            "  GRANT SELECT ON ALL TABLES IN SCHEMA public TO rag_readonly;\n"
            "  ALTER DEFAULT PRIVILEGES IN SCHEMA public "
            "GRANT SELECT ON TABLES TO rag_readonly;"
        )
    try:
        conn = psycopg2.connect(db_url, connect_timeout=10)
        conn.autocommit = False
        cur = conn.cursor()
        cur.execute("SET default_transaction_read_only = on;")
        cur.execute("SET statement_timeout = '30s';")
        conn.commit()
        cur.close()
        logger.info("Read-only DB connection established.")
        return conn
    except psycopg2.Error as e:
        raise RuntimeError(f"Database connection failed: {e}") from e


# ── Entity matcher ────────────────────────────────────────────────────────────

class EntityMatcher:
    """
    Matches entity names from the cache against tokens in a user question.

    Strategy:
        1. Exact match:  lowercase(token) in cache (O(1) hash lookup).
        2. Fuzzy match:  SequenceMatcher ratio >= FUZZY_THRESHOLD.
    """

    STOP_WORDS: Set[str] = {
        "what", "which", "where", "when", "who", "how", "why",
        "is", "are", "was", "were", "be", "been", "being",
        "the", "a", "an", "and", "or", "of", "for", "in",
        "on", "at", "to", "by", "with", "from", "about",
        "give", "tell", "show", "find", "get", "list", "do",
        "does", "did", "has", "have", "had", "will", "can",
        "please", "me", "my", "its", "their", "i", "you",
        "name", "drug", "herb", "medicine", "dose", "dosage",
        "use", "uses", "composition", "side", "effect", "effects",
        "patient", "patients", "disease", "diseases", "treatment",
        # Finance / generic words that cause false positives
        "price", "cost", "stock", "share", "market", "pharma",
        "current", "today", "rate", "value", "amount", "number",
        # Medical system names -- too generic, indexed in ayushformulation.system
        # and would swamp real entity matches (e.g. "Abhishyanda")
        "ayurveda", "unani", "siddha", "homeopathy", "homoeopathy",
        # Dosage-form / groupname tokens from janaushadhi that would beat
        # real entity names (e.g. 'ayurvedic' beating 'tulsi').
        "ayurvedic", "tablet", "tablets", "capsule", "capsules",
        "injection", "injections", "syrup", "solution", "suspension",
        "ointment", "cream", "drops", "powder", "test", "unit", "units",
    }

    def __init__(self, cache: EntityCache, fuzzy: bool = True):
        self._cache    = cache
        self._fuzzy    = fuzzy
        self._all_keys: Optional[List[str]] = None

    def _get_all_keys(self) -> List[str]:
        if self._all_keys is None:
            self._all_keys = self._cache.all_keys()
        return self._all_keys

    def _tokenize(self, question: str) -> List[str]:
        words = re.findall(r"[\w'-]+", question.lower())
        words = [w for w in words if w not in self.STOP_WORDS and len(w) >= MIN_TOKEN_LENGTH]
        tokens = list(words)
        tokens += [words[i] + " " + words[i+1] for i in range(len(words)-1)]
        tokens += [
            words[i] + " " + words[i+1] + " " + words[i+2]
            for i in range(len(words)-2)
        ]
        return tokens

    def match(self, question: str) -> List[EntityMatch]:
        """Find all entity matches in the question."""
        if not self._cache.is_loaded:
            return []

        tokens    = self._tokenize(question)
        seen_keys: Dict[str, EntityMatch] = {}

        for token in tokens:
            # 1. Exact match
            entries = self._cache.get_entries(token)
            if entries:
                tables   = list({e["table"] for e in entries})
                original = entries[0]["original"]
                m = EntityMatch(
                    original_token=token, matched_value=original,
                    matched_key=token, tables=tables,
                    match_type="exact", score=1.0,
                )
                if token not in seen_keys or seen_keys[token].score < 1.0:
                    seen_keys[token] = m
                continue

            # 2. Fuzzy match
            if self._fuzzy and len(token) >= MIN_TOKEN_LENGTH:
                best_key, best_score = "", 0.0
                for key in self._get_all_keys():
                    if abs(len(key) - len(token)) > 8:
                        continue
                    score = SequenceMatcher(None, token, key).ratio()
                    if score > best_score:
                        best_score = score
                        best_key   = key

                if best_score >= FUZZY_THRESHOLD:
                    entries = self._cache.get_entries(best_key)
                    if entries:
                        tables   = list({e["table"] for e in entries})
                        original = entries[0]["original"]
                        m = EntityMatch(
                            original_token=token, matched_value=original,
                            matched_key=best_key, tables=tables,
                            match_type="fuzzy", score=best_score,
                        )
                        if best_key not in seen_keys or seen_keys[best_key].score < best_score:
                            seen_keys[best_key] = m

        return sorted(seen_keys.values(), key=lambda m: m.score, reverse=True)


# ── Main structured lookup class ──────────────────────────────────────────────

class StructuredLookup:
    """
    Orchestrates entity matching -> template/LLM SQL -> result merging.

    Usage:
        with StructuredLookup() as sl:
            result = sl.lookup(question)
            context = sl.merge_with_chunks(result, chunks)
    """

    def __init__(
        self,
        cache_path: Path = CACHE_PATH,
        db_url_readonly: Optional[str] = None,
    ):
        self._cache   = EntityCache(cache_path)
        self._matcher = EntityMatcher(self._cache)
        self._schema  = _load_schema_snapshot()
        self._conn: Optional[psycopg2.extensions.connection] = None
        self._db_url  = db_url_readonly

    def _get_conn(self) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            if self._db_url:
                self._conn = psycopg2.connect(self._db_url, connect_timeout=10)
                self._conn.autocommit = False
            else:
                self._conn = open_readonly_connection()
        return self._conn

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def lookup(self, question: str) -> StructuredResult:
        """
        Full structured lookup for a question.

        Steps:
            1. Entity matching (no LLM, no DB).
            2. Template check (fast path, priority).
            3. LLM SQL fallback.
            4. No-match return.
        """
        # 1. Entity matching
        matches = self._matcher.match(question)

        if not matches:
            logger.info("No entity match for '%s...'", question[:60])
            return StructuredResult(
                path="no_match", template_id=None, question=question,
                entity_matches=[], matched_tables=[], sql=None,
                rows=[], columns=[], rows_returned=0,
                provenance_label="", error=None,
            )

        logger.info(
            "%d entity match(es): %s",
            len(matches),
            [(m.matched_key, m.tables, m.match_type) for m in matches[:3]]
        )

        # Collect matched tables (cap at MAX_ENTITY_TABLES)
        all_tables: List[str] = []
        for m in matches:
            for t in m.tables:
                if t not in all_tables:
                    all_tables.append(t)
        matched_tables = all_tables[:MAX_ENTITY_TABLES]

        # Default entity value: best-scored match
        entity_value = matches[0].matched_value

        # 2. Template path (priority)
        from retrieval.structured.query_templates import (
            find_template_for_entity,
            execute_template,
        )

        tmpl_result = find_template_for_entity(
            entity_tables=matched_tables,
            question_lower=question.lower(),
        )

        if tmpl_result is not None:
            template_id, template = tmpl_result

            # Pick the entity value from a match whose tables overlap with the
            # template's target tables -- avoids running the wrong entity
            # (e.g. 'ayurvedic' from janaushadhi) when a better match exists
            # for the actual template table (e.g. 'tulsi' from herb).
            template_entity_tables = set(template.entity_tables)
            _COMPOUND_TABLES = {"janaushadhi", "medicinedetails"}
            for m in matches:
                if set(m.tables) & template_entity_tables:
                    # For compound-name tables the cache stores the full drug
                    # string (e.g. "Aceclofenac 100mg and Paracetamol 325mg
                    # Tablets") as matched_value. The LIKE template wraps the
                    # value in % wildcards, but using the full compound name
                    # returns only that exact drug instead of all drugs
                    # containing the ingredient. Use the original_token
                    # (e.g. "Paracetamol") so the LIKE search is broader.
                    if set(m.tables) & _COMPOUND_TABLES:
                        entity_value = m.original_token
                    else:
                        entity_value = m.matched_value
                    break

            logger.info("Using template '%s' for entity '%s'.", template_id, entity_value)
            try:
                conn = self._get_conn()
                raw  = execute_template(template_id, entity_value, conn)
            except RuntimeError as e:
                return StructuredResult(
                    path="template", template_id=template_id, question=question,
                    entity_matches=matches, matched_tables=matched_tables,
                    sql=None, rows=[], columns=[], rows_returned=0,
                    provenance_label="DB:" + ",".join(template.tables),
                    error=str(e),
                )

            return StructuredResult(
                path="template", template_id=template_id, question=question,
                entity_matches=matches, matched_tables=template.tables,
                sql=raw.get("sql"),
                rows=raw.get("rows", []),
                columns=raw.get("columns", []),
                rows_returned=len(raw.get("rows", [])),
                provenance_label="DB:" + ",".join(template.tables),
                error=raw.get("error"),
            )

        # 3. LLM SQL path
        logger.info("No template -- LLM SQL for tables %s.", matched_tables)
        try:
            conn = self._get_conn()
            from retrieval.structured.llm_sql_generator import run_llm_sql
            raw  = run_llm_sql(
                question=question, matched_tables=matched_tables,
                full_schema=self._schema, conn=conn,
            )
        except RuntimeError as e:
            return StructuredResult(
                path="llm_sql", template_id=None, question=question,
                entity_matches=matches, matched_tables=matched_tables,
                sql=None, rows=[], columns=[], rows_returned=0,
                provenance_label="DB:" + ",".join(matched_tables),
                error=str(e),
            )

        return StructuredResult(
            path="llm_sql", template_id=None, question=question,
            entity_matches=matches, matched_tables=matched_tables,
            sql=raw.get("cleaned_sql"),
            rows=raw.get("rows", []),
            columns=raw.get("columns", []),
            rows_returned=raw.get("rows_returned", 0),
            provenance_label="DB:" + ",".join(matched_tables),
            error=raw.get("error"),
        )

    @staticmethod
    def merge_with_chunks(
        structured: StructuredResult,
        chunks: List[Dict],
    ) -> LabeledContext:
        """Merge structured lookup result with vector search chunks."""
        has_structured = (
            structured.path != "no_match"
            and structured.rows_returned > 0
            and structured.error is None
        )
        return LabeledContext(
            chunks=chunks,
            structured=structured if has_structured else None,
            has_structured=has_structured,
        )
