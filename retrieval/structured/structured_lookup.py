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
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import psycopg2
from dotenv import load_dotenv

try:
    from rapidfuzz import fuzz as _rfuzz, process as _rfprocess
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False
    import difflib as _difflib   # fallback to original difflib

load_dotenv()
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_PATH    = _PROJECT_ROOT / "data" / "structured_cache" / "entity_cache.json"
SCHEMA_PATH   = _PROJECT_ROOT / "data" / "structured_cache" / "schema_snapshot.json"

# ── Config ────────────────────────────────────────────────────────────────────

# Fuzzy match threshold — rapidfuzz WRatio score (0–100).
# 85 ≈ the former difflib 0.85 ratio; lower values increase recall at the
# cost of false positives.  Tune by reviewing logs/entity_match_log.jsonl.
FUZZY_THRESHOLD: int = 85          # rapidfuzz WRatio (0–100)
FUZZY_THRESHOLD_DIFFLIB: float = 0.85  # kept for fallback if rapidfuzz absent

MIN_TOKEN_LENGTH: int = 3          # ignore shorter tokens in entity matching
MAX_ENTITY_TABLES: int = 4         # cap tables passed to LLM

# Set True to log every fuzzy attempt (hit or miss) to logs/entity_match_log.jsonl.
# Useful for threshold tuning; leave False in production to avoid large logs.
LOG_ALL_MATCH_ATTEMPTS: bool = True


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
        """
        Find all entity matches in the question.

        Two-pass strategy per token:
            1. Exact match:  O(1) hash lookup in the entity cache.
            2. Fuzzy match:  rapidfuzz.process.extractOne over all cache keys
               (WRatio scorer, score_cutoff=FUZZY_THRESHOLD).  Falls back to
               difflib SequenceMatcher if rapidfuzz is unavailable.

        Every fuzzy attempt (hit AND miss) is optionally logged to
        logs/entity_match_log.jsonl when LOG_ALL_MATCH_ATTEMPTS=True,
        so you can review borderline scores and tune FUZZY_THRESHOLD.
        """
        if not self._cache.is_loaded:
            return []

        tokens    = self._tokenize(question)
        seen_keys: Dict[str, EntityMatch] = {}

        # Lazy-initialise log file handle
        _log_path = _PROJECT_ROOT / "logs" / "entity_match_log.jsonl"
        _log_fh   = None
        if LOG_ALL_MATCH_ATTEMPTS:
            _log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                _log_fh = open(_log_path, "a", encoding="utf-8")
            except OSError as e:
                logger.warning("Could not open entity match log: %s", e)

        import json as _json
        import time as _time
        _ts = _time.strftime("%Y-%m-%dT%H:%M:%S")

        def _log(record: dict):
            if _log_fh:
                _log_fh.write(_json.dumps(record, ensure_ascii=False) + "\n")

        all_keys = self._get_all_keys()

        try:
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
                    _log({"ts": _ts, "question": question[:80], "token": token,
                          "match_type": "exact", "matched_key": token,
                          "score": 100, "hit": True, "tables": tables})
                    continue

                # 2. Fuzzy match
                if not self._fuzzy or len(token) < MIN_TOKEN_LENGTH:
                    _log({"ts": _ts, "question": question[:80], "token": token,
                          "match_type": "skip", "hit": False,
                          "reason": "fuzzy disabled or token too short"})
                    continue

                if _RAPIDFUZZ_AVAILABLE:
                    # rapidfuzz.process.extractOne returns (match, score, key_index)
                    # WRatio handles abbreviations, partial matches, case differences.
                    result = _rfprocess.extractOne(
                        token, all_keys,
                        scorer=_rfuzz.WRatio,
                        score_cutoff=FUZZY_THRESHOLD,
                    )
                    if result is not None:
                        best_key, best_score_rf, _ = result
                        # normalise to 0–1 for EntityMatch.score field consistency
                        best_score = best_score_rf / 100.0
                        entries = self._cache.get_entries(best_key)
                        if entries:
                            tables   = list({e["table"] for e in entries})
                            original = entries[0]["original"]
                            m = EntityMatch(
                                original_token=token, matched_value=original,
                                matched_key=best_key, tables=tables,
                                match_type="fuzzy_rapidfuzz",
                                score=best_score,
                            )
                            if best_key not in seen_keys or seen_keys[best_key].score < best_score:
                                seen_keys[best_key] = m
                            _log({"ts": _ts, "question": question[:80], "token": token,
                                  "match_type": "fuzzy_rapidfuzz", "matched_key": best_key,
                                  "score": best_score_rf, "hit": True, "tables": tables})
                        else:
                            _log({"ts": _ts, "question": question[:80], "token": token,
                                  "match_type": "fuzzy_rapidfuzz", "matched_key": best_key,
                                  "score": best_score_rf, "hit": False,
                                  "reason": "no cache entries for matched key"})
                    else:
                        # Log the best score we found even though it didn't pass threshold
                        # so the user can tune FUZZY_THRESHOLD.
                        try:
                            top = _rfprocess.extractOne(token, all_keys, scorer=_rfuzz.WRatio)
                            best_score_miss = top[1] if top else 0
                            best_key_miss   = top[0] if top else ""
                        except Exception:
                            best_score_miss, best_key_miss = 0, ""
                        _log({"ts": _ts, "question": question[:80], "token": token,
                              "match_type": "fuzzy_rapidfuzz", "hit": False,
                              "best_miss_key": best_key_miss,
                              "best_miss_score": best_score_miss,
                              "threshold": FUZZY_THRESHOLD})

                else:
                    # difflib fallback
                    import difflib as _difflib_mod
                    best_key, best_score = "", 0.0
                    for key in all_keys:
                        if abs(len(key) - len(token)) > 8:
                            continue
                        score = _difflib_mod.SequenceMatcher(None, token, key).ratio()
                        if score > best_score:
                            best_score = score
                            best_key   = key

                    if best_score >= FUZZY_THRESHOLD_DIFFLIB:
                        entries = self._cache.get_entries(best_key)
                        if entries:
                            tables   = list({e["table"] for e in entries})
                            original = entries[0]["original"]
                            m = EntityMatch(
                                original_token=token, matched_value=original,
                                matched_key=best_key, tables=tables,
                                match_type="fuzzy_difflib",
                                score=best_score,
                            )
                            if best_key not in seen_keys or seen_keys[best_key].score < best_score:
                                seen_keys[best_key] = m
                        _log({"ts": _ts, "question": question[:80], "token": token,
                              "match_type": "fuzzy_difflib", "matched_key": best_key,
                              "score": round(best_score * 100, 1), "hit": bool(entries)})
                    else:
                        _log({"ts": _ts, "question": question[:80], "token": token,
                              "match_type": "fuzzy_difflib", "hit": False,
                              "best_score": round(best_score * 100, 1),
                              "threshold": FUZZY_THRESHOLD_DIFFLIB * 100})
        finally:
            if _log_fh:
                _log_fh.close()

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
            run_all_matched_templates,
        )

        tmpl_result = find_template_for_entity(
            entity_tables=matched_tables,
            question_lower=question.lower(),
        )

        _COMPOUND_TABLES = {"janaushadhi", "medicinedetails"}

        def _pick_entity_value(template_entity_tables):
            """
            Pick the entity value string to use as the LIKE query parameter.

            Priority order for the chosen value:
              1. matched_value, if it is a single word (no compound drug name
                 pollution) — preserves cache capitalisation.
              2. matched_key, if the match was fuzzy (matched_key holds the
                 correctly-spelled cache key, e.g. 'ashwagandha', whereas
                 original_token may be misspelled, e.g. 'ashwaganda').
              3. original_token for compound-name tables (janaushadhi /
                 medicinedetails) — broadens LIKE beyond one specific compound.
              4. entity_value as a last-resort fallback.
            """
            for m in matches:
                if set(m.tables) & set(template_entity_tables):
                    mv_words = m.matched_value.split()
                    if len(mv_words) == 1:
                        # Simple single-word value — use it directly
                        return m.matched_value
                    if m.match_type.startswith("fuzzy"):
                        # Fuzzy hit: matched_key has correct spelling; use it
                        return m.matched_key
                    # Exact hit on a compound name — use original_token to broaden
                    return m.original_token
            return entity_value  # fallback to best-scored match

        if tmpl_result is not None:
            # ── Single-template path (clear winner) ──────────────────────────
            template_id, template = tmpl_result
            ev = _pick_entity_value(template.entity_tables)

            logger.info("Using template '%s' for entity '%s'.", template_id, ev)
            try:
                conn = self._get_conn()
                raw  = execute_template(template_id, ev, conn)
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

        else:
            # ── Multi-table path (ambiguous — run all matched templates) ──────
            logger.info(
                "Ambiguous entity '%s' — running all templates for tables %s.",
                entity_value, matched_tables,
            )
            try:
                conn = self._get_conn()
                all_raw = run_all_matched_templates(
                    entity_tables=matched_tables,
                    entity_value=_pick_entity_value(matched_tables),
                    conn=conn,
                )
            except RuntimeError as e:
                return StructuredResult(
                    path="multi_template", template_id=None, question=question,
                    entity_matches=matches, matched_tables=matched_tables,
                    sql=None, rows=[], columns=[], rows_returned=0,
                    provenance_label="DB:" + ",".join(matched_tables),
                    error=str(e),
                )

            # Merge rows from all templates; label each row with its source template
            merged_rows: List[Dict] = []
            all_columns: List[str] = []
            all_tables_found: List[str] = []
            errors = []
            for raw in all_raw:
                if raw.get("error"):
                    errors.append(f"{raw.get('template_id','?')}: {raw['error']}")
                    continue
                for row in raw.get("rows", []):
                    labeled_row = dict(row)
                    labeled_row["_source_template"] = raw.get("template_id", "?")
                    labeled_row["_source_table"]    = raw.get("source_table", "?")
                    merged_rows.append(labeled_row)
                for col in raw.get("columns", []):
                    if col not in all_columns:
                        all_columns.append(col)
                for tbl in raw.get("source_table", "").split(","):
                    if tbl and tbl not in all_tables_found:
                        all_tables_found.append(tbl)

            return StructuredResult(
                path="multi_template", template_id=None, question=question,
                entity_matches=matches, matched_tables=all_tables_found or matched_tables,
                sql=None,
                rows=merged_rows,
                columns=all_columns,
                rows_returned=len(merged_rows),
                provenance_label="DB:" + ",".join(all_tables_found or matched_tables),
                error="; ".join(errors) if errors else None,
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
