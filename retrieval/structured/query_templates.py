"""
Fast-Path SQL Query Templates
==============================

Parameterised, validated SELECT-only templates for the most critical
lookup patterns. These take priority over LLM-generated SQL when an
entity matches a templated pattern -- guaranteeing correct, exact SQL
for safety-critical facts (dosages, drug prices, compositions, etc.).

All templates:
  - Are SELECT-only (no INSERT / UPDATE / DELETE / DDL)
  - Include LIMIT to bound result set size
  - Use %s parameterisation (psycopg2) -- never f-strings -- to prevent
    SQL injection through user input
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import psycopg2

logger = logging.getLogger(__name__)


@dataclass
class Template:
    """
    A single fast-path query template.

    Attributes:
        template_id:   Unique string key (snake_case).
        description:   Human-readable description of what this looks up.
        tables:        Tables touched by this query (for documentation).
        entity_tables: Tables whose entity cache entries trigger this template.
        sql:           Parameterised SQL. Use %s for the entity value.
                       Must be SELECT-only with a LIMIT clause.
        param_columns: Column(s) the %s parameter is compared against.
        row_limit:     Max rows returned (must match LIMIT in sql).
    """
    template_id:   str
    description:   str
    tables:        List[str]
    entity_tables: List[str]
    sql:           str
    param_columns: List[str]
    row_limit:     int = 20


# ── Template definitions ─────────────────────────────────────────────────────
# Order matters: first matching template wins when multiple candidates exist.

TEMPLATES: Dict[str, Template] = {

    "drug_price_lookup": Template(
        template_id   = "drug_price_lookup",
        description   = "Price / MRP / cost of a generic drug from Jan Aushadhi catalog",
        tables        = ["janaushadhi"],
        entity_tables = ["janaushadhi"],
        sql           = """
            SELECT genericname, unitsize, mrp, groupname, drugcode
            FROM janaushadhi
            WHERE LOWER(genericname) LIKE LOWER(%s)
            ORDER BY mrp
            LIMIT 20
        """,
        param_columns = ["genericname"],
    ),

    "drug_uses_lookup": Template(
        template_id   = "drug_uses_lookup",
        description   = "Uses / indications for a medicine from the medicine details catalog",
        tables        = ["medicinedetails"],
        entity_tables = ["medicinedetails"],
        sql           = """
            SELECT medicinename, uses, composition, manufacturer
            FROM medicinedetails
            WHERE LOWER(medicinename) LIKE LOWER(%s)
            LIMIT 20
        """,
        param_columns = ["medicinename"],
    ),

    "drug_composition_lookup": Template(
        template_id   = "drug_composition_lookup",
        description   = "Composition / active ingredients of a medicine",
        tables        = ["medicinedetails"],
        entity_tables = ["medicinedetails"],
        sql           = """
            SELECT medicinename, composition, manufacturer
            FROM medicinedetails
            WHERE LOWER(medicinename) LIKE LOWER(%s)
            LIMIT 20
        """,
        param_columns = ["medicinename"],
    ),

    "drug_sideeffects_lookup": Template(
        template_id   = "drug_sideeffects_lookup",
        description   = "Side effects of a medicine",
        tables        = ["medicinedetails"],
        entity_tables = ["medicinedetails"],
        sql           = """
            SELECT medicinename, sideeffects, composition
            FROM medicinedetails
            WHERE LOWER(medicinename) LIKE LOWER(%s)
            LIMIT 20
        """,
        param_columns = ["medicinename"],
    ),

    "herb_profile_lookup": Template(
        template_id   = "herb_profile_lookup",
        description   = "Full profile of an Ayurveda herb (name, botanical, family, properties)",
        tables        = ["herb", "herbsynonym"],
        entity_tables = ["herb", "herbsynonym"],
        sql           = """
            SELECT h.herbid, h.name, h.botanicalname, h.family, h.englishname,
                   h.virya, h.vipaka, h.tridosha, h.preview,
                   array_agg(DISTINCT s.synonym) FILTER (WHERE s.synonym IS NOT NULL)
                       AS synonyms
            FROM herb h
            LEFT JOIN herbsynonym s ON s.herbid = h.herbid
            WHERE LOWER(h.name)          LIKE LOWER(%s)
               OR LOWER(h.botanicalname) LIKE LOWER(%s)
               OR LOWER(h.englishname)   LIKE LOWER(%s)
               OR LOWER(s.synonym)       LIKE LOWER(%s)
            GROUP BY h.herbid
            LIMIT 20
        """,
        param_columns = ["name", "botanicalname", "englishname"],
    ),

    "herb_dosha_lookup": Template(
        template_id   = "herb_dosha_lookup",
        description   = "Dosha effects (Kapha/Vata/Pitta) of an Ayurveda herb",
        tables        = ["herb", "herbdoshaeffect", "dosha"],
        entity_tables = ["herb", "herbsynonym"],
        sql           = """
            SELECT h.name AS herb_name, h.botanicalname,
                   d.name AS dosha, hde.effect
            FROM herb h
            JOIN herbdoshaeffect hde ON hde.herbid = h.herbid
            JOIN dosha d             ON d.doshaid  = hde.doshaid
            WHERE LOWER(h.name)          LIKE LOWER(%s)
               OR LOWER(h.botanicalname) LIKE LOWER(%s)
               OR LOWER(h.englishname)   LIKE LOWER(%s)
            ORDER BY d.name
            LIMIT 20
        """,
        param_columns = ["name", "botanicalname", "englishname"],
    ),

    "herb_indications_lookup": Template(
        template_id   = "herb_indications_lookup",
        description   = "Classical indications (Ayurveda uses) of an herb",
        tables        = ["herb", "herbindication", "classicalindication"],
        entity_tables = ["herb", "herbsynonym", "classicalindication"],
        sql           = """
            SELECT h.name AS herb_name, h.botanicalname,
                   ci.name AS indication
            FROM herb h
            JOIN herbindication hi      ON hi.herbid = h.herbid
            JOIN classicalindication ci ON ci.classicalindicationid = hi.classicalindicationid
            WHERE LOWER(h.name)          LIKE LOWER(%s)
               OR LOWER(h.botanicalname) LIKE LOWER(%s)
               OR LOWER(h.englishname)   LIKE LOWER(%s)
            ORDER BY ci.name
            LIMIT 20
        """,
        param_columns = ["name", "botanicalname", "englishname"],
    ),

    "formulation_dose_lookup": Template(
        template_id   = "formulation_dose_lookup",
        description   = "Dose, dosage, precaution and preferred use of an Ayurveda/Unani formulation",
        tables        = ["ayushformulation"],
        entity_tables = ["ayushformulation"],
        sql           = """
            SELECT name, system, category, dose, precaution, preferreduse,
                   packsize, referencetext
            FROM ayushformulation
            WHERE LOWER(name) LIKE LOWER(%s)
            LIMIT 20
        """,
        param_columns = ["name"],
    ),

    "formulation_by_indication": Template(
        template_id   = "formulation_by_indication",
        description   = "Ayurveda formulations recommended for a classical indication",
        tables        = ["classicalindication", "formulationindication", "ayushformulation"],
        entity_tables = ["classicalindication"],
        sql           = """
            SELECT af.name AS formulation, af.system, af.category,
                   af.dose, af.precaution, ci.name AS indication
            FROM classicalindication ci
            JOIN formulationindication fi ON fi.classicalindicationid = ci.classicalindicationid
            JOIN ayushformulation af      ON af.formulationid = fi.formulationid
            WHERE LOWER(ci.name) LIKE LOWER(%s)
            ORDER BY af.system, af.name
            LIMIT 20
        """,
        param_columns = ["name"],
    ),

    "lab_reference_lookup": Template(
        template_id   = "lab_reference_lookup",
        description   = "What a lab test measures, its unit, and category",
        tables        = ["labtesttype"],
        entity_tables = ["labtesttype"],
        sql           = """
            SELECT testname, unit, category
            FROM labtesttype
            WHERE LOWER(testname) LIKE LOWER(%s)
               OR LOWER(category)  LIKE LOWER(%s)
            LIMIT 20
        """,
        param_columns = ["testname", "category"],
    ),
}

# ── Table -> template index ───────────────────────────────────────────────────
_TABLE_TO_TEMPLATE_IDS: Dict[str, List[str]] = {}
for _tid, _tmpl in TEMPLATES.items():
    for _et in _tmpl.entity_tables:
        _TABLE_TO_TEMPLATE_IDS.setdefault(_et, []).append(_tid)


def find_template_for_entity(
    entity_tables: List[str],
    question_lower: str,
) -> Optional[Tuple[str, Template]]:
    """
    Find the highest-priority matching template given entity tables + question text.

    Priority: definition order in TEMPLATES (first match wins).
    Keyword hints disambiguate when multiple templates share the same entity_tables.

    Returns:
        (template_id, Template) or None if no match.
    """
    KEYWORD_OVERRIDES = {
        "drug_price_lookup": [
            "mrp", "price", "cost", "how much", "rate", "maximum retail",
        ],
        "drug_sideeffects_lookup": [
            "side effect", "side-effect", "adverse", "reaction", "danger", "risk",
        ],
        "drug_composition_lookup": [
            "composition", "ingredient", "contain", "made of", "active", "formula",
        ],
        "drug_uses_lookup": [
            "use", "indication", "treat", "prescribed", "given for", "purpose",
        ],
        "herb_dosha_lookup": [
            "dosha", "vata", "pitta", "kapha", "balance",
        ],
        # formulation_by_indication must win over herb_indications_lookup
        # when the question asks for formulations for a classical indication.
        "formulation_by_indication": [
            "formulation", "formulat", "preparation", "indicated for",
            "which formulation", "recommended for", "prescribe",
        ],
        "herb_indications_lookup": [
            "herb indicat", "classical use of herb", "herb for which",
        ],
        "formulation_dose_lookup": [
            "dose", "dosage", "how much", "precaution", "preferred use",
        ],
        "lab_reference_lookup": [
            "measure", "unit", "what is", "reference range", "normal range",
        ],
    }

    candidate_ids: List[str] = []
    for et in entity_tables:
        for tid in _TABLE_TO_TEMPLATE_IDS.get(et, []):
            if tid not in candidate_ids:
                candidate_ids.append(tid)

    if not candidate_ids:
        return None

    if len(candidate_ids) == 1:
        tid = candidate_ids[0]
        return tid, TEMPLATES[tid]

    # Score by keyword hints; fall back to first candidate
    best_id, best_score = candidate_ids[0], 0
    for tid in candidate_ids:
        score = sum(
            1 for kw in KEYWORD_OVERRIDES.get(tid, [])
            if kw in question_lower
        )
        if score > best_score:
            best_score = score
            best_id = tid

    return best_id, TEMPLATES[best_id]


def build_template_params(template: Template, entity_value: str) -> Tuple:
    """
    Build the psycopg2 parameter tuple for a template.

    Wraps entity_value in SQL LIKE wildcards. The number of %s placeholders
    is counted from the SQL so multi-param templates are handled correctly.
    """
    like_value = f"%{entity_value}%"
    n_params = template.sql.count("%s")
    return tuple(like_value for _ in range(n_params))


def execute_template(
    template_id: str,
    entity_value: str,
    conn: psycopg2.extensions.connection,
) -> Dict[str, Any]:
    """
    Execute a fast-path template and return a structured result dict.

    Args:
        template_id:  Key from TEMPLATES dict.
        entity_value: The matched entity string (used as LIKE parameter).
        conn:         Open psycopg2 connection (must be read-only).

    Returns:
        Dict with keys: path, template_id, sql, rows, columns, entity_value, error.
    """
    template = TEMPLATES.get(template_id)
    if template is None:
        return {
            "path": "template", "template_id": template_id,
            "sql": "", "rows": [], "columns": [],
            "entity_value": entity_value,
            "error": f"Unknown template_id: {template_id}",
        }

    params = build_template_params(template, entity_value)
    cur = conn.cursor()
    try:
        cur.execute(template.sql, params)
        col_names = [desc[0] for desc in cur.description]
        rows = [dict(zip(col_names, row)) for row in cur.fetchall()]
        logger.info(
            "Template '%s' for entity '%s': %d rows.",
            template_id, entity_value, len(rows)
        )
        return {
            "path":         "template",
            "template_id":  template_id,
            "sql":          cur.mogrify(template.sql, params).decode("utf-8"),
            "rows":         rows,
            "columns":      col_names,
            "entity_value": entity_value,
            "error":        None,
        }
    except Exception as e:
        logger.error("Template '%s' failed: %s", template_id, e)
        try:
            conn.rollback()
        except Exception:
            pass
        return {
            "path": "template", "template_id": template_id,
            "sql": "", "rows": [], "columns": [],
            "entity_value": entity_value, "error": str(e),
        }
    finally:
        cur.close()
