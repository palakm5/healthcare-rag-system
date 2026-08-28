"""
Structured Database — Table Descriptions & Signal Keywords
===========================================================

Two purposes:

1. **Disambiguation routing** (`SIGNAL_KEYWORDS`)
   When an entity matches multiple tables (e.g. "Ashwagandha" hits both
   ``herb`` and ``janaushadhi``), ``find_template_for_entity`` scans the
   question for the signal keywords defined here to choose the right table.
   Keywords are matched as lowercased substrings.  If no signal fires the
   code runs lookups against ALL matched tables and returns combined results,
   which is always safer than silently picking the wrong one.

2. **Developer documentation** (`TABLE_DESCRIPTIONS`)
   Plain-English description of what each table contains and when it should
   be used.  Referenced by:
     - ``find_template_for_entity`` (LLM secondary disambiguation step)
     - Human maintainers writing new templates or entity-cache entries
   The ``limitations`` field makes known gaps explicit so future
   template-writers don't accidentally query a missing column (e.g. the
   HbA1c bug where ``normal_range`` was assumed to exist in ``labtesttype``).

Adding a new table
------------------
1. Add an entry to TABLE_DESCRIPTIONS.
2. Add signal keywords to SIGNAL_KEYWORDS (at least the most
   discriminating words; leave empty dict if the table has no
   near-duplicates).
3. Re-run ``build_entity_cache`` if the table holds entity values.
"""

from typing import Dict, List


# ── Signal keywords ──────────────────────────────────────────────────────────
# Maps table_name → list of lowercase keyword substrings.
# When a question contains ANY of these keywords, that table is strongly
# preferred over other tables that matched the same entity.
# Keep keywords specific enough that they don't fire on unrelated questions.
# Multiple tables can list the same keyword — scores are compared; highest wins.

SIGNAL_KEYWORDS: Dict[str, List[str]] = {

    # ── Drug pricing (Jan Aushadhi scheme) ───────────────────────────────────
    "janaushadhi": [
        "jan aushadhi", "janaushadhi", "price", "cost", "mrp",
        "how much", "maximum retail", "rupee", "rs.", "₹",
        "cheap", "affordable", "generic price",
    ],

    # ── Detailed medicine info (brand, composition, side effects, uses) ──────
    "medicinedetails": [
        "side effect", "side-effect", "adverse", "adverse effect",
        "composition", "ingredient", "active ingredient", "made of",
        "contain", "manufacturer", "brand", "uses of", "indication of",
        "prescribed for", "what is it used for",
    ],

    # ── Ayurvedic herb properties (Virya, Vipaka, Tridosha, botanical) ───────
    "herb": [
        "virya", "vipaka", "tridosha", "dosha", "rasa", "guna",
        "botanical", "ayurvedic property", "ayurveda herb",
        "herb properties", "ayurvedic herb", "properties of",
        "latin name", "botanical name", "family of", "english name",
    ],

    # ── Herb synonyms — routes to herb_profile_lookup same as herb ───────────
    "herbsynonym": [
        "also known as", "synonym", "other name",
    ],

    # ── Lab test metadata (unit, category — NO reference ranges) ─────────────
    "labtesttype": [
        "lab test", "laboratory test", "blood test", "urine test",
        "test measure", "what does the test", "what is measured",
        "unit of", "reported in", "what unit",
        "hba1c", "creatinine", "haemoglobin", "hemoglobin",
        "glucose", "bilirubin", "cholesterol", "platelet",
    ],

    # ── Ayurveda classical formulations ──────────────────────────────────────
    "ayushformulation": [
        "formulation", "churna", "asava", "arishta", "ghrita",
        "taila", "kwath", "bhasma", "leha", "vati",
        "preparation", "classical preparation", "dose of formulation",
    ],

    # ── AyurKosh herb catalog (Devanagari / Sanskrit names) ──────────────────
    "ayurkoshherb": [
        "ayurkosh", "sanskrit name", "devanagari",
    ],

    # ── Classical Ayurveda indications ────────────────────────────────────────
    "classicalindication": [
        "classical indication", "ayurvedic indication",
        "indicated for", "classical use",
    ],

    # ── Vital sign types ──────────────────────────────────────────────────────
    "vitalsigntype": [
        "vital sign", "blood pressure", "pulse", "temperature",
        "respiratory rate", "spo2", "oxygen saturation",
    ],

    # ── Dosha names ───────────────────────────────────────────────────────────
    "dosha": [
        "vata", "pitta", "kapha", "tridosha",
    ],
}


# ── Table descriptions ────────────────────────────────────────────────────────

TABLE_DESCRIPTIONS: Dict[str, Dict] = {

    "janaushadhi": {
        "description": (
            "Contains the Jan Aushadhi scheme generic drug catalog. "
            "Each row is a drug with: generic name (may be compound like "
            "'Paracetamol 500mg Tablets'), pack size (unitsize), MRP price, "
            "and therapeutic group. "
            "Use for: drug cost / price / MRP questions."
        ),
        "key_columns": ["genericname", "unitsize", "mrp", "groupname"],
        "row_count_approx": 2439,
        "limitations": [
            "No drug composition, mechanism, or side-effect data.",
            "No brand names — only generic names.",
            "MRP may be 0.00 for some entries (data quality issue in source).",
            "Does NOT contain Ayurvedic properties (Virya, Vipaka, etc.) "
            "even though some Ayurvedic drugs appear in the catalog.",
        ],
    },

    "medicinedetails": {
        "description": (
            "Detailed medicine information catalog: brand/generic name, "
            "composition (active ingredients), uses/indications, side effects, "
            "manufacturer. "
            "Use for: questions about what a medicine is used for, what it "
            "contains, its side effects."
        ),
        "key_columns": ["medicinename", "uses", "composition", "sideeffects", "manufacturer"],
        "row_count_approx": 17531,
        "limitations": [
            "No pricing data.",
            "Does not contain Ayurvedic herb properties.",
        ],
    },

    "herb": {
        "description": (
            "Ayurvedic herb database from the NHP corpus. Each herb has: "
            "name (Sanskrit/common), botanical name, family, English name, "
            "Virya (potency: Ushna/Sheeta), Vipaka (post-digestive effect: "
            "Madhura/Katu/Amla), Tridosha flag, and a plain-English preview. "
            "Use for: questions about Ayurvedic herb properties, Virya, Vipaka, "
            "Tridosha, botanical classification."
        ),
        "key_columns": ["name", "botanicalname", "family", "englishname", "virya", "vipaka", "tridosha"],
        "row_count_approx": 360,
        "limitations": [
            "Does NOT contain dosage or MRP/price data.",
            "Does NOT contain normal reference ranges.",
            "Tridosha is stored as a boolean flag (True/False), "
            "not a per-dosha breakdown — use herbdoshaeffect join for that.",
        ],
    },

    "herbsynonym": {
        "description": (
            "Alternate names (synonyms) for herbs in the herb table. "
            "Joined to herb via herbid. "
            "Use for: entity matching when a question uses an alternate name."
        ),
        "key_columns": ["synonym", "herbid"],
        "row_count_approx": 1187,
        "limitations": [
            "Lookup-only; routes to herb_profile_lookup template.",
        ],
    },

    "labtesttype": {
        "description": (
            "Clinical lab test reference table: test name, measurement unit, "
            "and test category (Chemistry, Haematology, etc.). "
            "Use for: questions about what unit a lab test is reported in, "
            "or what category it belongs to."
        ),
        "key_columns": ["testname", "unit", "category"],
        "row_count_approx": 31,
        "limitations": [
            "⚠️  Does NOT contain normal/reference ranges. "
            "The table has no 'normal_range' column. "
            "For reference ranges, use the unstructured ICMR/NHP guideline chunks.",
            "Only 31 test types — many common tests may be absent.",
        ],
    },

    "ayushformulation": {
        "description": (
            "AYUSH (Ayurveda/Unani/Siddha) classical formulations: name, "
            "system (Ayurveda/Unani), category, dose, precaution, preferred use, "
            "pack size, and reference text. "
            "Use for: dose/dosage/precaution questions about classical formulations, "
            "or formulation name lookups."
        ),
        "key_columns": ["name", "system", "category", "dose", "precaution"],
        "row_count_approx": 863,
        "limitations": [
            "Dose is free-text (not structured numeric).",
        ],
    },

    "classicalindication": {
        "description": (
            "Classical Ayurvedic indications (Sanskrit/English). "
            "Used via join with formulationindication and herbindication. "
            "Use for: looking up which formulations or herbs are indicated for "
            "a given classical condition."
        ),
        "key_columns": ["name"],
        "row_count_approx": 1140,
        "limitations": [
            "Names are Romanised Sanskrit — spelling variants may not match.",
        ],
    },

    "ayurkoshherb": {
        "description": (
            "AyurKosh herb catalog (comprehensive Ayurveda database). "
            "May include Devanagari or Sanskrit names not in the 'herb' table. "
            "Use for: AyurKosh-specific herb lookups."
        ),
        "key_columns": ["name", "latinname"],
        "row_count_approx": 2647,
        "limitations": [
            "Does not have Virya/Vipaka columns — those are in the 'herb' table.",
        ],
    },

    "vitalsigntype": {
        "description": (
            "Small lookup table of clinical vital sign types "
            "(Blood Pressure, Pulse, Temperature, etc.)."
        ),
        "key_columns": ["name"],
        "row_count_approx": 5,
        "limitations": [
            "Only 5 rows — extremely limited.",
        ],
    },

    "dosha": {
        "description": "Vata, Pitta, Kapha dosha names.",
        "key_columns": ["name"],
        "row_count_approx": 4,
        "limitations": ["Lookup-only; used in joins."],
    },
}
