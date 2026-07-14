"""
Load the updated 48-table corpus into your new Supabase schema.
===================================================================
Reads unified_clinical_pharma_ayush_corpus.xlsx sheet-by-sheet, in
FK-safe dependency order, into the schema defined by coep_schema.sql.

This replaces the old 58-table loader: the entire normalized Pharmacy
domain (manufacturer/medicine/activeingredient/... 12 tables) is gone,
replaced by two flat, unnormalized reference tables loaded as-is:
janaushadhi and medicinedetails.

Usage:
    pip install pandas openpyxl sqlalchemy psycopg2-binary python-dotenv
    1. Run coep_schema.sql against your Supabase project first (creates
       all 48 empty tables).
    2. Put this script + the xlsx in the same folder, with a .env file
       containing DATABASE_URL=postgresql://... (see earlier setup).
    3. python load_v3_to_postgres.py
"""
import os
import re
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://user:password@your-host:5432/postgres"
)
WORKBOOK_PATH = "unified_clinical_pharma_ayush_corpus.xlsx"

# Dependency order matches coep_schema.sql's CREATE TABLE order exactly --
# do not reorder without checking that file's FK references.
TABLE_ORDER = [
    "dataset", "patient", "admission", "comorbiditytype", "patientcomorbidity",
    "diagnosistype", "patientdiagnosis", "labtesttype", "labresult", "vitalsigntype",
    "vitalsign", "reproductiveprofile", "diabetesprofile", "cardiovascularprofile",
    "renalprofile", "liverprofile", "janaushadhi", "medicinedetails", "herb",
    "herbsynonym", "partused", "herbpartused", "classicalindication", "herbindication",
    "dosha", "herbdoshaeffect", "rasagunaattribute", "herbrasaguna", "ayushformulation",
    "formulationindication", "potency", "formulationpotency", "ayurkoshherb",
    "ayurkoshremedy", "ayurkoshgroup", "ayurkoshherbgroup", "ayurkoshherbalternate",
    "ayurkoshquality", "ayurkoshherbquality", "ayurkoshcompound", "ayurkoshherbcompound",
    "ayurkoshdisease", "ayurkoshlakshan", "ayurkoshdiseaseremedy", "ayurkoshdiseaselakshan",
    "ayurkoshcompounddisease", "ayurkoshcompoundlakshan", "ayurkoshlakshanprakritidhatu",
]

# db table name (lowercase) -> actual sheet name in the xlsx (original CamelCase)
SHEET_NAME = {
    "dataset": "Dataset", "patient": "Patient", "admission": "Admission",
    "comorbiditytype": "ComorbidityType", "patientcomorbidity": "PatientComorbidity",
    "diagnosistype": "DiagnosisType", "patientdiagnosis": "PatientDiagnosis",
    "labtesttype": "LabTestType", "labresult": "LabResult", "vitalsigntype": "VitalSignType",
    "vitalsign": "VitalSign", "reproductiveprofile": "ReproductiveProfile",
    "diabetesprofile": "DiabetesProfile", "cardiovascularprofile": "CardiovascularProfile",
    "renalprofile": "RenalProfile", "liverprofile": "LiverProfile",
    "janaushadhi": "JanAushadhi", "medicinedetails": "MedicineDetails", "herb": "Herb",
    "herbsynonym": "HerbSynonym", "partused": "PartUsed", "herbpartused": "HerbPartUsed",
    "classicalindication": "ClassicalIndication", "herbindication": "HerbIndication",
    "dosha": "Dosha", "herbdoshaeffect": "HerbDoshaEffect",
    "rasagunaattribute": "RasaGunaAttribute", "herbrasaguna": "HerbRasaGuna",
    "ayushformulation": "AyushFormulation", "formulationindication": "FormulationIndication",
    "potency": "Potency", "formulationpotency": "FormulationPotency",
    "ayurkoshherb": "AyurKoshHerb", "ayurkoshremedy": "AyurKoshRemedy",
    "ayurkoshgroup": "AyurKoshGroup", "ayurkoshherbgroup": "AyurKoshHerbGroup",
    "ayurkoshherbalternate": "AyurKoshHerbAlternate", "ayurkoshquality": "AyurKoshQuality",
    "ayurkoshherbquality": "AyurKoshHerbQuality", "ayurkoshcompound": "AyurKoshCompound",
    "ayurkoshherbcompound": "AyurKoshHerbCompound", "ayurkoshdisease": "AyurKoshDisease",
    "ayurkoshlakshan": "AyurKoshLakshan", "ayurkoshdiseaseremedy": "AyurKoshDiseaseRemedy",
    "ayurkoshdiseaselakshan": "AyurKoshDiseaseLakshan",
    "ayurkoshcompounddisease": "AyurKoshCompoundDisease",
    "ayurkoshcompoundlakshan": "AyurKoshCompoundLakshan",
    "ayurkoshlakshanprakritidhatu": "AyurKoshLakshanPrakritiDhatu",
}

# Every column this schema expects, per table (used to auto-match source headers
# regardless of spacing/casing/punctuation differences).
SCHEMA_COLUMNS = {
    "dataset": ["datasetid", "datasetname", "sourceorg", "domain", "originalrowcount", "originalcolcount"],
    "patient": ["patientkey", "datasetid", "originalpatientid", "age", "gender"],
    "admission": ["admissionid", "patientkey", "admissiondate", "dischargedate", "admissiontype",
                  "monthyear", "lengthofstaydays", "iculengthofstaydays", "ruralurban", "outcome"],
    "comorbiditytype": ["comorbiditytypeid", "name"],
    "patientcomorbidity": ["patientkey", "comorbiditytypeid", "presentflag"],
    "diagnosistype": ["diagnosistypeid", "name", "category"],
    "patientdiagnosis": ["patientkey", "diagnosistypeid", "presentflag"],
    "labtesttype": ["labtesttypeid", "testname", "unit", "category"],
    "labresult": ["labresultid", "patientkey", "labtesttypeid", "resultvalue"],
    "vitalsigntype": ["vitalsigntypeid", "name", "unit"],
    "vitalsign": ["vitalsignid", "patientkey", "vitalsigntypeid", "value"],
    "reproductiveprofile": ["patientkey", "pcosflag", "betahcg_i", "betahcg_ii", "amh"],
    "diabetesprofile": ["patientkey", "diagnosistemp", "diabetesstatus", "diabetestarget"],
    "cardiovascularprofile": ["patientkey", "chestpaintype", "restingecg", "exerciseangina",
                              "oldpeak", "slope", "nummajorvessels", "classificationtarget"],
    "renalprofile": ["patientkey", "specificgravity", "urinealbumin", "urinesugar", "rbcurine",
                      "puscell", "puscellclumps", "bacteria", "appetiteflag", "pedaledemaflag", "classtarget"],
    "liverprofile": ["patientkey", "selector"],
    "janaushadhi": ["srno", "drugcode", "genericname", "unitsize", "mrp", "groupname"],
    "medicinedetails": ["medicinedetailid", "medicinename", "composition", "uses", "sideeffects",
                         "manufacturer", "excellentreviewpct", "averagereviewpct", "poorreviewpct"],
    "herb": ["herbid", "name", "botanicalname", "family", "englishname", "virya", "vipaka",
             "tridosha", "preview", "link"],
    "herbsynonym": ["herbsynonymid", "herbid", "synonym"],
    "partused": ["partusedid", "name"],
    "herbpartused": ["herbid", "partusedid"],
    "classicalindication": ["classicalindicationid", "name"],
    "herbindication": ["herbid", "classicalindicationid"],
    "dosha": ["doshaid", "name"],
    "herbdoshaeffect": ["herbdoshaeffectid", "herbid", "doshaid", "effect"],
    "rasagunaattribute": ["attributeid", "attributetype", "value"],
    "herbrasaguna": ["herbid", "attributeid"],
    "ayushformulation": ["formulationid", "system", "category", "name", "referencetext",
                          "packsize", "dose", "precaution", "preferreduse"],
    "formulationindication": ["formulationid", "classicalindicationid"],
    "potency": ["potencyid", "name"],
    "formulationpotency": ["formulationid", "potencyid"],
    "ayurkoshherb": ["ayurkoshherbid", "name", "latinname", "family", "herbid"],
    "ayurkoshremedy": ["remedyid", "name", "ayurkoshherbid"],
    "ayurkoshgroup": ["groupid", "name"],
    "ayurkoshherbgroup": ["ayurkoshherbid", "groupid"],
    "ayurkoshherbalternate": ["ayurkoshherbid", "alternateayurkoshherbid"],
    "ayurkoshquality": ["qualityid", "name"],
    "ayurkoshherbquality": ["ayurkoshherbid", "qualityid"],
    "ayurkoshcompound": ["compoundid", "name"],
    "ayurkoshherbcompound": ["ayurkoshherbid", "compoundid"],
    "ayurkoshdisease": ["diseaseid", "name"],
    "ayurkoshlakshan": ["lakshanid", "name"],
    "ayurkoshdiseaseremedy": ["diseaseid", "remedyid", "reference"],
    "ayurkoshdiseaselakshan": ["diseaseid", "lakshanid", "reference"],
    "ayurkoshcompounddisease": ["compoundid", "diseaseid"],
    "ayurkoshcompoundlakshan": ["compoundid", "lakshanid"],
    "ayurkoshlakshanprakritidhatu": ["rowid", "effectonparameter_1", "effectonparameter_2", "name",
                                     "parameter", "lakshan", "reference", "lakshaninenglish"],
}


def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def align_columns(table, df):
    """Rename/drop/add columns so df matches SCHEMA_COLUMNS[table] exactly."""
    expected = SCHEMA_COLUMNS[table]
    norm_to_expected = {norm(c): c for c in expected}

    rename = {}
    for col in df.columns:
        target = norm_to_expected.get(norm(col))
        if target:
            rename[col] = target
    df = df.rename(columns=rename)

    # drop anything not in the schema (stray "Unnamed:" artifacts etc.)
    df = df[[c for c in df.columns if c in expected]]

    # medicinedetails has no source ID column -- generate one
    if table == "medicinedetails" and "medicinedetailid" not in df.columns:
        df.insert(0, "medicinedetailid", range(1, len(df) + 1))

    # reorder to match schema column order (cosmetic, but keeps things predictable)
    df = df[[c for c in expected if c in df.columns]]
    return df


def main():
    if "your-host" in DATABASE_URL:
        raise SystemExit(
            "DATABASE_URL isn't set. Create a .env file next to this script with:\n"
            "  DATABASE_URL=postgresql://postgres:[PASSWORD]@db.xxxx.supabase.co:5432/postgres"
        )

    engine = create_engine(DATABASE_URL)
    xls = pd.ExcelFile(WORKBOOK_PATH)

    with engine.connect() as conn:
        conn.execute(text("SET session_replication_role = 'replica';"))
        conn.commit()

        for table in TABLE_ORDER:
            sheet_name = SHEET_NAME[table]
            if sheet_name not in xls.sheet_names:
                print(f"SKIP   {table}: sheet '{sheet_name}' not found in workbook")
                continue

            df = pd.read_excel(xls, sheet_name=sheet_name)
            df = align_columns(table, df)

            try:
                df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=5000)
                conn.commit()
                print(f"LOADED {table}: {len(df)} rows")
            except Exception as e:
                conn.rollback()
                print(f"FAILED {table}: {type(e).__name__}: {str(e)[:300]}")
                print("  -> skipped, moving to next table. Re-run just this one after fixing.")
                continue

        conn.execute(text("SET session_replication_role = 'origin';"))
        conn.commit()

    print("\nDone. Run verify_counts_v3.sql to confirm row counts match.")


if __name__ == "__main__":
    main()