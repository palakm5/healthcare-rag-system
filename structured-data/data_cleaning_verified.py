"""
Data cleaning & parsing -- Clinical + Pharmacy/Drug-Reference domains
========================================================================
This is the verified, tested code for the two domains built from raw
source CSVs in this conversation:

  1. CLINICAL domain (16 tables) -- from 6 raw CSVs: D1 (Hero DMC
     cardiology admissions), D3 (PCOS infertility), D4 (Diabetes_MCDD),
     D5 (Cardiovascular Disease Dataset), D6 (Chronic Kidney Disease),
     D7 (ILPD liver dataset).

  2. PHARMACY/DRUG-REFERENCE domain (2 tables) -- CURRENT version.
     janaushadhi and medicinedetails are loaded as flat, unnormalized
     reference tables straight from their source CSVs (per your later
     decision to drop the earlier normalized 12-table version --
     Manufacturer/Medicine/ActiveIngredient/etc -- entirely).

NOT INCLUDED: the Herb / AyushFormulation / AyurKOSH domains (32 tables).
That data arrived pre-built in an uploaded workbook partway through our
conversation -- I never saw or wrote the raw-CSV parsing for those
tables, so there's no verified code for them to show here. If you have
the original raw source files for that domain, send them over and I'll
write and test that ETL the same way as what's below.
"""
import pandas as pd
import numpy as np
import re

# =============================================================================
# PART 1 -- CLINICAL DOMAIN  (16 tables: dataset -> liverprofile)
# =============================================================================
def build_clinical_tables(base):
    """
    base: folder containing the 6 raw clinical CSVs, with this structure:
        {base}/Punjab-Heart-Disease-Data/HDHI Admission data.csv          (D1)
        {base}/pcos-data/PCOS_infertility.csv                              (D3)
        {base}/ieee-diabetes-dataset/Diabetes_MCDD.csv                     (D4)
        {base}/ieee-cardiovascular-disease/Cardiovascular_Disease_Dataset.csv (D5)
        {base}/chronic-Kidney-disease/csv_result-chronic_kidney_disease_full.csv (D6)
        {base}/Indian Liver Patient Dataset (ILPD) copy.csv                (D7)
    """
    def clean_cols(df):
        df.columns = [c.strip().strip("'") for c in df.columns]
        return df

    # ---- Load raw sources ---------------------------------------------------
    d1 = clean_cols(pd.read_csv(f"{base}/Punjab-Heart-Disease-Data/HDHI Admission data.csv", encoding="utf-8-sig"))
    d3 = clean_cols(pd.read_csv(f"{base}/pcos-data/PCOS_infertility.csv", encoding="utf-8-sig"))
    d4 = clean_cols(pd.read_csv(f"{base}/ieee-diabetes-dataset/Diabetes_MCDD.csv", encoding="utf-8-sig"))
    d5 = clean_cols(pd.read_csv(f"{base}/ieee-cardiovascular-disease/Cardiovascular_Disease_Dataset.csv", encoding="utf-8-sig"))
    d6 = clean_cols(pd.read_csv(f"{base}/chronic-Kidney-disease/csv_result-chronic_kidney_disease_full.csv",
                                 encoding="utf-8-sig", engine="python", on_bad_lines="warn"))
    d6 = d6.iloc[:, :26]  # drop stray trailing empty column from malformed source rows (rows with an extra comma)
    d7 = pd.read_csv(f"{base}/Indian Liver Patient Dataset (ILPD) copy.csv", header=None,
                      names=["Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "AGRatio", "Selector"])

    d3.columns = ["SlNo", "PatientFileNo", "PCOS_YN", "BetaHCG_I", "BetaHCG_II", "AMH"]

    for df in (d1, d3, d4, d5, d6, d7):
        df.replace("?", np.nan, inplace=True)
    d1["MRD No."] = d1["MRD No."].astype(str)

    # ---- 1. dataset -------------------------------------------------------
    dataset_rows = [
        ("D1", "Hero DMC Heart Institute Admissions", "Hero DMC Cardiology", "Cardiology", len(d1), len(d1.columns)),
        ("D3", "PCOS Infertility", "Gynecology / Endocrinology", "Reproductive Health", len(d3), len(d3.columns)),
        ("D4", "Diabetes_MCDD", "IEEE Diabetes Dataset", "Endocrinology", len(d4), len(d4.columns)),
        ("D5", "Cardiovascular Disease Dataset", "IEEE Cardiovascular Dataset", "Cardiology", len(d5), len(d5.columns)),
        ("D6", "Chronic Kidney Disease", "UCI CKD Dataset", "Nephrology", len(d6), len(d6.columns)),
        ("D7", "ILPD (Indian Liver Patient Dataset)", "UCI ILPD Dataset", "Hepatology", len(d7), len(d7.columns)),
    ]
    Dataset = pd.DataFrame(dataset_rows, columns=["DatasetID", "DatasetName", "SourceOrg", "Domain", "OriginalRowCount", "OriginalColCount"])

    # ---- 2. patient (one row per unique patient; D1 deduped by MRD No. since
    #        patients can be readmitted -- multiple admissions, one patient) ---
    patient_frames = []

    p1_dedup = d1.drop_duplicates(subset=["MRD No."], keep="first")
    patient_frames.append(pd.DataFrame({
        "DatasetID": "D1", "OriginalPatientID": p1_dedup["MRD No."],
        "Age": pd.to_numeric(p1_dedup["AGE"], errors="coerce"),
        "Gender": p1_dedup["GENDER"].map({"M": "M", "F": "F"}),
    }))
    patient_frames.append(pd.DataFrame({
        "DatasetID": "D3", "OriginalPatientID": d3["PatientFileNo"].astype(str),
        "Age": np.nan, "Gender": np.nan,  # D3 carries neither column
    }))
    patient_frames.append(pd.DataFrame({
        "DatasetID": "D4", "OriginalPatientID": (d4.index + 1).astype(str),  # no native ID column
        "Age": pd.to_numeric(d4["Age"], errors="coerce"),
        "Gender": d4["Sex"].map({1: "M", 0: "F"}),
    }))
    patient_frames.append(pd.DataFrame({
        "DatasetID": "D5", "OriginalPatientID": d5["patientid"].astype(str),
        "Age": pd.to_numeric(d5["age"], errors="coerce"),
        "Gender": d5["gender"].map({1: "M", 0: "F"}),
    }))
    patient_frames.append(pd.DataFrame({
        "DatasetID": "D6", "OriginalPatientID": d6["id"].astype(str),
        "Age": pd.to_numeric(d6["age"], errors="coerce"),
        "Gender": np.nan,  # D6 carries no Gender column
    }))
    patient_frames.append(pd.DataFrame({
        "DatasetID": "D7", "OriginalPatientID": (d7.index + 1).astype(str),  # no native ID column
        "Age": pd.to_numeric(d7["Age"], errors="coerce"),
        "Gender": d7["Gender"].map({"Male": "M", "Female": "F"}),
    }))

    Patient = pd.concat(patient_frames, ignore_index=True)
    Patient.insert(0, "PatientKey", range(1, len(Patient) + 1))

    pk_lookup = Patient.set_index(["DatasetID", "OriginalPatientID"])["PatientKey"]

    def get_pk(dataset_id, original_id_series):
        idx = pd.MultiIndex.from_arrays([[dataset_id] * len(original_id_series), original_id_series.astype(str)])
        return pk_lookup.reindex(idx).values

    pk_d1 = get_pk("D1", d1["MRD No."])
    pk_d3 = get_pk("D3", d3["PatientFileNo"].astype(str))
    pk_d4 = get_pk("D4", pd.Series((d4.index + 1).astype(str)))
    pk_d5 = get_pk("D5", d5["patientid"])
    pk_d6 = get_pk("D6", d6["id"])
    pk_d7 = get_pk("D7", pd.Series((d7.index + 1).astype(str)))

    # ---- 3. admission (D1 only, one row per visit) -------------------------
    Admission = pd.DataFrame({
        "PatientKey": pk_d1,
        "AdmissionDate": pd.to_datetime(d1["D.O.A"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "DischargeDate": pd.to_datetime(d1["D.O.D"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "AdmissionType": d1["TYPE OF ADMISSION-EMERGENCY/OPD"],
        "MonthYear": d1["month year"],
        "LengthOfStayDays": pd.to_numeric(d1["DURATION OF STAY"], errors="coerce"),
        "ICULengthOfStayDays": pd.to_numeric(d1["duration of intensive unit stay"], errors="coerce"),
        "RuralUrban": d1["RURAL"],
        "Outcome": d1["OUTCOME"],
    })
    Admission.insert(0, "AdmissionID", range(1, len(Admission) + 1))

    # ---- 4/5. comorbiditytype + patientcomorbidity (D1 + D6, merged concepts) --
    comorbidity_types = ["Diabetes Mellitus", "Hypertension", "Coronary Artery Disease", "Cardiomyopathy",
                          "Chronic Kidney Disease", "Smoking", "Alcohol", "Severe Anaemia", "Anaemia"]
    ComorbidityType = pd.DataFrame({"ComorbidityTypeID": range(1, len(comorbidity_types) + 1), "Name": comorbidity_types})
    ct_id = dict(zip(ComorbidityType["Name"], ComorbidityType["ComorbidityTypeID"]))

    pc_rows = []
    d1_map = {"DM": "Diabetes Mellitus", "HTN": "Hypertension", "CAD": "Coronary Artery Disease",
              "PRIOR CMP": "Cardiomyopathy", "CKD": "Chronic Kidney Disease", "SMOKING": "Smoking",
              "ALCOHOL": "Alcohol", "SEVERE ANAEMIA": "Severe Anaemia", "ANAEMIA": "Anaemia"}
    for col, name in d1_map.items():
        hit = pd.to_numeric(d1[col], errors="coerce") == 1
        pc_rows.append(pd.DataFrame({"PatientKey": pk_d1[hit.values], "ComorbidityTypeID": ct_id[name]}))

    d6_map = {"htn": "Hypertension", "dm": "Diabetes Mellitus", "cad": "Coronary Artery Disease", "ane": "Anaemia"}
    for col, name in d6_map.items():
        hit = d6[col].astype(str).str.strip().str.lower() == "yes"
        pc_rows.append(pd.DataFrame({"PatientKey": pk_d6[hit.values], "ComorbidityTypeID": ct_id[name]}))

    PatientComorbidity = pd.concat(pc_rows, ignore_index=True).dropna(subset=["PatientKey"])
    PatientComorbidity["PatientKey"] = PatientComorbidity["PatientKey"].astype(int)
    PatientComorbidity["PresentFlag"] = True
    PatientComorbidity.drop_duplicates(subset=["PatientKey", "ComorbidityTypeID"], inplace=True)

    # ---- 6/7. diagnosistype + patientdiagnosis (D1 only) -----------------------
    diagnosis_cols = ["STABLE ANGINA", "ACS", "STEMI", "ATYPICAL CHEST PAIN", "HEART FAILURE", "HFREF", "HFNEF",
                       "VALVULAR", "CHB", "SSS", "AKI", "CVA INFRACT", "CVA BLEED", "AF", "VT", "PSVT", "CONGENITAL",
                       "UTI", "NEURO CARDIOGENIC SYNCOPE", "ORTHOSTATIC", "INFECTIVE ENDOCARDITIS", "DVT",
                       "CARDIOGENIC SHOCK", "SHOCK", "PULMONARY EMBOLISM", "CHEST INFECTION"]
    DiagnosisType = pd.DataFrame({"DiagnosisTypeID": range(1, len(diagnosis_cols) + 1), "Name": diagnosis_cols, "Category": "Cardiac"})
    dt_id = dict(zip(DiagnosisType["Name"], DiagnosisType["DiagnosisTypeID"]))

    pd_rows = []
    for col in diagnosis_cols:
        hit = pd.to_numeric(d1[col], errors="coerce") == 1
        pd_rows.append(pd.DataFrame({"PatientKey": pk_d1[hit.values], "DiagnosisTypeID": dt_id[col]}))
    PatientDiagnosis = pd.concat(pd_rows, ignore_index=True).dropna(subset=["PatientKey"])
    PatientDiagnosis["PatientKey"] = PatientDiagnosis["PatientKey"].astype(int)
    PatientDiagnosis["PresentFlag"] = True
    PatientDiagnosis.drop_duplicates(subset=["PatientKey", "DiagnosisTypeID"], inplace=True)

    # ---- 8/9. labtesttype + labresult ------------------------------------------
    lab_types = [
        ("Glucose-Fasting", "mg/dL", "Chemistry"), ("Glucose-PostPrandial", "mg/dL", "Chemistry"),
        ("Glucose-Random", "mg/dL", "Chemistry"), ("Blood Urea", "mg/dL", "Chemistry"),
        ("Serum Creatinine", "mg/dL", "Chemistry"), ("Haemoglobin", "g/dL", "Haematology"),
        ("Total Leukocyte Count", "x10^3/uL", "Haematology"), ("Platelets", "x10^3/uL", "Haematology"),
        ("BNP", "pg/mL", "Cardiac Marker"), ("Raised Cardiac Enzymes", "Y/N", "Cardiac Marker"),
        ("Ejection Fraction", "%", "Cardiac Function"), ("HbA1c", "%", "Chemistry"),
        ("Total Cholesterol", "mg/dL", "Lipid Panel"), ("HDL Cholesterol", "mg/dL", "Lipid Panel"),
        ("LDL Cholesterol", "mg/dL", "Lipid Panel"), ("Triglycerides", "mg/dL", "Lipid Panel"),
        ("TSH", "uIU/mL", "Endocrine"), ("Serum Cholesterol", "mg/dL", "Lipid Panel"),
        ("Sodium", "mEq/L", "Electrolyte"), ("Potassium", "mEq/L", "Electrolyte"),
        ("Packed Cell Volume", "%", "Haematology"), ("White Blood Cell Count", "cells/cumm", "Haematology"),
        ("Red Blood Cell Count", "millions/cumm", "Haematology"), ("Total Bilirubin", "mg/dL", "Liver Function"),
        ("Direct Bilirubin", "mg/dL", "Liver Function"), ("Alkaline Phosphotase", "IU/L", "Liver Function"),
        ("Alanine Aminotransferase", "IU/L", "Liver Function"), ("Aspartate Aminotransferase", "IU/L", "Liver Function"),
        ("Total Proteins", "g/dL", "Liver Function"), ("Albumin", "g/dL", "Liver Function"),
        ("Albumin/Globulin Ratio", "ratio", "Liver Function"),
    ]
    LabTestType = pd.DataFrame(lab_types, columns=["TestName", "Unit", "Category"])
    LabTestType.insert(0, "LabTestTypeID", range(1, len(LabTestType) + 1))
    lt_id = dict(zip(LabTestType["TestName"], LabTestType["LabTestTypeID"]))

    def lab_rows(pk_array, values, test_name):
        v = pd.to_numeric(values, errors="coerce")
        keep = v.notna()
        return pd.DataFrame({"PatientKey": pk_array[keep.values], "LabTestTypeID": lt_id[test_name], "ResultValue": v[keep].values})

    lr_rows = [
        lab_rows(pk_d1, d1["GLUCOSE"], "Glucose-Random"), lab_rows(pk_d1, d1["UREA"], "Blood Urea"),
        lab_rows(pk_d1, d1["CREATININE"], "Serum Creatinine"), lab_rows(pk_d1, d1["HB"], "Haemoglobin"),
        lab_rows(pk_d1, d1["TLC"], "Total Leukocyte Count"), lab_rows(pk_d1, d1["PLATELETS"], "Platelets"),
        lab_rows(pk_d1, d1["BNP"], "BNP"), lab_rows(pk_d1, d1["RAISED CARDIAC ENZYMES"], "Raised Cardiac Enzymes"),
        lab_rows(pk_d1, d1["EF"], "Ejection Fraction"),
        lab_rows(pk_d4, d4["FBS_mg_dL"], "Glucose-Fasting"), lab_rows(pk_d4, d4["PPBS_mg_dL"], "Glucose-PostPrandial"),
        lab_rows(pk_d4, d4["HbA1c_percent"], "HbA1c"), lab_rows(pk_d4, d4["Total_Cholesterol_mg_dL"], "Total Cholesterol"),
        lab_rows(pk_d4, d4["HDL_Cholesterol_mg_dL"], "HDL Cholesterol"), lab_rows(pk_d4, d4["LDL_Cholesterol_mg_dL"], "LDL Cholesterol"),
        lab_rows(pk_d4, d4["Triglycerides_mg_dL"], "Triglycerides"), lab_rows(pk_d4, d4["Serum_Creatinine_mg_dL"], "Serum Creatinine"),
        lab_rows(pk_d4, d4["TSH_uIU_mL"], "TSH"),
        lab_rows(pk_d5, d5["fastingbloodsugar"], "Glucose-Fasting"), lab_rows(pk_d5, d5["serumcholestrol"], "Serum Cholesterol"),
        lab_rows(pk_d6, d6["bgr"], "Glucose-Random"), lab_rows(pk_d6, d6["bu"], "Blood Urea"),
        lab_rows(pk_d6, d6["sc"], "Serum Creatinine"), lab_rows(pk_d6, d6["sod"], "Sodium"),
        lab_rows(pk_d6, d6["pot"], "Potassium"), lab_rows(pk_d6, d6["hemo"], "Haemoglobin"),
        lab_rows(pk_d6, d6["pcv"], "Packed Cell Volume"), lab_rows(pk_d6, d6["wbcc"], "White Blood Cell Count"),
        lab_rows(pk_d6, d6["rbcc"], "Red Blood Cell Count"),
        lab_rows(pk_d7, d7["TB"], "Total Bilirubin"), lab_rows(pk_d7, d7["DB"], "Direct Bilirubin"),
        lab_rows(pk_d7, d7["Alkphos"], "Alkaline Phosphotase"), lab_rows(pk_d7, d7["Sgpt"], "Alanine Aminotransferase"),
        lab_rows(pk_d7, d7["Sgot"], "Aspartate Aminotransferase"), lab_rows(pk_d7, d7["TP"], "Total Proteins"),
        lab_rows(pk_d7, d7["ALB"], "Albumin"), lab_rows(pk_d7, d7["AGRatio"], "Albumin/Globulin Ratio"),
    ]
    LabResult = pd.concat(lr_rows, ignore_index=True)
    LabResult["PatientKey"] = LabResult["PatientKey"].astype(int)
    LabResult.drop_duplicates(subset=["PatientKey", "LabTestTypeID"], inplace=True)
    LabResult.insert(0, "LabResultID", range(1, len(LabResult) + 1))

    # ---- 10/11. vitalsigntype + vitalsign --------------------------------------
    vital_types = [("Systolic BP", "mmHg"), ("Pulse Rate", "bpm"), ("Weight", "kg"), ("Height", "cm"), ("BMI", "kg/m^2")]
    VitalSignType = pd.DataFrame(vital_types, columns=["Name", "Unit"])
    VitalSignType.insert(0, "VitalSignTypeID", range(1, len(VitalSignType) + 1))
    vt_id = dict(zip(VitalSignType["Name"], VitalSignType["VitalSignTypeID"]))

    def vital_rows(pk_array, values, type_name):
        v = pd.to_numeric(values, errors="coerce")
        keep = v.notna()
        return pd.DataFrame({"PatientKey": pk_array[keep.values], "VitalSignTypeID": vt_id[type_name], "Value": v[keep].values})

    vs_rows = [
        vital_rows(pk_d4, d4["BP_Systolic_mmHg"], "Systolic BP"), vital_rows(pk_d4, d4["Pulse_bpm"], "Pulse Rate"),
        vital_rows(pk_d4, d4["Weight_kg"], "Weight"), vital_rows(pk_d4, d4["Height_cm"], "Height"),
        vital_rows(pk_d4, d4["BMI"], "BMI"), vital_rows(pk_d5, d5["restingBP"], "Systolic BP"),
        vital_rows(pk_d5, d5["maxheartrate"], "Pulse Rate"), vital_rows(pk_d6, d6["bp"], "Systolic BP"),
    ]
    VitalSign = pd.concat(vs_rows, ignore_index=True)
    VitalSign["PatientKey"] = VitalSign["PatientKey"].astype(int)
    VitalSign.drop_duplicates(subset=["PatientKey", "VitalSignTypeID"], inplace=True)
    VitalSign.insert(0, "VitalSignID", range(1, len(VitalSign) + 1))

    # ---- 12-16. domain-specific subtype profile tables -------------------------
    ReproductiveProfile = pd.DataFrame({
        "PatientKey": pk_d3, "PCOSFlag": d3["PCOS_YN"].map({1: True, 0: False}),
        "BetaHCG_I": pd.to_numeric(d3["BetaHCG_I"], errors="coerce"),
        "BetaHCG_II": pd.to_numeric(d3["BetaHCG_II"], errors="coerce"),
        "AMH": pd.to_numeric(d3["AMH"], errors="coerce"),
    }).dropna(subset=["PatientKey"])
    ReproductiveProfile["PatientKey"] = ReproductiveProfile["PatientKey"].astype(int)

    DiabetesProfile = pd.DataFrame({
        "PatientKey": pk_d4, "DiagnosisTemp": pd.to_numeric(d4["Diagnosis_Temp"], errors="coerce"),
        "DiabetesStatus": d4["Diabetes_Status"], "DiabetesTarget": pd.to_numeric(d4["Diabetes_Target"], errors="coerce"),
    }).dropna(subset=["PatientKey"])
    DiabetesProfile["PatientKey"] = DiabetesProfile["PatientKey"].astype(int)

    CardiovascularProfile = pd.DataFrame({
        "PatientKey": pk_d5, "ChestPainType": d5["chestpain"], "RestingECG": d5["restingrelectro"],
        "ExerciseAngina": d5["exerciseangia"].map({1: True, 0: False}),
        "Oldpeak": pd.to_numeric(d5["oldpeak"], errors="coerce"), "Slope": pd.to_numeric(d5["slope"], errors="coerce"),
        "NumMajorVessels": pd.to_numeric(d5["noofmajorvessels"], errors="coerce"),
        "ClassificationTarget": pd.to_numeric(d5["target"], errors="coerce"),
    }).dropna(subset=["PatientKey"])
    CardiovascularProfile["PatientKey"] = CardiovascularProfile["PatientKey"].astype(int)

    RenalProfile = pd.DataFrame({
        "PatientKey": pk_d6, "SpecificGravity": pd.to_numeric(d6["sg"], errors="coerce"),
        "UrineAlbumin": pd.to_numeric(d6["al"], errors="coerce"), "UrineSugar": pd.to_numeric(d6["su"], errors="coerce"),
        "RBCUrine": d6["rbc"], "PusCell": d6["pc"], "PusCellClumps": d6["pcc"], "Bacteria": d6["ba"],
        "AppetiteFlag": d6["appet"].map({"good": True, "poor": False}),
        "PedalEdemaFlag": d6["pe"].astype(str).str.strip().str.lower().map({"yes": True, "no": False}),
        "ClassTarget": d6["class"],
    }).dropna(subset=["PatientKey"])
    RenalProfile["PatientKey"] = RenalProfile["PatientKey"].astype(int)

    LiverProfile = pd.DataFrame({
        "PatientKey": pk_d7, "Selector": pd.to_numeric(d7["Selector"], errors="coerce"),
    }).dropna(subset=["PatientKey"])
    LiverProfile["PatientKey"] = LiverProfile["PatientKey"].astype(int)

    return {
        "dataset": Dataset, "patient": Patient, "admission": Admission,
        "comorbiditytype": ComorbidityType, "patientcomorbidity": PatientComorbidity,
        "diagnosistype": DiagnosisType, "patientdiagnosis": PatientDiagnosis,
        "labtesttype": LabTestType, "labresult": LabResult,
        "vitalsigntype": VitalSignType, "vitalsign": VitalSign,
        "reproductiveprofile": ReproductiveProfile, "diabetesprofile": DiabetesProfile,
        "cardiovascularprofile": CardiovascularProfile, "renalprofile": RenalProfile, "liverprofile": LiverProfile,
    }


# =============================================================================
# PART 2 -- PHARMACY / DRUG-REFERENCE DOMAIN  (2 tables, current version)
# =============================================================================
def build_pharmacy_reference_tables(uploads_dir):
    """
    Loads the two drug-reference CSVs as flat, unnormalized tables.
    This is intentionally NOT normalized (no separate Manufacturer,
    ActiveIngredient, Indication, SideEffect tables) -- that earlier
    12-table normalized design was built, then removed by request in
    favor of this simpler flat-load approach.

    uploads_dir: folder containing:
        Jan Aushadhi Drug List (PMBI Official).csv
        Medicine_Details.csv
    """
    jan = pd.read_csv(f"{uploads_dir}/Jan Aushadhi Drug List (PMBI Official).csv", encoding="utf-8-sig")
    jan.columns = [c.strip() for c in jan.columns]

    JanAushadhi = pd.DataFrame({
        "SrNo": pd.to_numeric(jan["Sr No"], errors="coerce"),
        "DrugCode": pd.to_numeric(jan["Drug Code"], errors="coerce"),
        "GenericName": jan["Generic Name"].str.strip(),
        "UnitSize": jan["Unit Size"],
        "MRP": pd.to_numeric(jan["MRP"], errors="coerce"),
        "GroupName": jan["Group Name"].str.strip(),
    })

    med = pd.read_csv(f"{uploads_dir}/Medicine_Details.csv", encoding="utf-8-sig")
    med.columns = [c.strip() for c in med.columns]

    MedicineDetails = pd.DataFrame({
        "MedicineName": med["Medicine Name"].str.strip(),
        "Composition": med["Composition"],
        "Uses": med["Uses"],
        "SideEffects": med["Side_effects"],
        "Manufacturer": med["Manufacturer"].str.strip(),
        "ExcellentReviewPct": pd.to_numeric(med["Excellent Review %"], errors="coerce"),
        "AverageReviewPct": pd.to_numeric(med["Average Review %"], errors="coerce"),
        "PoorReviewPct": pd.to_numeric(med["Poor Review %"], errors="coerce"),
        # NOTE: "Image URL" column from the source CSV is intentionally dropped here,
        # per the decision to exclude it as out of scope for this schema.
    })
    MedicineDetails.insert(0, "MedicineDetailID", range(1, len(MedicineDetails) + 1))

    return {
        "janaushadhi": JanAushadhi,
        "medicinedetails": MedicineDetails,
    }


# =============================================================================
# Usage
# =============================================================================
if __name__ == "__main__":
    clinical_tables = build_clinical_tables("final-dataset/clinical domain")
    pharmacy_tables = build_pharmacy_reference_tables("final-dataset/pharmacy-domain")

    all_tables = {}
    all_tables.update(clinical_tables)
    all_tables.update(pharmacy_tables)

    for name, df in all_tables.items():
        print(f"{name}: {len(df)} rows, columns={list(df.columns)}")
