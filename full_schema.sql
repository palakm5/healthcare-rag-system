-- ==============================================================================
-- Unified Clinical + Jan Aushadhi/Medicine Reference + AYUSH Corpus -- 48-Table Schema
-- Target: PostgreSQL 14+ (Supabase / Neon / RDS all compatible)
-- Regenerated from Data_Sources_and_Preprocessing_Report_Updated.docx +
-- unified_clinical_pharma_ayush_corpus.xlsx (Overview sheet).
--
-- CHANGES FROM THE PRIOR 58-TABLE SCHEMA:
--  - The entire Pharmacy domain (12 tables: manufacturer, medicine,
--    activeingredient, medicinecomposition, indication, medicineindication,
--    sideeffect, medicinesideeffect, medicinesubstitute, therapeuticgroup,
--    genericdruglisting, genericdrugcomposition) has been REMOVED.
--  - Replaced with two flat, unnormalized reference tables loaded as-is
--    from their source CSVs: janaushadhi (2,439 rows) and medicinedetails
--    (11,825 rows). Neither table has foreign keys -- both are standalone.
--  - Per request, medicinedetails omits the Image URL column present in
--    the source CSV (Medicine Name, Composition, Uses, Side_effects,
--    Manufacturer, and the three review-percentage columns only).
--  - Clinical, Herb/AyushFormulation, and AyurKOSH domains are unchanged.
-- ==============================================================================

-- Dataset  (Lookup, ~6 rows)
CREATE TABLE dataset (
    datasetid TEXT PRIMARY KEY,
    datasetname TEXT,
    sourceorg TEXT,
    domain TEXT,
    originalrowcount INT,
    originalcolcount INT
);

-- Patient  (Super-type, ~31,118 rows)
CREATE TABLE patient (
    patientkey INT PRIMARY KEY,
    datasetid TEXT,
    originalpatientid TEXT,
    age INT,
    gender TEXT,
    FOREIGN KEY (datasetid) REFERENCES dataset (datasetid)
);

-- Admission  (Encounter, ~15,757 rows)
CREATE TABLE admission (
    admissionid INT PRIMARY KEY,
    patientkey INT,
    admissiondate DATE,
    dischargedate DATE,
    admissiontype TEXT,
    monthyear TEXT,
    lengthofstaydays INT,
    iculengthofstaydays INT,
    ruralurban TEXT,
    outcome TEXT,
    FOREIGN KEY (patientkey) REFERENCES patient (patientkey)
);

-- ComorbidityType  (Lookup, ~9 rows)
CREATE TABLE comorbiditytype (
    comorbiditytypeid INT PRIMARY KEY,
    name TEXT
);

-- PatientComorbidity  (Junction (M:N), ~25,802 rows)
CREATE TABLE patientcomorbidity (
    patientkey INT,
    comorbiditytypeid INT,
    presentflag BOOLEAN,
    PRIMARY KEY (patientkey,comorbiditytypeid),
    FOREIGN KEY (patientkey) REFERENCES patient (patientkey),
    FOREIGN KEY (comorbiditytypeid) REFERENCES comorbiditytype (comorbiditytypeid)
);

-- DiagnosisType  (Lookup, ~26 rows)
CREATE TABLE diagnosistype (
    diagnosistypeid INT PRIMARY KEY,
    name TEXT,
    category TEXT
);

-- PatientDiagnosis  (Junction (M:N), ~24,908 rows)
CREATE TABLE patientdiagnosis (
    patientkey INT,
    diagnosistypeid INT,
    presentflag BOOLEAN,
    PRIMARY KEY (patientkey,diagnosistypeid),
    FOREIGN KEY (patientkey) REFERENCES patient (patientkey),
    FOREIGN KEY (diagnosistypeid) REFERENCES diagnosistype (diagnosistypeid)
);

-- LabTestType  (Lookup, ~31 rows)
CREATE TABLE labtesttype (
    labtesttypeid INT PRIMARY KEY,
    testname TEXT,
    unit TEXT,
    category TEXT
);

-- LabResult  (Junction (M:N), ~174,255 rows)
CREATE TABLE labresult (
    labresultid INT PRIMARY KEY,
    patientkey INT,
    labtesttypeid INT,
    resultvalue NUMERIC(12,4),
    FOREIGN KEY (patientkey) REFERENCES patient (patientkey),
    FOREIGN KEY (labtesttypeid) REFERENCES labtesttype (labtesttypeid)
);

-- VitalSignType  (Lookup, ~5 rows)
CREATE TABLE vitalsigntype (
    vitalsigntypeid INT PRIMARY KEY,
    name TEXT,
    unit TEXT
);

-- VitalSign  (Junction (M:N), ~77,017 rows)
CREATE TABLE vitalsign (
    vitalsignid INT PRIMARY KEY,
    patientkey INT,
    vitalsigntypeid INT,
    value NUMERIC(12,4),
    FOREIGN KEY (patientkey) REFERENCES patient (patientkey),
    FOREIGN KEY (vitalsigntypeid) REFERENCES vitalsigntype (vitalsigntypeid)
);

-- ReproductiveProfile  (Sub-type profile, ~541 rows)
CREATE TABLE reproductiveprofile (
    patientkey INT PRIMARY KEY,
    pcosflag BOOLEAN,
    betahcg_i NUMERIC(12,4),
    betahcg_ii NUMERIC(12,4),
    amh NUMERIC(12,4),
    FOREIGN KEY (patientkey) REFERENCES patient (patientkey)
);

-- DiabetesProfile  (Sub-type profile, ~16,353 rows)
CREATE TABLE diabetesprofile (
    patientkey INT PRIMARY KEY,
    diagnosistemp TEXT,
    diabetesstatus TEXT,
    diabetestarget INT,
    FOREIGN KEY (patientkey) REFERENCES patient (patientkey)
);

-- CardiovascularProfile  (Sub-type profile, ~1,000 rows)
CREATE TABLE cardiovascularprofile (
    patientkey INT PRIMARY KEY,
    chestpaintype INT,
    restingecg INT,
    exerciseangina BOOLEAN,
    oldpeak NUMERIC(12,4),
    slope INT,
    nummajorvessels INT,
    classificationtarget INT,
    FOREIGN KEY (patientkey) REFERENCES patient (patientkey)
);

-- RenalProfile  (Sub-type profile, ~397 rows)
CREATE TABLE renalprofile (
    patientkey INT PRIMARY KEY,
    specificgravity NUMERIC(12,4),
    urinealbumin INT,
    urinesugar INT,
    rbcurine TEXT,
    puscell TEXT,
    puscellclumps TEXT,
    bacteria TEXT,
    appetiteflag INT,
    pedaledemaflag INT,
    classtarget TEXT,
    FOREIGN KEY (patientkey) REFERENCES patient (patientkey)
);

-- LiverProfile  (Sub-type profile, ~583 rows)
CREATE TABLE liverprofile (
    patientkey INT PRIMARY KEY,
    selector INT,
    FOREIGN KEY (patientkey) REFERENCES patient (patientkey)
);

-- ==============================================================================
-- JAN AUSHADHI / MEDICINE REFERENCE DOMAIN -- 2 tables (new, flat/unnormalized)
-- Loaded as-is from source CSVs. No foreign keys -- both are standalone.
-- ==============================================================================

-- JanAushadhi  (Flat reference, ~2,439 rows)
CREATE TABLE janaushadhi (
    srno INT PRIMARY KEY,
    drugcode INT,
    genericname TEXT,
    unitsize TEXT,
    mrp NUMERIC(12,2),
    groupname TEXT
);

-- MedicineDetails  (Flat reference, ~11,825 rows)
-- NOTE: Image URL column from the source CSV is intentionally omitted.
CREATE TABLE medicinedetails (
    medicinedetailid INT PRIMARY KEY,
    medicinename TEXT,
    composition TEXT,
    uses TEXT,
    sideeffects TEXT,
    manufacturer TEXT,
    excellentreviewpct NUMERIC(5,2),
    averagereviewpct NUMERIC(5,2),
    poorreviewpct NUMERIC(5,2)
);

-- Herb  (Core entity, ~360 rows)
CREATE TABLE herb (
    herbid INT PRIMARY KEY,
    name TEXT,
    botanicalname TEXT,
    family TEXT,
    englishname TEXT,
    virya TEXT,
    vipaka TEXT,
    tridosha BOOLEAN,
    preview TEXT,
    link TEXT
);

-- HerbSynonym  (Junction, ~1,310 rows)
CREATE TABLE herbsynonym (
    herbsynonymid INT PRIMARY KEY,
    herbid INT,
    synonym TEXT,
    FOREIGN KEY (herbid) REFERENCES herb (herbid)
);

-- PartUsed  (Lookup, ~32 rows)
CREATE TABLE partused (
    partusedid INT PRIMARY KEY,
    name TEXT
);

-- HerbPartUsed  (Junction (M:N), ~651 rows)
CREATE TABLE herbpartused (
    herbid INT,
    partusedid INT,
    PRIMARY KEY (herbid,partusedid),
    FOREIGN KEY (herbid) REFERENCES herb (herbid),
    FOREIGN KEY (partusedid) REFERENCES partused (partusedid)
);

-- ClassicalIndication  (Lookup, ~1,140 rows)
CREATE TABLE classicalindication (
    classicalindicationid INT PRIMARY KEY,
    name TEXT
);

-- HerbIndication  (Junction (M:N), ~1,738 rows)
CREATE TABLE herbindication (
    herbid INT,
    classicalindicationid INT,
    PRIMARY KEY (herbid,classicalindicationid),
    FOREIGN KEY (herbid) REFERENCES herb (herbid),
    FOREIGN KEY (classicalindicationid) REFERENCES classicalindication (classicalindicationid)
);

-- Dosha  (Lookup, ~4 rows)
CREATE TABLE dosha (
    doshaid INT PRIMARY KEY,
    name TEXT
);

-- HerbDoshaEffect  (Junction (M:N), ~1,039 rows)
CREATE TABLE herbdoshaeffect (
    herbdoshaeffectid INT PRIMARY KEY,
    herbid INT,
    doshaid INT,
    effect TEXT,
    FOREIGN KEY (herbid) REFERENCES herb (herbid),
    FOREIGN KEY (doshaid) REFERENCES dosha (doshaid)
);

-- RasaGunaAttribute  (Lookup, ~94 rows)
CREATE TABLE rasagunaattribute (
    attributeid INT PRIMARY KEY,
    attributetype TEXT,
    value TEXT
);

-- HerbRasaGuna  (Junction (M:N), ~2,044 rows)
CREATE TABLE herbrasaguna (
    herbid INT,
    attributeid INT,
    PRIMARY KEY (herbid,attributeid),
    FOREIGN KEY (herbid) REFERENCES herb (herbid),
    FOREIGN KEY (attributeid) REFERENCES rasagunaattribute (attributeid)
);

-- AyushFormulation  (Core entity, ~801 rows)
CREATE TABLE ayushformulation (
    formulationid INT PRIMARY KEY,
    system TEXT,
    category TEXT,
    name TEXT,
    referencetext TEXT,
    packsize TEXT,
    dose TEXT,
    precaution TEXT,
    preferreduse TEXT
);

-- FormulationIndication  (Junction (M:N), ~1,963 rows)
CREATE TABLE formulationindication (
    formulationid INT,
    classicalindicationid INT,
    PRIMARY KEY (formulationid,classicalindicationid),
    FOREIGN KEY (formulationid) REFERENCES ayushformulation (formulationid),
    FOREIGN KEY (classicalindicationid) REFERENCES classicalindication (classicalindicationid)
);

-- Potency  (Lookup, ~7 rows)
CREATE TABLE potency (
    potencyid INT PRIMARY KEY,
    name TEXT
);

-- FormulationPotency  (Junction (M:N), ~454 rows)
CREATE TABLE formulationpotency (
    formulationid INT,
    potencyid INT,
    PRIMARY KEY (formulationid,potencyid),
    FOREIGN KEY (formulationid) REFERENCES ayushformulation (formulationid),
    FOREIGN KEY (potencyid) REFERENCES potency (potencyid)
);

-- AyurKoshHerb  (Core entity, ~2,440 rows)
CREATE TABLE ayurkoshherb (
    ayurkoshherbid INT PRIMARY KEY,
    name TEXT,
    latinname TEXT,
    family TEXT,
    herbid INT,
    FOREIGN KEY (herbid) REFERENCES herb (herbid)
);

-- AyurKoshRemedy  (Bridge, ~8,860 rows)
CREATE TABLE ayurkoshremedy (
    remedyid INT PRIMARY KEY,
    name TEXT,
    ayurkoshherbid INT,
    FOREIGN KEY (ayurkoshherbid) REFERENCES ayurkoshherb (ayurkoshherbid)
);

-- AyurKoshGroup  (Lookup, ~262 rows)
CREATE TABLE ayurkoshgroup (
    groupid INT PRIMARY KEY,
    name TEXT
);

-- AyurKoshHerbGroup  (Junction (M:N), ~512 rows)
CREATE TABLE ayurkoshherbgroup (
    ayurkoshherbid INT,
    groupid INT,
    FOREIGN KEY (ayurkoshherbid) REFERENCES ayurkoshherb (ayurkoshherbid),
    FOREIGN KEY (groupid) REFERENCES ayurkoshgroup (groupid)
);

-- AyurKoshHerbAlternate  (Junction (self-ref), ~1,953 rows)
CREATE TABLE ayurkoshherbalternate (
    ayurkoshherbid INT,
    alternateayurkoshherbid INT,
    FOREIGN KEY (ayurkoshherbid) REFERENCES ayurkoshherb (ayurkoshherbid)
);

-- AyurKoshQuality  (Lookup, ~58 rows)
CREATE TABLE ayurkoshquality (
    qualityid INT PRIMARY KEY,
    name TEXT
);

-- AyurKoshHerbQuality  (Junction (M:N), ~864 rows)
CREATE TABLE ayurkoshherbquality (
    ayurkoshherbid INT,
    qualityid INT,
    FOREIGN KEY (ayurkoshherbid) REFERENCES ayurkoshherb (ayurkoshherbid),
    FOREIGN KEY (qualityid) REFERENCES ayurkoshquality (qualityid)
);

-- AyurKoshCompound  (Core entity, ~433 rows)
CREATE TABLE ayurkoshcompound (
    compoundid INT PRIMARY KEY,
    name TEXT
);

-- AyurKoshHerbCompound  (Junction (M:N), ~1,107 rows)
CREATE TABLE ayurkoshherbcompound (
    ayurkoshherbid INT,
    compoundid INT,
    FOREIGN KEY (ayurkoshherbid) REFERENCES ayurkoshherb (ayurkoshherbid),
    FOREIGN KEY (compoundid) REFERENCES ayurkoshcompound (compoundid)
);

-- AyurKoshDisease  (Core entity, ~68 rows)
CREATE TABLE ayurkoshdisease (
    diseaseid INT PRIMARY KEY,
    name TEXT
);

-- AyurKoshLakshan  (Lookup, ~1,927 rows)
CREATE TABLE ayurkoshlakshan (
    lakshanid INT PRIMARY KEY,
    name TEXT
);

-- AyurKoshDiseaseRemedy  (Junction (M:N), ~18,093 rows)
CREATE TABLE ayurkoshdiseaseremedy (
    diseaseid INT,
    remedyid INT,
    reference TEXT,
    FOREIGN KEY (diseaseid) REFERENCES ayurkoshdisease (diseaseid),
    FOREIGN KEY (remedyid) REFERENCES ayurkoshremedy (remedyid)
);

-- AyurKoshDiseaseLakshan  (Junction (M:N), ~6,081 rows)
CREATE TABLE ayurkoshdiseaselakshan (
    diseaseid INT,
    lakshanid INT,
    reference TEXT,
    FOREIGN KEY (diseaseid) REFERENCES ayurkoshdisease (diseaseid),
    FOREIGN KEY (lakshanid) REFERENCES ayurkoshlakshan (lakshanid)
);

-- AyurKoshCompoundDisease  (Junction (M:N), ~4 rows)
CREATE TABLE ayurkoshcompounddisease (
    compoundid INT,
    diseaseid INT,
    FOREIGN KEY (compoundid) REFERENCES ayurkoshcompound (compoundid),
    FOREIGN KEY (diseaseid) REFERENCES ayurkoshdisease (diseaseid)
);

-- AyurKoshCompoundLakshan  (Junction (M:N), ~227 rows)
CREATE TABLE ayurkoshcompoundlakshan (
    compoundid INT,
    lakshanid INT,
    FOREIGN KEY (compoundid) REFERENCES ayurkoshcompound (compoundid),
    FOREIGN KEY (lakshanid) REFERENCES ayurkoshlakshan (lakshanid)
);

-- AyurKoshLakshanPrakritiDhatu  (Reference (denormalized), ~142 rows)
CREATE TABLE ayurkoshlakshanprakritidhatu (
    rowid INT PRIMARY KEY,
    effectonparameter_1 TEXT,
    effectonparameter_2 TEXT,
    name TEXT,
    parameter TEXT,
    lakshan TEXT,
    reference TEXT,
    lakshaninenglish TEXT
);


-- ==============================================================================
-- Enable Row Level Security on every table (Supabase requirement)
-- No policies are created -- this locks out the public anon/authenticated
-- API entirely. Your direct Postgres connection (psql, SQLAlchemy, DBeaver)
-- is unaffected since RLS only restricts the PostgREST API roles.
-- ==============================================================================
ALTER TABLE dataset ENABLE ROW LEVEL SECURITY;
ALTER TABLE patient ENABLE ROW LEVEL SECURITY;
ALTER TABLE admission ENABLE ROW LEVEL SECURITY;
ALTER TABLE comorbiditytype ENABLE ROW LEVEL SECURITY;
ALTER TABLE patientcomorbidity ENABLE ROW LEVEL SECURITY;
ALTER TABLE diagnosistype ENABLE ROW LEVEL SECURITY;
ALTER TABLE patientdiagnosis ENABLE ROW LEVEL SECURITY;
ALTER TABLE labtesttype ENABLE ROW LEVEL SECURITY;
ALTER TABLE labresult ENABLE ROW LEVEL SECURITY;
ALTER TABLE vitalsigntype ENABLE ROW LEVEL SECURITY;
ALTER TABLE vitalsign ENABLE ROW LEVEL SECURITY;
ALTER TABLE reproductiveprofile ENABLE ROW LEVEL SECURITY;
ALTER TABLE diabetesprofile ENABLE ROW LEVEL SECURITY;
ALTER TABLE cardiovascularprofile ENABLE ROW LEVEL SECURITY;
ALTER TABLE renalprofile ENABLE ROW LEVEL SECURITY;
ALTER TABLE liverprofile ENABLE ROW LEVEL SECURITY;
ALTER TABLE janaushadhi ENABLE ROW LEVEL SECURITY;
ALTER TABLE medicinedetails ENABLE ROW LEVEL SECURITY;
ALTER TABLE herb ENABLE ROW LEVEL SECURITY;
ALTER TABLE herbsynonym ENABLE ROW LEVEL SECURITY;
ALTER TABLE partused ENABLE ROW LEVEL SECURITY;
ALTER TABLE herbpartused ENABLE ROW LEVEL SECURITY;
ALTER TABLE classicalindication ENABLE ROW LEVEL SECURITY;
ALTER TABLE herbindication ENABLE ROW LEVEL SECURITY;
ALTER TABLE dosha ENABLE ROW LEVEL SECURITY;
ALTER TABLE herbdoshaeffect ENABLE ROW LEVEL SECURITY;
ALTER TABLE rasagunaattribute ENABLE ROW LEVEL SECURITY;
ALTER TABLE herbrasaguna ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayushformulation ENABLE ROW LEVEL SECURITY;
ALTER TABLE formulationindication ENABLE ROW LEVEL SECURITY;
ALTER TABLE potency ENABLE ROW LEVEL SECURITY;
ALTER TABLE formulationpotency ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshherb ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshremedy ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshgroup ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshherbgroup ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshherbalternate ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshquality ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshherbquality ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshcompound ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshherbcompound ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshdisease ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshlakshan ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshdiseaseremedy ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshdiseaselakshan ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshcompounddisease ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshcompoundlakshan ENABLE ROW LEVEL SECURITY;
ALTER TABLE ayurkoshlakshanprakritidhatu ENABLE ROW LEVEL SECURITY;