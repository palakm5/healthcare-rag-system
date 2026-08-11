# Structured Database Schema Reference

> **Auto-generated** by `retrieval/structured/build_entity_cache.py`  
> Last updated: 2026-08-11 06:03:37 UTC  
> Total tables: 48  
> Do not edit manually -- re-run the script to refresh.

---

## `admission` (15757 rows) *(skipped: patient-level dates/stay data)*

| Column | Type | Nullable |
|--------|------|----------|
| `admissionid` | integer | NO |
| `patientkey` | integer | YES |
| `admissiondate` | date | YES |
| `dischargedate` | date | YES |
| `admissiontype` | text | YES |
| `monthyear` | text | YES |
| `lengthofstaydays` | integer | YES |
| `iculengthofstaydays` | integer | YES |
| `ruralurban` | text | YES |
| `outcome` | text | YES |

---

## `ayurkoshcompound` (433 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `compoundid` | integer | NO |
| `name` | text | YES |

---

## `ayurkoshcompounddisease` (4 rows) *(skipped: junction table -- integer FKs only)*

| Column | Type | Nullable |
|--------|------|----------|
| `compoundid` | integer | YES |
| `diseaseid` | integer | YES |

---

## `ayurkoshcompoundlakshan` (227 rows) *(skipped: junction table -- integer FKs only)*

| Column | Type | Nullable |
|--------|------|----------|
| `compoundid` | integer | YES |
| `lakshanid` | integer | YES |

---

## `ayurkoshdisease` (68 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `diseaseid` | integer | NO |
| `name` | text | YES |

---

## `ayurkoshdiseaselakshan` (6081 rows) *(skipped: junction table -- integer FKs only)*

| Column | Type | Nullable |
|--------|------|----------|
| `diseaseid` | integer | YES |
| `lakshanid` | integer | YES |
| `reference` | text | YES |

---

## `ayurkoshdiseaseremedy` (18093 rows) *(skipped: junction table -- integer FKs only)*

| Column | Type | Nullable |
|--------|------|----------|
| `diseaseid` | integer | YES |
| `remedyid` | integer | YES |
| `reference` | text | YES |

---

## `ayurkoshgroup` (262 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `groupid` | integer | NO |
| `name` | text | YES |

---

## `ayurkoshherb` (2440 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `ayurkoshherbid` | integer | NO |
| `name` | text | YES |
| `latinname` | text | YES |
| `family` | text | YES |
| `herbid` | integer | YES |

---

## `ayurkoshherbalternate` (1953 rows) *(skipped: junction table -- integer FKs only)*

| Column | Type | Nullable |
|--------|------|----------|
| `ayurkoshherbid` | integer | YES |
| `alternateayurkoshherbid` | integer | YES |

---

## `ayurkoshherbcompound` (1107 rows) *(skipped: junction table -- integer FKs only)*

| Column | Type | Nullable |
|--------|------|----------|
| `ayurkoshherbid` | integer | YES |
| `compoundid` | integer | YES |

---

## `ayurkoshherbgroup` (512 rows) *(skipped: junction table -- integer FKs only)*

| Column | Type | Nullable |
|--------|------|----------|
| `ayurkoshherbid` | integer | YES |
| `groupid` | integer | YES |

---

## `ayurkoshherbquality` (864 rows) *(skipped: junction table -- integer FKs only)*

| Column | Type | Nullable |
|--------|------|----------|
| `ayurkoshherbid` | integer | YES |
| `qualityid` | integer | YES |

---

## `ayurkoshlakshan` (1927 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `lakshanid` | integer | NO |
| `name` | text | YES |

---

## `ayurkoshlakshanprakritidhatu` (142 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `rowid` | integer | NO |
| `effectonparameter_1` | text | YES |
| `effectonparameter_2` | text | YES |
| `name` | text | YES |
| `parameter` | text | YES |
| `lakshan` | text | YES |
| `reference` | text | YES |
| `lakshaninenglish` | text | YES |

---

## `ayurkoshquality` (58 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `qualityid` | integer | NO |
| `name` | text | YES |

---

## `ayurkoshremedy` (8860 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `remedyid` | integer | NO |
| `name` | text | YES |
| `ayurkoshherbid` | integer | YES |

---

## `ayushformulation` (801 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `formulationid` | integer | NO |
| `system` | text | YES |
| `category` | text | YES |
| `name` | text | YES |
| `referencetext` | text | YES |
| `packsize` | text | YES |
| `dose` | text | YES |
| `precaution` | text | YES |
| `preferreduse` | text | YES |

---

## `cardiovascularprofile` (1000 rows) *(skipped: patient-level clinical profile)*

| Column | Type | Nullable |
|--------|------|----------|
| `patientkey` | integer | NO |
| `chestpaintype` | integer | YES |
| `restingecg` | integer | YES |
| `exerciseangina` | boolean | YES |
| `oldpeak` | numeric | YES |
| `slope` | integer | YES |
| `nummajorvessels` | integer | YES |
| `classificationtarget` | integer | YES |

---

## `classicalindication` (1140 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `classicalindicationid` | integer | NO |
| `name` | text | YES |

---

## `comorbiditytype` (9 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `comorbiditytypeid` | integer | NO |
| `name` | text | YES |

---

## `dataset` (8 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `datasetid` | text | NO |
| `datasetname` | text | YES |
| `sourceorg` | text | YES |
| `domain` | text | YES |
| `originalrowcount` | integer | YES |
| `originalcolcount` | integer | YES |

---

## `diabetesprofile` (16353 rows) *(skipped: patient-level clinical profile)*

| Column | Type | Nullable |
|--------|------|----------|
| `patientkey` | integer | NO |
| `diagnosistemp` | text | YES |
| `diabetesstatus` | text | YES |
| `diabetestarget` | integer | YES |

---

## `diagnosistype` (26 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `diagnosistypeid` | integer | NO |
| `name` | text | YES |
| `category` | text | YES |

---

## `dosha` (4 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `doshaid` | integer | NO |
| `name` | text | YES |

---

## `formulationindication` (1963 rows) *(skipped: junction table -- integer FKs only)*

| Column | Type | Nullable |
|--------|------|----------|
| `formulationid` | integer | NO |
| `classicalindicationid` | integer | NO |

---

## `formulationpotency` (454 rows) *(skipped: junction table -- integer FKs only)*

| Column | Type | Nullable |
|--------|------|----------|
| `formulationid` | integer | NO |
| `potencyid` | integer | NO |

---

## `herb` (360 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `herbid` | integer | NO |
| `name` | text | YES |
| `botanicalname` | text | YES |
| `family` | text | YES |
| `englishname` | text | YES |
| `virya` | text | YES |
| `vipaka` | text | YES |
| `tridosha` | boolean | YES |
| `preview` | text | YES |
| `link` | text | YES |

---

## `herbdoshaeffect` (1039 rows) *(skipped: effect column too generic for entity matching)*

| Column | Type | Nullable |
|--------|------|----------|
| `herbdoshaeffectid` | integer | NO |
| `herbid` | integer | YES |
| `doshaid` | integer | YES |
| `effect` | text | YES |

---

## `herbindication` (1738 rows) *(skipped: junction table -- integer FKs only)*

| Column | Type | Nullable |
|--------|------|----------|
| `herbid` | integer | NO |
| `classicalindicationid` | integer | NO |

---

## `herbpartused` (651 rows) *(skipped: junction table -- integer FKs only)*

| Column | Type | Nullable |
|--------|------|----------|
| `herbid` | integer | NO |
| `partusedid` | integer | NO |

---

## `herbrasaguna` (2044 rows) *(skipped: junction table -- integer FKs only)*

| Column | Type | Nullable |
|--------|------|----------|
| `herbid` | integer | NO |
| `attributeid` | integer | NO |

---

## `herbsynonym` (1310 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `herbsynonymid` | integer | NO |
| `herbid` | integer | YES |
| `synonym` | text | YES |

---

## `janaushadhi` (2439 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `srno` | integer | NO |
| `drugcode` | integer | YES |
| `genericname` | text | YES |
| `unitsize` | text | YES |
| `mrp` | numeric | YES |
| `groupname` | text | YES |

---

## `labresult` (174255 rows) *(skipped: patient-level measurement data)*

| Column | Type | Nullable |
|--------|------|----------|
| `labresultid` | integer | NO |
| `patientkey` | integer | YES |
| `labtesttypeid` | integer | YES |
| `resultvalue` | numeric | YES |

---

## `labtesttype` (31 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `labtesttypeid` | integer | NO |
| `testname` | text | YES |
| `unit` | text | YES |
| `category` | text | YES |

---

## `liverprofile` (583 rows) *(skipped: patient-level clinical profile)*

| Column | Type | Nullable |
|--------|------|----------|
| `patientkey` | integer | NO |
| `selector` | integer | YES |

---

## `medicinedetails` (11825 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `medicinedetailid` | integer | NO |
| `medicinename` | text | YES |
| `composition` | text | YES |
| `uses` | text | YES |
| `sideeffects` | text | YES |
| `manufacturer` | text | YES |
| `excellentreviewpct` | numeric | YES |
| `averagereviewpct` | numeric | YES |
| `poorreviewpct` | numeric | YES |

---

## `partused` (32 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `partusedid` | integer | NO |
| `name` | text | YES |

---

## `patient` (31118 rows) *(skipped: patient-level numeric/ID data)*

| Column | Type | Nullable |
|--------|------|----------|
| `patientkey` | integer | NO |
| `datasetid` | text | YES |
| `originalpatientid` | text | YES |
| `age` | integer | YES |
| `gender` | text | YES |

---

## `patientcomorbidity` (25802 rows) *(skipped: patient-level boolean flags)*

| Column | Type | Nullable |
|--------|------|----------|
| `patientkey` | integer | NO |
| `comorbiditytypeid` | integer | NO |
| `presentflag` | boolean | YES |

---

## `patientdiagnosis` (24908 rows) *(skipped: patient-level boolean flags)*

| Column | Type | Nullable |
|--------|------|----------|
| `patientkey` | integer | NO |
| `diagnosistypeid` | integer | NO |
| `presentflag` | boolean | YES |

---

## `potency` (7 rows) *(skipped: small lookup (7 rows) -- not a primary entity)*

| Column | Type | Nullable |
|--------|------|----------|
| `potencyid` | integer | NO |
| `name` | text | YES |

---

## `rasagunaattribute` (94 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `attributeid` | integer | NO |
| `attributetype` | text | YES |
| `value` | text | YES |

---

## `renalprofile` (397 rows) *(skipped: patient-level clinical profile)*

| Column | Type | Nullable |
|--------|------|----------|
| `patientkey` | integer | NO |
| `specificgravity` | numeric | YES |
| `urinealbumin` | integer | YES |
| `urinesugar` | integer | YES |
| `rbcurine` | text | YES |
| `puscell` | text | YES |
| `puscellclumps` | text | YES |
| `bacteria` | text | YES |
| `appetiteflag` | integer | YES |
| `pedaledemaflag` | integer | YES |
| `classtarget` | text | YES |

---

## `reproductiveprofile` (541 rows) *(skipped: patient-level clinical profile)*

| Column | Type | Nullable |
|--------|------|----------|
| `patientkey` | integer | NO |
| `pcosflag` | boolean | YES |
| `betahcg_i` | numeric | YES |
| `betahcg_ii` | numeric | YES |
| `amh` | numeric | YES |

---

## `vitalsign` (77017 rows) *(skipped: patient-level measurement data)*

| Column | Type | Nullable |
|--------|------|----------|
| `vitalsignid` | integer | NO |
| `patientkey` | integer | YES |
| `vitalsigntypeid` | integer | YES |
| `value` | numeric | YES |

---

## `vitalsigntype` (5 rows) `[entity-indexed]`

| Column | Type | Nullable |
|--------|------|----------|
| `vitalsigntypeid` | integer | NO |
| `name` | text | YES |
| `unit` | text | YES |

---
