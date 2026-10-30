# Comorbidity trajectories

Built by `scripts/02_build_trajectories.py` from events_clinical.parquet. A condition is active from its event date through its STOP date or the patient's death, whichever comes first; with neither, it stays active to the end of the data (2026-10-30). Conditions active in the same window are comorbid.

## Cohort

| Measure | Count |
| --- | ---: |
| Patients with a clinical event | 2,953 |
| Patients with at least 2 distinct codes (the cohort) | 2,912 |
| Events of cohort patients, all dates | 40,911 |
| Events without a STOP date (chronic) | 13,979 (34.2%) |
| Codes in the vocabulary | 223 |

## Yearly windows, 2017-2025

Events dated before 2017: 14,652, of which 69.6% have no STOP date. From 2017: 26,259, of which 14.4% have no STOP date. Earlier years hold mostly conditions still open when Synthea's full export history begins, so windows start at the first full year after it (see config.yaml). Earlier conditions that are still active count in the windows.

Active patients have at least one active condition in the window. The mean, median and max count active conditions per active patient. Events dated counts the events (new conditions or episodes) dated within the year.

| Year | Active patients | Share of cohort | Mean active conditions | Median | Max | Events dated |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2017 | 2,158 | 74.1% | 5.22 | 4 | 29 | 2,498 |
| 2018 | 2,171 | 74.6% | 5.27 | 4 | 30 | 2,533 |
| 2019 | 2,196 | 75.4% | 5.29 | 4 | 33 | 2,547 |
| 2020 | 2,234 | 76.7% | 5.42 | 4 | 32 | 2,816 |
| 2021 | 2,257 | 77.5% | 5.44 | 4 | 33 | 2,804 |
| 2022 | 2,241 | 77.0% | 5.49 | 4 | 33 | 2,647 |
| 2023 | 2,272 | 78.0% | 5.53 | 4 | 31 | 2,744 |
| 2024 | 2,262 | 77.7% | 5.61 | 4 | 32 | 2,623 |
| 2025 | 2,299 | 78.9% | 5.66 | 4 | 31 | 2,796 |

## Transitions

Each code enters a patient's trajectory once, at its onset. A -> B means B was the first new condition after A's onset; codes with the same onset time share a step, with no transition between them. Transitions use every event, including those before the first window and after the prediction cutoff.

34,178 transitions from 2,907 patients, 6,647 distinct code pairs.

| From | To | Patients |
| --- | --- | ---: |
| Gingivitis (disorder) | Gingival disease (disorder) | 561 |
| Gingivitis (disorder) | Primary dental caries (disorder) | 320 |
| Prediabetes (finding) | Anemia (disorder) | 319 |
| Viral sinusitis (disorder) | Gingivitis (disorder) | 259 |
| Gingivitis (disorder) | Viral sinusitis (disorder) | 254 |
| Acute viral pharyngitis (disorder) | Gingivitis (disorder) | 192 |
| Ischemic heart disease (disorder) | History of coronary artery bypass grafting (situation) | 189 |
| Body mass index 30+ - obesity (finding) | Prediabetes (finding) | 180 |
| Gingivitis (disorder) | Acute bronchitis (disorder) | 180 |
| Anemia (disorder) | Body mass index 30+ - obesity (finding) | 178 |

## Example trajectories (3, random with seed 42)

Every clinical event, in time order. Windows lists the years from 2017-2025 in which the condition is active.

### Patient 1a12f042-780c-f260-823f-71330c6fc5e9

M, born 1940, died 2010-06-24. 33 events, 22 distinct conditions.

| Date | Condition | Code | Active until | Windows |
| --- | --- | --- | --- | --- |
| 1945-12-14 | Chronic sinusitis (disorder) | 40055000 | 2010-06-24 (death) | - |
| 1962-12-08 | Body mass index 30+ - obesity (finding) | 162864005 | 2010-06-24 (death) | - |
| 1990-10-06 | Prediabetes (finding) | 714628002 | 2010-06-24 (death) | - |
| 1991-10-12 | Metabolic syndrome X (disorder) | 237602007 | 2010-06-24 (death) | - |
| 1991-10-12 | Essential hypertension (disorder) | 59621000 | 2010-06-24 (death) | - |
| 1993-10-23 | Anemia (disorder) | 271737000 | 2010-06-24 (death) | - |
| 1994-10-29 | Disorder of kidney due to diabetes mellitus (disorder) | 127013003 | 2010-06-24 (death) | - |
| 1994-10-29 | Chronic kidney disease stage 1 (disorder) | 431855005 | 2010-06-24 (death) | - |
| 1996-11-09 | Hyperlipidemia (disorder) | 55822004 | 2010-06-24 (death) | - |
| 1997-03-08 | Ischemic heart disease (disorder) | 414545008 | 2010-06-24 (death) | - |
| 1998-08-08 | Chronic kidney disease stage 2 (disorder) | 431856006 | 2010-06-24 (death) | - |
| 1998-08-08 | Microalbuminuria due to type 2 diabetes mellitus (disorder) | 90781000119102 | 2010-06-24 (death) | - |
| 1999-09-25 | Loss of teeth (disorder) | 37320007 | 2010-06-24 (death) | - |
| 2000-11-18 | Acute viral pharyngitis (disorder) | 195662009 | 2000-11-28 | - |
| 2002-06-22 | Gingivitis (disorder) | 66383009 | 2002-07-06 | - |
| 2004-04-24 | Gingivitis (disorder) | 66383009 | 2004-05-22 | - |
| 2004-05-29 | Viral sinusitis (disorder) | 444814009 | 2004-06-07 | - |
| 2004-07-17 | Gingivitis (disorder) | 66383009 | 2004-08-28 | - |
| 2004-08-28 | Gingival disease (disorder) | 18718003 | 2004-08-28 | - |
| 2004-09-04 | Viral sinusitis (disorder) | 444814009 | 2004-09-13 | - |
| 2006-02-25 | Acute infective cystitis (disorder) | 307426000 | 2006-07-14 | - |
| 2008-01-19 | Neoplasm of prostate (disorder) | 126906006 | 2010-06-24 (death) | - |
| 2008-01-19 | Metastatic malignant neoplasm to prostate (disorder) | 94503003 | 2010-06-24 (death) | - |
| 2008-09-13 | Gingivitis (disorder) | 66383009 | 2008-09-27 | - |
| 2008-09-27 | Gingival disease (disorder) | 18718003 | 2008-09-27 | - |
| 2009-02-14 | Gingivitis (disorder) | 66383009 | 2009-02-28 | - |
| 2009-02-28 | Gingival disease (disorder) | 18718003 | 2009-02-28 | - |
| 2009-07-11 | Proteinuria due to type 2 diabetes mellitus (disorder) | 157141000119108 | 2010-06-24 (death) | - |
| 2009-07-11 | Chronic kidney disease stage 3 (disorder) | 433144002 | 2010-06-24 (death) | - |
| 2009-07-11 | Gingivitis (disorder) | 66383009 | 2009-07-25 | - |
| 2010-03-06 | Gingivitis (disorder) | 66383009 | 2010-03-20 | - |
| 2010-03-20 | Gingival disease (disorder) | 18718003 | 2010-03-20 | - |
| 2010-03-21 | Viral sinusitis (disorder) | 444814009 | 2010-04-07 | - |

### Patient ae084928-abe4-bdc1-6f57-ff3edc4b6898

M, born 2002. 12 events, 9 distinct conditions.

| Date | Condition | Code | Active until | Windows |
| --- | --- | --- | --- | --- |
| 2018-02-02 | Gingivitis (disorder) | 66383009 | 2018-02-16 | 2018 |
| 2018-12-22 | Acute viral pharyngitis (disorder) | 195662009 | 2019-01-03 | 2018-2019 |
| 2019-02-08 | Gingivitis (disorder) | 66383009 | 2019-02-23 | 2019 |
| 2019-02-22 | Gingival disease (disorder) | 18718003 | 2019-02-23 | 2019 |
| 2019-12-20 | Acute bronchitis (disorder) | 10509002 | 2019-12-28 | 2019 |
| 2021-01-31 | Chronic neck pain (finding) | 1121000119107 | ongoing | 2021-2025 |
| 2021-01-31 | Chronic low back pain (finding) | 278860009 | ongoing | 2021-2025 |
| 2021-01-31 | Chronic pain (finding) | 82423001 | ongoing | 2021-2025 |
| 2021-12-22 | Acute viral pharyngitis (disorder) | 195662009 | 2022-01-04 | 2021-2022 |
| 2025-02-28 | Body mass index 30+ - obesity (finding) | 162864005 | ongoing | 2025 |
| 2025-02-28 | Gingivitis (disorder) | 66383009 | 2025-03-14 | 2025 |
| 2025-03-14 | Primary dental caries (disorder) | 109570002 | 2025-03-14 | 2025 |

### Patient c818549a-9a99-4095-e7e0-19661c4e0251

M, born 1993. 7 events, 7 distinct conditions.

| Date | Condition | Code | Active until | Windows |
| --- | --- | --- | --- | --- |
| 2007-12-27 | Chronic intractable migraine without aura (disorder) | 124171000119105 | ongoing | 2017-2025 |
| 2008-03-01 | Chronic pain (finding) | 82423001 | ongoing | 2017-2025 |
| 2008-12-25 | Impacted molars (disorder) | 196416002 | ongoing | 2017-2025 |
| 2015-11-27 | Unhealthy alcohol drinking behavior (finding) | 10939881000119105 | 2024-12-06 | 2017-2024 |
| 2015-11-27 | Body mass index 40+ - severely obese (finding) | 408512008 | ongoing | 2017-2025 |
| 2018-03-15 | Streptococcal sore throat (disorder) | 43878008 | 2018-03-26 | 2018 |
| 2021-12-06 | Acute viral pharyngitis (disorder) | 195662009 | 2021-12-14 | 2021 |
