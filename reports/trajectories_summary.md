# Comorbidity trajectories

Built by `scripts/02_build_trajectories.py` from events_clinical.parquet. A condition is active from its event date through its STOP date or the patient's death, whichever comes first; with neither, it stays active to the end of the data (2026-10-30). Conditions active in the same window are comorbid.

## Cohort

| Measure | Count |
| --- | ---: |
| Patients with a clinical event | 2,953 |
| Patients with at least 2 distinct codes | 2,912 |
| Of those, with no condition active in any window 2017-2025 (dropped) | 266 |
| Patients in the cohort | 2,646 |
| Events of cohort patients, all dates | 36,172 |
| Events without a STOP date (chronic) | 12,100 (33.5%) |
| Codes in the vocabulary | 214 |

## Yearly windows, 2017-2025

Events dated before 2017: 9,934, of which 83.8% have no STOP date. From 2017: 26,238, of which 14.4% have no STOP date. Earlier years hold mostly conditions still open when Synthea's full export history begins, so windows start at the first full year after it (see config.yaml). Earlier conditions that are still active count in the windows.

Active patients have at least one active condition in the window. The mean, median and max count active conditions per active patient. Events dated counts the events (new conditions or episodes) dated within the year.

| Year | Active patients | Share of cohort | Mean active conditions | Median | Max | Events dated |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2017 | 2,158 | 81.6% | 5.22 | 4 | 29 | 2,498 |
| 2018 | 2,171 | 82.0% | 5.27 | 4 | 30 | 2,533 |
| 2019 | 2,196 | 83.0% | 5.29 | 4 | 33 | 2,547 |
| 2020 | 2,234 | 84.4% | 5.42 | 4 | 32 | 2,816 |
| 2021 | 2,257 | 85.3% | 5.44 | 4 | 33 | 2,804 |
| 2022 | 2,241 | 84.7% | 5.49 | 4 | 33 | 2,647 |
| 2023 | 2,272 | 85.9% | 5.53 | 4 | 31 | 2,744 |
| 2024 | 2,262 | 85.5% | 5.61 | 4 | 32 | 2,623 |
| 2025 | 2,299 | 86.9% | 5.66 | 4 | 31 | 2,796 |

## Transitions

Each code enters a patient's trajectory once, at its onset. A -> B means B was the first new condition after A's onset; codes with the same onset time share a step, with no transition between them. Transitions use every event, including those before the first window and after the prediction cutoff.

30,314 transitions from 2,643 patients, 6,098 distinct code pairs. 22,959 of them end before the prediction cutoff (2023-01-01); only those may be used as features. transitions.parquet and transitions.npz hold all of them, for analysis only.

| From | To | Patients |
| --- | --- | ---: |
| Gingivitis (disorder) | Gingival disease (disorder) | 509 |
| Gingivitis (disorder) | Primary dental caries (disorder) | 286 |
| Prediabetes (finding) | Anemia (disorder) | 283 |
| Viral sinusitis (disorder) | Gingivitis (disorder) | 248 |
| Gingivitis (disorder) | Viral sinusitis (disorder) | 235 |
| Acute viral pharyngitis (disorder) | Gingivitis (disorder) | 176 |
| Gingivitis (disorder) | Acute bronchitis (disorder) | 162 |
| Acute bronchitis (disorder) | Gingivitis (disorder) | 157 |
| Body mass index 30+ - obesity (finding) | Prediabetes (finding) | 157 |
| Anemia (disorder) | Body mass index 30+ - obesity (finding) | 157 |

## Example trajectories (3, random with seed 42)

Every clinical event, in time order. Windows lists the years from 2017-2025 in which the condition is active.

### Patient 1a8c7157-083f-3910-2c19-34fd909f095e

M, born 1983. 17 events, 15 distinct conditions.

| Date | Condition | Code | Active until | Windows |
| --- | --- | --- | --- | --- |
| 1989-10-20 | Chronic sinusitis (disorder) | 40055000 | ongoing | 2017-2025 |
| 2001-05-15 | Anemia (disorder) | 271737000 | ongoing | 2017-2025 |
| 2001-05-15 | Prediabetes (finding) | 714628002 | ongoing | 2017-2025 |
| 2011-05-31 | Diabetes mellitus type 2 (disorder) | 44054006 | ongoing | 2017-2025 |
| 2014-06-03 | Metabolic syndrome X (disorder) | 237602007 | ongoing | 2017-2025 |
| 2014-06-03 | Hypertriglyceridemia (disorder) | 302870006 | ongoing | 2017-2025 |
| 2014-06-03 | Essential hypertension (disorder) | 59621000 | ongoing | 2017-2025 |
| 2016-12-28 | Acute viral pharyngitis (disorder) | 195662009 | 2017-01-07 | 2017 |
| 2017-06-20 | Gingivitis (disorder) | 66383009 | 2017-07-05 | 2017 |
| 2017-07-04 | Gingival disease (disorder) | 18718003 | 2017-07-05 | 2017 |
| 2017-07-07 | Viral sinusitis (disorder) | 444814009 | 2017-07-27 | 2017 |
| 2018-05-16 | Laceration of thigh (disorder) | 283385000 | 2018-06-07 | 2018 |
| 2018-05-16 | Laceration - injury (disorder) | 312608009 | 2018-06-07 | 2018 |
| 2021-04-12 | Viral sinusitis (disorder) | 444814009 | 2021-04-25 | 2021 |
| 2022-08-02 | Infection of tooth (disorder) | 427898007 | 2022-08-03 | 2022 |
| 2022-09-26 | Acute bronchitis (disorder) | 10509002 | 2022-10-15 | 2022 |
| 2023-07-25 | Gingivitis (disorder) | 66383009 | 2023-08-09 | 2023 |

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

### Patient c7dc98ce-297e-e09a-a087-9c5943fc2086

M, born 2019. 8 events, 8 distinct conditions.

| Date | Condition | Code | Active until | Windows |
| --- | --- | --- | --- | --- |
| 2019-12-16 | Acute allergic reaction (disorder) | 241929008 | 2019-12-16 | 2019 |
| 2021-05-27 | Perennial allergic rhinitis with seasonal variation (disorder) | 232353008 | ongoing | 2021-2025 |
| 2021-10-11 | Primary dental caries (disorder) | 109570002 | 2021-10-11 | 2021 |
| 2022-03-28 | Fracture of bone (disorder) | 125605004 | 2022-07-20 | 2022 |
| 2022-03-28 | Fracture of ankle (disorder) | 16114001 | 2022-07-20 | 2022 |
| 2023-01-08 | Otitis media (disorder) | 65363002 | 2023-04-10 | 2023 |
| 2025-07-24 | Laceration - injury (disorder) | 312608009 | 2025-08-09 | 2025 |
| 2025-07-24 | Facial laceration (disorder) | 370247008 | 2025-08-09 | 2025 |
