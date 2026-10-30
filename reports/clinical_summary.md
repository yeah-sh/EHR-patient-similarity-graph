# Clinical code filter

Merges codes that duplicate a concept, then restricts the condition events to clinical conditions, dropping social, administrative and symptom codes. Unfiltered events with their original codes stay in events.parquet; the merged, filtered ones are in events_clinical.parquet.

## Code merges (1, `data.code_merge`)

Each old code is renamed to its canonical code before the filter, so the concept counts once; the events keep their own description. Patients and events are those recorded under the old code. Every count below is after merging.

| Old code | Old description | Canonical code | Canonical description | Patients | Events |
| --- | --- | --- | --- | ---: | ---: |
| 15777000 | not in data | 714628002 | Prediabetes (finding) | 0 | 0 |

## Rule

A code is kept if any of its descriptions ends with `(disorder)` or equals one of the whitelist entries below (both case-insensitive). Every other code is dropped. Any wording counts because Synthea versions word some codes differently.

### Whitelist (31 entries, `data.code_filter`)

| Description | Codes matched | Patients |
| --- | --- | ---: |
| Prediabetes (finding) | 714628002 | 1,142 |
| Prediabetes | no match | 0 |
| Body mass index 30+ - obesity (finding) | 162864005 | 1,409 |
| Body mass index 40+ - severely obese (finding) | 408512008 | 22 |
| Hypertension | no match | 0 |
| Hyperlipidemia | no match | 0 |
| Chronic pain (finding) | 82423001 | 703 |
| Chronic low back pain (finding) | 278860009 | 502 |
| Chronic neck pain (finding) | 1121000119107 | 306 |
| Severe anxiety (panic) (finding) | 80583007 | 347 |
| Bacteremia (finding) | no match | 0 |
| Sprain (morphologic abnormality) | 384709000 | 387 |
| Burn injury (morphologic abnormality) | 48333001 | 122 |
| Coronary Heart Disease | no match | 0 |
| Cardiac Arrest | no match | 0 |
| Drug overdose | no match | 0 |
| Escherichia coli urinary tract infection | no match | 0 |
| Cystitis | no match | 0 |
| Gout | 90560007 | 23 |
| Brain damage - traumatic | no match | 0 |
| Chronic paralysis due to lesion of spinal cord | no match | 0 |
| Cystic Fibrosis | no match | 0 |
| Infection caused by Pseudomonas aeruginosa | no match | 0 |
| Male Infertility | no match | 0 |
| Smokes tobacco daily (finding) | 449868002 | 29 |
| Unhealthy alcohol drinking behavior (finding) | 10939881000119105 | 445 |
| Misuses drugs (finding) | 361055000 | 213 |
| History of myocardial infarction (situation) | 399211009 | 145 |
| History of coronary artery bypass grafting (situation) | 399261000 | 206 |
| History of cardiac arrest (situation) | no match | 0 |
| History of aortic valve replacement (situation) | 1231000119100 | 23 |

## Result

|  | Kept | Dropped | Total | Kept share |
| --- | ---: | ---: | ---: | ---: |
| Codes | 220 | 76 | 296 | 74.3% |
| Events | 40,871 | 66,055 | 106,926 | 38.2% |

Kept codes by rule: 204 by suffix, 16 by whitelist only. Before merging there were 296 codes and 106,926 events.

| Patients | Count | Share of all patients |
| --- | ---: | ---: |
| With any condition event | 3,000 | 100.0% |
| With at least 1 clinical code | 2,953 | 98.4% |
| With at least 2 distinct clinical codes | 2,912 | 97.1% |

## Top 20 clinical conditions

Ranked by the number of patients with at least one event of the code.

| Rank | Code | Description | Patients | Share of patients | Events |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | 66383009 | Gingivitis (disorder) | 2,392 | 79.7% | 7,799 |
| 2 | 444814009 | Viral sinusitis (disorder) | 1,901 | 63.4% | 3,028 |
| 3 | 18718003 | Gingival disease (disorder) | 1,419 | 47.3% | 2,349 |
| 4 | 162864005 | Body mass index 30+ - obesity (finding) | 1,409 | 47.0% | 1,409 |
| 5 | 195662009 | Acute viral pharyngitis (disorder) | 1,323 | 44.1% | 1,742 |
| 6 | 10509002 | Acute bronchitis (disorder) | 1,186 | 39.5% | 1,478 |
| 7 | 109570002 | Primary dental caries (disorder) | 1,181 | 39.4% | 1,707 |
| 8 | 714628002 | Prediabetes (finding) | 1,142 | 38.1% | 1,142 |
| 9 | 271737000 | Anemia (disorder) | 1,048 | 34.9% | 1,048 |
| 10 | 59621000 | Essential hypertension (disorder) | 709 | 23.6% | 709 |
| 11 | 82423001 | Chronic pain (finding) | 703 | 23.4% | 703 |
| 12 | 40055000 | Chronic sinusitis (disorder) | 615 | 20.5% | 615 |
| 13 | 414545008 | Ischemic heart disease (disorder) | 513 | 17.1% | 513 |
| 14 | 278860009 | Chronic low back pain (finding) | 502 | 16.7% | 502 |
| 15 | 237602007 | Metabolic syndrome X (disorder) | 450 | 15.0% | 450 |
| 16 | 10939881000119105 | Unhealthy alcohol drinking behavior (finding) | 445 | 14.8% | 486 |
| 17 | 125605004 | Fracture of bone (disorder) | 433 | 14.4% | 471 |
| 18 | 427898007 | Infection of tooth (disorder) | 430 | 14.3% | 509 |
| 19 | 37320007 | Loss of teeth (disorder) | 420 | 14.0% | 420 |
| 20 | 312608009 | Laceration - injury (disorder) | 389 | 13.0% | 415 |

## Patients with an active clinical condition, 2000-2025

A condition is active in a year if it started before the year ended, had not stopped before the year began (no STOP date means still active), and the patient had not died before the year began. Alive counts patients born before the year ended and not dead before it began.

| Year | Alive | With active clinical condition | Share of alive | Flag |
| ---: | ---: | ---: | ---: | --- |
| 2000 | 2,053 | 1,201 | 58.5% |  |
| 2001 | 2,085 | 1,232 | 59.1% |  |
| 2002 | 2,121 | 1,278 | 60.3% |  |
| 2003 | 2,151 | 1,320 | 61.4% |  |
| 2004 | 2,183 | 1,359 | 62.3% |  |
| 2005 | 2,210 | 1,390 | 62.9% |  |
| 2006 | 2,241 | 1,417 | 63.2% |  |
| 2007 | 2,276 | 1,473 | 64.7% |  |
| 2008 | 2,298 | 1,499 | 65.2% |  |
| 2009 | 2,323 | 1,536 | 66.1% |  |
| 2010 | 2,360 | 1,567 | 66.4% |  |
| 2011 | 2,386 | 1,601 | 67.1% |  |
| 2012 | 2,415 | 1,635 | 67.7% |  |
| 2013 | 2,416 | 1,662 | 68.8% |  |
| 2014 | 2,433 | 1,760 | 72.3% |  |
| 2015 | 2,451 | 1,803 | 73.6% |  |
| 2016 | 2,477 | 1,933 | 78.0% |  |
| 2017 | 2,499 | 2,162 | 86.5% |  |
| 2018 | 2,513 | 2,174 | 86.5% |  |
| 2019 | 2,533 | 2,201 | 86.9% |  |
| 2020 | 2,557 | 2,237 | 87.5% |  |
| 2021 | 2,566 | 2,262 | 88.2% |  |
| 2022 | 2,579 | 2,245 | 87.0% |  |
| 2023 | 2,591 | 2,277 | 87.9% |  |
| 2024 | 2,609 | 2,272 | 87.1% |  |
| 2025 | 2,621 | 2,304 | 87.9% |  |

Years below 500: none.

## Dropped codes (76)

| Code | Description | Patients | Events |
| --- | --- | ---: | ---: |
| 314529007 | Medication review due (situation) | 3,000 | 20,722 |
| 73595000 | Stress (finding) | 2,365 | 8,203 |
| 160903007 | Full-time employment (finding) | 2,365 | 7,717 |
| 160904001 | Part-time employment (finding) | 1,899 | 4,841 |
| 423315002 | Limited social contact (finding) | 1,673 | 3,092 |
| 422650009 | Social isolation (finding) | 1,640 | 2,989 |
| 224299000 | Received higher education (finding) | 1,462 | 1,462 |
| 741062008 | Not in labor force (finding) | 1,404 | 2,654 |
| 706893006 | Victim of intimate partner abuse (finding) | 1,302 | 1,934 |
| 424393004 | Reports of violence in the environment (finding) | 1,019 | 1,599 |
| 73438004 | Unemployed (finding) | 935 | 1,128 |
| 473461003 | Educated to high school level (finding) | 731 | 731 |
| 160968000 | Risk activity involvement (finding) | 724 | 775 |
| 266948004 | Has a criminal record (finding) | 613 | 613 |
| 161744009 | Past pregnancy history of miscarriage (situation) | 578 | 578 |
| 72892002 | Normal pregnancy (finding) | 525 | 1,157 |
| 274531002 | Abnormal findings diagnostic imaging heart+coronary circulat (finding) | 516 | 516 |
| 105531004 | Housing unsatisfactory (finding) | 376 | 376 |
| 267020005 | History of tubal ligation (situation) | 320 | 320 |
| 278558000 | Dental filling lost (finding) | 316 | 357 |
| 278588009 | Fractured dental filling (finding) | 305 | 338 |
| 278598003 | Leaking dental filling (finding) | 303 | 350 |
| 266934004 | Transport problem (finding) | 295 | 295 |
| 278602001 | Loose dental filling (finding) | 294 | 331 |
| 713458007 | Lack of access to transportation (finding) | 290 | 290 |
| 840544004 | Suspected disease caused by Severe acute respiratory coronavirus 2 (situation) | 267 | 267 |
| 224295006 | Only received primary school education (finding) | 234 | 234 |
| 386661006 | Fever (finding) | 233 | 233 |
| 49727002 | Cough (finding) | 183 | 183 |
| 1187604002 | Serving in military service (finding) | 148 | 148 |
| 36955009 | Loss of taste (finding) | 139 | 139 |
| 428251008 | History of appendectomy (situation) | 134 | 134 |
| 446654005 | Refugee (person) | 122 | 122 |
| 698306007 | Awaiting transplantation of kidney (situation) | 105 | 105 |
| 84229001 | Fatigue (finding) | 102 | 102 |
| 1290882004 | History of seizure (situation) | 101 | 101 |
| 248595008 | Sputum finding (finding) | 90 | 90 |
| 161665007 | History of renal transplant (situation) | 81 | 81 |
| 183996000 | Sterilization requested (situation) | 76 | 76 |
| 32911000 | Homeless (finding) | 58 | 65 |
| 1551000119108 | Nonproliferative diabetic retinopathy due to type II diabetes mellitus | 55 | 55 |
| 267036007 | Dyspnea (finding) | 47 | 47 |
| 56018004 | Wheezing (finding) | 47 | 47 |
| 315268008 | Suspected prostate cancer (situation) | 43 | 52 |
| 271825005 | Respiratory distress (finding) | 41 | 41 |
| 57676002 | Joint pain | 40 | 40 |
| 68962001 | Muscle pain (finding) | 40 | 40 |
| 267102003 | Sore throat (finding) | 39 | 39 |
| 43724002 | Chill (finding) | 33 | 33 |
| 5602001 | Opioid abuse | 28 | 28 |
| 25064002 | Headache (finding) | 27 | 27 |
| 162573006 | Suspected lung cancer (situation) | 23 | 23 |
| 876882001 | Died in hospice (finding) | 19 | 19 |
| 249497008 | Vomiting symptom (finding) | 15 | 15 |
| 422587007 | Nausea (finding) | 15 | 15 |
| 68235000 | Nasal congestion (finding) | 15 | 15 |
| 221360009 | Spasticity (finding) | 9 | 9 |
| 236077008 | Protracted diarrhea (finding) | 9 | 9 |
| 108631000119101 | History of autologous bone marrow transplant (situation) | 8 | 8 |
| 267060006 | Diarrhea symptom (finding) | 8 | 8 |
| 398152000 | Poor muscle tone (finding) | 6 | 6 |
| 22253000 | Pain (finding) | 4 | 4 |
| 698303004 | Awaiting transplantation of bone marrow (situation) | 4 | 4 |
| 1501000119109 | Proliferative diabetic retinopathy due to type II diabetes mellitus | 3 | 3 |
| 153351000119102 | History of peripheral stem cell transplant (situation) | 3 | 3 |
| 428915008 | Toxoplasma gondii antibody detected (finding) | 3 | 3 |
| 14760008 | Constipation (finding) | 2 | 2 |
| 160701002 | Social migrant (finding) | 2 | 2 |
| 288959006 | Unable to swallow saliva (finding) | 2 | 2 |
| 62718007 | Dribbling from mouth (finding) | 2 | 2 |
| 1003755004 | Allergy to Hevea brasiliensis latex protein (finding) | 1 | 1 |
| 161622006 | History of lower limb amputation (situation) | 1 | 1 |
| 161679004 | History of artificial joint (situation) | 1 | 1 |
| 225444004 | At increased risk for suicide (finding) | 1 | 1 |
| 246677007 | Passive conjunctival congestion (finding) | 1 | 1 |
| 66857006 | Hemoptysis (finding) | 1 | 1 |
