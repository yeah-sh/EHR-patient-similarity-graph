# Labels

Built by `scripts/05_build_labels.py` for the model cohort (2,581 patients, every one a graph node whatever its labels). Inputs use windows before the cutoff 2023-01-01; labels use the horizon 2023-01-01 to 2026-01-01 (exclusive). Events, encounters and deaths on or after the horizon end are never read. Labels are 1, 0 or masked (NaN); every count and rate below uses unmasked entries only.

## Targets

- **Primary:** `any_onset`: 1 if the patient has a new onset in any onset group in 2023-2025; 0 if at least one group label is observed and every observed one is 0; masked if every group label is masked.
- **Secondary:** the per-group labels `onset_<group>`.
- **Not a target:** `mortality`, kept for descriptive statistics only (see below).

## Onset groups

A group's onset is its first event, of any of its codes. Precedence: died before the cutoff (masked) > selected by post-cutoff codes (masked) > had it before the cutoff (masked) > (a) lost to follow-up (masked) > onset in the horizon (1) > (b) died in the horizon without it (masked) > 0.

| Group | Codes |
| --- | --- |
| ckd | Chronic kidney disease stage 1 (disorder) (431855005); Chronic kidney disease stage 2 (disorder) (431856006); Chronic kidney disease stage 3 (disorder) (433144002); Chronic kidney disease stage 4 (disorder) (431857002); End-stage renal disease (disorder) (46177005); Disorder of kidney due to diabetes mellitus (disorder) (127013003) |
| prediabetes | Prediabetes (finding) (714628002) |
| hypertension | Essential hypertension (disorder) (59621000) |
| ischemic_heart_disease | Ischemic heart disease (disorder) (414545008); Preinfarction syndrome (disorder) (4557003); Myocardial infarction (disorder) (22298006); Acute ST segment elevation myocardial infarction (disorder) (401303003); Acute non-ST segment elevation myocardial infarction (disorder) (401314000); History of myocardial infarction (situation) (399211009) |
| metabolic_syndrome | Metabolic syndrome X (disorder) (237602007) |
| hyperlipidemia | Hyperlipidemia (disorder) (55822004); Hypertriglyceridemia (disorder) (302870006) |
| obesity | Body mass index 30+ - obesity (finding) (162864005); Body mass index 40+ - severely obese (finding) (408512008) |
| chronic_pain | Chronic pain (finding) (82423001); Chronic low back pain (finding) (278860009); Chronic neck pain (finding) (1121000119107); Fibromyalgia (disorder) (203082005); Primary fibromyalgia syndrome (disorder) (95417003); Chronic intractable migraine without aura (disorder) (124171000119105); Transformed migraine (disorder) (427419006) |
| substance_use | Dependent drug abuse (disorder) (6525002); Opioid abuse (5602001); Misuses drugs (finding) (361055000); Alcoholism (disorder) (7200002); Unhealthy alcohol drinking behavior (finding) (10939881000119105); Overdose (disorder) (1149222004) |

| Group | Had it before cutoff (masked) | New onsets 2023-2025 | Negatives | Positive rate | Masked: (a) lost (after onset) | Masked: (b) died | Masked: died before | Masked: selected by future | Review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ckd | 172 | 32 | 2,022 | 1.6% | 170 (0) | 22 | 77 | 86 | keep, **borderline** |
| prediabetes | 901 | 51 | 1,330 | 3.7% | 129 (7) | 7 | 77 | 86 | keep |
| hypertension | 508 | 51 | 1,678 | 2.9% | 168 (0) | 13 | 77 | 86 | keep |
| ischemic_heart_disease | 347 | 55 | 1,832 | 2.9% | 169 (1) | 15 | 77 | 86 | keep |
| metabolic_syndrome | 308 | 31 | 1,893 | 1.6% | 170 (0) | 16 | 77 | 86 | keep, **borderline** |
| hyperlipidemia | 313 | 56 | 1,859 | 2.9% | 169 (0) | 21 | 77 | 86 | keep |
| obesity | 1,101 | 72 | 1,116 | 6.1% | 123 (9) | 6 | 77 | 86 | keep |
| chronic_pain | 644 | 34 | 1,574 | 2.1% | 148 (0) | 18 | 77 | 86 | keep |
| substance_use | 532 | 109 | 1,611 | 6.3% | 143 (7) | 23 | 77 | 86 | keep |

Groups in `labels.borderline_groups` are kept but flagged **borderline**: they have few new onsets for a review threshold of 30, so their metrics will be noisy.

## Primary target: any_onset

Positives / observed: 414 / 2,221 (18.6%); masked: 360. 66 of the positives have new onsets in more than one group.

| any_onset status | Value | Patients |
| --- | --- | ---: |
| positive | 1 | 414 |
| negative | 0 | 1,807 |
| died before cutoff | masked | 77 |
| selected by post-cutoff codes | masked | 86 |
| had every group before cutoff | masked | 2 |
| lost to follow-up after onset | masked | 21 |
| lost to follow-up | masked | 149 |
| died without onset | masked | 25 |

Patients with every target label masked (any_onset and every group): 360. With mortality masked too, so every label: 163 (died before the cutoff or selected by post-cutoff codes).

## Follow-up and censoring

| Rule | Patients | Onset label entries masked |
| --- | ---: | ---: |
| Died before the cutoff: every label masked | 77 | all |
| Selected by post-cutoff codes (fewer than 2 distinct codes before the cutoff): every label masked | 86 | all |
| (a) Alive through the horizon, last encounter before 2025-01-01: every onset label masked | 177 | 1,389 |
| (b) Died in the horizon: onset labels without that onset masked | 29 | 141 |

Rule (a) also masks 24 onsets that fall in the horizon (the patient was lost later); rule (b) keeps the 7 onsets seen before death as positives. Patients who died in the horizon are judged by rule (b) only, so (a) does not mask their onsets before death.

## Mortality (descriptive only)

Observed: 2,418 patients alive on 2023-01-01 and not selected by post-cutoff codes. Died 2023-01-01 to 2026-01-01 (exclusive): 29 (1.2%). Masked: 163.

Mortality is not a model target: 29 deaths are too few to train on or to evaluate (per-split counts in the split table). The column stays in `labels.parquet` for descriptive statistics only.

## Cohort selection check

86 model-cohort patients have fewer than 2 distinct codes before the cutoff: they are in the cohort (Phase 2 rule, applied to all dates) only thanks to later codes, so they are certain to have a new condition after the cutoff and to be alive at it. They stay in the graph as nodes, with their pre-cutoff features, but every label of theirs is masked: each onset group, `any_onset` and `mortality`.

## Split

Patient-level 70%/15%/15%, stratified by Phase 4 phenotype cluster x `any_onset` label (masked is its own value), seed 42; saved to `splits.json`. Positives / observed (rate):

| Label | train | val | test |
| --- | ---: | ---: | ---: |
| Patients | 1,810 | 389 | 382 |
| any_onset (primary) | 290 / 1,556 (18.6%) | 63 / 335 (18.8%) | 61 / 330 (18.5%) |
| onset_ckd | 19 / 1,439 (1.3%) | 7 / 312 (2.2%) | 6 / 303 (2.0%) |
| onset_prediabetes | 37 / 966 (3.8%) | 3 / 207 (1.4%) | 11 / 208 (5.3%) |
| onset_hypertension | 35 / 1,215 (2.9%) | 11 / 267 (4.1%) | 5 / 247 (2.0%) |
| onset_ischemic_heart_disease | 40 / 1,319 (3.0%) | 6 / 285 (2.1%) | 9 / 283 (3.2%) |
| onset_metabolic_syndrome | 20 / 1,343 (1.5%) | 6 / 300 (2.0%) | 5 / 281 (1.8%) |
| onset_hyperlipidemia | 39 / 1,329 (2.9%) | 6 / 295 (2.0%) | 11 / 291 (3.8%) |
| onset_obesity | 50 / 829 (6.0%) | 13 / 188 (6.9%) | 9 / 171 (5.3%) |
| onset_chronic_pain | 24 / 1,134 (2.1%) | 6 / 243 (2.5%) | 4 / 231 (1.7%) |
| onset_substance_use | 76 / 1,209 (6.3%) | 16 / 263 (6.1%) | 17 / 248 (6.9%) |
| mortality (descriptive only) | 19 / 1,692 (1.1%) | 6 / 367 (1.6%) | 4 / 359 (1.1%) |

## Phenotypes (Phase 4, attached for analysis only)

The k = 7 clusters of the 2022 graph, reused as computed in Phase 4, with its top over-represented chronic conditions. They stratify the split and support analysis; they are never a training target.

| Cluster | Patients | Top over-represented conditions |
| ---: | ---: | --- |
| 1 | 786 | Childhood asthma, Perennial allergic rhinitis, Seasonal allergic rhinitis, Perennial allergic rhinitis with seasonal variation, Child attention deficit disorder |
| 2 | 630 | Neuropathy due to type 2 diabetes mellitus, Nonproliferative diabetic retinopathy due to type II diabetes mellitus, Chronic kidney disease stage 4, Acute non-ST segment elevation myocardial infarction, History of coronary artery bypass grafting |
| 3 | 621 | Sepsis, Malignant neoplasm of breast, Recurrent urinary tract infection, Unhealthy alcohol drinking behavior, Localized  primary osteoarthritis of the hand |
| 4 | 279 | Chronic neck pain, Chronic low back pain, Fibromyalgia, Chronic pain, Unhealthy alcohol drinking behavior |
| 5 | 116 | Impacted molars, Dependent drug abuse, Chronic intractable migraine without aura, Chronic pain, Sleep disorder |
| 6 | 102 | (no chronic condition common enough to list) |
| 7 | 47 | Obstructive sleep apnea syndrome, Sleep disorder, Essential hypertension, Metabolic syndrome X, Ischemic heart disease |
