# Phase 8: predictions for 4 example patients

Built by `scripts/08_predict_demo.py` with `inference.Predictor`. Each patient is placed among the 2,581 reference (model-cohort) patients without refitting anything: features from the saved transforms; in each year's k-NN graph the patient is linked to their 10 most similar reference patients (normalization updated for the affected nodes only); the Nystrom extension places them in the 2022 spectral embedding, where the phenotype, the similar patients and the neighbour outcomes are read. Predictor loaded in 0.52 s. Synthetic data (Synthea).

The examples are realistic profiles, not chosen to land in different phenotypes: the multimorbid elderly patient lands in the same phenotype as the metabolic-syndrome pattern. Phenotype 2 holds 132 of the 173 reference patients aged 75+ with 6+ chronic conditions (a cardio-metabolic continuum).

| # | Patient | Phenotype | At risk for | Neighbour any_onset, 10 nearest | Neighbour any_onset, 50 nearest | All reference patients |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 1 | Healthy young adult | 1 | 9 of 9 groups | 3/8 (38%) | 13/44 (30%) | 414/2221 (19%) |
| 2 | Metabolic-syndrome pattern | 2 | 5 of 9 groups | 3/10 (30%) | 15/50 (30%) | 414/2221 (19%) |
| 3 | Multimorbid elderly | 2 | 5 of 9 groups | 3/10 (30%) | 15/49 (31%) | 414/2221 (19%) |
| 4 | Chronic pain and musculoskeletal | 4 | 5 of 9 groups | 1/8 (12%) | 8/43 (19%) | 414/2221 (19%) |

**Risk set.** `any_onset` is a new onset in 2023-2025 in any of the 9 onset groups; a group a patient already has is masked, so each neighbour counts only for the groups they were still at risk of. A new onset can only occur in a group the patient does not already have, so a multimorbid patient has fewer groups left at risk, and so do their similar patients. A healthy patient's neighbours can therefore show a higher any_onset rate than a multimorbid patient's: more groups in which a new onset can occur, not more disease.

## 1. Healthy young adult

Female, born 1999-03-15 (age 23.8 at the cutoff 2023-01-01). 6 history entries; prediction in 0.24 s.

**Conditions used** (onset year; chronic = active from onset through the last window):

| Onset | Condition | Code | Chronic |
| --- | --- | --- | --- |
| 2018 | Viral sinusitis | 444814009 | no |
| 2020 | Acute viral pharyngitis | 195662009 | no |
| 2021 | Gingivitis | 66383009 | no |
| 2022 | Sprain of ankle | 44465007 | no |

**Input notes.**

- 1 code(s) not in the vocabulary were ignored: 12345678
- 384709000 (Sprain, 2022-01-01) not used: redundant: recorded with a more specific code, dropped from the features as in Phase 3

**Nearest phenotype: cluster 1** (786 reference patients). Most over-represented chronic conditions: Childhood asthma (6% vs 2% overall); Perennial allergic rhinitis (5% vs 2% overall); Seasonal allergic rhinitis (2% vs 1% overall); Perennial allergic rhinitis with seasonal variation (6% vs 2% overall); Child attention deficit disorder (3% vs 1% overall).

**At risk for 9 of 9 groups** (ckd, prediabetes, hypertension, ischemic heart disease, metabolic syndrome, hyperlipidemia, obesity, chronic pain, substance use). Already has: none.

**Supervised `any_onset` probability: 21.8%** (TemporalGCN, Platt-calibrated; test AUROC 0.559 (0.483-0.632), an interval that includes 0.5, so near chance; reference base rate 18.6%).

**Neighbour outcomes**: new onsets in 2023-2025 among the nearest reference patients in the spectral embedding, over those whose label is observed (masked labels never count).

| Onset | Patient already had it | 10 nearest | 50 nearest | All reference patients |
| --- | --- | ---: | ---: | ---: |
| any of the groups |  | 3/8 (38%) | 13/44 (30%) | 414/2221 (19%) |
| ckd | no | 0/8 (0%) | 1/44 (2%) | 32/2054 (2%) |
| prediabetes | no | 0/8 (0%) | 2/40 (5%) | 51/1381 (4%) |
| hypertension | no | 0/8 (0%) | 0/41 (0%) | 51/1729 (3%) |
| ischemic heart disease | no | 0/8 (0%) | 0/42 (0%) | 55/1887 (3%) |
| metabolic syndrome | no | 0/8 (0%) | 0/44 (0%) | 31/1924 (2%) |
| hyperlipidemia | no | 0/7 (0%) | 1/40 (2%) | 56/1915 (3%) |
| obesity | no | 0/8 (0%) | 4/38 (11%) | 72/1188 (6%) |
| chronic pain | no | 1/8 (12%) | 1/41 (2%) | 34/1608 (2%) |
| substance use | no | 2/8 (25%) | 6/43 (14%) | 109/1720 (6%) |

**10 most similar reference patients** (cosine similarity in the 2022 spectral embedding) and the chronic conditions they share with this patient:

| # | Patient | Similarity | Age | Sex | Phenotype | Shared / their chronic | Shared chronic conditions |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | c2345b3a | 0.996 | 11 | F | 1 | 0 / 0 | - |
| 2 | 84f5c190 | 0.962 | 10 | M | 1 | 0 / 0 | - |
| 3 | e8d6fcd2 | 0.942 | 26 | M | 1 | 0 / 1 | - |
| 4 | a1a70a51 | 0.938 | 54 | M | 1 | 0 / 1 | - |
| 5 | dc838d03 | 0.918 | 16 | M | 1 | 0 / 0 | - |
| 6 | 8f84f321 | 0.917 | 17 | F | 1 | 0 / 0 | - |
| 7 | e9f768e0 | 0.917 | 13 | F | 1 | 0 / 0 | - |
| 8 | e999bf4e | 0.913 | 20 | M | 1 | 0 / 0 | - |
| 9 | 296a793c | 0.910 | 21 | M | 1 | 0 / 0 | - |
| 10 | aa4e14e1 | 0.908 | 11 | F | 1 | 0 / 0 | - |

**Explanation**: the patient's chronic conditions most over-represented among the 50 nearest reference patients (lift = share there / share among all reference patients):

No chronic condition: the placement rests on acute conditions and transitions.

**Trajectory** (TemporalGraphAE GRU state after each window, inserted into each year's graph; the same space as the reference patients' trajectories):

| Window | Graph neighbours | Mean similarity | Step from previous year |
| --- | ---: | ---: | ---: |
| 2017 | 0 | - (isolated) |  |
| 2018 | 10 | 1.000 | 1.944 |
| 2019 | 10 | 1.000 | 0.991 |
| 2020 | 10 | 0.828 | 0.618 |
| 2021 | 10 | 0.791 | 0.482 |
| 2022 | 10 | 0.704 | 0.182 |

## 2. Metabolic-syndrome pattern

Male, born 1972-06-01 (age 50.6 at the cutoff 2023-01-01). 5 history entries; prediction in 0.25 s.

**Conditions used** (onset year; chronic = active from onset through the last window):

| Onset | Condition | Code | Chronic |
| --- | --- | --- | --- |
| 2017 | Body mass index 30+ - obesity | 162864005 | yes |
| 2018 | Prediabetes | 714628002 | yes |
| 2019 | Viral sinusitis | 444814009 | no |
| 2019 | Essential hypertension | 59621000 | yes |
| 2021 | Metabolic syndrome X | 237602007 | yes |

**Nearest phenotype: cluster 2** (630 reference patients). Most over-represented chronic conditions: Neuropathy due to type 2 diabetes mellitus (6% vs 1% overall); Nonproliferative diabetic retinopathy due to type II diabetes mellitus (6% vs 1% overall); Chronic kidney disease stage 4 (5% vs 1% overall); Acute non-ST segment elevation myocardial infarction (8% vs 2% overall); History of coronary artery bypass grafting (20% vs 5% overall).

**At risk for 5 of 9 groups** (ckd, ischemic heart disease, hyperlipidemia, chronic pain, substance use). Already has: prediabetes, hypertension, metabolic syndrome, obesity.

**Supervised `any_onset` probability: 17.6%** (TemporalGCN, Platt-calibrated; test AUROC 0.559 (0.483-0.632), an interval that includes 0.5, so near chance; reference base rate 18.6%).

**Neighbour outcomes**: new onsets in 2023-2025 among the nearest reference patients in the spectral embedding, over those whose label is observed (masked labels never count).

| Onset | Patient already had it | 10 nearest | 50 nearest | All reference patients |
| --- | --- | ---: | ---: | ---: |
| any of the groups |  | 3/10 (30%) | 15/50 (30%) | 414/2221 (19%) |
| ckd | no | 0/10 (0%) | 2/39 (5%) | 32/2054 (2%) |
| prediabetes | yes | 0/0 | 0/3 (0%) | 51/1381 (4%) |
| hypertension | yes | 0/0 | 0/0 | 51/1729 (3%) |
| ischemic heart disease | no | 1/8 (12%) | 2/38 (5%) | 55/1887 (3%) |
| metabolic syndrome | yes | 0/1 (0%) | 2/15 (13%) | 31/1924 (2%) |
| hyperlipidemia | no | 2/5 (40%) | 6/36 (17%) | 56/1915 (3%) |
| obesity | yes | 0/0 | 1/10 (10%) | 72/1188 (6%) |
| chronic pain | no | 0/6 (0%) | 1/38 (3%) | 34/1608 (2%) |
| substance use | no | 1/9 (11%) | 4/36 (11%) | 109/1720 (6%) |

**10 most similar reference patients** (cosine similarity in the 2022 spectral embedding) and the chronic conditions they share with this patient:

| # | Patient | Similarity | Age | Sex | Phenotype | Shared / their chronic | Shared chronic conditions |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 8816182b | 0.928 | 49 | F | 2 | 4 / 6 | Body mass index 30+ - obesity; Metabolic syndrome X; Essential hypertension; Prediabetes |
| 2 | 1de30e29 | 0.915 | 64 | F | 2 | 4 / 6 | Body mass index 30+ - obesity; Metabolic syndrome X; Essential hypertension; Prediabetes |
| 3 | aad9f1c3 | 0.911 | 71 | M | 2 | 4 / 8 | Body mass index 30+ - obesity; Metabolic syndrome X; Essential hypertension; Prediabetes |
| 4 | 8b7fda01 | 0.902 | 52 | F | 2 | 4 / 4 | Body mass index 30+ - obesity; Metabolic syndrome X; Essential hypertension; Prediabetes |
| 5 | 343849ae | 0.890 | 57 | M | 2 | 4 / 10 | Body mass index 30+ - obesity; Metabolic syndrome X; Essential hypertension; Prediabetes |
| 6 | e5133e96 | 0.889 | 61 | M | 2 | 4 / 9 | Body mass index 30+ - obesity; Metabolic syndrome X; Essential hypertension; Prediabetes |
| 7 | e15491eb | 0.877 | 58 | M | 2 | 4 / 9 | Body mass index 30+ - obesity; Metabolic syndrome X; Essential hypertension; Prediabetes |
| 8 | e83ed9d0 | 0.873 | 62 | M | 2 | 3 / 10 | Body mass index 30+ - obesity; Essential hypertension; Prediabetes |
| 9 | 0510677e | 0.873 | 94 | M | 2 | 4 / 11 | Body mass index 30+ - obesity; Metabolic syndrome X; Essential hypertension; Prediabetes |
| 10 | 64352eed | 0.873 | 54 | M | 2 | 4 / 9 | Body mass index 30+ - obesity; Metabolic syndrome X; Essential hypertension; Prediabetes |

**Explanation**: the patient's chronic conditions most over-represented among the 50 nearest reference patients (lift = share there / share among all reference patients):

| Condition | Neighbours with it | All reference patients | Lift |
| --- | ---: | ---: | ---: |
| Metabolic syndrome X | 35/50 (70%) | 13.1% | 5.3 |
| Essential hypertension | 50/50 (100%) | 21.2% | 4.7 |
| Prediabetes | 47/50 (94%) | 36.8% | 2.6 |
| Body mass index 30+ - obesity | 40/50 (80%) | 44.6% | 1.8 |

**Trajectory** (TemporalGraphAE GRU state after each window, inserted into each year's graph; the same space as the reference patients' trajectories):

| Window | Graph neighbours | Mean similarity | Step from previous year |
| --- | ---: | ---: | ---: |
| 2017 | 10 | 1.000 |  |
| 2018 | 10 | 0.891 | 1.532 |
| 2019 | 10 | 0.841 | 0.930 |
| 2020 | 10 | 0.773 | 0.853 |
| 2021 | 10 | 0.742 | 0.736 |
| 2022 | 10 | 0.736 | 0.440 |

## 3. Multimorbid elderly

Female, born 1944-09-10 (age 78.3 at the cutoff 2023-01-01). 9 history entries; prediction in 0.24 s.

**Conditions used** (onset year; chronic = active from onset through the last window):

| Onset | Condition | Code | Chronic |
| --- | --- | --- | --- |
| 2010 | Essential hypertension | 59621000 | yes |
| 2012 | Hyperlipidemia | 55822004 | yes |
| 2014 | Diabetes mellitus type 2 | 44054006 | yes |
| 2016 | Ischemic heart disease | 414545008 | yes |
| 2017 | Osteoarthritis of knee | 239873007 | yes |
| 2018 | Chronic kidney disease stage 2 | 431856006 | yes |
| 2019 | Anemia | 271737000 | yes |
| 2020 | Osteoporosis | 64859006 | yes |
| 2021 | Chronic congestive heart failure | 88805009 | yes |

**Nearest phenotype: cluster 2** (630 reference patients). Most over-represented chronic conditions: Neuropathy due to type 2 diabetes mellitus (6% vs 1% overall); Nonproliferative diabetic retinopathy due to type II diabetes mellitus (6% vs 1% overall); Chronic kidney disease stage 4 (5% vs 1% overall); Acute non-ST segment elevation myocardial infarction (8% vs 2% overall); History of coronary artery bypass grafting (20% vs 5% overall).

**At risk for 5 of 9 groups** (prediabetes, metabolic syndrome, obesity, chronic pain, substance use). Already has: ckd, hypertension, ischemic heart disease, hyperlipidemia.

**Supervised `any_onset` probability: 7.2%** (TemporalGCN, Platt-calibrated; test AUROC 0.559 (0.483-0.632), an interval that includes 0.5, so near chance; reference base rate 18.6%).

**Neighbour outcomes**: new onsets in 2023-2025 among the nearest reference patients in the spectral embedding, over those whose label is observed (masked labels never count).

| Onset | Patient already had it | 10 nearest | 50 nearest | All reference patients |
| --- | --- | ---: | ---: | ---: |
| any of the groups |  | 3/10 (30%) | 15/49 (31%) | 414/2221 (19%) |
| ckd | yes | 1/8 (12%) | 2/38 (5%) | 32/2054 (2%) |
| prediabetes | no | 0/0 | 0/2 (0%) | 51/1381 (4%) |
| hypertension | yes | 0/0 | 0/0 | 51/1729 (3%) |
| ischemic heart disease | yes | 1/6 (17%) | 3/36 (8%) | 55/1887 (3%) |
| metabolic syndrome | no | 0/1 (0%) | 2/13 (15%) | 31/1924 (2%) |
| hyperlipidemia | yes | 1/6 (17%) | 5/34 (15%) | 56/1915 (3%) |
| obesity | no | 0/2 (0%) | 1/9 (11%) | 72/1188 (6%) |
| chronic pain | no | 0/7 (0%) | 1/35 (3%) | 34/1608 (2%) |
| substance use | no | 1/7 (14%) | 4/36 (11%) | 109/1720 (6%) |

**10 most similar reference patients** (cosine similarity in the 2022 spectral embedding) and the chronic conditions they share with this patient:

| # | Patient | Similarity | Age | Sex | Phenotype | Shared / their chronic | Shared chronic conditions |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 55ddc877 | 0.847 | 63 | F | 2 | 1 / 6 | Essential hypertension |
| 2 | 76a2c3b0 | 0.819 | 60 | F | 2 | 2 / 6 | Ischemic heart disease; Essential hypertension |
| 3 | 4a7ca231 | 0.804 | 79 | M | 2 | 5 / 14 | Anemia; Ischemic heart disease; Hyperlipidemia; Essential hypertension; Osteoporosis |
| 4 | 47c1d86a | 0.785 | 52 | F | 2 | 2 / 5 | Anemia; Essential hypertension |
| 5 | 9daea916 | 0.754 | 66 | F | 2 | 3 / 13 | Anemia; Diabetes mellitus type 2; Essential hypertension |
| 6 | 8739f725 | 0.751 | 105 | M | 2 | 5 / 6 | Anemia; Ischemic heart disease; Hyperlipidemia; Essential hypertension; Osteoporosis |
| 7 | 279fbb62 | 0.741 | 68 | M | 2 | 2 / 8 | Anemia; Essential hypertension |
| 8 | 5e2ff3ca | 0.735 | 56 | M | 2 | 2 / 6 | Anemia; Essential hypertension |
| 9 | 55b5c2df | 0.734 | 47 | M | 2 | 2 / 7 | Anemia; Essential hypertension |
| 10 | 42c1a845 | 0.733 | 91 | F | 2 | 4 / 16 | Anemia; Ischemic heart disease; Hyperlipidemia; Essential hypertension |

**Explanation**: the patient's chronic conditions most over-represented among the 50 nearest reference patients (lift = share there / share among all reference patients):

| Condition | Neighbours with it | All reference patients | Lift |
| --- | ---: | ---: | ---: |
| Essential hypertension | 50/50 (100%) | 21.2% | 4.7 |
| Anemia | 40/50 (80%) | 31.2% | 2.6 |
| Hyperlipidemia | 11/50 (22%) | 8.6% | 2.5 |
| Osteoporosis | 5/50 (10%) | 4.0% | 2.5 |
| Chronic kidney disease stage 2 | 6/50 (12%) | 5.5% | 2.2 |

**Trajectory** (TemporalGraphAE GRU state after each window, inserted into each year's graph; the same space as the reference patients' trajectories):

| Window | Graph neighbours | Mean similarity | Step from previous year |
| --- | ---: | ---: | ---: |
| 2017 | 10 | 0.569 |  |
| 2018 | 10 | 0.564 | 1.192 |
| 2019 | 10 | 0.553 | 0.600 |
| 2020 | 10 | 0.545 | 0.303 |
| 2021 | 10 | 0.539 | 0.196 |
| 2022 | 10 | 0.535 | 0.201 |

## 4. Chronic pain and musculoskeletal

Female, born 1944-09-10 (age 78.3 at the cutoff 2023-01-01). 8 history entries; prediction in 0.26 s.

**Conditions used** (onset year; chronic = active from onset through the last window):

| Onset | Condition | Code | Chronic |
| --- | --- | --- | --- |
| 2013 | Body mass index 30+ - obesity | 162864005 | yes |
| 2015 | Chronic low back pain | 278860009 | yes |
| 2015 | Chronic pain | 82423001 | yes |
| 2016 | Osteoarthritis of knee | 239873007 | yes |
| 2017 | Ischemic heart disease | 414545008 | yes |
| 2018 | Osteoporosis | 64859006 | yes |
| 2019 | Anemia | 271737000 | yes |
| 2020 | Hyperlipidemia | 55822004 | yes |

**Nearest phenotype: cluster 4** (279 reference patients). Most over-represented chronic conditions: Chronic neck pain (58% vs 11% overall); Chronic low back pain (84% vs 18% overall); Fibromyalgia (9% vs 2% overall); Chronic pain (98% vs 25% overall); Unhealthy alcohol drinking behavior (14% vs 9% overall).

**At risk for 5 of 9 groups** (ckd, prediabetes, hypertension, metabolic syndrome, substance use). Already has: ischemic heart disease, hyperlipidemia, obesity, chronic pain.

**Supervised `any_onset` probability: 15.7%** (TemporalGCN, Platt-calibrated; test AUROC 0.559 (0.483-0.632), an interval that includes 0.5, so near chance; reference base rate 18.6%).

**Neighbour outcomes**: new onsets in 2023-2025 among the nearest reference patients in the spectral embedding, over those whose label is observed (masked labels never count).

| Onset | Patient already had it | 10 nearest | 50 nearest | All reference patients |
| --- | --- | ---: | ---: | ---: |
| any of the groups |  | 1/8 (12%) | 8/43 (19%) | 414/2221 (19%) |
| ckd | no | 0/8 (0%) | 0/42 (0%) | 32/2054 (2%) |
| prediabetes | no | 0/7 (0%) | 0/29 (0%) | 51/1381 (4%) |
| hypertension | no | 0/7 (0%) | 1/36 (3%) | 51/1729 (3%) |
| ischemic heart disease | yes | 0/8 (0%) | 1/40 (2%) | 55/1887 (3%) |
| metabolic syndrome | no | 0/7 (0%) | 1/41 (2%) | 31/1924 (2%) |
| hyperlipidemia | yes | 0/8 (0%) | 1/40 (2%) | 56/1915 (3%) |
| obesity | yes | 0/3 (0%) | 3/19 (16%) | 72/1188 (6%) |
| chronic pain | yes | 0/0 | 0/0 | 34/1608 (2%) |
| substance use | no | 1/8 (12%) | 3/37 (8%) | 109/1720 (6%) |

**10 most similar reference patients** (cosine similarity in the 2022 spectral embedding) and the chronic conditions they share with this patient:

| # | Patient | Similarity | Age | Sex | Phenotype | Shared / their chronic | Shared chronic conditions |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 25482b3d | 0.951 | 43 | F | 4 | 3 / 3 | Body mass index 30+ - obesity; Chronic low back pain; Chronic pain |
| 2 | daafe999 | 0.951 | 47 | F | 4 | 3 / 4 | Body mass index 30+ - obesity; Chronic low back pain; Chronic pain |
| 3 | e0a1de09 | 0.950 | 45 | M | 4 | 2 / 3 | Body mass index 30+ - obesity; Chronic pain |
| 4 | b5eadd82 | 0.945 | 35 | M | 4 | 1 / 14 | Chronic pain |
| 5 | 6b076264 | 0.942 | 44 | M | 4 | 3 / 5 | Body mass index 30+ - obesity; Chronic low back pain; Chronic pain |
| 6 | 4274cdbe | 0.939 | 54 | M | 4 | 2 / 6 | Body mass index 30+ - obesity; Chronic pain |
| 7 | 56313fa2 | 0.939 | 24 | F | 4 | 2 / 3 | Chronic low back pain; Chronic pain |
| 8 | c65a6686 | 0.933 | 70 | F | 4 | 3 / 4 | Body mass index 30+ - obesity; Chronic low back pain; Chronic pain |
| 9 | 3c9f6085 | 0.933 | 23 | F | 4 | 2 / 4 | Chronic low back pain; Chronic pain |
| 10 | 056bef15 | 0.932 | 30 | F | 4 | 2 / 2 | Chronic low back pain; Chronic pain |

**Explanation**: the patient's chronic conditions most over-represented among the 50 nearest reference patients (lift = share there / share among all reference patients):

| Condition | Neighbours with it | All reference patients | Lift |
| --- | ---: | ---: | ---: |
| Chronic low back pain | 41/50 (82%) | 17.9% | 4.6 |
| Chronic pain | 50/50 (100%) | 24.9% | 4.0 |
| Body mass index 30+ - obesity | 27/50 (54%) | 44.6% | 1.2 |
| Osteoporosis | 2/50 (4%) | 4.0% | 1.0 |
| Osteoarthritis of knee | 2/50 (4%) | 4.5% | 0.9 |

**Trajectory** (TemporalGraphAE GRU state after each window, inserted into each year's graph; the same space as the reference patients' trajectories):

| Window | Graph neighbours | Mean similarity | Step from previous year |
| --- | ---: | ---: | ---: |
| 2017 | 10 | 0.682 |  |
| 2018 | 10 | 0.610 | 1.372 |
| 2019 | 10 | 0.591 | 0.683 |
| 2020 | 10 | 0.547 | 0.342 |
| 2021 | 10 | 0.524 | 0.705 |
| 2022 | 10 | 0.524 | 0.551 |
