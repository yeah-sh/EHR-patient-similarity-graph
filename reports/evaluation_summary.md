# Evaluation summary

Plain-language findings of Phase 7 (`scripts/07_evaluate.py`); every number, with its setup, is in `results.md`.

**The question.** From a patient's recorded conditions up to the end of 2022, predict whether they are diagnosed with a new chronic condition in one of 9 groups (ckd, prediabetes, hypertension, ischemic heart disease, metabolic syndrome, hyperlipidemia, obesity, chronic pain, substance use) from 2023-01-01 to the end of 2025. 18.6% of the 2,221 patients with a known outcome are. The data are synthetic (Synthea).

**How the models are compared.** Eight models, from age and sex alone to the full temporal graph network, each scored by 5-fold cross-validation on all 2,221 patients; differences are paired bootstrap intervals (95%). A difference counts as shown only if its interval excludes 0.

## Main findings

1. **Test set (scored once).** On the 330 held-out test patients (61 with a new onset), the final TemporalGCN has AUROC 0.559 (0.483 to 0.632): it ranks a patient who goes on to a new onset above one who does not 56% of the time, against 50% for guessing. Its interval includes 0.5: on this test set the model cannot be told apart from guessing. AUPRC is 0.218 (0.164 to 0.310) against 0.185 for a random score. Its validation AUROC, 0.678, lies above the test interval. A single split moves a lot: out of fold, the same architecture scores these test patients at 0.640 and the validation patients at 0.633. The pooled cross-validated AUROC, 0.579 (0.549 to 0.608), is the steadier estimate.
2. **Does the graph help? (model 8 vs model 6, the same network with no graph)** **Not shown.** The interval includes 0, so these data cannot tell the two apart (the point estimate favours the no-graph model): pooled AUROC 0.579 vs 0.583, difference -0.003 (95% CI -0.028 to +0.021); the full model is better in 4 of 5 folds.
3. **Does time help? (model 8 vs model 5, a GCN on the final window only)** **Not shown.** The interval includes 0, so these data cannot tell the two apart (the point estimate favours the full model): pooled AUROC 0.579 vs 0.556, difference +0.023 (95% CI -0.010 to +0.057); the full model is better in 3 of 5 folds. Model 5 is not blind to the past: its 2022 window is a time-decayed sum of earlier windows, so this tests the sequence model, not history itself. Caveat: model 5 ranks patients almost exactly as age and sex alone do (mean within-fold rank correlation with model 1: 0.97; full model: 0.46). With the hyperparameters fixed at the full model's setting it learned little beyond the demographics, so this comparison mostly measures what the full model adds to age and sex.
4. **Does the real similarity graph beat a random one? (model 8 vs model 7, degree-preserving rewired graphs)** **Not shown.** The interval includes 0, so these data cannot tell the two apart (the point estimate favours the full model): pooled AUROC 0.579 vs 0.556, difference +0.023 (95% CI -0.009 to +0.057); the full model is better in 3 of 5 folds. Caveat: model 7 ranks patients almost exactly as age and sex alone do (mean within-fold rank correlation with model 1: 0.97; full model: 0.46). With the hyperparameters fixed at the full model's setting it learned little beyond the demographics, so this comparison mostly measures what the full model adds to age and sex.
5. **Simple baselines.** Age and sex alone reach a pooled out-of-fold AUROC of 0.554; the best model without a GCN is model 2 (Logistic regression, 2022 window + age, sex) at 0.585. The full TemporalGCN (0.579) ranks 4th of 8. **Not shown.** The interval includes 0, so these data cannot tell the two apart (the point estimate favours model 2): pooled AUROC 0.579 vs 0.585, difference -0.005 (95% CI -0.029 to +0.021); the full model is better in 2 of 5 folds.
6. **Calibration.** Training with `pos_weight` pushes the raw probabilities up (mean 0.422 on test against an observed rate of 0.185). Platt scaling fitted on validation brings the mean to 0.192 and the ECE from 0.238 to 0.075 (0.059 to 0.130). The test Brier score, 0.155 (0.129 to 0.180), is not distinguishable from always predicting the observed rate (0.151): the probabilities now match the base rate, but carry little information beyond it.

## Where the TemporalGCN does not beat a simpler model

Models whose pooled AUROC the full TemporalGCN does not exceed with an interval excluding 0:

- **Model 1, Age and sex only** (age at the cutoff and sex, logistic regression): pooled AUROC 0.554 vs 0.579; full minus this +0.025 (-0.007 to +0.059), full model better in 3 of 5 folds.
- **Model 2, Logistic regression** (2022 window + age, sex): pooled AUROC 0.585 vs 0.579; full minus this -0.005 (-0.029 to +0.021), full model better in 2 of 5 folds.
- **Model 3, Logistic regression** (6 windows 2017-2022, concatenated + age, sex): pooled AUROC 0.584 vs 0.579; full minus this -0.005 (-0.032 to +0.023), full model better in 2 of 5 folds.
- **Model 4, Spectral embedding + logistic regression** (16 Laplacian eigenvectors of the 2022 graph + age, sex): pooled AUROC 0.553 vs 0.579; full minus this +0.027 (-0.002 to +0.054), full model better in 4 of 5 folds.
- **Model 5, Static GCN** (2022 window and its graph; age, sex in the heads): pooled AUROC 0.556 vs 0.579; full minus this +0.023 (-0.010 to +0.057), full model better in 3 of 5 folds.
- **Model 6, Temporal model, no graph** (6 windows 2017-2022, A_hat = I in every window; age, sex in the heads): pooled AUROC 0.583 vs 0.579; full minus this -0.003 (-0.028 to +0.021), full model better in 4 of 5 folds.
- **Model 7, TemporalGCN, rewired graphs** (6 windows 2017-2022 + degree-preserving random graphs; age, sex in the heads): pooled AUROC 0.556 vs 0.579; full minus this +0.023 (-0.009 to +0.057), full model better in 3 of 5 folds.
- **Onset groups (secondary targets):** another model has the highest pooled AUROC on 8 of 9 groups (ckd: model 2, hypertension: model 7, ischemic_heart_disease: model 1, metabolic_syndrome: model 7, hyperlipidemia: model 5, obesity: model 1, chronic_pain: model 2, substance_use: model 1). With 31-109 positives per group, the intervals overlap widely.

## What the embeddings show

- **Phenotypes:** by the silhouette score, the trajectory embedding h separates the 7 Phase 4 phenotype clusters no better than shuffled clusters do (-0.335 vs -0.127). The 2022 input features score 0.041 and the spectral embedding the clusters were defined on 0.047. h is trained to predict onsets, not to reproduce the clusters.
- **Spread:** h's first principal component holds 99.8% of its variance, so patients differ mostly along that one direction; nearest neighbours lie at distances of 0.0025-0.0077 against a median of 0.8157 between random patients.
- **Neighbours:** a patient's 5 nearest neighbours in h share on average 13% of their chronic conditions (Jaccard), against 10% for random patients and 51% for the input-feature neighbours that build the graph; 28% are in the same phenotype cluster (random: 22%).
- **Trajectories:** the per-year embeddings of five example patients are drawn as paths in `results.md` (figure `trajectories_umap.png`), with the years their chronic conditions began.

## Caveats

- **Synthetic data.** Synthea simulates disease modules with fixed rules; the results say how these methods behave on such data, not how they would on real records.
- **Slightly optimistic GCN estimates.** The GCN hyperparameters were tuned on the original validation split, whose patients are inside the cross-validation folds. The test set was not used for any choice.
- **One setting for every GCN.** The ablations (models 5-7) use the full model's tuned hyperparameters; tuned on their own they might do better. Nothing was retuned here.
- **What the intervals cover.** They resample patients; they do not include the variation from retraining (seeds and folds), so they understate the full uncertainty.
- **Transductive.** Every patient, including held-out ones, is a node of the graphs with their pre-cutoff features (never their labels). Scoring a new patient means placing them in the graph.
- **Small numbers.** 61 test positives, and 31-109 positives per onset group across all folds: group-level results are indicative only.
