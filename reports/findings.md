# Findings, Phases 4-7b

Synthea synthetic EHR, 2,581 model-cohort patients, yearly windows 2017-2022 before the prediction cutoff (2023-01-01). Brackets are 95% percentile bootstrap intervals over patients; a difference counts as shown only if its interval excludes 0. *Neighbour concordance* is the mean Jaccard overlap of chronic conditions between a test patient and their 5 nearest neighbours in an embedding (cosine); random patients score 0.103 (0.094-0.112). Details: `graphs_summary.md`, `results.md`, `evaluation_summary.md`, `embedding_results.md`.

## 1. The patient-similarity graph carries clinical structure

The same self-supervised temporal graph autoencoder (GraphAE) trained on the real k-NN graphs, on degree-preserving rewired graphs, and with no graph (A_hat = I):

| GraphAE trained on | Concordance | Real graph minus this |
| --- | ---: | ---: |
| Real similarity graphs | 0.358 (0.332-0.381) | |
| Rewired graphs (same degrees) | 0.123 (0.111-0.134) | +0.235 (+0.213 to +0.256) |
| No graph | 0.318 (0.297-0.340) | +0.040 (+0.020 to +0.060) |

Rewiring while keeping every patient's degree brings concordance down to near chance, so it is who is linked to whom that carries the structure. The 2022 graph's Laplacian spectrum yields 7 phenotypes (eigengap) with coherent profiles: chronic pain in 98% of cluster 4, obstructive sleep apnoea in 98% of cluster 7 (40.7 times its overall rate), diabetic and cardiac complications in cluster 2. The spectrum rises smoothly (λ2 = 0.028 to λ20 = 0.126), so the phenotypes partition a continuum rather than separate communities.

## 2. Linear-algebraic embeddings preserve comorbidity similarity best

Concordance of the 32-dimensional embeddings, highest first: spectral (Laplacian eigenvectors of the 2022 graph) 0.431 (0.408-0.455); PCA of every window 0.414 (0.387-0.442); GraphAE with the link term only 0.403 (0.379-0.426); PCA of 2022 0.394 (0.369-0.419); TemporalGraphAE 0.358; GraphAE without a graph 0.318; supervised TemporalGCN 0.143 (0.130-0.156); rewired 0.123. The TemporalGraphAE is shown below the spectral embedding (-0.074, -0.095 to -0.055) and both PCAs (-0.057 and -0.036), and adding its next-window objective lowered concordance (-0.046, -0.062 to -0.030). Spectral ranks first on point estimate; its interval overlaps PCA's (that difference was not tested). The supervised embedding collapsed onto one direction (99.8% of its variance on the first principal component). No embedding reaches the 236-dimensional input features themselves (0.503, 0.481-0.527).

## 3. Per-year embeddings trace trajectories; phenotypes occupy distinct regions

See `figures/ae_trajectories_umap.png` (the GraphAE's state after each year 2017-2022 for 5 patients, chronic onsets marked) and `figures/ae_umap_phenotype.png`.

- The mean yearly step in the 32-dimensional space shrinks from 1.382 (2017 to 2018) to 0.267 (2021 to 2022). The long early steps are mostly the recurrent state settling as it reads more years; in the examples, later moves coincide with new chronic onsets (e.g. hypertension in 2020 and obesity in 2022 for one patient).
- Phenotypes 1 (childhood asthma, allergies), 2 (diabetic complications), 4 (chronic pain) and 5 (drug abuse, migraine) occupy distinct UMAP regions; 3 spreads across the middle, 6 overlaps 1, and 7 (47 patients) is scattered. The silhouette against the phenotypes is -0.012 (shuffled labels: -0.181): better than chance, but clusters touch, as the smooth spectrum implies.

## 4. Onset forecasting is near chance for every method on Synthea

- **Cross-validation (Phase 7).** 2,221 patients, 414 (18.6%) with a new chronic onset in 2023-2025. The full TemporalGCN reaches a pooled AUROC of 0.579 (0.549-0.608). All 8 models lie between 0.553 and 0.585 (logistic regression on 2022 features plus age and sex, 0.585 (0.556-0.614); age and sex alone, 0.554 (0.525-0.582)). No difference with the TemporalGCN is shown, including removing the graph (-0.003, -0.028 to +0.021) or time (+0.023, -0.010 to +0.057).
- **Held-out test.** 330 patients, 61 positives: AUROC 0.559 (0.483-0.632), an interval that includes 0.5. Brier 0.155 (0.129-0.180), against 0.151 for always predicting the base rate.
- **Next year.** New conditions in 2022 from 2021 embeddings. The GraphAE's pooled AUROC, 0.755 (0.719-0.792), matches the conditions' base rates alone, 0.760 (difference -0.005, -0.034 to +0.026). Within a condition, the 8 embeddings rank patients at 0.503-0.624, and only 2 have an interval above 0.5.
- **Reading.** Every method falls in one narrow band, and age and sex alone are the best model for ischaemic heart disease (0.88). This is consistent with a limit of the data rather than of the methods: Synthea's rule-based disease modules appear to leave earlier conditions little information about later onsets beyond age and sex. This reading is not tested here. Future work: real EHR data with longer histories, validated on an external cohort.

## Limitations

- **Synthetic data.** Synthea simulates care with fixed disease modules; the results describe how the methods behave on such data, not on real records.
- **10-year history window.** Records are complete only from late 2016 (Synthea's default export history), so features cover six yearly windows (2017-2022), and earlier conditions appear only if still active.
- **Transductive training.** Every patient, test patients included, was a graph node with their pre-cutoff features during training (never their labels). Scoring a new patient requires inserting them into the graph, which is an approximation.
- **Test-patient edges in the link term.** The GraphAE reconstructs every edge, test patients' included, so its encoder has seen their graph neighbourhoods. This may favour graph-based embeddings in the test-patient concordance comparisons.
- **Small positive counts.** 61 test positives; 31-109 positives per onset group across all folds and 4-17 on the test set; 122 next-year onsets over 12 conditions, 5 of them with 3 or fewer. Group-level results are indicative only, and the intervals omit retraining (seed) variation.
