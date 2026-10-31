# Embedding summary (Phase 7b)

Plain-language findings of `scripts/07b_embeddings.py`; every number, with its setup, is in `embedding_results.md`.

**Why this phase.** Phase 7 found that the supervised TemporalGCN's trajectory embedding had collapsed onto almost a single direction, and that a patient's nearest neighbours in it shared few of their chronic conditions compared with the neighbours in the input features. Phase 7b trains the same encoder (graph convolutions per year, a GRU over the years) without any outcome labels: it learns to reconstruct each year's patient-similarity graph and to predict which conditions are active the next year. Its embeddings are compared with seven others on held-out (test) patients.

**How results are read.** 95% paired bootstrap intervals over test patients; a difference counts as shown only if its interval excludes 0. The data are synthetic (Synthea).

**What the four measures mean.**

- *Next-year new conditions*: from a patient's embedding at the end of 2021, predict which conditions they are diagnosed with for the first time in 2022. Only 12 conditions are common enough to score, mostly short acute ones (most frequent: Viral sinusitis, Gingivitis, Gingival disease). The micro AUROC pools all conditions, and the conditions' base rates alone already reach 0.760 on it; the macro AUROC (mean over conditions) measures how well patients are ranked within a condition, where 0.5 is guessing.
- *Neighbour concordance*: the share of chronic conditions a patient has in common with their 5 nearest neighbours in the embedding (Jaccard; random patients: 10.3%, input features: 50.3%).
- *`any_onset` probe*: the Phase 7 prediction task (a new chronic condition in 2023-2025), by logistic regression on the embedding plus age and sex, cross-validated on Phase 7's folds; compared with Phase 7's model 2 (the 2022 features plus age and sex).

## Did the collapse go away?

- The GraphAE's final embedding puts 64.7% of its variance on the first principal component (supervised h: 99.8%; the GraphAE across the three tuning seeds: 42.9%-64.7%). It is no longer collapsed onto one line, but one direction still dominates; PCA spreads its variance far more evenly (8.1%).
- Its nearest neighbours share 35.8% of their chronic conditions, against 14.3% for the supervised h: a clear improvement. It is still below PCA (39.4%), the spectral embedding (43.1%) and the input features (50.3%).

## The questions

**Does the graph help?** (TemporalGraphAE (i) vs GraphAE, no graph (vi))

*In short, the GraphAE is better on neighbour concordance, `any_onset` probe; not shown different on next-year micro AUROC, next-year macro AUROC.*

  - Next-year micro AUROC: **Not shown** (the interval includes 0; the point estimate favours GraphAE, no graph (vi)): micro AUROC 0.755 vs 0.766, difference -0.011 (95% CI -0.041 to +0.019).
  - Next-year macro AUROC: **Not shown** (the interval includes 0; the point estimate favours GraphAE, no graph (vi)): macro AUROC 0.571 vs 0.624, difference -0.054 (95% CI -0.129 to +0.036).
  - Neighbour concordance: **Yes**, the GraphAE is better: mean Jaccard 0.358 vs 0.318, difference +0.040 (95% CI +0.020 to +0.060).
  - `any_onset` probe: **Yes**, the GraphAE is better: pooled AUROC 0.581 vs 0.531, difference +0.050 (95% CI +0.009 to +0.085).
  - Caution on the probe: in 1 of 5 folds the calibration of GraphAE, no graph (vi) has a slope <= 0 and reverses that fold's ranking, which pulls its pooled AUROC down. Uncalibrated, the mean within-fold AUROC is 0.586 for the GraphAE and 0.539 for GraphAE, no graph (vi).

**Is the real graph better than a random one?** (TemporalGraphAE (i) vs GraphAE, rewired graphs (vii))

*In short, the GraphAE is better on neighbour concordance, `any_onset` probe; not shown different on next-year micro AUROC, next-year macro AUROC.*

  - Next-year micro AUROC: **Not shown** (the interval includes 0; the point estimate favours GraphAE, rewired graphs (vii)): micro AUROC 0.755 vs 0.769, difference -0.014 (95% CI -0.044 to +0.017).
  - Next-year macro AUROC: **Not shown** (the interval includes 0; the point estimate favours the GraphAE): macro AUROC 0.571 vs 0.557, difference +0.014 (95% CI -0.091 to +0.121).
  - Neighbour concordance: **Yes**, the GraphAE is better: mean Jaccard 0.358 vs 0.123, difference +0.235 (95% CI +0.213 to +0.256).
  - `any_onset` probe: **Yes**, the GraphAE is better: pooled AUROC 0.581 vs 0.468, difference +0.113 (95% CI +0.073 to +0.154).
  - Caution on the probe: in 3 of 5 folds the calibration of GraphAE, rewired graphs (vii) has a slope <= 0 and reverses that fold's ranking, which pulls its pooled AUROC down. Uncalibrated, the mean within-fold AUROC is 0.586 for the GraphAE and 0.546 for GraphAE, rewired graphs (vii).

**Does the temporal (next-window) objective help?** (TemporalGraphAE (i) vs GraphAE, link only (viii))

*In short, the GraphAE is worse on neighbour concordance; not shown different on next-year micro AUROC, next-year macro AUROC, `any_onset` probe.*

  - Next-year micro AUROC: **Not shown** (the interval includes 0; the point estimate favours the GraphAE): micro AUROC 0.755 vs 0.742, difference +0.014 (95% CI -0.008 to +0.034).
  - Next-year macro AUROC: **Not shown** (the interval includes 0; the point estimate favours GraphAE, link only (viii)): macro AUROC 0.571 vs 0.574, difference -0.004 (95% CI -0.047 to +0.084).
  - Neighbour concordance: **No, the reverse**: GraphAE, link only (viii) is better: mean Jaccard 0.358 vs 0.403, difference -0.046 (95% CI -0.062 to -0.030).
  - `any_onset` probe: **Not shown** (the interval includes 0; the point estimate favours GraphAE, link only (viii)): pooled AUROC 0.581 vs 0.586, difference -0.005 (95% CI -0.036 to +0.025).

**Does the GraphAE beat PCA of the final window?** (TemporalGraphAE (i) vs PCA, final window (ii))

*In short, the GraphAE is worse on neighbour concordance; not shown different on next-year micro AUROC, next-year macro AUROC, `any_onset` probe.*

  - Next-year micro AUROC: **Not shown** (the interval includes 0; the point estimate favours the GraphAE): micro AUROC 0.755 vs 0.752, difference +0.003 (95% CI -0.023 to +0.028).
  - Next-year macro AUROC: **Not shown** (the interval includes 0; the point estimate favours PCA, final window (ii)): macro AUROC 0.571 vs 0.624, difference -0.054 (95% CI -0.108 to +0.047).
  - Neighbour concordance: **No, the reverse**: PCA, final window (ii) is better: mean Jaccard 0.358 vs 0.394, difference -0.036 (95% CI -0.053 to -0.019).
  - `any_onset` probe: **Not shown** (the interval includes 0; the point estimate favours the GraphAE): pooled AUROC 0.581 vs 0.575, difference +0.006 (95% CI -0.029 to +0.038).

**Does the GraphAE beat PCA of every window?** (TemporalGraphAE (i) vs PCA, every window (iii))

*In short, the GraphAE is worse on neighbour concordance; not shown different on next-year micro AUROC, next-year macro AUROC, `any_onset` probe.*

  - Next-year micro AUROC: **Not shown** (the interval includes 0; the point estimate favours the GraphAE): micro AUROC 0.755 vs 0.727, difference +0.028 (95% CI -0.001 to +0.057).
  - Next-year macro AUROC: **Not shown** (the interval includes 0; the point estimate favours the GraphAE): macro AUROC 0.571 vs 0.571, difference +0.000 (95% CI -0.054 to +0.088).
  - Neighbour concordance: **No, the reverse**: PCA, every window (iii) is better: mean Jaccard 0.358 vs 0.414, difference -0.057 (95% CI -0.077 to -0.037).
  - `any_onset` probe: **Not shown** (the interval includes 0; the point estimate favours the GraphAE): pooled AUROC 0.581 vs 0.558, difference +0.023 (95% CI -0.011 to +0.053).

**Does the GraphAE beat the spectral embedding?** (TemporalGraphAE (i) vs Spectral embedding (iv))

*In short, the GraphAE is better on next-year micro AUROC; worse on neighbour concordance; not shown different on next-year macro AUROC, `any_onset` probe.*

  - Next-year micro AUROC: **Yes**, the GraphAE is better: micro AUROC 0.755 vs 0.713, difference +0.042 (95% CI +0.016 to +0.067).
  - Next-year macro AUROC: **Not shown** (the interval includes 0; the point estimate favours the GraphAE): macro AUROC 0.571 vs 0.548, difference +0.023 (95% CI -0.042 to +0.117).
  - Neighbour concordance: **No, the reverse**: Spectral embedding (iv) is better: mean Jaccard 0.358 vs 0.431, difference -0.074 (95% CI -0.095 to -0.055).
  - `any_onset` probe: **Not shown** (the interval includes 0; the point estimate favours the GraphAE): pooled AUROC 0.581 vs 0.562, difference +0.019 (95% CI -0.010 to +0.046).

**Does the GraphAE beat the supervised TemporalGCN's h?** (TemporalGraphAE (i) vs Supervised TemporalGCN h (v))

*In short, the GraphAE is better on neighbour concordance; worse on `any_onset` probe; not shown different on next-year micro AUROC, next-year macro AUROC.*

  - Next-year micro AUROC: **Not shown** (the interval includes 0; the point estimate favours the GraphAE): micro AUROC 0.755 vs 0.753, difference +0.002 (95% CI -0.032 to +0.037).
  - Next-year macro AUROC: **Not shown** (the interval includes 0; the point estimate favours the GraphAE): macro AUROC 0.571 vs 0.503, difference +0.068 (95% CI -0.037 to +0.165).
  - Neighbour concordance: **Yes**, the GraphAE is better: mean Jaccard 0.358 vs 0.143, difference +0.215 (95% CI +0.189 to +0.240).
  - `any_onset` probe: **No, the reverse**: Supervised TemporalGCN h (v) is better: pooled AUROC 0.581 vs 0.691, difference -0.110 (95% CI -0.142 to -0.080).

## Overall

- **Next-year prediction is weak for every embedding.** Within a condition (macro AUROC) the embeddings rank patients at 0.503-0.624; 2 of 8 have an interval above 0.5 (PCA, final window (ii) and GraphAE, no graph (vi)), and the GraphAE's is 0.499 to 0.643. Pooled over conditions (micro AUROC, 0.713-0.769), 2 of 8 score above the 0.760 that the conditions' base rates give with no patient information; the GraphAE minus base rates is -0.005 (95% CI -0.035 to +0.026). Across the tuning seeds the GraphAE's micro AUROC ranges 0.741-0.755, a spread the intervals do not include.
- **`any_onset` probe:** the GraphAE + age, sex reaches 0.581; minus Phase 7 model 2: -0.004 (95% CI -0.035 to +0.027): not distinguishable from model 2.

## Caveats

- **Next-window weight 2** was chosen on validation next-window loss; that criterion favours whichever setting weights the next-window term most, so it says nothing about the link term's value. No other hyperparameter was searched (the Phase 6 TemporalGCN setting is reused).
- **Link-only (viii) has an untrained GRU.** With the next-window gradient cut, nothing trains the recurrent part: its final state is a fixed random recurrent map of the link-trained per-year embeddings. A linear readout trained alongside (on detached states) only picks its epoch.
- **No graph (vi) also has no link term** (there are no edges to reconstruct), so i vs vi tests the graph as a whole, message passing and link supervision together.
- **Probe calibration.** The probe follows Phase 7's protocol (Platt scaling on each fold's inner validation patients, then pooled). For embeddings with almost no signal the fitted slope can be negative and reverse a fold's ranking, which can push the pooled AUROC below 0.5; the uncalibrated within-fold AUROC is reported next to it.
- **Transductive self-supervision.** Test patients' features and edges, including their 2022 graph, enter message passing and the link term, so the shared weights have seen their 2022 feature rows (never as next-window targets). The 2021 embeddings used for the next-year task read no 2022 data for any patient, but the encoder weights were shaped partly by the 2022 graph. The IDF weights and the transition SVD of the features are fitted on 2017-2022 windows (population statistics, Phase 3).
- **The supervised h (v)** was trained on training-split `any_onset` labels; its probe AUROC is optimistic (those patients are in the probe's held-out folds).
- **The spectral embedding (iv)** is scored against phenotypes derived from the same graph's eigenvectors, so its silhouette is circular.
- **What the intervals cover.** They resample test patients; they do not include retraining variation (seeds), which the seed check shows is not negligible.
- **Small numbers.** 122 test onsets over 12 conditions, 5 of them with 3 or fewer: per-condition AUROCs, and so the macro AUROC, are noisy.
