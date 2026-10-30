# Baselines (validation only)

Built by `scripts/06a_baselines.py`. Target `any_onset`, same train/validation split and masked labels as the TemporalGCN: every model is fitted on the 1,556 training patients with an observed label and scored on the 335 validation patients with one (63 positives, so a random score has AUPRC 0.188). Test labels are never loaded.

| Model | Inputs | Features | Val AUROC (95% CI) | Val AUPRC (95% CI) | Phase 6 GCN AUROC minus this (95% CI) | Train AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 2022 window | 236 | 0.650 (0.577 to 0.718) | 0.284 (0.215 to 0.396) | +0.029 (-0.035 to +0.092) | 0.728 |
| Logistic regression | 6 windows 2017-2022, concatenated | 1,416 | 0.646 (0.569 to 0.721) | 0.287 (0.210 to 0.395) | +0.033 (-0.036 to +0.100) | 0.855 |
| Logistic regression | 2022 window + age, sex | 238 | 0.661 (0.588 to 0.727) | 0.287 (0.213 to 0.395) | +0.018 (-0.047 to +0.082) | 0.729 |
| Logistic regression | 6 windows 2017-2022, concatenated + age, sex | 1,418 | 0.648 (0.571 to 0.723) | 0.287 (0.209 to 0.395) | +0.031 (-0.036 to +0.096) | 0.855 |
| TemporalGCN, Phase 6 setting | 6 windows + graphs | 236 per window | 0.679 (0.607 to 0.743) | 0.266 (0.202 to 0.349) | reference | 0.806 |
| TemporalGCN, Phase 6 setting | 6 windows + graphs, age and sex in the heads | 236 per window | 0.662 (0.590 to 0.725) | 0.255 (0.195 to 0.336) | +0.017 (-0.017 to +0.049) | 0.798 |

## Notes

- **Logistic regression:** scikit-learn, L2 penalty with C = 1.0, lbfgs, class weights balanced; inputs as stored (each feature block is already row-L2-normalized), age standardized with the training split's mean and sd, sex 0/1. Not tuned.
- **The 2022 window is cumulative:** its TF-IDF block is a time-decayed sum over windows 2017-2022 and its transition block holds every transition before the cutoff, so "2022 only" already summarizes earlier years.
- **Reference GCN:** the Phase 6 setting (hidden 64, embedding 32, dropout 0.5, weight decay 0.0005, no smoothness loss, no DropEdge) retrained with seed 42, without demographics (as in Phase 6) and with age and sex in the heads.
- **Intervals:** percentile bootstrap over the validation patients, 1,000 resamples drawn with seed 42; every model is scored on the same resamples, so the AUROC differences are paired. An interval that excludes 0 marks a difference larger than the resampling noise of this validation set; it is not a test-set result.
