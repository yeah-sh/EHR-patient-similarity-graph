# Pipeline run from scratch

`scripts/run_all.py`, started 2026-10-31 21:50:52, deleted `data/processed/` and `artifacts/` first. Python 3.12.16 on Windows 11 (AMD64). Wall-clock time per step, each in a fresh process.

| Phase | Script | Produces | Seconds | m:ss | Share |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `01_prepare_data.py` | events, code dictionary, data reports | 4.5 | 0:04 | 0.5% |
| 2 | `02_build_trajectories.py` | yearly windows and transitions | 2.4 | 0:02 | 0.3% |
| 3 | `03_build_features.py` | TF-IDF + transition features, comorbidity, cohort | 5.4 | 0:05 | 0.6% |
| 4 | `04_build_graphs.py` | k-NN graphs, A_hat, spectrum, phenotypes | 5.9 | 0:06 | 0.6% |
| 5 | `05_build_labels.py` | onset labels and the train/validation/test split | 3.4 | 0:03 | 0.4% |
| 6a | `06a_baselines.py` | validation baselines that 06_train.py compares with | 38.1 | 0:38 | 4.1% |
| 6 | `06_train.py` | final TemporalGCN and its Platt calibration | 33.8 | 0:34 | 3.6% |
| 7 | `07_evaluate.py` | cross-validation, test set, embedding analysis | 493.1 | 8:13 | 53.1% |
| 7b | `07b_embeddings.py` | TemporalGraphAE and the embedding comparison | 298.5 | 4:58 | 32.1% |
| 9 | `09_app_assets.py` | assets of the Streamlit app | 43.6 | 0:44 | 4.7% |
|  | **Total** |  | 928.8 | 15:29 | 100.0% |

All 10 steps finished.
