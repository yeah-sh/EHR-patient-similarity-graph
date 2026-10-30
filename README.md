# EHR comorbidity trajectory embeddings

Electronic health record networks: time-series embedding of comorbidity trajectories via
linear-algebraic patient-similarity graphs.

Each patient's yearly comorbidity history (Synthea synthetic EHR, SNOMED-CT conditions) becomes
a sequence of feature vectors. For every year a cosine k-NN patient-similarity graph is built and
normalized, and a temporal graph convolutional network (a shared GCN per year, then a GRU over the
years) learns an embedding per patient and year plus one trajectory embedding per patient. These
embeddings are trained to predict new onsets of chronic condition groups.

## Running the pipeline

Use the `ehrgraph` conda environment. Every parameter is in `config/config.yaml`, and the seed is 42.

```
conda run -n ehrgraph python scripts/01_prepare_data.py      # events, code dictionary, data reports
conda run -n ehrgraph python scripts/02_build_trajectories.py # yearly windows 2017-2025
conda run -n ehrgraph python scripts/03_build_features.py     # TF-IDF + transition features, model cohort
conda run -n ehrgraph python scripts/04_build_graphs.py       # k-NN graphs, A_hat, spectra, phenotypes
conda run -n ehrgraph python scripts/05_build_labels.py       # onset labels and the train/val/test split
conda run -n ehrgraph python scripts/06a_baselines.py         # logistic-regression baselines (validation)
conda run -n ehrgraph python scripts/06b_tune.py              # hyperparameter search (validation)
conda run -n ehrgraph python scripts/06_train.py [--quick]    # temporal GCN, embeddings, calibration
conda run -n ehrgraph python -m pytest                        # tests
```

Each step writes a report in `reports/`. `06b_tune.py` reports the chosen setting; copy it into
the `model` section of `config/config.yaml` before running `06_train.py`. Steps 06a and 06b use
validation labels only.

## Prediction setup and leakage

- **Inputs:** the yearly windows that close by the cutoff 2023-01-01, which are 2017-2022 (read only through
  `features.load_model_inputs`).
- **Labels:** new onsets in the horizon 2023-01-01 to 2026-01-01. The primary target `any_onset`
  pools 9 onset groups; the per-group onsets are secondary targets. Labels are 1, 0 or masked;
  masked labels never count.

## Training is transductive

`scripts/06_train.py` trains full batch on one graph per window that holds **all 2,581
model-cohort patients**:

- **Message passing uses every node.** Training, validation and test patients, and patients
  whose labels are masked, all pass messages through the graph with their pre-cutoff features.
  The model therefore sees the features (never the labels) of validation and test patients
  during training. Its numbers describe this transductive setting: scoring a new patient
  means placing them in the graph.
- **The loss uses only the observed labels of training patients.** Validation labels are used
  only to choose the epoch (early stopping on validation AUROC of `any_onset`) and to report
  metrics.
- **Test labels are not loaded during training.** `train.load_training_data` sets them to NaN.
- **Demographics enter the heads only.** With `model.use_demographics`, age on the cutoff date
  (standardized with training-split statistics) and sex are concatenated to the trajectory
  embedding before the prediction heads. They never enter the graphs or the node features.
- **Calibration.** `06_train.py` fits a temperature on the validation logits of `any_onset`
  (`artifacts/temperature.json`); calibrated probability = `sigmoid(logit / T)`.

Use `--quick` for a smoke test: 5 epochs on 300 patients and the subgraph they induce, written
to `artifacts/quick/`. Results are in `reports/training_summary.md`.
