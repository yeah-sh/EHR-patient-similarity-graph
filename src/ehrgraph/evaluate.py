"""Evaluation of ``any_onset``: baselines and calibration (Phase 6), and Phase 7.

Phase 6 works on ``train.TrainingData``, whose test labels are NaN on loading, and
scores the validation split only.

Baselines (run end to end by ``scripts/06a_baselines.py``)::

    data = load_training_data(cfg)
    results, val_scores = run_baselines(data, cfg)
    save_baselines(results, val_scores, cfg)
    write_baselines_summary(results, data, cfg)

Calibration of the final model (``scripts/06_train.py``, after ``train.fit``)::

    logits, targets, _ = validation_logits(model, data)
    calibration = calibrate(logits, targets, cfg.calibration.method, cfg.evaluation, cfg.seed)
    save_calibration(calibration, outputs.calibration_json)
    plot_reliability(calibration, outputs.reliability_fig, cfg)

Both calibration methods are a ``Calibrator``, ``sigmoid(slope * logit + intercept)``:
Platt scaling fits the slope and the intercept, temperature scaling only the slope
``1 / T``.

Phase 7 (run end to end by ``scripts/07_evaluate.py``), every hyperparameter fixed::

    model_data = load_model_data(cfg)
    cv = run_cross_validation(model_data, load_spectral_embedding(cfg, n), cfg)
    table, groups = cv_results(cv, cfg), cv_group_results(cv, cfg)
    test = evaluate_test(...)  # the final model of 06_train.py, scored once
    analysis = analyse_embeddings(...)
    write_results(...)
    write_evaluation_summary(...)

Test labels are read in two places only: ``held_out_targets``, for the single-shot
test evaluation of the final model, and ``run_cross_validation``, which by design
uses every patient with an observed label; there a fold's labels are hidden while
its models are fitted and read only to score it.

Confidence intervals are percentile bootstrap intervals over the patients with an
observed label. Models scored on the same patients use the same resamples, drawn
with the global seed, so the differences between them are paired.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
import torch
import yaml
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    davies_bouldin_score,
    precision_recall_curve,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ehrgraph.config import (
    BaselinesConfig,
    Config,
    EvaluationConfig,
    ModelConfig,
    feature_window_years,
    reference_model_config,
)
from ehrgraph.data import chart_style, markdown_table, save_figure
from ehrgraph.features import chronic_mask, chronic_presence, load_model_cohort
from ehrgraph.graph import cluster_phenotypes, normalize_adjacency
from ehrgraph.labels import (
    ANY_ONSET,
    ONSET_PREFIX,
    PHENOTYPE,
    SPLIT,
    split_strata,
    stratified_split,
)
from ehrgraph.models import OnsetModel
from ehrgraph.train import (
    TEST,
    TRAIN,
    VAL,
    ModelData,
    TrainingData,
    binary_metrics,
    fit,
    make_training_data,
    predict,
    split_targets,
)
from ehrgraph.trajectories import load_trajectories

logger = logging.getLogger(__name__)

# The Phase 6 TemporalGCN retrained (baselines.reference_model): the baselines' reference.
REFERENCE = "gcn_phase6"
COMPARISON_COLUMNS: list[str] = [
    "key", "val_observed", "val_positives", "val_auroc", "val_auroc_low", "val_auroc_high",
    "val_auprc", "val_auprc_low", "val_auprc_high", "delta_auroc", "delta_auroc_low",
    "delta_auroc_high",
]
BASELINE_COLUMNS: list[str] = [
    "key", "model", "inputs", "n_features", *COMPARISON_COLUMNS[1:], "train_auroc",
]
RELIABILITY_COLUMNS: list[str] = ["bin", "n", "mean_predicted", "observed_rate"]

# Phase 7 model kinds, the reference model, and the split name of patients whose
# any_onset is masked (graph nodes without a label in any fold).
LOGISTIC, GCN = "logistic", "gcn"
FULL_MODEL = "tgcn"
DEMOGRAPHICS_MODEL = "age_sex"
UNLABELED = "unlabeled"
RESULT_COLUMNS: list[str] = [
    "number", "key", "model", "inputs", "graph", "windows",
    "fold_auroc_mean", "fold_auroc_sd", "fold_auprc_mean", "fold_auprc_sd",
    "oof_observed", "oof_positives", "oof_auroc", "oof_auroc_low", "oof_auroc_high",
    "oof_auprc", "oof_auprc_low", "oof_auprc_high",
    "delta_auroc", "delta_auroc_low", "delta_auroc_high", "folds_full_better",
    "oof_brier", "oof_ece", "oof_mean_predicted",
    "age_sex_rank_corr_mean", "age_sex_rank_corr_min", "age_sex_rank_corr_max",
]
FOLD_COLUMNS: list[str] = ["fold", "key", "observed", "positives", "auroc", "auprc"]
GROUP_RESULT_COLUMNS: list[str] = [
    "group", "key", "observed", "positives", "auroc", "auroc_low", "auroc_high",
]
FIT_COLUMNS: list[str] = [
    "fold", "key", "seed", "best_epoch", "epochs_run", "inner_val_auroc", "slope", "intercept",
    "inner_train", "inner_val", "held_out", "held_out_positives",
]
# Test-set metrics with bootstrap CIs (evaluate_test), in report order.
TEST_METRICS: tuple[str, ...] = ("auroc", "auprc", "brier", "ece", "f1", "precision", "recall")
# (question, model compared with the full TemporalGCN), for the ablation table.
ABLATIONS: tuple[tuple[str, str], ...] = (
    ("Does the graph help?", "temporal_no_graph"),
    ("Does time help?", "gcn_static"),
    ("Does the real similarity structure matter?", "tgcn_rewired"),
)

# Chart tokens from the dataviz reference palette (as in train.py): categorical slots 1
# and 2 (before and after calibration; a path and its onset years), secondary ink for
# annotations, baseline gray for context points.
_SERIES_COLOR = "#2a78d6"
_ACCENT_COLOR = "#eb6834"
_BEFORE_COLOR = _SERIES_COLOR
_AFTER_COLOR = _ACCENT_COLOR
_SURFACE = "#fcfcfb"
_TEXT_SECONDARY = "#52514e"
_BASELINE = "#c3c2b7"
_CONTEXT_POINT = "#e1e0d9"  # every other point of a UMAP panel, one step off the surface
_POINTS_PER_INCH = 72
_FIG_SIZE_IN = (6.4, 6.0)
_LINE_PX = 2
_MARKER_PX = 10
_RING_PX = 2  # surface ring around each marker, so overlapping markers stay distinct
_LABEL_OFFSET_PT = 6
_CHANCE_AUROC = 0.5
_DOT_AREA_PT2 = 3.0  # UMAP points: dense scatters need small dots
_PANEL_IN = 3.4  # width and height of one small-multiple panel
_ROW_IN = 0.42  # height of one model row in the cross-validation figure
_ZOOM_MIN_SHARE = 0.3  # a trajectory panel shows at least this share of the UMAP's range
_SEMANTIC_TAG = re.compile(r"\s*\([^()]*\)$")  # " (disorder)", dropped from condition names


@dataclass(frozen=True)
class BaselineSpec:
    """One logistic-regression baseline.

    Attributes:
        key: Column name of its scores and row key of its results.
        windows: "last" (the last training window) or "all" (every window, concatenated).
        demographics: Whether age and sex (``TrainingData.demographics``) are added.
    """

    key: str
    windows: str
    demographics: bool


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec("lr_last", "last", False),
    BaselineSpec("lr_all", "all", False),
    BaselineSpec("lr_last_demo", "last", True),
    BaselineSpec("lr_all_demo", "all", True),
)


@dataclass(frozen=True)
class BootstrapSamples:
    """AUROC and AUPRC of each model on the same bootstrap resamples.

    Attributes:
        auroc: Model key -> AUROC per resample used.
        auprc: Model key -> AUPRC per resample used.
        drawn: Resamples drawn.
        used: Resamples with both classes (the others are skipped for every model).
    """

    auroc: dict[str, np.ndarray]
    auprc: dict[str, np.ndarray]
    drawn: int
    used: int


@dataclass(frozen=True)
class Calibrator:
    """A fitted map from logits to probabilities: ``sigmoid(slope * logit + intercept)``.

    Attributes:
        method: "platt" (slope and intercept fitted) or "temperature" (``slope = 1 / T``,
            intercept 0).
        slope: Multiplies the logit; a positive slope keeps the ranking.
        intercept: Added after the slope.
    """

    method: str
    slope: float
    intercept: float

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Return the calibrated probabilities of ``logits``."""
        return sigmoid(self.slope * np.asarray(logits, dtype=np.float64) + self.intercept)

    def describe(self) -> str:
        """Short description of the fitted parameters, e.g. "a = 0.92, b = -1.43"."""
        if self.method == "temperature":
            return f"T = {1.0 / self.slope:.3f}"
        return f"a = {self.slope:.3f}, b = {self.intercept:+.3f}"


@dataclass(frozen=True)
class Calibration:
    """Calibration of the ``any_onset`` logits on the validation patients.

    Attributes:
        calibrator: The map fitted on every validation patient with a label.
        observed: Validation patients with an observed label.
        positives: Of them, positives.
        before: ``probability_metrics`` of ``sigmoid(logit)``.
        after: ``probability_metrics`` of ``calibrator.apply(logit)``: fitted and
            scored on the same patients (in-sample).
        cross_fitted: ``probability_metrics`` with each patient's calibrator fitted on
            the other folds (out-of-sample).
        fold_calibrators: The calibrator of each fold.
        bins_before: ``reliability_bins`` before calibration.
        bins_after: ``reliability_bins`` after.
    """

    calibrator: Calibrator
    observed: int
    positives: int
    before: dict[str, float]
    after: dict[str, float]
    cross_fitted: dict[str, float]
    fold_calibrators: tuple[Calibrator, ...]
    bins_before: pd.DataFrame
    bins_after: pd.DataFrame


# --- Bootstrap -------------------------------------------------------------------------


def bootstrap_metrics(
    targets: np.ndarray, scores: Mapping[str, np.ndarray], n_resamples: int, seed: int
) -> BootstrapSamples:
    """Resample patients with replacement and score every model on each resample.

    Args:
        targets: [n] observed labels, 1.0 or 0.0 (no NaN).
        scores: Model key -> [n] scores of the same patients.
        n_resamples: Resamples to draw.
        seed: Random seed of the resamples.

    Returns:
        The per-resample metrics.

    Raises:
        ValueError: If a target is NaN or a score array has the wrong length.
    """
    y = np.asarray(targets, dtype=float)
    if np.isnan(y).any():
        raise ValueError("bootstrap_metrics takes observed targets only (no NaN)")
    bad = [key for key, s in scores.items() if len(s) != len(y)]
    if bad:
        raise ValueError(f"scores of {bad} do not match the {len(y)} targets")
    draws = np.random.default_rng(seed).integers(0, len(y), size=(n_resamples, len(y)))
    positives = y[draws].sum(axis=1)
    draws = draws[(positives > 0) & (positives < len(y))]
    auroc = {
        key: np.array([roc_auc_score(y[i], s[i]) for i in draws]) for key, s in scores.items()
    }
    auprc = {
        key: np.array([average_precision_score(y[i], s[i]) for i in draws])
        for key, s in scores.items()
    }
    if len(draws) < n_resamples:
        logger.warning("%d of %d bootstrap resamples had one class and were skipped",
                       n_resamples - len(draws), n_resamples)
    return BootstrapSamples(auroc, auprc, n_resamples, len(draws))


def percentile_interval(values: np.ndarray, level: float) -> tuple[float, float]:
    """Return the central ``level`` percentile interval of ``values``."""
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def compare_models(
    targets: np.ndarray,
    scores: Mapping[str, np.ndarray],
    reference: str,
    evaluation: EvaluationConfig,
    seed: int,
) -> pd.DataFrame:
    """Score models on the same patients, with bootstrap CIs and paired differences.

    ``delta_auroc`` is the AUROC of ``reference`` minus that of the model (positive:
    the reference ranks better), with a CI from the same resamples; NaN for the
    reference itself.

    Args:
        targets: [n] observed labels (no NaN).
        scores: Model key -> [n] scores; must include ``reference``.
        reference: Key of the model the others are compared with.
        evaluation: The ``evaluation`` section (resamples, CI level).
        seed: Random seed of the resamples.

    Returns:
        ``COMPARISON_COLUMNS``, one row per model in ``scores`` order.
    """
    if reference not in scores:
        raise ValueError(f"the reference {reference!r} has no scores")
    y = np.asarray(targets, dtype=float)
    boot = bootstrap_metrics(y, scores, evaluation.bootstrap_resamples, seed)
    level = evaluation.ci_level
    rows = []
    for key, s in scores.items():
        point = binary_metrics(y, np.asarray(s, dtype=float))
        delta = float(roc_auc_score(y, scores[reference]) - point["auroc"])
        delta_ci = percentile_interval(boot.auroc[reference] - boot.auroc[key], level)
        is_reference = key == reference
        rows.append({
            "key": key,
            "val_observed": point["observed"],
            "val_positives": point["positives"],
            "val_auroc": point["auroc"],
            **dict(zip(("val_auroc_low", "val_auroc_high"),
                       percentile_interval(boot.auroc[key], level))),
            "val_auprc": point["auprc"],
            **dict(zip(("val_auprc_low", "val_auprc_high"),
                       percentile_interval(boot.auprc[key], level))),
            "delta_auroc": np.nan if is_reference else delta,
            "delta_auroc_low": np.nan if is_reference else delta_ci[0],
            "delta_auroc_high": np.nan if is_reference else delta_ci[1],
        })
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


# --- Baselines -------------------------------------------------------------------------


def baseline_inputs(data: TrainingData, windows: str, demographics: bool) -> np.ndarray:
    """Return the logistic-regression inputs of every node.

    Args:
        data: Training data.
        windows: "last": the last training window's features; "all": every
            window's features, oldest first, concatenated.
        demographics: Append ``data.demographics`` (standardized age, sex).

    Returns:
        Dense float64 [N, F]; N x (windows x features), never N x N.

    Raises:
        ValueError: For another ``windows``.
    """
    if windows not in ("last", "all"):
        raise ValueError(f"windows must be 'last' or 'all', got {windows!r}")
    blocks = [data.features[-1]] if windows == "last" else list(data.features)
    if demographics:
        blocks.append(data.demographics)
    return torch.cat(blocks, dim=1).numpy().astype(np.float64)


def fit_logistic(X: np.ndarray, y: np.ndarray, settings: BaselinesConfig) -> LogisticRegression:
    """Fit the configured logistic regression on the rows of ``X`` whose label is observed.

    Args:
        X: [N, F] inputs.
        y: [N] labels: 1.0, 0.0 or NaN (not fitted).
        settings: The ``baselines`` section (C, max_iter, class weights).

    Returns:
        The fitted model.

    Raises:
        ValueError: If the observed labels hold a single class.
    """
    seen = ~np.isnan(y)
    model = LogisticRegression(
        C=settings.C,
        max_iter=settings.max_iter,
        class_weight="balanced" if settings.balanced_class_weight else None,
    )
    model.fit(X[seen], y[seen].astype(int))
    if int(np.max(model.n_iter_)) >= settings.max_iter:
        logger.warning("Logistic regression did not converge in %d iterations", settings.max_iter)
    return model


def logistic_scores(X: np.ndarray, data: TrainingData, cfg: Config) -> np.ndarray:
    """Fit a logistic regression of ``any_onset`` on the training split; score every node.

    Only training patients with an observed label are fitted; validation and test
    labels are not used.

    Args:
        X: [N, F] inputs (``baseline_inputs``).
        data: Training data.
        cfg: Project configuration (``baselines`` section).

    Returns:
        [N] predicted probabilities.
    """
    y = split_targets(data, TRAIN)[0].numpy()
    return fit_logistic(X, y, cfg.baselines).predict_proba(X)[:, 1]


def gcn_scores(data: TrainingData, model_cfg: ModelConfig, seed: int) -> np.ndarray:
    """Train a TemporalGCN with ``train.fit`` and return its ``any_onset`` probabilities [N]."""
    result = fit(data, model_cfg, seed, verbose=False)
    return torch.sigmoid(predict(result.model, data).any_onset_logits).numpy().astype(np.float64)


def run_baselines(data: TrainingData, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the ``BASELINES`` and the Phase 6 reference GCN on the validation split.

    The reference is ``config.reference_model_config`` retrained with ``cfg.seed``,
    without and with demographics (key ``REFERENCE`` and ``REFERENCE + "_demo"``).

    Args:
        data: Training data (test labels already NaN).
        cfg: Project configuration.

    Returns:
        ``BASELINE_COLUMNS`` (delta_auroc: reference AUROC minus the model's), and
        the validation patients with an observed label: patient_id, ``ANY_ONSET``
        and one score column per model key.
    """
    years, width = data.years, data.features[0].shape[1]
    window_names = {
        "last": (f"{years[-1]} window", width),
        "all": (f"{len(years)} windows {years[0]}-{years[-1]}, concatenated", len(years) * width),
    }
    n_demo = data.demographics.shape[1]
    meta: dict[str, tuple[str, str, int]] = {}
    scores: dict[str, np.ndarray] = {}
    for spec in BASELINES:
        name, n_features = window_names[spec.windows]
        if spec.demographics:
            name, n_features = f"{name} + age, sex", n_features + n_demo
        meta[spec.key] = ("Logistic regression", name, n_features)
        scores[spec.key] = logistic_scores(
            baseline_inputs(data, spec.windows, spec.demographics), data, cfg
        )
        logger.info("Baseline %s: %d inputs", spec.key, n_features)
    reference = reference_model_config(cfg)
    for key, demographics in ((REFERENCE, reference.use_demographics),
                              (REFERENCE + "_demo", True)):
        model_cfg = replace(reference, use_demographics=demographics)
        inputs = f"{len(years)} windows + graphs"
        if demographics:
            inputs += ", age and sex in the heads"
        meta[key] = ("TemporalGCN, Phase 6 setting", inputs, width)
        scores[key] = gcn_scores(data, model_cfg, cfg.seed)
        logger.info("Reference %s trained", key)

    val_y = split_targets(data, VAL)[0].numpy()
    train_y = split_targets(data, TRAIN)[0].numpy()
    seen = ~np.isnan(val_y)
    table = compare_models(
        val_y[seen], {key: s[seen] for key, s in scores.items()}, REFERENCE, cfg.evaluation,
        cfg.seed,
    )
    table.insert(1, "model", [meta[key][0] for key in table["key"]])
    table.insert(2, "inputs", [meta[key][1] for key in table["key"]])
    table.insert(3, "n_features", [meta[key][2] for key in table["key"]])
    table["train_auroc"] = [binary_metrics(train_y, scores[key])["auroc"] for key in table["key"]]
    val_scores = pd.DataFrame({
        "patient_id": np.asarray(data.patient_ids)[seen],
        ANY_ONSET: val_y[seen],
        **{key: s[seen] for key, s in scores.items()},
    })
    return table[BASELINE_COLUMNS], val_scores


def save_baselines(results: pd.DataFrame, val_scores: pd.DataFrame, cfg: Config) -> None:
    """Write the baseline table and the validation scores (paths from ``cfg.paths``)."""
    for path in (cfg.paths.baseline_results_csv, cfg.paths.baseline_val_scores_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(cfg.paths.baseline_results_csv, index=False)
    val_scores.to_csv(cfg.paths.baseline_val_scores_csv, index=False)
    logger.info("Wrote %s and %s", cfg.paths.baseline_results_csv,
                cfg.paths.baseline_val_scores_csv)


def load_baseline_scores(cfg: Config) -> pd.DataFrame | None:
    """Read the baselines' validation scores, or None if 06a_baselines.py has not run."""
    path = cfg.paths.baseline_val_scores_csv
    return pd.read_csv(path, dtype={"patient_id": str}) if path.exists() else None


def write_baselines_summary(results: pd.DataFrame, data: TrainingData, cfg: Config) -> str:
    """Write the baseline report to ``cfg.paths.baselines_summary_md``.

    Args:
        results: ``run_baselines`` table.
        data: Training data.
        cfg: Project configuration.

    Returns:
        The markdown written.
    """
    evaluation, settings = cfg.evaluation, cfg.baselines
    reference = cfg.baselines.reference_model
    first = results.iloc[0]
    train_y = split_targets(data, TRAIN)[0].numpy()
    rows = [
        [row.model, row.inputs,
         f"{row.n_features:,}" + (" per window" if row.key.startswith(REFERENCE) else ""),
         _with_ci(row.val_auroc, row.val_auroc_low, row.val_auroc_high),
         _with_ci(row.val_auprc, row.val_auprc_low, row.val_auprc_high),
         "reference" if row.key == REFERENCE
         else _with_ci(row.delta_auroc, row.delta_auroc_low, row.delta_auroc_high, signed=True),
         f"{row.train_auroc:.3f}"]
        for row in results.itertuples()
    ]
    level = f"{evaluation.ci_level:.0%}"
    lines = [
        "# Baselines (validation only)",
        "",
        f"Built by `scripts/06a_baselines.py`. Target `{ANY_ONSET}`, same train/validation "
        "split and masked labels as the TemporalGCN: every model is fitted on the "
        f"{int((~np.isnan(train_y)).sum()):,} training patients with an observed label and "
        f"scored on the {first.val_observed:,} validation patients with one "
        f"({first.val_positives:,} positives, so a random score has AUPRC "
        f"{first.val_positives / first.val_observed:.3f}). Test labels are never loaded.",
        "",
        markdown_table(
            ["Model", "Inputs", "Features", f"Val AUROC ({level} CI)",
             f"Val AUPRC ({level} CI)", f"Phase 6 GCN AUROC minus this ({level} CI)",
             "Train AUROC"],
            rows,
            "llrrrrr",
        ),
        "",
        "## Notes",
        "",
        f"- **Logistic regression:** scikit-learn, L2 penalty with C = {settings.C}, lbfgs, "
        f"class weights {'balanced' if settings.balanced_class_weight else 'equal'}; inputs "
        "as stored (each feature block is already row-L2-normalized), age standardized with "
        "the training split's mean and sd, sex 0/1. Not tuned.",
        f"- **The {data.years[-1]} window is cumulative:** its TF-IDF block is a time-decayed "
        f"sum over windows {data.years[0]}-{data.years[-1]} and its transition block holds "
        f"every transition before the cutoff, so \"{data.years[-1]} only\" already summarizes "
        "earlier years.",
        f"- **Reference GCN:** the Phase 6 setting (hidden {reference.hidden_dim}, embedding "
        f"{reference.embed_dim}, dropout {reference.dropout}, weight decay "
        f"{reference.weight_decay}, no smoothness loss, no DropEdge) retrained with seed "
        f"{cfg.seed}, without demographics (as in Phase 6) and with age and sex in the heads.",
        f"- **Intervals:** percentile bootstrap over the validation patients, "
        f"{evaluation.bootstrap_resamples:,} resamples drawn with seed {cfg.seed}; every model "
        "is scored on the same resamples, so the AUROC differences are paired. An interval "
        "that excludes 0 marks a difference larger than the resampling noise of this "
        "validation set; it is not a test-set result.",
        "",
    ]
    text = "\n".join(lines)
    path = cfg.paths.baselines_summary_md
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)
    return text


def comparison_section(
    table: pd.DataFrame,
    labels: Mapping[str, tuple[str, str]],
    final: str,
    evaluation: EvaluationConfig,
) -> list[str]:
    """Markdown comparing the final model with the baselines (``compare_models`` table).

    Args:
        table: ``compare_models`` output with ``final`` as the reference.
        labels: Model key -> (model, inputs) for display.
        final: Key of the final model.
        evaluation: The ``evaluation`` section.

    Returns:
        Lines to pass as ``train.write_training_summary(extra_sections=...)``.
    """
    level = f"{evaluation.ci_level:.0%}"
    ordered = pd.concat([table[table["key"] == final], table[table["key"] != final]])
    rows = [
        [f"**{labels[row.key][0]}**" if row.key == final else labels[row.key][0],
         labels[row.key][1],
         _with_ci(row.val_auroc, row.val_auroc_low, row.val_auroc_high),
         _with_ci(row.val_auprc, row.val_auprc_low, row.val_auprc_high),
         "" if row.key == final
         else _with_ci(row.delta_auroc, row.delta_auroc_low, row.delta_auroc_high, signed=True)]
        for row in ordered.itertuples()
    ]
    return [
        "## Comparison with the baselines (validation)",
        "",
        f"Same {int(table.iloc[0].val_observed):,} validation patients for every model; "
        f"percentile bootstrap, {evaluation.bootstrap_resamples:,} resamples shared by all "
        "models, so the differences are paired. Baselines from "
        "`scripts/06a_baselines.py` (see `baselines_summary.md`).",
        "",
        markdown_table(
            ["Model", "Inputs", f"Val AUROC ({level} CI)", f"Val AUPRC ({level} CI)",
             f"Final model AUROC minus this ({level} CI)"],
            rows,
            "llrrr",
        ),
        "",
    ]


# --- Calibration -----------------------------------------------------------------------


def validation_logits(
    model: OnsetModel, data: TrainingData
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ``any_onset`` logits, labels and IDs of the observed validation patients."""
    y = split_targets(data, VAL)[0].numpy()
    seen = ~np.isnan(y)
    logits = predict(model, data).any_onset_logits.numpy().astype(np.float64)
    return logits[seen], y[seen].astype(np.float64), np.asarray(data.patient_ids)[seen]


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return np.exp(-np.logaddexp(0.0, -logits))


def logit_nll(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean binary cross-entropy of ``sigmoid(logits)`` (unweighted), computed stably."""
    return float(np.mean(np.logaddexp(0.0, logits) - targets * logits))


def fit_temperature(
    logits: np.ndarray, targets: np.ndarray, bounds: tuple[float, float]
) -> float:
    """Return the T in ``bounds`` minimizing the NLL of ``sigmoid(logits / T)``.

    Bounded scalar search on log T. A fit at a bound is logged: logits that barely
    rank the labels can drive T towards infinity (every probability 0.5).

    Args:
        logits: [n] logits.
        targets: [n] labels, 1.0 or 0.0.
        bounds: (lowest, highest) temperature (``evaluation.temperature_bounds``).

    Returns:
        The temperature.
    """
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    log_bounds = (float(np.log(bounds[0])), float(np.log(bounds[1])))
    result = minimize_scalar(
        lambda log_t: logit_nll(z * np.exp(-log_t), y), bounds=log_bounds, method="bounded"
    )
    temperature = float(np.exp(result.x))
    if np.isclose(result.x, log_bounds, atol=1e-3).any():
        logger.warning("Temperature %.4g is at the search bound %s", temperature, bounds)
    return temperature


def fit_platt(logits: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Platt scaling: the (slope, intercept) minimizing the NLL of ``sigmoid(a z + b)``.

    A logistic regression of the label on the logit, with an intercept, fitted by
    unpenalized maximum likelihood (BFGS with the exact gradient, from (1, 0)).
    Logits that separate the labels perfectly have no finite optimum; the fit then
    stops at the iteration limit, which is logged.

    Args:
        logits: [n] logits.
        targets: [n] labels, 1.0 or 0.0.

    Returns:
        The slope and the intercept.
    """
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(targets, dtype=np.float64)

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        s = params[0] * z + params[1]
        residual = sigmoid(s) - y
        return logit_nll(s, y), np.array([np.mean(residual * z), np.mean(residual)])

    result = minimize(objective, x0=np.array([1.0, 0.0]), jac=True, method="BFGS")
    if not result.success:
        logger.warning("Platt scaling did not converge: %s", result.message)
    return float(result.x[0]), float(result.x[1])


def fit_calibrator(
    logits: np.ndarray, targets: np.ndarray, method: str, evaluation: EvaluationConfig
) -> Calibrator:
    """Fit ``calibration.method`` ("platt" or "temperature") on labelled logits.

    Args:
        logits: [n] logits.
        targets: [n] labels, 1.0 or 0.0 (no NaN).
        method: "platt" or "temperature".
        evaluation: The ``evaluation`` section (``temperature_bounds``).

    Returns:
        The fitted calibrator.

    Raises:
        ValueError: For another method.
    """
    if method == "platt":
        slope, intercept = fit_platt(logits, targets)
        if slope <= 0:
            logger.warning("Platt slope %.4g <= 0: the calibration reverses the ranking", slope)
        return Calibrator(method, slope, intercept)
    if method == "temperature":
        temperature = fit_temperature(logits, targets, evaluation.temperature_bounds)
        return Calibrator(method, 1.0 / temperature, 0.0)
    raise ValueError(f"unknown calibration method {method!r}")


def reliability_bins(
    targets: np.ndarray, probabilities: np.ndarray, n_bins: int
) -> pd.DataFrame:
    """Equal-count bins of the predictions, lowest first, with their observed rate.

    Args:
        targets: [n] labels.
        probabilities: [n] predicted probabilities.
        n_bins: Number of bins (sizes differ by at most one).

    Returns:
        ``RELIABILITY_COLUMNS``.
    """
    order = np.argsort(probabilities, kind="stable")
    rows = [
        {"bin": b + 1, "n": len(part), "mean_predicted": float(probabilities[part].mean()),
         "observed_rate": float(targets[part].mean())}
        for b, part in enumerate(np.array_split(order, n_bins))
        if len(part)
    ]
    return pd.DataFrame(rows, columns=RELIABILITY_COLUMNS)


def probability_metrics(
    targets: np.ndarray, probabilities: np.ndarray, n_bins: int
) -> dict[str, float]:
    """AUROC, AUPRC, Brier score, NLL, expected calibration error and mean prediction.

    The ECE is the patient-weighted mean of |mean predicted - observed rate| over
    ``reliability_bins``.

    Args:
        targets: [n] labels, 1.0 or 0.0.
        probabilities: [n] predicted probabilities.
        n_bins: Bins for the ECE.

    Returns:
        auroc, auprc, brier, nll, ece and mean_predicted.
    """
    y, p = np.asarray(targets, dtype=np.float64), np.asarray(probabilities, dtype=np.float64)
    tiny = np.finfo(np.float64).eps
    clipped = np.clip(p, tiny, 1.0 - tiny)
    bins = reliability_bins(y, p, n_bins)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "ece": float((bins["n"] * (bins["mean_predicted"] - bins["observed_rate"]).abs()).sum()
                     / len(y)),
        "mean_predicted": float(p.mean()),
    }


def calibrate(
    logits: np.ndarray,
    targets: np.ndarray,
    method: str,
    evaluation: EvaluationConfig,
    seed: int,
) -> Calibration:
    """Fit a calibrator on the validation logits and measure its effect.

    The calibrator is fitted on every validation patient with an observed label
    and scored on the same patients (``after``: in-sample, optimistic). As an
    out-of-sample check, ``cross_fitted`` scores each patient with a calibrator
    fitted on the other ``calibration_cv_folds`` - 1 stratified folds.

    Args:
        logits: [n] validation logits of ``any_onset``.
        targets: [n] their labels (no NaN).
        method: ``calibration.method``: "platt" or "temperature".
        evaluation: The ``evaluation`` section.
        seed: Random seed of the folds.

    Returns:
        The calibrator and the metrics before and after.
    """
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    n_bins = evaluation.calibration_bins
    calibrator = fit_calibrator(z, y, method, evaluation)
    out_of_fold = np.empty_like(z)
    fold_calibrators = []
    folds = StratifiedKFold(evaluation.calibration_cv_folds, shuffle=True, random_state=seed)
    for fit_rows, score_rows in folds.split(z, y):
        fold_calibrator = fit_calibrator(z[fit_rows], y[fit_rows], method, evaluation)
        fold_calibrators.append(fold_calibrator)
        out_of_fold[score_rows] = fold_calibrator.apply(z[score_rows])
    before, after = sigmoid(z), calibrator.apply(z)
    calibration = Calibration(
        calibrator=calibrator,
        observed=len(y),
        positives=int(y.sum()),
        before=probability_metrics(y, before, n_bins),
        after=probability_metrics(y, after, n_bins),
        cross_fitted=probability_metrics(y, out_of_fold, n_bins),
        fold_calibrators=tuple(fold_calibrators),
        bins_before=reliability_bins(y, before, n_bins),
        bins_after=reliability_bins(y, after, n_bins),
    )
    logger.info(
        "Calibration (%s) %s (folds: %s): Brier %.4f -> %.4f (cross-fitted %.4f), "
        "ECE %.4f -> %.4f (cross-fitted %.4f)",
        method, calibrator.describe(), "; ".join(c.describe() for c in fold_calibrators),
        calibration.before["brier"], calibration.after["brier"],
        calibration.cross_fitted["brier"], calibration.before["ece"], calibration.after["ece"],
        calibration.cross_fitted["ece"],
    )
    return calibration


def save_calibration(calibration: Calibration, path: Path) -> None:
    """Write the fitted calibrator and its fit statistics as JSON."""
    calibrator = calibration.calibrator
    payload: dict[str, Any] = {
        "target": ANY_ONSET,
        "method": calibrator.method,
        "slope": calibrator.slope,
        "intercept": calibrator.intercept,
        "apply_as": "probability = sigmoid(slope * logit + intercept)",
        "fitted_on": "validation patients with an observed label",
        "observed": calibration.observed,
        "positives": calibration.positives,
        "nll_before": calibration.before["nll"],
        "nll_after": calibration.after["nll"],
        "cross_fitted": {key: calibration.cross_fitted[key]
                         for key in ("brier", "ece", "nll", "mean_predicted")},
        "folds": [{"slope": c.slope, "intercept": c.intercept}
                  for c in calibration.fold_calibrators],
    }
    if calibrator.method == "temperature":
        payload["temperature"] = 1.0 / calibrator.slope
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%s: %s)", path, calibrator.method, calibrator.describe())


def load_calibration(path: Path) -> Calibrator:
    """Read the calibrator written by ``save_calibration``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Calibrator(payload["method"], float(payload["slope"]), float(payload["intercept"]))


def plot_reliability(calibration: Calibration, path: Path, cfg: Config) -> None:
    """Save the validation reliability diagram of ``any_onset`` before and after calibration.

    Args:
        calibration: ``calibrate`` output.
        path: Output PNG.
        cfg: Project configuration (``summary.figure_dpi``).
    """
    plot_reliability_bins(
        calibration.bins_before,
        calibration.bins_after,
        calibrated_label(calibration.calibrator),
        f"Reliability of {ANY_ONSET}, validation (n = {calibration.observed}, "
        f"{len(calibration.bins_before)} equal-count bins)",
        path,
        cfg,
    )


def plot_reliability_bins(
    bins_before: pd.DataFrame,
    bins_after: pd.DataFrame,
    after_label: str,
    title: str,
    path: Path,
    cfg: Config,
) -> None:
    """Save a reliability diagram of two ``reliability_bins`` series, before and after.

    Args:
        bins_before: Bins of the uncalibrated probabilities.
        bins_after: Bins of the calibrated probabilities.
        after_label: Label of the calibrated series.
        title: Chart title.
        path: Output PNG.
        cfg: Project configuration (``summary.figure_dpi``).
    """
    dpi = cfg.summary.figure_dpi
    px = _POINTS_PER_INCH / dpi
    series = [
        (bins_before, "before", _BEFORE_COLOR),
        (bins_after, after_label, _AFTER_COLOR),
    ]
    limit = 1.0  # probabilities: the full unit square
    with matplotlib.rc_context(chart_style(dpi, cfg.summary.figure_save_dpi)):
        fig = Figure(figsize=_FIG_SIZE_IN, layout="constrained")
        ax = fig.subplots()
        ax.plot([0, limit], [0, limit], color=_BASELINE, linewidth=px, linestyle=":")
        ax.annotate("perfect calibration", xy=(limit, limit), xytext=(-_LABEL_OFFSET_PT, 0),
                    textcoords="offset points", ha="right", va="top", color=_TEXT_SECONDARY)
        ends = [(float(bins["observed_rate"].iloc[-1]), float(bins["mean_predicted"].iloc[-1]))
                for bins, _, _ in series]
        for (bins, name, color), (end, x_end), (other, x_other) in zip(series, ends, ends[::-1]):
            ax.plot(bins["mean_predicted"], bins["observed_rate"], color=color,
                    linewidth=_LINE_PX * px, marker="o", markersize=_MARKER_PX * px,
                    markeredgecolor=_SURFACE, markeredgewidth=_RING_PX * px, label=name)
            # The higher end label sits above its point and the other below, so the two
            # stay apart. A monotone calibration keeps equal-count bins, so the ends often
            # tie in height: then the leftmost end's label goes above, clear of the other
            # curve, which runs below and to its right.
            leftmost = x_end < x_other or (x_end == x_other and name == series[0][1])
            above = end > other or (end == other and leftmost)
            ax.annotate(name.split(" (")[0], xy=(x_end, end),
                        xytext=(_LABEL_OFFSET_PT, _LABEL_OFFSET_PT * (1 if above else -1)),
                        textcoords="offset points", ha="left",
                        va="bottom" if above else "top", color=_TEXT_SECONDARY)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed onset rate")
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.legend(frameon=False, loc="upper left")
        save_figure(fig, path)


def calibration_section(
    calibration: Calibration, figure: str, calibration_file: str, cfg: Config
) -> list[str]:
    """Markdown for the calibration result.

    Args:
        calibration: ``calibrate`` output.
        figure: The reliability figure's path relative to the report.
        calibration_file: Name of the saved calibration file.
        cfg: Project configuration.

    Returns:
        Lines to pass as ``train.write_training_summary(extra_sections=...)``.
    """
    calibrator = calibration.calibrator
    before, after, cross = calibration.before, calibration.after, calibration.cross_fitted
    rows = [
        [name, f"{before[key]:.3f}", f"{after[key]:.3f}", f"{cross[key]:.3f}"]
        for name, key in (("AUROC", "auroc"), ("AUPRC", "auprc"), ("Brier score", "brier"),
                          ("Log loss", "nll"), ("ECE", "ece"),
                          ("Mean predicted probability", "mean_predicted"))
    ]
    prevalence = calibration.positives / calibration.observed
    folds = cfg.evaluation.calibration_cv_folds
    fitted_on = (
        f"on the {calibration.observed:,} validation patients with an observed label "
        f"({calibration.positives:,} positives, rate {prevalence:.3f}); saved to "
        f"`{calibration_file}`"
    )
    if calibrator.method == "platt":
        title = "Platt scaling"
        method = (
            f"`{ANY_ONSET}` probabilities are `sigmoid(a * logit + b)` with **"
            f"{calibrator.describe()}**: Platt scaling, a logistic regression of the label on "
            f"the logit with an intercept, fitted by maximum likelihood {fitted_on}. A positive "
            "slope preserves the ranking, so AUROC and AUPRC do not change."
        )
        shift = ("the intercept removes that shift, so the mean predicted probability should "
                 "now be close to the observed rate.")
    else:
        title = "temperature scaling"
        method = (
            f"`{ANY_ONSET}` probabilities are `sigmoid(logit / T)` with **"
            f"{calibrator.describe()}**, fitted by minimizing the log loss {fitted_on}. A "
            "temperature preserves the ranking, so AUROC and AUPRC do not change."
        )
        shift = ("a temperature only rescales logits around 0 and cannot remove such a shift, "
                 "so compare the mean predicted probability with the observed rate.")
    return [
        f"## Calibration ({title})",
        "",
        f"{method} \"After\" is scored on the patients it was fitted on (optimistic); "
        f"\"cross-fitted\" scores each patient with a calibrator fitted on the other "
        f"{folds - 1} of {folds} stratified folds (fold fits: "
        f"{'; '.join(c.describe() for c in calibration.fold_calibrators)}); its AUROC and "
        "AUPRC pool folds scored with different calibrators, so they move slightly.",
        "",
        markdown_table(["Validation", "Before", "After", "After, cross-fitted"], rows, "lrrr"),
        "",
        f"ECE: expected calibration error over {cfg.evaluation.calibration_bins} equal-count "
        "bins. The model is trained with `pos_weight` = negatives / positives, which shifts "
        f"its logits upward; {shift}",
        "",
        f"![Reliability diagram]({figure})",
        "",
    ]


def calibrated_label(calibrator: Calibrator) -> str:
    """Legend label of calibrated probabilities, e.g. "after Platt (a = 0.92, b = -1.43)"."""
    name = "Platt" if calibrator.method == "platt" else "temperature"
    return f"after {name} ({calibrator.describe()})"


# --- Phase 7: models and their inputs --------------------------------------------------


@dataclass(frozen=True)
class CVModel:
    """One model of the Phase 7 comparison.

    Attributes:
        number: Its number in the comparison: 1 (age and sex only) to 8 (full TemporalGCN).
        key: Name of its prediction columns.
        name: Display name.
        inputs: Display description of its inputs.
        graph: The graph it uses: "real", "random" or "none".
        windows: The windows it reads.
        kind: ``LOGISTIC`` or ``GCN``.
    """

    number: int
    key: str
    name: str
    inputs: str
    graph: str
    windows: str
    kind: str


def cv_models(years: Sequence[int], spectral_dim: int) -> tuple[CVModel, ...]:
    """Return the eight Phase 7 models, numbered as in the report.

    Every model sees age at the cutoff and sex: the logistic regressions as inputs,
    the GCNs in their heads.

    Args:
        years: The training windows, oldest first.
        spectral_dim: Width of the spectral embedding (model 4).

    Returns:
        Models 1 to 8; the full TemporalGCN (``FULL_MODEL``) is model 8.
    """
    last = str(years[-1])
    every = f"{len(years)} windows {years[0]}-{years[-1]}"
    heads = "; age, sex in the heads"
    return (
        CVModel(1, DEMOGRAPHICS_MODEL, "Age and sex only",
                "age at the cutoff and sex, logistic regression", "none", "none", LOGISTIC),
        CVModel(2, "lr_last", "Logistic regression", f"{last} window + age, sex", "none", last,
                LOGISTIC),
        CVModel(3, "lr_all", "Logistic regression", f"{every}, concatenated + age, sex", "none",
                every, LOGISTIC),
        CVModel(4, "spectral_lr", "Spectral embedding + logistic regression",
                f"{spectral_dim} Laplacian eigenvectors of the {last} graph + age, sex", "real",
                last, LOGISTIC),
        CVModel(5, "gcn_static", "Static GCN", f"{last} window and its graph{heads}", "real",
                last, GCN),
        CVModel(6, "temporal_no_graph", "Temporal model, no graph",
                f"{every}, A_hat = I in every window{heads}", "none", every, GCN),
        CVModel(7, "tgcn_rewired", "TemporalGCN, rewired graphs",
                f"{every} + degree-preserving random graphs{heads}", "random", every, GCN),
        CVModel(8, FULL_MODEL, "TemporalGCN (full)", f"{every} + similarity graphs{heads}",
                "real", every, GCN),
    )


def rewire_graph(W: sp.spmatrix, swaps_per_edge: int, rng: np.random.Generator) -> sp.csr_matrix:
    """Rewire a graph by degree-preserving double edge swaps (Maslov and Sneppen).

    Each attempt draws two edges (a, b) and (c, d), orients the second at random,
    and replaces them by (a, d) and (c, b), each new edge keeping the weight of the
    edge it replaces. An attempt that would make a self-loop or a repeated edge is
    skipped. Every node keeps its number of neighbours and the edge weights keep
    their distribution; which patients are linked becomes random.

    Args:
        W: Symmetric (n, n) adjacency without self-loops.
        swaps_per_edge: Attempts per edge.
        rng: Random generator.

    Returns:
        CSR (n, n), symmetric, no self-loops, with the degrees of ``W``.
    """
    upper = sp.triu(sp.csr_matrix(W), k=1).tocoo()
    heads, tails = upper.row.tolist(), upper.col.tolist()  # heads[e] < tails[e]
    m, n = len(heads), W.shape[0]
    if m < 2:
        return sp.csr_matrix(W, copy=True)
    edges = set(zip(heads, tails))
    attempts = swaps_per_edge * m
    picks = rng.integers(0, m, size=(attempts, 2)).tolist()
    flips = (rng.random(attempts) < 0.5).tolist()
    accepted = 0
    for (i, j), flip in zip(picks, flips):
        if i == j:
            continue
        a, b = heads[i], tails[i]
        c, d = (tails[j], heads[j]) if flip else (heads[j], tails[j])
        if a == d or c == b:
            continue
        first = (a, d) if a < d else (d, a)
        second = (c, b) if c < b else (b, c)
        if first in edges or second in edges:
            continue
        edges.remove((heads[i], tails[i]))
        edges.remove((heads[j], tails[j]))
        edges.update((first, second))
        heads[i], tails[i] = first
        heads[j], tails[j] = second
        accepted += 1
    logger.info("Rewired %d edges: %d of %d swaps accepted", m, accepted, attempts)
    rows, cols = np.array(heads + tails), np.array(tails + heads)
    return sp.csr_matrix((np.concatenate([upper.data, upper.data]), (rows, cols)), shape=(n, n))


@dataclass(frozen=True)
class GraphInputs:
    """The windows, features and graphs one GCN model is trained on."""

    years: tuple[int, ...]
    matrices: list[sp.csr_matrix]
    weights: list[sp.csr_matrix]
    adjacency: list[sp.csr_matrix]


def gcn_inputs(model_data: ModelData, swaps_per_edge: int, seed: int) -> dict[str, GraphInputs]:
    """Return the inputs of each GCN model of ``cv_models``, by key.

    * ``gcn_static``: the final window and its graph only.
    * ``temporal_no_graph``: every window with an empty graph, so ``A_hat = I``.
    * ``tgcn_rewired``: every window's graph rewired (``rewire_graph``), with one
      generator seeded with ``seed``; no labels are involved.
    * ``FULL_MODEL``: every window and its similarity graph.

    Args:
        model_data: ``train.load_model_data`` output.
        swaps_per_edge: ``cross_validation.rewire_swaps_per_edge``.
        seed: Seed of the rewiring.

    Returns:
        Key -> inputs.
    """
    n = len(model_data.patient_ids)
    empty = [sp.csr_matrix((n, n)) for _ in model_data.years]
    rng = np.random.default_rng(seed)
    rewired = [rewire_graph(W, swaps_per_edge, rng) for W in model_data.weights]
    return {
        "gcn_static": GraphInputs(model_data.years[-1:], model_data.matrices[-1:],
                                  model_data.weights[-1:], model_data.adjacency[-1:]),
        "temporal_no_graph": GraphInputs(model_data.years, model_data.matrices, empty,
                                         [normalize_adjacency(W) for W in empty]),
        "tgcn_rewired": GraphInputs(model_data.years, model_data.matrices, rewired,
                                    [normalize_adjacency(W) for W in rewired]),
        FULL_MODEL: GraphInputs(model_data.years, model_data.matrices, model_data.weights,
                                model_data.adjacency),
    }


def load_spectral_embedding(cfg: Config, n_nodes: int) -> np.ndarray:
    """Read the Phase 4 spectral embedding of the final window's graph, in node order.

    Label-free: the smallest non-trivial Laplacian eigenvectors of the graph's largest
    component. Nodes outside that component get zeros.

    Args:
        cfg: Project configuration.
        n_nodes: Number of graph nodes (the model cohort).

    Returns:
        [n_nodes, graph.spectral_dim].
    """
    year = feature_window_years(cfg)[-1]
    saved = np.load(cfg.paths.graphs_dir / f"spectral_{year}.npz")
    embedding = np.zeros((n_nodes, saved["embedding"].shape[1]))
    embedding[saved["nodes"]] = saved["embedding"]
    return embedding


# --- Phase 7: folds --------------------------------------------------------------------


def stratified_folds(strata: pd.Series, n_folds: int, seed: int) -> pd.Series:
    """Deal patients into folds 1..n_folds so that every stratum is spread evenly.

    Strata are taken in sorted order and each is shuffled with ``seed``; patients are
    dealt round-robin, continuing from where the previous stratum stopped. Each
    stratum's fold counts, and the fold sizes, differ by at most one, however small
    the stratum.

    Args:
        strata: Stratum per patient, indexed by patient_id.
        n_folds: Number of folds.
        seed: Random seed.

    Returns:
        Fold number per patient, indexed like ``strata``.
    """
    rng = np.random.default_rng(seed)
    fold = pd.Series(0, index=strata.index, dtype=np.int64, name="fold")
    start = 0
    for stratum in sorted(strata.unique()):
        members = strata.index[(strata == stratum).to_numpy()]
        members = members[rng.permutation(len(members))]
        fold[members] = (start + np.arange(len(members))) % n_folds + 1
        start = (start + len(members)) % n_folds
    return fold


def cross_validation_folds(labels: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Fold of every patient with an observed ``ANY_ONSET``, stratified by phenotype x label.

    Args:
        labels: ``labels.load_labels`` output (with ``PHENOTYPE``).
        n_folds: ``cross_validation.folds``.
        seed: Random seed.

    Returns:
        Fold number (1..n_folds) indexed by patient_id, in label order.
    """
    seen = labels[labels[ANY_ONSET].notna()]
    return stratified_folds(split_strata(seen), n_folds, seed)


def fold_split(
    labels: pd.DataFrame, folds: pd.Series, fold: int, inner_val: float, seed: int
) -> np.ndarray:
    """Return the split of every node for one fold of the cross-validation.

    The fold's patients are ``TEST`` (their labels are dropped by
    ``make_training_data``); the other folds' patients are divided into ``TRAIN``
    and ``VAL`` (the inner validation set, a share ``inner_val``) with
    ``labels.stratified_split`` on phenotype x label; patients without an observed
    ``ANY_ONSET`` are ``UNLABELED`` (graph nodes only).

    Args:
        labels: ``labels.load_labels`` output, node order.
        folds: ``cross_validation_folds`` output.
        fold: The held-out fold.
        inner_val: ``cross_validation.inner_val``.
        seed: Random seed of the inner split.

    Returns:
        Split name per node.
    """
    strata = split_strata(labels)
    in_fold = folds.to_numpy() == fold
    inner = stratified_split(strata.loc[folds.index[~in_fold]],
                             {TRAIN: 1.0 - inner_val, VAL: inner_val}, seed)
    split = pd.Series(UNLABELED, index=pd.Index(labels["patient_id"]), dtype=object)
    split[folds.index[in_fold]] = TEST
    split[inner.index] = inner.to_numpy(dtype=object)
    return split.to_numpy(dtype=str)


def fold_training_data(
    model_data: ModelData, inputs: GraphInputs, split: np.ndarray
) -> TrainingData:
    """Training data of one fold and one model: ``TEST`` labels dropped, age restandardized.

    Args:
        model_data: ``train.load_model_data`` output.
        inputs: The model's windows and graphs (``gcn_inputs``).
        split: ``fold_split`` output.

    Returns:
        The training data; age is standardized with the fold's ``TRAIN`` patients.
    """
    return make_training_data(
        model_data.patient_ids, inputs.years, inputs.matrices, inputs.weights, inputs.adjacency,
        model_data.labels.assign(**{SPLIT: split}), model_data.groups, model_data.demographics,
    )


# --- Phase 7: fitting ------------------------------------------------------------------


def logistic_inputs(key: str, data: TrainingData, spectral: np.ndarray) -> np.ndarray:
    """Return the inputs of logistic model ``key`` (``cv_models``) for every node.

    Age (standardized with the ``TRAIN`` patients) and sex are always included. The
    spectral eigenvectors are standardized with the ``TRAIN`` patients' mean and sd:
    their scale (unit norm over all nodes) is arbitrary and tiny, so the L2 penalty
    would otherwise erase them.

    Args:
        key: "age_sex", "lr_last", "lr_all" or "spectral_lr".
        data: The fold's training data.
        spectral: ``load_spectral_embedding`` output.

    Returns:
        Dense [N, F].

    Raises:
        ValueError: For another key.
    """
    demographics = data.demographics.numpy().astype(np.float64)
    if key == DEMOGRAPHICS_MODEL:
        return demographics
    if key == "lr_last":
        return baseline_inputs(data, "last", True)
    if key == "lr_all":
        return baseline_inputs(data, "all", True)
    if key == "spectral_lr":
        scaled = StandardScaler().fit(spectral[data.split == TRAIN]).transform(spectral)
        return np.hstack([scaled, demographics])
    raise ValueError(f"no logistic model {key!r}")


def logistic_logits(
    X: np.ndarray, data: TrainingData, settings: BaselinesConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Fit logistic regressions of ``any_onset`` and of each group on the ``TRAIN`` labels.

    Args:
        X: [N, F] inputs (``logistic_inputs``).
        data: The fold's training data.
        settings: The ``baselines`` section.

    Returns:
        The decision function (logit) of every node: [N] for ``any_onset`` and [N, G]
        for the groups; a group with a single class among the ``TRAIN`` labels gets
        logit 0 everywhere (logged).
    """
    any_targets, onset_targets = (t.numpy() for t in split_targets(data, TRAIN))
    logits = fit_logistic(X, any_targets, settings).decision_function(X)
    group_logits = np.zeros((X.shape[0], onset_targets.shape[1]))
    for g, group in enumerate(data.groups):
        observed = onset_targets[:, g][~np.isnan(onset_targets[:, g])]
        if len(np.unique(observed)) < 2:
            logger.warning("Group %s has one class among the training labels; logit 0", group)
            continue
        group_logits[:, g] = fit_logistic(X, onset_targets[:, g], settings).decision_function(X)
    return logits, group_logits


def gcn_logits(
    data: TrainingData, model_cfg: ModelConfig, seeds: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Train one ``OnsetModel`` per seed with ``train.fit``; average their logits.

    Each run stops early on the ``VAL`` patients' AUROC of ``any_onset``.

    Args:
        data: The fold's training data for this model.
        model_cfg: The ``model`` section (fixed hyperparameters).
        seeds: ``cross_validation.seeds``.

    Returns:
        The seeds' mean logits, [N] for ``any_onset`` and [N, G] for the groups, and
        one record per seed: seed, best_epoch, epochs_run, inner_val_auroc.
    """
    any_logits, group_logits, runs = [], [], []
    for seed in seeds:
        result = fit(data, model_cfg, seed, verbose=False)
        output = predict(result.model, data)
        any_logits.append(output.any_onset_logits.numpy().astype(np.float64))
        group_logits.append(output.onset_logits.numpy().astype(np.float64))
        best = result.history.iloc[result.best_epoch - 1]
        runs.append({"seed": seed, "best_epoch": result.best_epoch,
                     "epochs_run": result.epochs_run, "inner_val_auroc": float(best["val_auroc"])})
    return np.mean(any_logits, axis=0), np.mean(group_logits, axis=0), runs


# --- Phase 7: cross-validation ---------------------------------------------------------


@dataclass(frozen=True)
class CrossValidation:
    """Out-of-fold predictions of every Phase 7 model.

    Attributes:
        models: ``cv_models`` output.
        groups: Onset group names.
        predictions: One row per patient with an observed ``ANY_ONSET``: patient_id,
            node, fold, ``PHENOTYPE`` (0 = unclustered), ``ANY_ONSET``, the onset
            labels (NaN = masked), and per model its calibrated ``any_onset``
            probability (column ``key``), its uncalibrated logit
            (``prediction_column(key, "logit")``) and each group's uncalibrated logit
            (``prediction_column(key, group)``).
        fits: ``FIT_COLUMNS``: one row per fold and model (per seed for the GCNs).
    """

    models: tuple[CVModel, ...]
    groups: tuple[str, ...]
    predictions: pd.DataFrame
    fits: pd.DataFrame


def prediction_column(key: str, part: str) -> str:
    """Name of a ``CrossValidation.predictions`` column: model ``key``'s ``part``."""
    return f"{key}.{part}"


def run_cross_validation(
    model_data: ModelData, spectral: np.ndarray, cfg: Config
) -> CrossValidation:
    """Run the Phase 7 cross-validation of every ``cv_models`` model.

    For each fold (``cross_validation_folds``): the fold is held out; the other
    folds are split into inner training and inner validation patients
    (``fold_split``). Every model is fitted on the inner training labels; the GCNs
    stop early on the inner validation AUROC, one run per seed, logits averaged.
    ``calibration.method`` is fitted on each model's inner validation logits and
    applied to the held-out fold. The held-out labels are read only to score it.

    Args:
        model_data: ``train.load_model_data`` output.
        spectral: ``load_spectral_embedding`` output.
        cfg: Project configuration (fixed ``model`` hyperparameters).

    Returns:
        The out-of-fold predictions.
    """
    settings = cfg.cross_validation
    models = cv_models(model_data.years, spectral.shape[1])
    labels = model_data.labels
    folds = cross_validation_folds(labels, settings.folds, cfg.seed)
    nodes = pd.Index(labels["patient_id"]).get_indexer(folds.index)
    row_of = np.full(len(labels), -1)
    row_of[nodes] = np.arange(len(nodes))
    onset_columns = [ONSET_PREFIX + group for group in model_data.groups]
    frame = pd.DataFrame({
        "patient_id": np.asarray(folds.index, dtype=str),
        "node": nodes,
        "fold": folds.to_numpy(),
        PHENOTYPE: labels[PHENOTYPE].fillna(0).to_numpy(dtype=np.int64)[nodes],
        ANY_ONSET: labels[ANY_ONSET].to_numpy(dtype=float)[nodes],
        **{column: labels[column].to_numpy(dtype=float)[nodes] for column in onset_columns},
    })
    columns = {
        name: np.full(len(nodes), np.nan)
        for model in models
        for name in (model.key, prediction_column(model.key, "logit"),
                     *(prediction_column(model.key, g) for g in model_data.groups))
    }
    graph_inputs = gcn_inputs(model_data, settings.rewire_swaps_per_edge, cfg.seed)
    all_targets = labels[ANY_ONSET].to_numpy(dtype=float)
    fits: list[dict[str, Any]] = []
    start = perf_counter()
    for fold in range(1, settings.folds + 1):
        split = fold_split(labels, folds, fold, settings.inner_val, cfg.seed)
        base = fold_training_data(model_data, graph_inputs[FULL_MODEL], split)
        train_y = split_targets(base, TRAIN)[0].numpy()
        val_y = split_targets(base, VAL)[0].numpy()
        val_seen = ~np.isnan(val_y)
        held_out = np.flatnonzero(split == TEST)
        rows = row_of[held_out]
        sizes = {"inner_train": int((~np.isnan(train_y)).sum()), "inner_val": int(val_seen.sum()),
                 "held_out": len(held_out),
                 "held_out_positives": int(np.nansum(all_targets[held_out]))}
        logger.info("Fold %d/%d: %s", fold, settings.folds, sizes)
        for model in models:
            if model.kind == LOGISTIC:
                X = logistic_inputs(model.key, base, spectral)
                logits, group_logits = logistic_logits(X, base, cfg.baselines)
                runs: list[dict[str, Any]] = [{
                    "seed": np.nan, "best_epoch": np.nan, "epochs_run": np.nan,
                    "inner_val_auroc": binary_metrics(val_y, logits)["auroc"],
                }]
            else:
                data = base if model.key == FULL_MODEL else fold_training_data(
                    model_data, graph_inputs[model.key], split
                )
                logits, group_logits, runs = gcn_logits(data, cfg.model, settings.seeds)
            calibrator = fit_calibrator(logits[val_seen], val_y[val_seen],
                                        cfg.calibration.method, cfg.evaluation)
            probabilities = calibrator.apply(logits[held_out])
            columns[model.key][rows] = probabilities
            columns[prediction_column(model.key, "logit")][rows] = logits[held_out]
            for g, group in enumerate(model_data.groups):
                columns[prediction_column(model.key, group)][rows] = group_logits[held_out, g]
            fits.extend({"fold": fold, "key": model.key, **run, "slope": calibrator.slope,
                         "intercept": calibrator.intercept, **sizes} for run in runs)
            logger.info(
                "Fold %d/%d, model %d (%s): held-out AUROC %.4f; Platt %s (%.0f s elapsed)",
                fold, settings.folds, model.number, model.key,
                roc_auc_score(all_targets[held_out], probabilities), calibrator.describe(),
                perf_counter() - start,
            )
    predictions = pd.concat([frame, pd.DataFrame(columns)], axis=1)
    return CrossValidation(models, model_data.groups, predictions,
                           pd.DataFrame(fits, columns=FIT_COLUMNS))


def save_cross_validation(cv: CrossValidation, cfg: Config) -> None:
    """Write the out-of-fold predictions and the fits (paths from ``cfg.paths``)."""
    for path in (cfg.paths.cv_predictions_csv, cfg.paths.cv_fits_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    cv.predictions.to_csv(cfg.paths.cv_predictions_csv, index=False)
    cv.fits.to_csv(cfg.paths.cv_fits_csv, index=False)
    logger.info("Wrote %s and %s", cfg.paths.cv_predictions_csv, cfg.paths.cv_fits_csv)


def load_cross_validation(cfg: Config, model_data: ModelData, spectral_dim: int) -> CrossValidation:
    """Read the predictions written by ``save_cross_validation``.

    Args:
        cfg: Project configuration.
        model_data: ``train.load_model_data`` output.
        spectral_dim: Width of the spectral embedding (for the model descriptions).

    Returns:
        The cross-validation.

    Raises:
        ValueError: If the saved predictions are for other patients, folds or models.
    """
    predictions = pd.read_csv(cfg.paths.cv_predictions_csv, dtype={"patient_id": str})
    fits = pd.read_csv(cfg.paths.cv_fits_csv)
    models = cv_models(model_data.years, spectral_dim)
    labels = model_data.labels
    expected = set(labels.loc[labels[ANY_ONSET].notna(), "patient_id"].astype(str))
    needed = {prediction_column(m.key, part) for m in models
              for part in ("logit", *model_data.groups)} | {m.key for m in models}
    if (set(predictions["patient_id"]) != expected
            or predictions["fold"].nunique() != cfg.cross_validation.folds
            or not needed <= set(predictions.columns)):
        raise ValueError(f"{cfg.paths.cv_predictions_csv.name} does not match the labels, folds "
                         "and models; run scripts/07_evaluate.py without --reuse-cv")
    return CrossValidation(models, model_data.groups, predictions, fits)


# --- Phase 7: cross-validation metrics -------------------------------------------------


def cv_fold_metrics(cv: CrossValidation) -> pd.DataFrame:
    """AUROC and AUPRC of every model on every held-out fold.

    Args:
        cv: ``run_cross_validation`` output.

    Returns:
        ``FOLD_COLUMNS``, one row per fold and model.
    """
    rows = []
    for fold, part in cv.predictions.groupby("fold", sort=True):
        y = part[ANY_ONSET].to_numpy(dtype=float)
        for model in cv.models:
            metrics = binary_metrics(y, part[model.key].to_numpy(dtype=float))
            rows.append({"fold": int(fold), "key": model.key, **metrics})
    return pd.DataFrame(rows, columns=FOLD_COLUMNS)


def cv_results(cv: CrossValidation, cfg: Config) -> pd.DataFrame:
    """The main Phase 7 table: per-fold and pooled out-of-fold metrics of every model.

    Pooled metrics use every patient's calibrated out-of-fold probability, with
    percentile bootstrap CIs over patients; every model is scored on the same
    resamples, so ``delta_auroc`` (the full TemporalGCN's AUROC minus the model's)
    is paired. ``folds_full_better`` counts the folds where the full model's AUROC
    is higher.

    Args:
        cv: ``run_cross_validation`` output.
        cfg: Project configuration (``evaluation`` section, seed).

    Returns:
        ``RESULT_COLUMNS``, sorted by pooled AUROC, best first.
    """
    predictions = cv.predictions
    y = predictions[ANY_ONSET].to_numpy(dtype=float)
    scores = {model.key: predictions[model.key].to_numpy(dtype=float) for model in cv.models}
    pooled = compare_models(y, scores, FULL_MODEL, cfg.evaluation, cfg.seed).set_index("key")
    per_fold = cv_fold_metrics(cv).pivot(index="fold", columns="key")
    full_folds = per_fold["auroc"][FULL_MODEL]
    agreement = rank_agreement(cv, DEMOGRAPHICS_MODEL)
    rows = []
    for model in cv.models:
        row = pooled.loc[model.key]
        rho = agreement[model.key]
        fold_auroc, fold_auprc = per_fold["auroc"][model.key], per_fold["auprc"][model.key]
        calibration = probability_metrics(y, scores[model.key], cfg.evaluation.calibration_bins)
        rows.append({
            "number": model.number, "key": model.key, "model": model.name,
            "inputs": model.inputs, "graph": model.graph, "windows": model.windows,
            "fold_auroc_mean": fold_auroc.mean(), "fold_auroc_sd": fold_auroc.std(ddof=1),
            "fold_auprc_mean": fold_auprc.mean(), "fold_auprc_sd": fold_auprc.std(ddof=1),
            "oof_observed": int(row.val_observed), "oof_positives": int(row.val_positives),
            "oof_auroc": row.val_auroc, "oof_auroc_low": row.val_auroc_low,
            "oof_auroc_high": row.val_auroc_high, "oof_auprc": row.val_auprc,
            "oof_auprc_low": row.val_auprc_low, "oof_auprc_high": row.val_auprc_high,
            "delta_auroc": row.delta_auroc, "delta_auroc_low": row.delta_auroc_low,
            "delta_auroc_high": row.delta_auroc_high,
            "folds_full_better": (np.nan if model.key == FULL_MODEL
                                  else float((full_folds > fold_auroc).sum())),
            "oof_brier": calibration["brier"], "oof_ece": calibration["ece"],
            "oof_mean_predicted": calibration["mean_predicted"],
            "age_sex_rank_corr_mean": rho.mean(), "age_sex_rank_corr_min": rho.min(),
            "age_sex_rank_corr_max": rho.max(),
        })
    table = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return table.sort_values(["oof_auroc", "number"], ascending=[False, True],
                             kind="stable").reset_index(drop=True)


def rank_agreement(cv: CrossValidation, reference: str) -> pd.DataFrame:
    """Within-fold Spearman correlation of every model's logits with ``reference``'s.

    Within a fold, a correlation near 1 means the model ranks the held-out patients
    as ``reference`` does: with ``DEMOGRAPHICS_MODEL``, that it learned little beyond
    age and sex.

    Args:
        cv: ``run_cross_validation`` output.
        reference: Key of the reference model.

    Returns:
        One row per fold, one column per model key.
    """
    logit_columns = {prediction_column(model.key, "logit"): model.key for model in cv.models}
    rows = {}
    for fold, part in cv.predictions.groupby("fold", sort=True):
        correlations = part[list(logit_columns)].rename(columns=logit_columns).corr("spearman")
        rows[int(fold)] = correlations[reference]
    return pd.DataFrame(rows).T.rename_axis("fold")


def cv_by_split(cv: CrossValidation, labels: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold AUROC of every model on the patients of each original (Phase 5) split.

    Args:
        cv: ``run_cross_validation`` output.
        labels: ``labels.load_labels`` output (with ``SPLIT``).

    Returns:
        split, patients, positives, then one AUROC column per model key.
    """
    split_of = labels.set_index(labels["patient_id"].astype(str))[SPLIT]
    predictions = cv.predictions.assign(
        **{SPLIT: split_of.reindex(cv.predictions["patient_id"].astype(str)).to_numpy()}
    )
    rows = []
    for name in (TRAIN, VAL, TEST):
        part = predictions[predictions[SPLIT] == name]
        y = part[ANY_ONSET].to_numpy(dtype=float)
        rows.append({SPLIT: name, "patients": len(part), "positives": int(y.sum()),
                     **{model.key: binary_metrics(y, part[model.key].to_numpy(dtype=float))["auroc"]
                        for model in cv.models}})
    return pd.DataFrame(rows)


def cv_group_results(cv: CrossValidation, cfg: Config) -> pd.DataFrame:
    """Pooled out-of-fold AUROC of every model on every onset group, with bootstrap CIs.

    Group scores are the uncalibrated logits (too few inner-validation positives per
    group to fit a calibration), so pooling mixes the folds' scales. Per group, every
    model is scored on the same resamples of the patients with an observed label.

    Args:
        cv: ``run_cross_validation`` output.
        cfg: Project configuration.

    Returns:
        ``GROUP_RESULT_COLUMNS``, by group (config order) then model number.
    """
    evaluation = cfg.evaluation
    rows = []
    for group in cv.groups:
        y = cv.predictions[ONSET_PREFIX + group].to_numpy(dtype=float)
        seen = ~np.isnan(y)
        if not 0 < y[seen].sum() < seen.sum():
            logger.warning("Group %s has a single observed class; no AUROC", group)
            rows += [{"group": group, "key": model.key, "observed": int(seen.sum()),
                      "positives": int(y[seen].sum()), "auroc": np.nan, "auroc_low": np.nan,
                      "auroc_high": np.nan} for model in cv.models]
            continue
        scores = {model.key: cv.predictions[prediction_column(model.key, group)]
                  .to_numpy(dtype=float)[seen] for model in cv.models}
        boot = bootstrap_metrics(y[seen], scores, evaluation.bootstrap_resamples, cfg.seed)
        for model in cv.models:
            low, high = percentile_interval(boot.auroc[model.key], evaluation.ci_level)
            rows.append({"group": group, "key": model.key, "observed": int(seen.sum()),
                         "positives": int(y[seen].sum()),
                         "auroc": float(roc_auc_score(y[seen], scores[model.key])),
                         "auroc_low": low, "auroc_high": high})
    return pd.DataFrame(rows, columns=GROUP_RESULT_COLUMNS)


def plot_cv_auroc(table: pd.DataFrame, path: Path, cfg: Config) -> None:
    """Save each model's pooled out-of-fold AUROC and its difference from the full model.

    Args:
        table: ``cv_results`` output (best first).
        path: Output PNG.
        cfg: Project configuration.
    """
    dpi = cfg.summary.figure_dpi
    px = _POINTS_PER_INCH / dpi
    level = f"{cfg.evaluation.ci_level:.0%}"
    rows = table.iloc[::-1].reset_index(drop=True)  # best at the top
    y = np.arange(len(rows))
    panels = [
        ("Pooled out-of-fold AUROC", f"AUROC ({level} CI)", "oof_auroc", _CHANCE_AUROC, False),
        ("Full TemporalGCN minus this model", f"Difference in AUROC ({level} CI, paired)",
         "delta_auroc", 0.0, True),
    ]
    with matplotlib.rc_context(chart_style(dpi, cfg.summary.figure_save_dpi)):
        fig = Figure(figsize=(11.0, _ROW_IN * len(rows) + 1.4), layout="constrained")
        axes = fig.subplots(1, 2, sharey=True)
        for ax, (title, xlabel, column, reference, signed) in zip(axes, panels):
            value = rows[column].to_numpy(dtype=float)
            low = rows[f"{column}_low"].to_numpy(dtype=float)
            high = rows[f"{column}_high"].to_numpy(dtype=float)
            shown = ~np.isnan(value)
            ax.axvline(reference, color=_BASELINE, linewidth=px)
            ax.hlines(y[shown], low[shown], high[shown], color=_SERIES_COLOR,
                      linewidth=_LINE_PX * px)
            ax.plot(value[shown], y[shown], linestyle="none", marker="o", color=_SERIES_COLOR,
                    markersize=_MARKER_PX * px, markeredgecolor=_SURFACE,
                    markeredgewidth=_RING_PX * px)
            for row, v, h in zip(y[shown], value[shown], high[shown]):
                ax.annotate(f"{v:+.3f}" if signed else f"{v:.3f}", xy=(h, row),
                            xytext=(_LABEL_OFFSET_PT, 0), textcoords="offset points",
                            va="center", color=_TEXT_SECONDARY)
            for row in y[~shown]:
                ax.annotate("reference", xy=(reference, row), xytext=(_LABEL_OFFSET_PT, 0),
                            textcoords="offset points", va="center", color=_TEXT_SECONDARY)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.margins(x=0.18)
            ax.grid(axis="x")
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
            ax.spines["left"].set_visible(False)
        axes[0].set_yticks(y, [f"{row.number}. {row.model}" for row in rows.itertuples()])
        save_figure(fig, path)


# --- Phase 7: the held-out test set ----------------------------------------------------


def held_out_targets(model_data: ModelData) -> tuple[np.ndarray, np.ndarray]:
    """Return the test split's labels: the only reader of test labels outside the CV.

    Call it once, to score the final model (``evaluate_test``).

    Args:
        model_data: ``train.load_model_data`` output.

    Returns:
        ``any_onset`` [N] and onset [N, G] labels, NaN for every non-test node (and
        for masked test labels).
    """
    labels = model_data.labels
    test = (labels[SPLIT] == TEST).to_numpy()
    any_onset = np.where(test, labels[ANY_ONSET].to_numpy(dtype=float), np.nan)
    onsets = labels[[ONSET_PREFIX + g for g in model_data.groups]].to_numpy(dtype=float, copy=True)
    onsets[~test] = np.nan
    logger.info("Read the test labels of %d patients", int(test.sum()))
    return any_onset, onsets


def f1_threshold(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the probability threshold with the highest F1 (predict 1 at or above it)."""
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    f1 = 2 * precision * recall / np.maximum(precision + recall, np.finfo(np.float64).tiny)
    return float(thresholds[np.argmax(f1[:-1])])  # the last point has no threshold


def threshold_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """F1, precision and recall of predicting 1 when the probability is >= ``threshold``."""
    y = np.asarray(targets) == 1
    predicted = np.asarray(probabilities) >= threshold
    tp = float((predicted & y).sum())
    precision = tp / predicted.sum() if predicted.any() else 0.0
    recall = tp / y.sum() if y.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall}


def bootstrap_statistics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], Mapping[str, float]],
    n_resamples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Evaluate ``statistic`` on bootstrap resamples of the patients.

    Drawn as in ``bootstrap_metrics`` (same seed, same resamples); resamples with a
    single class are skipped.

    Args:
        targets: [n] labels, 1.0 or 0.0.
        probabilities: [n] predictions.
        statistic: (targets, probabilities) -> named values.
        n_resamples: Resamples to draw.
        seed: Random seed.

    Returns:
        Name -> one value per resample used.
    """
    y, p = np.asarray(targets, dtype=float), np.asarray(probabilities, dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(y), size=(n_resamples, len(y)))
    positives = y[draws].sum(axis=1)
    draws = draws[(positives > 0) & (positives < len(y))]
    samples = [statistic(y[i], p[i]) for i in draws]
    return {name: np.array([sample[name] for sample in samples]) for name in samples[0]}


@dataclass(frozen=True)
class TestEvaluation:
    """The single-shot test evaluation of the final model.

    Attributes:
        observed: Test patients with an observed ``any_onset``.
        positives: Of them, positives.
        calibrator: The calibration fitted on validation (``calibration.json``).
        threshold: The calibrated-probability threshold with the best validation F1.
        calibrated: ``probability_metrics`` and ``threshold_metrics`` of the
            calibrated test probabilities.
        intervals: Bootstrap CI of each ``TEST_METRICS`` value.
        raw: ``probability_metrics`` of ``sigmoid(logit)``.
        groups: group, observed, positives, auroc, auroc_low, auroc_high.
        bins_raw: ``reliability_bins`` before calibration.
        bins_calibrated: ``reliability_bins`` after.
        resamples: Bootstrap resamples used.
    """

    observed: int
    positives: int
    calibrator: Calibrator
    threshold: float
    calibrated: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    raw: dict[str, float]
    groups: pd.DataFrame
    bins_raw: pd.DataFrame
    bins_calibrated: pd.DataFrame
    resamples: int


def evaluate_test(
    logits: np.ndarray,
    onset_logits: np.ndarray,
    targets: np.ndarray,
    onset_targets: np.ndarray,
    groups: Sequence[str],
    calibrator: Calibrator,
    threshold: float,
    cfg: Config,
) -> TestEvaluation:
    """Score the final model's test predictions, with bootstrap CIs.

    Args:
        logits: [N] ``any_onset`` logits of every node.
        onset_logits: [N, G] group logits.
        targets: [N] ``held_out_targets`` (NaN outside the test split).
        onset_targets: [N, G] the same for the groups.
        groups: Group names.
        calibrator: Fitted on validation (``load_calibration``).
        threshold: ``f1_threshold`` of the validation patients' calibrated probabilities.
        cfg: Project configuration.

    Returns:
        The evaluation.
    """
    evaluation = cfg.evaluation
    n_bins = evaluation.calibration_bins
    seen = ~np.isnan(targets)
    y, z = targets[seen], logits[seen]
    probabilities, raw = calibrator.apply(z), sigmoid(z)

    def statistic(yy: np.ndarray, pp: np.ndarray) -> dict[str, float]:
        return {**probability_metrics(yy, pp, n_bins), **threshold_metrics(yy, pp, threshold)}

    samples = bootstrap_statistics(y, probabilities, statistic, evaluation.bootstrap_resamples,
                                   cfg.seed)
    group_rows = []
    for g, group in enumerate(groups):
        group_seen = ~np.isnan(onset_targets[:, g])
        yg, sg = onset_targets[group_seen, g], onset_logits[group_seen, g]
        row = {"group": group, "observed": int(group_seen.sum()), "positives": int(yg.sum()),
               "auroc": np.nan, "auroc_low": np.nan, "auroc_high": np.nan}
        if 0 < row["positives"] < row["observed"]:
            boot = bootstrap_metrics(yg, {"final": sg}, evaluation.bootstrap_resamples, cfg.seed)
            row["auroc"] = float(roc_auc_score(yg, sg))
            row["auroc_low"], row["auroc_high"] = percentile_interval(boot.auroc["final"],
                                                                      evaluation.ci_level)
        group_rows.append(row)
    result = TestEvaluation(
        observed=len(y),
        positives=int(y.sum()),
        calibrator=calibrator,
        threshold=threshold,
        calibrated=statistic(y, probabilities),
        intervals={name: percentile_interval(samples[name], evaluation.ci_level)
                   for name in TEST_METRICS},
        raw=probability_metrics(y, raw, n_bins),
        groups=pd.DataFrame(group_rows),
        bins_raw=reliability_bins(y, raw, n_bins),
        bins_calibrated=reliability_bins(y, probabilities, n_bins),
        resamples=len(samples["auroc"]),
    )
    logger.info(
        "Test (%d patients, %d positives): AUROC %.4f %s, AUPRC %.4f, Brier %.4f, ECE %.4f",
        result.observed, result.positives, result.calibrated["auroc"],
        result.intervals["auroc"], result.calibrated["auprc"], result.calibrated["brier"],
        result.calibrated["ece"],
    )
    return result


def test_predictions_frame(
    patient_ids: Sequence[str],
    logits: np.ndarray,
    onset_logits: np.ndarray,
    targets: np.ndarray,
    onset_targets: np.ndarray,
    groups: Sequence[str],
    calibrator: Calibrator,
) -> pd.DataFrame:
    """The final model's predictions for the test patients with an observed ``any_onset``.

    Returns:
        patient_id, ``ANY_ONSET``, logit, probability (calibrated), and per group its
        label (``ONSET_PREFIX`` + group, NaN = masked) and score (sigmoid of its logit).
    """
    seen = ~np.isnan(targets)
    frame = pd.DataFrame({
        "patient_id": np.asarray(patient_ids)[seen],
        ANY_ONSET: targets[seen],
        "logit": logits[seen],
        "probability": calibrator.apply(logits[seen]),
    })
    for g, group in enumerate(groups):
        frame[ONSET_PREFIX + group] = onset_targets[seen, g]
        frame[f"score_{group}"] = sigmoid(onset_logits[seen, g])
    return frame


def require_trained_model(config_path: Path, cfg: Config) -> Calibrator:
    """Check that the saved final model matches the config; return its calibrator.

    The ``model`` section the weights were trained with (``config_used.yaml``) must
    equal the current one, and ``calibration.json`` must use ``calibration.method``.

    Args:
        config_path: The YAML ``cfg`` was loaded from.
        cfg: Project configuration.

    Returns:
        ``load_calibration`` of ``paths.calibration_json``.

    Raises:
        ValueError: If the model or the calibration is stale (re-run 06_train.py).
    """
    paths = cfg.paths
    missing = [p.name for p in (paths.model_pt, paths.config_used_yaml, paths.calibration_json)
               if not p.exists()]
    if missing:
        raise ValueError(f"missing {missing}; run scripts/06_train.py first")
    used = yaml.safe_load(paths.config_used_yaml.read_text(encoding="utf-8"))
    current = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if used.get("model") != current.get("model"):
        raise ValueError(f"{paths.model_pt.name} was trained with another model section; "
                         "re-run scripts/06_train.py")
    calibrator = load_calibration(paths.calibration_json)
    if calibrator.method != cfg.calibration.method:
        raise ValueError(f"{paths.calibration_json.name} uses {calibrator.method!r}, not "
                         f"calibration.method {cfg.calibration.method!r}; re-run 06_train.py")
    return calibrator


# --- Phase 7: embedding analysis -------------------------------------------------------


@dataclass(frozen=True)
class PatientContext:
    """What the embedding analysis shows about each node, in node order.

    Attributes:
        patient_ids: Node order.
        years: The training windows.
        phenotype: Phase 4 phenotype cluster (1 = largest; 0 = unclustered).
        age: Age at the cutoff, in years.
        sex: 1 = male, 0 = female.
        chronic: Binary CSR [N, n_codes]: chronic conditions active in the training
            windows (``features.chronic_presence``).
        codes: Column codes of ``chronic``.
        names: Code -> condition name without its semantic tag.
        nonempty: Bool [N, T]: the node's feature row in window t is non-zero.
        onsets: node, year, code: the first event of each chronic condition, for those
            first recorded in a training window.
    """

    patient_ids: tuple[str, ...]
    years: tuple[int, ...]
    phenotype: np.ndarray
    age: np.ndarray
    sex: np.ndarray
    chronic: sp.csr_matrix
    codes: tuple[str, ...]
    names: dict[str, str]
    nonempty: np.ndarray
    onsets: pd.DataFrame


def chronic_onsets(
    events: pd.DataFrame,
    codes: Sequence[str],
    years: Sequence[int],
    patient_ids: Sequence[str],
    min_duration_days: int,
) -> pd.DataFrame:
    """List the chronic conditions each patient first had in one of ``years``.

    A condition's onset is its first event for the patient; it counts if the patient
    has a chronic episode of it (``features.chronic_mask``) and the onset falls in
    one of ``years``.

    Args:
        events: ``Trajectories.events``.
        codes: Codes to consider (the feature vocabulary).
        years: The training windows.
        patient_ids: Node order.
        min_duration_days: ``features.chronic_min_duration_days``.

    Returns:
        node, year, code; sorted.
    """
    events = events[events["code"].isin(codes) & events["patient_id"].isin(patient_ids)]
    first = events.groupby(["patient_id", "code"], as_index=False)["date"].min()
    chronic = events.loc[chronic_mask(events, min_duration_days),
                         ["patient_id", "code"]].drop_duplicates()
    onsets = first.merge(chronic, on=["patient_id", "code"])
    onsets = onsets.assign(year=onsets["date"].dt.year.astype(int),
                           node=pd.Index(patient_ids).get_indexer(onsets["patient_id"]))
    onsets = onsets[onsets["year"].isin(years)]
    return (onsets[["node", "year", "code"]].astype({"code": str})
            .sort_values(["node", "year", "code"]).reset_index(drop=True))


def load_patient_context(cfg: Config, model_data: ModelData) -> PatientContext:
    """Collect the phenotypes, demographics, chronic conditions and onsets of every node.

    Args:
        cfg: Project configuration.
        model_data: ``train.load_model_data`` output.

    Returns:
        The context, in node order.

    Raises:
        ValueError: If the saved model cohort does not match ``model_data``.
    """
    trajectories = load_trajectories(cfg)
    presence, codes = chronic_presence(trajectories, cfg)
    cohort, rows = load_model_cohort(cfg)
    if cohort != model_data.patient_ids:
        raise ValueError("model_cohort.json does not match the model data")
    return PatientContext(
        patient_ids=model_data.patient_ids,
        years=model_data.years,
        phenotype=model_data.labels[PHENOTYPE].fillna(0).to_numpy(dtype=np.int64),
        age=model_data.demographics["age"].to_numpy(dtype=float),
        sex=model_data.demographics["sex"].to_numpy(dtype=float),
        chronic=sp.csr_matrix(presence[rows]),
        codes=tuple(codes),
        names={code: _SEMANTIC_TAG.sub("", description)
               for code, description in zip(trajectories.codes, trajectories.descriptions)},
        nonempty=np.column_stack([np.diff(sp.csr_matrix(X).indptr) > 0
                                  for X in model_data.matrices]),
        onsets=chronic_onsets(trajectories.events, codes, model_data.years,
                              model_data.patient_ids, cfg.features.chronic_min_duration_days),
    )


def bin_labels(edges: Sequence[float]) -> list[str]:
    """Label left-closed integer bins: [0, 18) -> "0-17", [1, 2) -> "1", [80, inf) -> "80+"."""
    labels = []
    for low, high in zip(edges[:-1], edges[1:]):
        if math.isinf(high):
            labels.append(f"{low:g}+")
        elif high - low == 1:
            labels.append(f"{low:g}")
        else:
            labels.append(f"{low:g}-{high - 1:g}")
    return labels


def categorize(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Return the ``bin_labels`` label of each value (object array)."""
    return np.asarray(pd.cut(values, bins=list(edges), right=False,
                             labels=bin_labels(edges)).astype(object))


def umap_coordinates(X: np.ndarray, cfg: Config) -> np.ndarray:
    """Project the rows of ``X`` to 2-D with UMAP, seeded with ``cfg.seed``.

    The neighbour search always uses nearest-neighbour descent, so no dense
    n x n distance matrix is built.

    Args:
        X: [n, d] embeddings.
        cfg: Project configuration (``embedding_analysis`` section).

    Returns:
        [n, 2].
    """
    import umap  # imported here: its numba compilation makes the import slow

    settings = cfg.embedding_analysis
    reducer = umap.UMAP(
        n_neighbors=settings.umap_n_neighbors, min_dist=settings.umap_min_dist,
        random_state=cfg.seed, n_jobs=1, force_approximation_algorithm=True,
    )
    return np.asarray(reducer.fit_transform(np.asarray(X, dtype=np.float32)), dtype=np.float64)


def pick_patients(eligible: np.ndarray, phenotype: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Draw ``n`` eligible nodes with ``seed``, from different phenotypes while possible.

    Args:
        eligible: Bool [N].
        phenotype: Cluster per node.
        n: Nodes to draw.
        seed: Random seed.

    Returns:
        Up to ``n`` node indices, in draw order.
    """
    order = np.random.default_rng(seed).permutation(np.flatnonzero(eligible))
    chosen: list[int] = []
    for node in order:
        if len(chosen) < n and phenotype[node] not in {phenotype[c] for c in chosen}:
            chosen.append(int(node))
    chosen += [int(node) for node in order if node not in chosen][: n - len(chosen)]
    return np.array(chosen, dtype=np.int64)


def trajectory_example_nodes(context: PatientContext, cfg: Config) -> np.ndarray:
    """The trajectory figure's patients (``pick_patients`` with ``cfg.seed``).

    Eligible: a non-empty feature row in every window and at least
    ``embedding_analysis.trajectory_min_onsets`` chronic-condition onsets in them.
    """
    settings = cfg.embedding_analysis
    onset_counts = np.bincount(context.onsets["node"], minlength=len(context.patient_ids))
    return pick_patients(
        context.nonempty.all(axis=1) & (onset_counts >= settings.trajectory_min_onsets),
        context.phenotype, settings.trajectory_patients, cfg.seed,
    )


def window_umap_coordinates(states: np.ndarray, nonempty: np.ndarray, cfg: Config) -> np.ndarray:
    """One UMAP of every non-empty patient-window's embedding; NaN for empty ones.

    Args:
        states: [N, T, d] per-window embeddings.
        nonempty: Bool [N, T] (``PatientContext.nonempty``).
        cfg: Project configuration.

    Returns:
        [N, T, 2].
    """
    coords = np.full((*states.shape[:2], 2), np.nan)
    coords[nonempty] = umap_coordinates(np.asarray(states, dtype=np.float64)[nonempty], cfg)
    return coords


def cluster_scores(
    representations: Mapping[str, np.ndarray],
    phenotype: np.ndarray,
    seed: int,
    working_memory_mb: int,
) -> pd.DataFrame:
    """Silhouette score and Davies-Bouldin index of each representation vs the phenotypes.

    Euclidean, over the clustered nodes. A last row scores the first representation
    against the phenotype labels shuffled with ``seed``: what chance gives.

    Args:
        representations: Name -> [N, d] array.
        phenotype: Cluster per node (0 = unclustered, left out).
        seed: Seed of the shuffle.
        working_memory_mb: Memory per block of the silhouette's distance computation.

    Returns:
        representation, dimensions, silhouette (higher = better separated, in [-1, 1]),
        davies_bouldin (lower = better, >= 0).
    """
    clustered = phenotype > 0
    labels = phenotype[clustered]
    first_name, first = next(iter(representations.items()))
    shuffled = np.random.default_rng(seed).permutation(labels)
    entries = [(name, X, labels) for name, X in representations.items()]
    entries.append((f"{first_name}, clusters shuffled (chance)", first, shuffled))
    rows = []
    for name, X, y in entries:
        X = np.asarray(X, dtype=np.float64)[clustered]
        with sklearn.config_context(working_memory=working_memory_mb):
            silhouette = float(silhouette_score(X, y))
        rows.append({"representation": name, "dimensions": X.shape[1], "silhouette": silhouette,
                     "davies_bouldin": float(davies_bouldin_score(X, y))})
    return pd.DataFrame(rows)


def chronic_jaccard(chronic: sp.spmatrix, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Jaccard index of the chronic conditions of node pairs (first[i], second[i]); 0 if both have none."""
    binary = sp.csr_matrix(chronic, dtype=np.float64, copy=True)
    binary.data[:] = 1.0
    shared = np.asarray(binary[first].multiply(binary[second]).sum(axis=1)).ravel()
    sizes = np.diff(binary.indptr)
    union = sizes[first] + sizes[second] - shared
    return np.divide(shared, union, out=np.zeros(len(first)), where=union > 0)


def neighbour_agreement(
    representations: Mapping[str, tuple[np.ndarray, str]],
    context: PatientContext,
    k: int,
    seed: int,
    working_memory_mb: int,
) -> pd.DataFrame:
    """How much nearest neighbours in each representation share chronic conditions.

    For every node with a chronic condition: the mean Jaccard index of its chronic
    conditions with each of its ``k`` nearest neighbours, and the share of those in
    its phenotype cluster. A last row uses ``k`` random other patients (chance).

    Args:
        representations: Name -> ([N, d] array, sklearn metric).
        context: ``load_patient_context`` output.
        k: Neighbours per node.
        seed: Seed of the random patients.
        working_memory_mb: Memory per block of the neighbour search.

    Returns:
        representation, metric, patients, neighbour_jaccard, same_phenotype.
    """
    queries = np.flatnonzero(np.diff(context.chronic.indptr) > 0)
    n = context.chronic.shape[0]

    def row(name: str, metric: str, neighbours: np.ndarray) -> dict[str, Any]:
        first, second = np.repeat(queries, neighbours.shape[1]), neighbours.ravel()
        return {"representation": name, "metric": metric, "patients": len(queries),
                "neighbour_jaccard": float(chronic_jaccard(context.chronic, first, second).mean()),
                "same_phenotype": float(np.mean(context.phenotype[first]
                                                == context.phenotype[second]))}

    rows = []
    for name, (X, metric) in representations.items():
        search = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="brute").fit(X)
        with sklearn.config_context(working_memory=working_memory_mb):
            neighbours = search.kneighbors(return_distance=False)  # never the node itself
        rows.append(row(name, metric, neighbours[queries]))
    draws = np.random.default_rng(seed).integers(0, n - 1, size=(len(queries), k))
    rows.append(row("Random other patients (chance)", "-", draws + (draws >= queries[:, None])))
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class EmbeddingAnalysis:
    """Phase 7 analysis of the final model's embeddings.

    Attributes:
        trajectory_umap: [N, 2] UMAP of the trajectory embeddings h.
        window_umap: [N, T, 2] one UMAP fitted on every non-empty patient-window's
            Z_t; NaN for empty patient-windows (whose Z_t is one shared constant).
        scores: ``cluster_scores`` output.
        agreement: ``neighbour_agreement`` output.
        trajectory_nodes: The trajectory figure's patients.
        query_nodes: The nearest-neighbour check's patients.
        neighbours: [len(query_nodes), n_neighbours] their nearest neighbours in h.
        distances: The Euclidean distances to them.
        pc_variance: Share of h's variance on each principal component, largest first.
        median_distance: Median Euclidean distance in h between a patient and a random
            other patient (the scale of ``distances``).
    """

    trajectory_umap: np.ndarray
    window_umap: np.ndarray
    scores: pd.DataFrame
    agreement: pd.DataFrame
    trajectory_nodes: np.ndarray
    query_nodes: np.ndarray
    neighbours: np.ndarray
    distances: np.ndarray
    pc_variance: np.ndarray
    median_distance: float


def analyse_embeddings(
    window_embeddings: np.ndarray,
    trajectory_embeddings: np.ndarray,
    spectral: np.ndarray,
    final_features: sp.spmatrix,
    context: PatientContext,
    cfg: Config,
) -> EmbeddingAnalysis:
    """Cluster separation, neighbour checks and UMAP projections of the embeddings.

    Args:
        window_embeddings: Z, [N, T, d], of the final model.
        trajectory_embeddings: h, [N, d].
        spectral: ``load_spectral_embedding`` output (a reference representation).
        final_features: The final window's [N, F] features (another reference).
        context: ``load_patient_context`` output.
        cfg: Project configuration.

    Returns:
        The analysis.
    """
    settings = cfg.embedding_analysis
    memory = cfg.graph.knn_working_memory_mb
    h = np.asarray(trajectory_embeddings, dtype=np.float64)
    Z = np.asarray(window_embeddings, dtype=np.float64)
    year = context.years[-1]
    final_window = Z[:, -1, :]
    features = sp.csr_matrix(final_features).toarray()  # N x F, not N x N
    scores = cluster_scores(
        {"Trajectory embedding h": h, f"Window embedding Z_{year}": final_window,
         f"Spectral embedding of the {year} graph": spectral,
         f"{year} window features (model input)": features},
        context.phenotype, cfg.seed, memory,
    )
    agreement = neighbour_agreement(
        {"Trajectory embedding h": (h, "euclidean"),
         f"Window embedding Z_{year}": (final_window, "euclidean"),
         f"{year} window features (the graph's neighbours)": (features, "cosine"),
         f"Spectral embedding of the {year} graph": (spectral, "euclidean")},
        context, settings.n_neighbours, cfg.seed, memory,
    )
    logger.info("Cluster separation:\n%s\nNeighbour agreement:\n%s", scores.to_string(index=False),
                agreement.to_string(index=False))
    trajectory_umap = umap_coordinates(h, cfg)
    window_umap = window_umap_coordinates(Z, context.nonempty, cfg)
    trajectory_nodes = trajectory_example_nodes(context, cfg)
    query_nodes = pick_patients(
        np.diff(context.chronic.indptr) >= settings.neighbour_min_conditions,
        context.phenotype, settings.neighbour_patients, cfg.seed,
    )
    k = settings.n_neighbours
    search = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(h)
    with sklearn.config_context(working_memory=memory):
        found_distances, found = search.kneighbors(h[query_nodes])
    others = found != query_nodes[:, None]
    neighbours = np.array([row[keep][:k] for row, keep in zip(found, others)])
    distances = np.array([row[keep][:k] for row, keep in zip(found_distances, others)])
    singular = np.linalg.svd(h - h.mean(axis=0), compute_uv=False)  # N x d, not N x N
    pc_variance = singular**2 / np.sum(singular**2)
    draws = np.random.default_rng(cfg.seed).integers(0, len(h) - 1, size=len(h))
    partners = draws + (draws >= np.arange(len(h)))  # anyone but the patient
    median_distance = float(np.median(np.linalg.norm(h - h[partners], axis=1)))
    logger.info("h: principal-component variance shares %s; median random distance %.4f",
                np.round(pc_variance[:3], 3).tolist(), median_distance)
    return EmbeddingAnalysis(trajectory_umap, window_umap, scores, agreement, trajectory_nodes,
                             query_nodes, neighbours, distances, pc_variance, median_distance)


def phenotype_names(context: PatientContext, cfg: Config) -> dict[int, str]:
    """Each phenotype cluster's most over-represented chronic condition (``graph.cluster_phenotypes``)."""
    clustered = context.phenotype > 0
    top = cluster_phenotypes(context.chronic[clustered], context.phenotype[clustered],
                             context.codes, 1, cfg.graph.phenotype_min_patients)
    names = {int(c): "no common chronic condition" for c in np.unique(context.phenotype[clustered])}
    names.update({int(row.cluster): context.names[row.code] for row in top.itertuples()})
    return names


def plot_umap_facets(
    coords: np.ndarray,
    categories: np.ndarray,
    order: Sequence[Any],
    titles: Mapping[Any, str],
    title: str,
    path: Path,
    cfg: Config,
) -> None:
    """Save small multiples of a 2-D projection: one panel per category, highlighted.

    Each panel draws every point in a light context gray and the category's points
    in series blue, so one hue carries the identity (any number of categories).

    Args:
        coords: [n, 2] coordinates.
        categories: [n] category of each point.
        order: Categories, one panel each, in order.
        titles: Category -> panel title.
        title: Figure title.
        path: Output PNG.
        cfg: Project configuration.
    """
    dpi = cfg.summary.figure_dpi
    ncols = 4 if len(order) > 6 else 3
    nrows = math.ceil(len(order) / ncols)
    with matplotlib.rc_context(chart_style(dpi, cfg.summary.figure_save_dpi)):
        fig = Figure(figsize=(_PANEL_IN * ncols, _PANEL_IN * nrows + 0.5), layout="constrained")
        axes = fig.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True)
        for ax, category in zip(axes.flat, order):
            inside = categories == category
            for mask, color in ((~inside, _CONTEXT_POINT), (inside, _SERIES_COLOR)):
                ax.scatter(coords[mask, 0], coords[mask, 1], s=_DOT_AREA_PT2, color=color,
                           linewidths=0, rasterized=True)
            ax.set_title(titles[category], fontsize="medium")
            _bare(ax)
        for ax in axes.flat[len(order):]:
            ax.set_visible(False)
        fig.suptitle(title, x=0.01, ha="left")
        save_figure(fig, path)


def plot_trajectories(analysis: EmbeddingAnalysis, context: PatientContext, path: Path,
                      cfg: Config) -> None:
    """Save each example patient's window embeddings Z_t as a path through the UMAP.

    See ``plot_window_paths``.

    Args:
        analysis: ``analyse_embeddings`` output.
        context: ``load_patient_context`` output.
        path: Output PNG.
        cfg: Project configuration.
    """
    plot_window_paths(analysis.window_umap, analysis.trajectory_nodes, context, path, cfg,
                      "Z_t", "Window embeddings")


def plot_window_paths(
    window_umap: np.ndarray,
    nodes: np.ndarray,
    context: PatientContext,
    path: Path,
    cfg: Config,
    symbol: str,
    description: str,
) -> None:
    """Save each patient's per-window embeddings as a path through one UMAP.

    One panel per patient over the cloud of every patient-window: arrows run from
    the first window to the last; orange points mark years with a new chronic
    condition, named beside them (up to ``max_onset_labels``, the rest as "+n").

    Args:
        window_umap: [N, T, 2] (``window_umap_coordinates``).
        nodes: The patients shown.
        context: ``load_patient_context`` output.
        path: Output PNG.
        cfg: Project configuration.
        symbol: Name of the per-window embedding, e.g. "Z_t".
        description: What it is, e.g. "Window embeddings".
    """
    settings = cfg.embedding_analysis
    dpi = cfg.summary.figure_dpi
    px = _POINTS_PER_INCH / dpi
    years = context.years
    background = window_umap[context.nonempty]
    span = np.ptp(background, axis=0)
    ncols = 3
    nrows = math.ceil((len(nodes) + 1) / ncols)
    with matplotlib.rc_context(chart_style(dpi, cfg.summary.figure_save_dpi)):
        fig = Figure(figsize=(5.0 * ncols, 4.8 * nrows), layout="constrained")
        axes = fig.subplots(nrows, ncols, squeeze=False)
        for ax, node in zip(axes.flat, nodes):
            points = window_umap[node]
            labels = onset_labels(context, node, settings.max_onset_labels,
                                  settings.onset_label_chars)
            ax.scatter(background[:, 0], background[:, 1], s=_DOT_AREA_PT2, color=_CONTEXT_POINT,
                       linewidths=0, rasterized=True)
            for t in range(len(years) - 1):
                ax.annotate("", xy=points[t + 1], xytext=points[t],
                            arrowprops={"arrowstyle": "-|>", "color": _SERIES_COLOR,
                                        "linewidth": _LINE_PX * px, "shrinkA": 5, "shrinkB": 5})
            colors = [_ACCENT_COLOR if year in labels else _SERIES_COLOR for year in years]
            ax.scatter(points[:, 0], points[:, 1], s=(_MARKER_PX * px) ** 2, c=colors,
                       edgecolors=_SURFACE, linewidths=_RING_PX * px, zorder=3)
            for t, year in enumerate(years):
                if year in labels or t in (0, len(years) - 1):
                    text = f"{year}: {labels[year]}" if year in labels else str(year)
                    ax.annotate(text, xy=points[t], xytext=(_LABEL_OFFSET_PT, _LABEL_OFFSET_PT),
                                textcoords="offset points", fontsize="small",
                                color=_TEXT_SECONDARY, zorder=4)
            center = (points.min(axis=0) + points.max(axis=0)) / 2
            half = np.maximum(np.ptp(points, axis=0) * 0.65, span * _ZOOM_MIN_SHARE / 2)
            ax.set_xlim(center[0] - half[0], center[0] + half[0])
            ax.set_ylim(center[1] - half[1], center[1] + half[1])
            ax.set_title(_patient_title(context, node), fontsize="medium")
            _bare(ax)
        key = axes.flat[len(nodes)]
        key.axis("off")
        key.legend(
            handles=[
                Line2D([], [], linestyle="none", marker="o", color=_CONTEXT_POINT,
                       label="every patient-window"),
                Line2D([], [], color=_SERIES_COLOR, marker="o", markeredgecolor=_SURFACE,
                       label=f"the patient's {symbol}, arrows {years[0]} to {years[-1]}"),
                Line2D([], [], linestyle="none", marker="o", color=_ACCENT_COLOR,
                       markeredgecolor=_SURFACE, label="a year with a new chronic condition"),
            ],
            loc="center left", frameon=False,
        )
        for ax in axes.flat[len(nodes) + 1:]:
            ax.set_visible(False)
        fig.suptitle(
            f"{description} {symbol} of {len(nodes)} patients, {years[0]}-{years[-1]} "
            "(one UMAP of every non-empty patient-window)", x=0.01, ha="left",
        )
        save_figure(fig, path)


def onset_labels(
    context: PatientContext, node: int, max_labels: int, max_chars: int
) -> dict[int, str]:
    """Year -> the node's new chronic conditions that year, e.g. "Prediabetes, Gout +1"."""
    labels = {}
    for year, part in context.onsets[context.onsets["node"] == node].groupby("year"):
        names = [_cut(context.names[code], max_chars) for code in part["code"]]
        rest = len(names) - max_labels
        labels[int(year)] = ", ".join(names[:max_labels]) + (f" +{rest}" if rest > 0 else "")
    return labels


def plot_embeddings(analysis: EmbeddingAnalysis, context: PatientContext, cfg: Config) -> None:
    """Save the three UMAP small-multiple figures of h and the trajectory figure."""
    paths, settings = cfg.paths, cfg.embedding_analysis
    coords = analysis.trajectory_umap
    clusters = sorted(int(c) for c in np.unique(context.phenotype) if c > 0)
    names = phenotype_names(context, cfg)
    phenotype = context.phenotype.astype(object)
    plot_umap_facets(
        coords, phenotype, clusters,
        {c: f"Cluster {c} (n = {int((context.phenotype == c).sum()):,})\n"
            f"{_cut(names[c], settings.onset_label_chars)}" for c in clusters},
        "UMAP of the trajectory embeddings h, by Phase 4 phenotype cluster "
        "(top over-represented chronic condition)", paths.umap_phenotype_fig, cfg,
    )
    age_bins = cfg.summary.age_bins
    ages = categorize(context.age, age_bins)
    plot_umap_facets(
        coords, ages, bin_labels(age_bins),
        {label: f"Age {label} (n = {int((ages == label).sum()):,})"
         for label in bin_labels(age_bins)},
        f"UMAP of the trajectory embeddings h, by age on {cfg.labels.cutoff_date:%Y-%m-%d}",
        paths.umap_age_fig, cfg,
    )
    count_bins = settings.chronic_count_bins
    counts = categorize(np.diff(context.chronic.indptr).astype(float), count_bins)
    plot_umap_facets(
        coords, counts, bin_labels(count_bins),
        {label: f"{label} chronic condition{'' if label == '1' else 's'} "
                f"(n = {int((counts == label).sum()):,})" for label in bin_labels(count_bins)},
        f"UMAP of the trajectory embeddings h, by number of chronic conditions "
        f"{context.years[0]}-{context.years[-1]}", paths.umap_chronic_fig, cfg,
    )
    plot_trajectories(analysis, context, paths.trajectory_umap_fig, cfg)


# --- Phase 7: reports ------------------------------------------------------------------


def write_results(
    table: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    group_table: pd.DataFrame,
    cv: CrossValidation,
    by_split: pd.DataFrame,
    test: TestEvaluation,
    analysis: EmbeddingAnalysis,
    context: PatientContext,
    val_auroc: float,
    cfg: Config,
) -> str:
    """Write every Phase 7 result to ``paths.results_md`` and the table to ``paths.results_csv``.

    Args:
        table: ``cv_results`` output.
        fold_metrics: ``cv_fold_metrics`` output.
        group_table: ``cv_group_results`` output.
        cv: The cross-validation.
        by_split: ``cv_by_split`` output.
        test: ``evaluate_test`` output.
        analysis: ``analyse_embeddings`` output.
        context: ``load_patient_context`` output.
        val_auroc: The final model's validation AUROC (``06_train.py``).
        cfg: Project configuration.

    Returns:
        The markdown written.
    """
    paths, evaluation, settings = cfg.paths, cfg.evaluation, cfg.cross_validation
    level = f"{evaluation.ci_level:.0%}"
    predictions, fits = cv.predictions, cv.fits
    n, positives = len(predictions), int(predictions[ANY_ONSET].sum())
    rate = positives / n
    by_key = table.set_index("key")
    full = by_key.loc[FULL_MODEL]
    models = {model.key: model for model in cv.models}
    years, model_cfg = context.years, cfg.model
    first_fits = fits.drop_duplicates(["fold", "key"]).drop_duplicates("fold")
    splits = by_split.set_index(SPLIT)

    def figure(path: Path) -> str:
        return Path(os.path.relpath(path, paths.results_md.parent)).as_posix()

    def span(column: str) -> str:
        low, high = int(first_fits[column].min()), int(first_fits[column].max())
        return f"{low:,}" if low == high else f"{low:,}-{high:,}"

    main_rows = [
        [str(rank), str(row.number),
         f"**{row.model}**" if row.key == FULL_MODEL else row.model, row.inputs,
         f"{row.fold_auroc_mean:.3f} +/- {row.fold_auroc_sd:.3f}",
         f"{row.fold_auprc_mean:.3f} +/- {row.fold_auprc_sd:.3f}",
         _with_ci(row.oof_auroc, row.oof_auroc_low, row.oof_auroc_high),
         _with_ci(row.oof_auprc, row.oof_auprc_low, row.oof_auprc_high),
         "reference" if row.key == FULL_MODEL
         else _with_ci(row.delta_auroc, row.delta_auroc_low, row.delta_auroc_high, signed=True),
         "" if row.key == FULL_MODEL else f"{int(row.folds_full_better)} / {settings.folds}",
         f"{row.oof_brier:.3f}"]
        for rank, row in enumerate(table.itertuples(), start=1)
    ]
    logistic = table[table["key"].map(lambda key: models[key].kind == LOGISTIC)]
    best_simple = logistic.iloc[0]
    comparisons = [(question, key) for question, key in ABLATIONS] + [
        ("Is it better than the best non-GCN model?", best_simple.key),
        ("Is it better than age and sex alone?", DEMOGRAPHICS_MODEL),
    ]
    ablation_rows = []
    for question, key in comparisons:
        row = by_key.loc[key]
        ablation_rows.append([
            question, f"8 vs {row.number}: {row.model} ({row.inputs})",
            f"{full.oof_auroc:.3f}", f"{row.oof_auroc:.3f}",
            _with_ci(row.delta_auroc, row.delta_auroc_low, row.delta_auroc_high, signed=True),
            f"{int(row.folds_full_better)} / {settings.folds}", _reading(row),
        ])
    numbers = [m.number for m in cv.models]
    fold_rows = []
    for fold, part in fold_metrics.groupby("fold", sort=True):
        first = part.iloc[0]
        aurocs = part.set_index("key")["auroc"]
        fold_rows.append([str(fold), f"{int(first.observed)} ({int(first.positives)})",
                          *(f"{aurocs[m.key]:.3f}" for m in cv.models)])
    group_rows = []
    flagged = set(cfg.labels.borderline_groups) | set(cfg.labels.near_threshold_groups)
    for group, part in group_table.groupby("group", sort=False):
        first = part.iloc[0]
        cells = part.set_index("key")
        group_rows.append([
            group + (" *" if group in flagged else ""),
            f"{int(first.positives)} / {int(first.observed):,}",
            *(_short_ci(cells.loc[m.key, "auroc"], cells.loc[m.key, "auroc_low"],
                        cells.loc[m.key, "auroc_high"]) for m in cv.models),
        ])
    seeds = ", ".join(str(seed) for seed in settings.seeds)
    n_clusters = len(np.unique(context.phenotype[context.phenotype > 0]))
    test_counts = test.groups["positives"]
    fit_rows = [
        [f"{models[key].number}. {models[key].name}",
         f"{part['best_epoch'].mean():.0f} ({int(part['best_epoch'].min())}-"
         f"{int(part['best_epoch'].max())})" if models[key].kind == GCN else "",
         f"{part['inner_val_auroc'].mean():.3f}",
         f"{part.drop_duplicates('fold')['slope'].min():.2f} to "
         f"{part.drop_duplicates('fold')['slope'].max():.2f}",
         "1 (itself)" if key == DEMOGRAPHICS_MODEL else
         f"{by_key.loc[key, 'age_sex_rank_corr_mean']:.2f} "
         f"({by_key.loc[key, 'age_sex_rank_corr_min']:.2f} to "
         f"{by_key.loc[key, 'age_sex_rank_corr_max']:.2f})"]
        for key, part in fits.groupby("key", sort=False)
    ]
    reduced = [models[key] for key in by_key.index if key != DEMOGRAPHICS_MODEL
               and by_key.loc[key, "age_sex_rank_corr_mean"] >= settings.demographic_rank_correlation]
    split_rows = [
        [name, f"{int(splits.loc[name, 'patients']):,} ({int(splits.loc[name, 'positives'])})",
         *(f"{splits.loc[name, m.key]:.3f}" for m in cv.models)]
        for name in splits.index
    ]

    prevalence = test.positives / test.observed
    c, ci, raw = test.calibrated, test.intervals, test.raw
    test_rows = [
        ["AUROC", _with_ci(c["auroc"], *ci["auroc"]), f"{raw['auroc']:.3f}", "0.500 (chance)"],
        ["AUPRC", _with_ci(c["auprc"], *ci["auprc"]), f"{raw['auprc']:.3f}",
         f"{prevalence:.3f} (a random score)"],
        ["Brier score", _with_ci(c["brier"], *ci["brier"]), f"{raw['brier']:.3f}",
         f"{prevalence * (1 - prevalence):.3f} (always the observed rate)"],
        ["ECE", _with_ci(c["ece"], *ci["ece"]), f"{raw['ece']:.3f}", "0 (perfect)"],
        ["Mean predicted probability", f"{c['mean_predicted']:.3f}",
         f"{raw['mean_predicted']:.3f}", f"{prevalence:.3f} (observed rate)"],
        [f"F1 at threshold {test.threshold:.3f}", _with_ci(c["f1"], *ci["f1"]), "",
         f"{2 * prevalence / (1 + prevalence):.3f} (predict an onset for everyone)"],
        ["Precision at that threshold", _with_ci(c["precision"], *ci["precision"]), "", ""],
        ["Recall at that threshold", _with_ci(c["recall"], *ci["recall"]), "", ""],
    ]
    test_group_rows = [
        [row.group + (" *" if row.group in flagged else ""),
         f"{row.positives} / {row.observed}",
         "n/a" if np.isnan(row.auroc) else _with_ci(row.auroc, row.auroc_low, row.auroc_high)]
        for row in test.groups.itertuples()
    ]
    score_rows = [[row.representation, str(row.dimensions), f"{row.silhouette:.3f}",
                   f"{row.davies_bouldin:.2f}"] for row in analysis.scores.itertuples()]
    agreement_rows = [[row.representation, row.metric, f"{row.patients:,}",
                       f"{row.neighbour_jaccard:.3f}", f"{row.same_phenotype:.1%}"]
                      for row in analysis.agreement.itertuples()]
    trajectory_rows = []
    for node in analysis.trajectory_nodes:
        onsets = context.onsets[context.onsets["node"] == node]
        for year, part in onsets.groupby("year"):
            trajectory_rows.append([_short_id(context.patient_ids[node]),
                                    _demographic_text(context, node),
                                    str(context.phenotype[node]), str(year),
                                    ", ".join(context.names[code] for code in part["code"])])
    n_windows = int(context.nonempty.sum())
    analysis_cfg = cfg.embedding_analysis
    k = analysis_cfg.n_neighbours

    lines = [
        "# Results",
        "",
        f"Built by `scripts/07_evaluate.py`. Primary target `{ANY_ONSET}`: a new onset in any "
        f"of the {len(cv.groups)} onset groups from {cfg.labels.cutoff_date:%Y-%m-%d} to the end "
        f"of {cfg.labels.cutoff_date.year + cfg.labels.horizon_years - 1} (see "
        f"`labels_summary.md`). Every model reads only the windows {years[0]}-{years[-1]}, "
        "before the cutoff. Every GCN keeps the hyperparameters tuned in Phase 6b (hidden/"
        f"embedding {model_cfg.hidden_dim}/{model_cfg.embed_dim}, dropout {model_cfg.dropout}, "
        f"weight decay {model_cfg.weight_decay}, lr {model_cfg.lr}, smoothness loss "
        f"{model_cfg.smoothness_weight:g}, DropEdge {model_cfg.drop_edge:g}) and has age and sex "
        "in its heads; nothing here was tuned. Plain-language findings: "
        f"`{paths.evaluation_summary_md.name}`.",
        "",
        f"## Main comparison: {settings.folds}-fold cross-validation",
        "",
        f"- **Patients:** all {n:,} patients with an observed `{ANY_ONSET}` ({positives:,} "
        f"positives, {rate:.1%}), in {settings.folds} folds stratified by phenotype cluster x "
        f"`{ANY_ONSET}` (seed {cfg.seed}); held-out folds of {span('held_out')} patients.",
        f"- **Within each fold:** {settings.inner_val:.0%} of the training folds, stratified "
        f"the same way, form an inner validation set ({span('inner_val')} patients). It picks "
        "each GCN's epoch (early stopping on its AUROC) and fits every model's Platt scaling. "
        f"Every model is fitted on the other {1 - settings.inner_val:.0%} ({span('inner_train')} "
        "patients). The held-out fold is scored once.",
        f"- **GCN models (5-8)** are trained with seeds {seeds} in every fold and their logits "
        "averaged before calibration. Training is transductive, as in Phase 6: all "
        f"{len(context.patient_ids):,} patients are graph nodes with their pre-cutoff features "
        "in every fold; only the inner training labels enter a loss. Model 7's graphs are each "
        f"window's graph rewired by {settings.rewire_swaps_per_edge} degree-preserving double "
        "edge swaps per edge (seed "
        f"{cfg.seed}): every patient keeps their number of neighbours, but who is linked to whom "
        "is random.",
        f"- **Logistic regressions (1-4):** L2 penalty, C = {cfg.baselines.C}, class weights "
        f"{'balanced' if cfg.baselines.balanced_class_weight else 'equal'}, as in "
        "`baselines_summary.md`. Model 4's eigenvectors are standardized with the inner training "
        "patients' mean and sd.",
        "- **Out-of-fold (OOF) predictions:** each patient's Platt-calibrated probability from "
        "the fold that held them out. Pooled metrics use all of them together; per-fold metrics "
        "average the folds.",
        "- **Optimism:** the GCN hyperparameters were chosen on the original validation split "
        "(Phase 6b), whose labelled patients are part of these folds, so the cross-validated "
        "GCN estimates (models 5-8) are slightly optimistic. The logistic regressions were not "
        "tuned.",
        "",
        markdown_table(
            ["Rank", "#", "Model", "Inputs", "Fold AUROC, mean +/- sd",
             "Fold AUPRC, mean +/- sd", f"Pooled OOF AUROC ({level} CI)",
             f"Pooled OOF AUPRC ({level} CI)", f"Full TemporalGCN minus this, AUROC ({level} CI)",
             "Folds full better", "Brier"],
            main_rows, "rrllrrrrrrr",
        ),
        "",
        f"Sorted by pooled OOF AUROC. Fold columns: mean and sd (ddof 1) over the "
        f"{settings.folds} held-out folds. Pooled columns: percentile bootstrap over the {n:,} "
        f"patients, {evaluation.bootstrap_resamples:,} resamples shared by every model, so the "
        "differences are paired. These intervals reflect which patients were sampled, not the "
        "variation from retraining the models. AUPRC of a random score: "
        f"{rate:.3f}. Brier: of the calibrated OOF probabilities; always predicting the observed "
        f"rate scores {rate * (1 - rate):.3f}. Also in `{paths.results_csv.name}` (with ECE).",
        "",
        f"![Pooled out-of-fold AUROC]({figure(paths.cv_auroc_fig)})",
        "",
        "## Ablations",
        "",
        f"Each row compares the full TemporalGCN (model 8) with one other model on the same "
        f"{n:,} patients (paired bootstrap of the pooled OOF AUROC).",
        "",
        markdown_table(
            ["Question", "Comparison", "Full AUROC", "Other AUROC",
             f"Full minus other ({level} CI)", "Folds full better", "Reading"],
            ablation_rows, "llrrrrl",
        ),
        "",
        f"Model 5 still sees the past in aggregate: its {years[-1]} window's TF-IDF block is a "
        "time-decayed sum of every earlier window and its transition block holds every "
        "transition before the cutoff. \"Time\" here is the sequence of windows and the GRU over "
        "them.",
        "",
        *([
            "**Models that reduce to age and sex.** "
            + _join([f"Model {m.number} ({m.name})"
                     for m in sorted(reduced, key=lambda m: m.number)])
            + (" ranks" if len(reduced) == 1 else " rank")
            + " the held-out patients almost exactly as age and sex alone do (mean "
            f"within-fold Spearman correlation with model 1 of at least "
            f"{settings.demographic_rank_correlation}; see Fits): with the hyperparameters fixed at "
            "the full model's tuned setting, they learned little beyond the demographics in their "
            "heads. A comparison with such a model says how much the full model adds to age and "
            "sex, not how much time or the real graph adds to a working static or random-graph "
            "model.",
            "",
        ] if reduced else []),
        "## Per-fold AUROC",
        "",
        markdown_table(["Fold", "Patients (positives)", *(str(x) for x in numbers)], fold_rows,
                       "rr" + "r" * len(numbers)),
        "",
        "Columns are the model numbers of the main table.",
        "",
        "## Fits",
        "",
        markdown_table(["Model", "Best epoch, mean (range)", "Inner validation AUROC, mean",
                        "Platt slope, range over folds",
                        "Rank correlation with model 1, mean (range over folds)"],
                       fit_rows, "lrrrr"),
        "",
        f"GCN rows: one fit per fold and seed ({settings.folds} x {len(settings.seeds)} per "
        "model); their inner validation AUROC is each seed's at its best epoch. Logistic rows: "
        "one fit per fold. The Platt slope is fitted per fold on the inner validation logits. "
        "Rank correlation: Spearman correlation between the model's and model 1's (age and sex "
        "only) out-of-fold logits, within each held-out fold.",
        "",
        "## Onset groups (secondary targets, pooled OOF AUROC)",
        "",
        "Pooled out-of-fold AUROC of each group's score with its "
        f"{level} CI, on the patients with that group's label observed. Group scores are "
        "uncalibrated (too few inner validation positives per group to fit Platt scaling), "
        "so pooling mixes the folds' scales; the logistic regressions fit one model per group.",
        "",
        markdown_table(["Group", "Positives / observed", *(str(x) for x in numbers)], group_rows,
                       "lr" + "r" * len(numbers)),
        "",
        "\\* borderline or near the review threshold of "
        f"{cfg.labels.min_onsets} onsets (see `labels_summary.md`).",
        "",
        "## Held-out test set (single shot)",
        "",
        "The final model of `06_train.py` (trained on the training split, epoch chosen on the "
        f"validation split, Platt scaling {test.calibrator.describe()} fitted on it; "
        f"`{paths.calibration_json.name}`), scored once on the {test.observed:,} test patients "
        f"with an observed `{ANY_ONSET}` ({test.positives} positives, {prevalence:.1%}). No "
        "choice was made with test labels. Percentile bootstrap over the test patients, "
        f"{test.resamples:,} resamples. The F1 threshold maximizes F1 on the validation "
        "patients' calibrated probabilities.",
        "",
        markdown_table(["Metric", f"Test, Platt-calibrated ({level} CI)", "Test, uncalibrated",
                        "Reference"], test_rows, "lrrl"),
        "",
        f"The same model's validation AUROC was {val_auroc:.3f}; the validation split chose its "
        "hyperparameters and its epoch, so that number is optimistic.",
        "",
        "### Out-of-fold AUROC by original split",
        "",
        "The cross-validation's out-of-fold AUROC on the patients of each Phase 5 split "
        "(columns: model numbers). The patients are the same as in the single-split numbers; "
        "only the training patients and seeds differ.",
        "",
        markdown_table(["Phase 5 split", "Patients (positives)", *(str(x) for x in numbers)],
                       split_rows, "lr" + "r" * len(numbers)),
        "",
        f"On the {test.observed} test patients, the full model scores {c['auroc']:.3f} when "
        f"trained once on the training split and {splits.loc[TEST, FULL_MODEL]:.3f} out of fold; "
        f"on the validation patients, {val_auroc:.3f} and {splits.loc[VAL, FULL_MODEL]:.3f}. "
        "The same patients are ranked quite differently by models trained on other patients "
        "and seeds, so a single split's AUROC is a noisy estimate; the pooled cross-validated "
        f"AUROC ({full.oof_auroc:.3f}) uses every labelled patient.",
        "",
        f"![Test reliability diagram]({figure(paths.test_reliability_fig)})",
        "",
        "Onset groups on the test set (secondary head, uncalibrated scores):",
        "",
        markdown_table(["Group", "Test positives / observed", f"AUROC ({level} CI)"],
                       test_group_rows, "lrr"),
        "",
        f"With {int(test_counts.min())}-{int(test_counts.max())} test positives per group, "
        "these AUROCs are noisy.",
        "",
        "## Embedding analysis (final model)",
        "",
        "Embeddings of the final model: the trajectory embedding h (the GRU's final state, "
        f"{model_cfg.embed_dim} numbers per patient) and the window embeddings Z_t (the GCN's "
        "output per window). Neither includes age or sex, which enter only the prediction heads.",
        "",
        "### UMAP of the trajectory embeddings",
        "",
        f"UMAP (n_neighbors {analysis_cfg.umap_n_neighbors}, min_dist "
        f"{analysis_cfg.umap_min_dist}, seed {cfg.seed}) of h for all "
        f"{len(context.patient_ids):,} patients. Each panel highlights one group in blue over "
        "everyone else in gray.",
        "",
        f"![UMAP by phenotype]({figure(paths.umap_phenotype_fig)})",
        "",
        f"![UMAP by age group]({figure(paths.umap_age_fig)})",
        "",
        f"![UMAP by number of chronic conditions]({figure(paths.umap_chronic_fig)})",
        "",
        "### Cluster separation",
        "",
        "Silhouette score (-1 to 1, higher = clusters better separated) and Davies-Bouldin "
        f"index (>= 0, lower = better) of each representation against the {n_clusters} Phase 4 "
        "phenotype clusters, Euclidean. The clusters were defined by k-means on the first "
        f"Laplacian eigenvectors of the {years[-1]} graph (rows normalized), so the spectral row "
        "is their reference; it stays low because the graph is one continuum (see "
        "`graphs_summary.md`).",
        "",
        markdown_table(["Representation", "Dimensions", "Silhouette", "Davies-Bouldin"],
                       score_rows, "lrrr"),
        "",
        "**Spread of h.** Its first three principal components hold "
        f"{', '.join(f'{share:.1%}' for share in analysis.pc_variance[:3])} of its variance, and "
        "the median distance between a patient and a random other patient is "
        f"{analysis.median_distance:.4f} (the nearest-neighbour distances below are on this "
        "scale).",
        "",
        "### Trajectories",
        "",
        "Z_t and h live in different spaces (GCN output vs GRU state), so the trajectories are "
        f"drawn in their own UMAP, fitted on the {n_windows:,} non-empty patient-windows' Z_t "
        f"(the {context.nonempty.size - n_windows:,} patient-windows with an empty feature row "
        "share one constant embedding and are left out). Patients drawn with seed "
        f"{cfg.seed}, one per phenotype, among those with a non-empty feature row in every "
        f"window and at least {analysis_cfg.trajectory_min_onsets} chronic-condition onsets in "
        f"{years[0]}-{years[-1]}.",
        "",
        f"![Trajectories]({figure(paths.trajectory_umap_fig)})",
        "",
        markdown_table(["Patient", "Age, sex", "Cluster", "Year", "New chronic conditions"],
                       trajectory_rows, "llrrl"),
        "",
        "### Nearest neighbours",
        "",
        f"For every patient with a chronic condition: the mean Jaccard index of their chronic "
        f"conditions ({years[0]}-{years[-1]}) with each of their {k} nearest neighbours, and the "
        "share of those neighbours in their phenotype cluster.",
        "",
        markdown_table(["Representation", "Metric", "Patients", f"Jaccard with {k} neighbours",
                        "Same phenotype"], agreement_rows, "llrrr"),
        "",
        *_neighbour_sections(analysis, context),
        "## Outputs",
        "",
        f"- `{paths.results_csv.name}`: the main table, one row per model.",
        f"- `artifacts/{paths.cv_predictions_csv.name}`: every patient's out-of-fold "
        f"predictions; `artifacts/{paths.cv_fits_csv.name}`: one row per fit (fold, model, seed); "
        f"`artifacts/{paths.test_predictions_csv.name}`: the final model's test predictions.",
        f"- Figures in `{figure(paths.figures_dir)}/`: `{paths.cv_auroc_fig.name}`, "
        f"`{paths.test_reliability_fig.name}`, `{paths.umap_phenotype_fig.name}`, "
        f"`{paths.umap_age_fig.name}`, `{paths.umap_chronic_fig.name}`, "
        f"`{paths.trajectory_umap_fig.name}`.",
        "",
    ]
    text = "\n".join(lines)
    paths.results_md.parent.mkdir(parents=True, exist_ok=True)
    paths.results_md.write_text(text, encoding="utf-8")
    table.to_csv(paths.results_csv, index=False)
    logger.info("Wrote %s and %s", paths.results_md, paths.results_csv)
    return text


def write_evaluation_summary(
    table: pd.DataFrame,
    group_table: pd.DataFrame,
    cv: CrossValidation,
    by_split: pd.DataFrame,
    test: TestEvaluation,
    analysis: EmbeddingAnalysis,
    val_auroc: float,
    cfg: Config,
) -> str:
    """Write the plain-language Phase 7 findings to ``paths.evaluation_summary_md``.

    Every verdict follows from a confidence interval: a difference counts as shown
    only when its interval excludes 0.

    Args:
        table: ``cv_results`` output.
        group_table: ``cv_group_results`` output.
        cv: The cross-validation.
        by_split: ``cv_by_split`` output.
        test: ``evaluate_test`` output.
        analysis: ``analyse_embeddings`` output.
        val_auroc: The final model's validation AUROC.
        cfg: Project configuration.

    Returns:
        The markdown written.
    """
    level = f"{cfg.evaluation.ci_level:.0%}"
    folds = cfg.cross_validation.folds
    by_key = table.set_index("key")
    full = by_key.loc[FULL_MODEL]
    models = {model.key: model for model in cv.models}
    predictions = cv.predictions
    rate = float(predictions[ANY_ONSET].mean())
    years = feature_window_years(cfg)
    cutoff = cfg.labels.cutoff_date
    c, ci = test.calibrated, test.intervals
    prevalence = test.positives / test.observed
    splits = by_split.set_index(SPLIT)

    low, high = ci["auroc"]
    chance = ("Its interval excludes 0.5, so it ranks better than chance."
              if low > _CHANCE_AUROC else
              "Its interval includes 0.5: on this test set the model cannot be told apart from "
              "guessing.")
    where = "above" if val_auroc > high else "below" if val_auroc < low else "inside"
    test_finding = (
        f"On the {test.observed} held-out test patients ({test.positives} with a new onset), "
        f"the final TemporalGCN has AUROC {_with_ci(c['auroc'], *ci['auroc'])}: it ranks a "
        "patient who goes on to a new onset above one who does not "
        f"{c['auroc']:.0%} of the time, against 50% for guessing. {chance} AUPRC is "
        f"{_with_ci(c['auprc'], *ci['auprc'])} against {prevalence:.3f} for a random score. "
        f"Its validation AUROC, {val_auroc:.3f}, lies {where} the test interval. A single split "
        "moves a lot: out of fold, the same architecture scores these test patients at "
        f"{splits.loc[TEST, FULL_MODEL]:.3f} and the validation patients at "
        f"{splits.loc[VAL, FULL_MODEL]:.3f}. The pooled cross-validated AUROC, "
        f"{_with_ci(full.oof_auroc, full.oof_auroc_low, full.oof_auroc_high)}, is the steadier "
        "estimate."
    )
    threshold = cfg.cross_validation.demographic_rank_correlation

    def reduced_note(key: str) -> str:
        row = by_key.loc[key]
        if row.age_sex_rank_corr_mean < threshold:
            return ""
        return (f" Caveat: model {row.number} ranks patients almost exactly as age and sex "
                f"alone do (mean within-fold rank correlation with model 1: "
                f"{row.age_sex_rank_corr_mean:.2f}; full model: "
                f"{full.age_sex_rank_corr_mean:.2f}). With the hyperparameters fixed at the full "
                "model's setting it learned little beyond the demographics, so this comparison "
                "mostly measures what the full model adds to age and sex.")

    best_simple = table[table["key"].map(lambda key: models[key].kind == LOGISTIC)].iloc[0]
    floor = by_key.loc[DEMOGRAPHICS_MODEL]
    rank = int(table.index[table["key"] == FULL_MODEL][0]) + 1
    baseline_finding = (
        f"Age and sex alone reach a pooled out-of-fold AUROC of {floor.oof_auroc:.3f}; the "
        f"best model without a GCN is model {best_simple.number} ({best_simple.model}, "
        f"{best_simple.inputs}) at {best_simple.oof_auroc:.3f}. The full TemporalGCN "
        f"({full.oof_auroc:.3f}) ranks {_ordinal(rank)} of {len(table)}. "
        + _difference_sentence(full, by_key.loc[best_simple.key], f"model {best_simple.number}",
                               folds, level)
    )
    shift = test.raw["mean_predicted"]
    base_brier = prevalence * (1 - prevalence)
    brier_low, brier_high = ci["brier"]
    if brier_high < base_brier:
        brier_verdict = f"better than always predicting the observed rate ({base_brier:.3f})"
    else:
        brier_verdict = (
            f"{'worse than' if brier_low > base_brier else 'not distinguishable from'} always "
            f"predicting the observed rate ({base_brier:.3f}): the probabilities now match the "
            "base rate, but carry little information beyond it"
        )
    calibration_finding = (
        f"Training with `pos_weight` pushes the raw probabilities up (mean {shift:.3f} on test "
        f"against an observed rate of {prevalence:.3f}). Platt scaling fitted on validation "
        f"brings the mean to {c['mean_predicted']:.3f} and the ECE from "
        f"{test.raw['ece']:.3f} to {_with_ci(c['ece'], *ci['ece'])}. The test Brier score, "
        f"{_with_ci(c['brier'], *ci['brier'])}, is {brier_verdict}."
    )
    not_better = [row for row in table.itertuples()
                  if row.key != FULL_MODEL and not row.delta_auroc_low > 0]
    if not_better:
        not_better_lines = [
            f"- **Model {row.number}, {row.model}** ({row.inputs}): pooled AUROC "
            f"{row.oof_auroc:.3f} vs {full.oof_auroc:.3f}; full minus this "
            f"{_with_ci(row.delta_auroc, row.delta_auroc_low, row.delta_auroc_high, signed=True)}"
            f", full model better in {int(row.folds_full_better)} of {folds} folds."
            for row in sorted(not_better, key=lambda r: r.number)
        ]
    else:
        not_better_lines = ["- None: every interval of the difference excludes 0."]
    scored = group_table.dropna(subset=["auroc"])
    best_per_group = scored.loc[scored.groupby("group", sort=False)["auroc"].idxmax()]
    lost = best_per_group[best_per_group["key"] != FULL_MODEL]
    full_groups = group_table[group_table["key"] == FULL_MODEL]
    chance_groups = full_groups[full_groups["auroc_low"] <= _CHANCE_AUROC]["group"].tolist()
    group_line = (
        f"- **Onset groups (secondary targets):** another model has the highest pooled AUROC "
        f"on {len(lost)} of {len(cv.groups)} groups"
        + (f" ({', '.join(f'{r.group}: model {models[r.key].number}' for r in lost.itertuples())})"
           if len(lost) else "")
        + ". With "
        f"{int(full_groups['positives'].min())}-{int(full_groups['positives'].max())} positives "
        "per group, the intervals overlap widely"
        + (f"; the full model's interval includes 0.5 for {', '.join(chance_groups)}."
           if chance_groups else ".")
    )
    scores = analysis.scores.set_index("representation")
    h_score = scores.iloc[0]
    chance_score = scores.iloc[-1]
    spectral_score = scores[scores.index.str.startswith("Spectral")].iloc[0]
    feature_score = scores[scores.index.str.contains("features")].iloc[0]
    separation = ("better than" if h_score.silhouette > chance_score.silhouette
                  else "no better than")
    first_pc = float(analysis.pc_variance[0])
    spread = ("patients differ mostly along that one direction"
              if first_pc > 0.5 else "the variation is spread over several directions")
    agreement = analysis.agreement.set_index("representation")
    h_agree, random_agree = agreement.iloc[0], agreement.iloc[-1]
    feature_agree = agreement[agreement.index.str.contains("features")].iloc[0]
    k = cfg.embedding_analysis.n_neighbours
    n_clusters = predictions.loc[predictions[PHENOTYPE] > 0, PHENOTYPE].nunique()
    group_names = ", ".join(group.replace("_", " ") for group in cv.groups)
    group_counts = full_groups["positives"]
    lines = [
        "# Evaluation summary",
        "",
        "Plain-language findings of Phase 7 (`scripts/07_evaluate.py`); every number, with "
        f"its setup, is in `{cfg.paths.results_md.name}`.",
        "",
        f"**The question.** From a patient's recorded conditions up to the end of {years[-1]}, "
        f"predict whether they are diagnosed with a new chronic condition in one of "
        f"{len(cv.groups)} groups ({group_names}) from {cutoff:%Y-%m-%d} to the end of "
        f"{cutoff.year + cfg.labels.horizon_years - 1}. "
        f"{rate:.1%} of the {len(predictions):,} patients with a known outcome are. The data "
        "are synthetic (Synthea).",
        "",
        f"**How the models are compared.** Eight models, from age and sex alone to the full "
        f"temporal graph network, each scored by {folds}-fold cross-validation on all "
        f"{len(predictions):,} patients; differences are paired bootstrap intervals "
        f"({level}). A difference counts as shown only if its interval excludes 0.",
        "",
        "## Main findings",
        "",
        f"1. **Test set (scored once).** {test_finding}",
        "2. **Does the graph help? (model 8 vs model 6, the same network with no graph)** "
        + _difference_sentence(full, by_key.loc["temporal_no_graph"], "the no-graph model",
                               folds, level),
        "3. **Does time help? (model 8 vs model 5, a GCN on the final window only)** "
        + _difference_sentence(full, by_key.loc["gcn_static"], "the static GCN", folds, level)
        + f" Model 5 is not blind to the past: its {years[-1]} window is a time-decayed sum "
        "of earlier windows, so this tests the sequence model, not history itself."
        + reduced_note("gcn_static"),
        "4. **Does the real similarity graph beat a random one? (model 8 vs model 7, "
        "degree-preserving rewired graphs)** "
        + _difference_sentence(full, by_key.loc["tgcn_rewired"], "the rewired-graph model",
                               folds, level)
        + reduced_note("tgcn_rewired"),
        f"5. **Simple baselines.** {baseline_finding}",
        f"6. **Calibration.** {calibration_finding}",
        "",
        "## Where the TemporalGCN does not beat a simpler model",
        "",
        "Models whose pooled AUROC the full TemporalGCN does not exceed with an interval "
        "excluding 0:",
        "",
        *not_better_lines,
        group_line,
        "",
        "## What the embeddings show",
        "",
        f"- **Phenotypes:** by the silhouette score, the trajectory embedding h separates the "
        f"{n_clusters} Phase 4 phenotype clusters {separation} shuffled clusters do "
        f"({h_score.silhouette:.3f} vs {chance_score.silhouette:.3f}). The {years[-1]} input "
        f"features score {feature_score.silhouette:.3f} and the spectral embedding the clusters "
        f"were defined on {spectral_score.silhouette:.3f}. h is trained to predict onsets, not "
        "to reproduce the clusters.",
        f"- **Spread:** h's first principal component holds {first_pc:.1%} of its variance, so "
        f"{spread}; nearest neighbours lie at distances of "
        f"{analysis.distances.min():.4f}-{analysis.distances.max():.4f} against a median of "
        f"{analysis.median_distance:.4f} between random patients.",
        f"- **Neighbours:** a patient's {k} nearest neighbours in h share on average "
        f"{h_agree.neighbour_jaccard:.0%} of their chronic conditions (Jaccard), against "
        f"{random_agree.neighbour_jaccard:.0%} for random patients and "
        f"{feature_agree.neighbour_jaccard:.0%} for the input-feature neighbours that build the "
        f"graph; {h_agree.same_phenotype:.0%} are in the same phenotype cluster "
        f"(random: {random_agree.same_phenotype:.0%}).",
        "- **Trajectories:** the per-year embeddings of five example patients are drawn as "
        f"paths in `{cfg.paths.results_md.name}` (figure `{cfg.paths.trajectory_umap_fig.name}`), "
        "with the years their chronic conditions began.",
        "",
        "## Caveats",
        "",
        "- **Synthetic data.** Synthea simulates disease modules with fixed rules; the results "
        "say how these methods behave on such data, not how they would on real records.",
        "- **Slightly optimistic GCN estimates.** The GCN hyperparameters were tuned on the "
        "original validation split, whose patients are inside the cross-validation folds. The "
        "test set was not used for any choice.",
        "- **One setting for every GCN.** The ablations (models 5-7) use the full model's tuned "
        "hyperparameters; tuned on their own they might do better. Nothing was retuned here.",
        "- **What the intervals cover.** They resample patients; they do not include the "
        "variation from retraining (seeds and folds), so they understate the full uncertainty.",
        "- **Transductive.** Every patient, including held-out ones, is a node of the graphs "
        "with their pre-cutoff features (never their labels). Scoring a new patient means "
        "placing them in the graph.",
        f"- **Small numbers.** {test.positives} test positives, and "
        f"{int(group_counts.min())}-{int(group_counts.max())} positives per onset group across "
        "all folds: group-level results are indicative only.",
        "",
    ]
    text = "\n".join(lines)
    path = cfg.paths.evaluation_summary_md
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)
    return text


def _difference_sentence(full: Any, other: Any, other_name: str, folds: int, level: str) -> str:
    """A plain-language verdict on the full model's pooled AUROC minus ``other``'s."""
    numbers = (
        f"pooled AUROC {full.oof_auroc:.3f} vs {other.oof_auroc:.3f}, difference "
        f"{other.delta_auroc:+.3f} ({level} CI {other.delta_auroc_low:+.3f} to "
        f"{other.delta_auroc_high:+.3f}); the full model is better in "
        f"{int(other.folds_full_better)} of {folds} folds"
    )
    if other.delta_auroc_low > 0:
        return f"**Yes.** The full model beats {other_name}: {numbers}."
    if other.delta_auroc_high < 0:
        return f"**No, the reverse.** {other_name[0].upper() + other_name[1:]} is better: {numbers}."
    favours = "the full model" if other.delta_auroc > 0 else other_name
    return (f"**Not shown.** The interval includes 0, so these data cannot tell the two apart "
            f"(the point estimate favours {favours}): {numbers}.")


def _reading(row: Any) -> str:
    """Short reading of a paired difference (full model minus this one)."""
    if row.delta_auroc_low > 0:
        return "full model better"
    if row.delta_auroc_high < 0:
        return "other model better"
    return "not shown (CI includes 0)"


def _join(items: Sequence[str]) -> str:
    """["a"] -> "a"; ["a", "b"] -> "a and b"; ["a", "b", "c"] -> "a, b and c"."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _ordinal(n: int) -> str:
    """1 -> "1st", 2 -> "2nd", 3 -> "3rd", 4 -> "4th"."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _neighbour_sections(analysis: EmbeddingAnalysis, context: PatientContext) -> list[str]:
    """Markdown listing each query patient's nearest neighbours in h and what they share."""
    lines = []
    for query, neighbours, distances in zip(analysis.query_nodes, analysis.neighbours,
                                            analysis.distances):
        own = set(_chronic_codes(context, query))
        lines += [
            f"**Patient {_short_id(context.patient_ids[query])}** "
            f"({_demographic_text(context, query)}, cluster {context.phenotype[query]}); "
            f"{len(own)} chronic conditions: {_names(context, sorted(own))}.",
            "",
        ]
        rows = []
        for rank, (node, distance) in enumerate(zip(neighbours, distances), start=1):
            theirs = set(_chronic_codes(context, node))
            shared = own & theirs
            union = own | theirs
            rows.append([str(rank), _short_id(context.patient_ids[node]), f"{distance:.3f}",
                         _demographic_text(context, node), str(context.phenotype[node]),
                         str(len(theirs)), _names(context, sorted(shared)) or "none",
                         f"{len(shared) / len(union):.2f}" if union else "n/a"])
        lines += [
            markdown_table(["Rank", "Neighbour", "Distance", "Age, sex", "Cluster",
                            "Chronic conditions", "Shared with the patient", "Jaccard"],
                           rows, "rlrlrrlr"),
            "",
        ]
    return lines


def _chronic_codes(context: PatientContext, node: int) -> list[str]:
    """The node's chronic condition codes."""
    row = context.chronic[node]
    return [context.codes[j] for j in row.indices]


def _names(context: PatientContext, codes: Sequence[str]) -> str:
    """Condition names of ``codes``, alphabetical, comma-separated."""
    return ", ".join(sorted(context.names[code] for code in codes))


def _demographic_text(context: PatientContext, node: int) -> str:
    """E.g. "age 67, male"."""
    return f"age {context.age[node]:.0f}, {'male' if context.sex[node] == 1 else 'female'}"


def _patient_title(context: PatientContext, node: int) -> str:
    """Panel title of one patient."""
    return (f"Patient {_short_id(context.patient_ids[node])}: "
            f"{_demographic_text(context, node)}, cluster {context.phenotype[node]}")


def _short_id(patient_id: str) -> str:
    """The first block of a patient UUID, enough to tell the shown patients apart."""
    return str(patient_id).split("-")[0]


def _cut(text: str, max_chars: int) -> str:
    """``text`` cut to ``max_chars`` characters, an ellipsis marking a cut."""
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def _short_ci(value: float, low: float, high: float) -> str:
    """Format ``value (low-high)`` with 2 decimals, for wide tables."""
    return f"{value:.2f} ({low:.2f}-{high:.2f})"


def _bare(ax: Any) -> None:
    """Remove ticks and spines: projection axes carry no units."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _with_ci(value: float, low: float, high: float, signed: bool = False) -> str:
    """Format ``value (low to high)`` with 3 decimals, signed if asked."""
    fmt = "{:+.3f}" if signed else "{:.3f}"
    return f"{fmt.format(value)} ({fmt.format(low)} to {fmt.format(high)})"
