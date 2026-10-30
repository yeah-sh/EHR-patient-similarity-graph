"""Validation-only evaluation of ``any_onset``: baselines, bootstrap CIs, calibration.

Nothing here reads test labels: every function works on ``train.TrainingData``,
whose test labels are NaN on loading, and scores the validation split only.

Baselines (run end to end by ``scripts/06a_baselines.py``)::

    data = load_training_data(cfg)
    results, val_scores = run_baselines(data, cfg)
    save_baselines(results, val_scores, cfg)
    write_baselines_summary(results, data, cfg)

Calibration of the final model (``scripts/06_train.py``, after ``train.fit``)::

    logits, targets, _ = validation_logits(model, data)
    calibration = calibrate(logits, targets, cfg.evaluation, cfg.seed)
    save_temperature(calibration, outputs.temperature_json)
    plot_reliability(calibration, outputs.reliability_fig, cfg)

Confidence intervals are percentile bootstrap intervals over the validation
patients with an observed label. Every model is scored on the same resamples,
drawn with the global seed, so the differences between models are paired.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ehrgraph.config import Config, EvaluationConfig, ModelConfig, reference_model_config
from ehrgraph.data import chart_style, markdown_table, save_figure
from ehrgraph.labels import ANY_ONSET
from ehrgraph.models import OnsetModel
from ehrgraph.train import TRAIN, VAL, TrainingData, binary_metrics, fit, predict, split_targets

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

# Chart tokens from the dataviz reference palette (as in train.py): categorical slots 1
# and 2 for before and after calibration, secondary ink for annotations.
_BEFORE_COLOR = "#2a78d6"
_AFTER_COLOR = "#eb6834"
_SURFACE = "#fcfcfb"
_TEXT_SECONDARY = "#52514e"
_BASELINE = "#c3c2b7"
_POINTS_PER_INCH = 72
_FIG_SIZE_IN = (6.4, 6.0)
_LINE_PX = 2
_MARKER_PX = 10
_RING_PX = 2  # surface ring around each marker, so overlapping markers stay distinct
_LABEL_OFFSET_PT = 6


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
class Calibration:
    """Temperature scaling of the ``any_onset`` logits on the validation patients.

    Attributes:
        temperature: T > 0; calibrated probability ``sigmoid(logit / T)``.
        observed: Validation patients with an observed label.
        positives: Of them, positives.
        before: ``probability_metrics`` of ``sigmoid(logit)``.
        after: ``probability_metrics`` of ``sigmoid(logit / T)``: fitted and scored on
            the same patients (in-sample).
        cross_fitted: ``probability_metrics`` with each patient's temperature fitted on
            the other folds (out-of-sample).
        fold_temperatures: The temperature of each fold.
        bins_before: ``reliability_bins`` before calibration.
        bins_after: ``reliability_bins`` after.
    """

    temperature: float
    observed: int
    positives: int
    before: dict[str, float]
    after: dict[str, float]
    cross_fitted: dict[str, float]
    fold_temperatures: tuple[float, ...]
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
    settings = cfg.baselines
    y = split_targets(data, TRAIN)[0].numpy()
    seen = ~np.isnan(y)
    model = LogisticRegression(
        C=settings.C,
        max_iter=settings.max_iter,
        class_weight="balanced" if settings.balanced_class_weight else None,
    )
    model.fit(X[seen], y[seen].astype(int))
    if int(np.max(model.n_iter_)) >= settings.max_iter:
        logger.warning("Logistic regression did not converge in %d iterations", settings.max_iter)
    return model.predict_proba(X)[:, 1]


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
    logits: np.ndarray, targets: np.ndarray, evaluation: EvaluationConfig, seed: int
) -> Calibration:
    """Fit temperature scaling on the validation logits and measure its effect.

    The temperature is fitted on every validation patient with an observed label
    and scored on the same patients (``after``: in-sample, optimistic). As an
    out-of-sample check, ``cross_fitted`` scores each patient with a temperature
    fitted on the other ``calibration_cv_folds`` - 1 stratified folds.

    Args:
        logits: [n] validation logits of ``any_onset``.
        targets: [n] their labels (no NaN).
        evaluation: The ``evaluation`` section.
        seed: Random seed of the folds.

    Returns:
        The temperature and the metrics before and after.
    """
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    n_bins = evaluation.calibration_bins
    temperature = fit_temperature(z, y, evaluation.temperature_bounds)
    out_of_fold = np.empty_like(z)
    fold_temperatures = []
    folds = StratifiedKFold(evaluation.calibration_cv_folds, shuffle=True, random_state=seed)
    for fit_rows, score_rows in folds.split(z, y):
        fold_t = fit_temperature(z[fit_rows], y[fit_rows], evaluation.temperature_bounds)
        fold_temperatures.append(fold_t)
        out_of_fold[score_rows] = sigmoid(z[score_rows] / fold_t)
    before, after = sigmoid(z), sigmoid(z / temperature)
    calibration = Calibration(
        temperature=temperature,
        observed=len(y),
        positives=int(y.sum()),
        before=probability_metrics(y, before, n_bins),
        after=probability_metrics(y, after, n_bins),
        cross_fitted=probability_metrics(y, out_of_fold, n_bins),
        fold_temperatures=tuple(fold_temperatures),
        bins_before=reliability_bins(y, before, n_bins),
        bins_after=reliability_bins(y, after, n_bins),
    )
    logger.info(
        "Temperature %.4f (folds %s): Brier %.4f -> %.4f (cross-fitted %.4f), ECE %.4f -> %.4f",
        temperature, ", ".join(f"{t:.3f}" for t in fold_temperatures),
        calibration.before["brier"], calibration.after["brier"],
        calibration.cross_fitted["brier"], calibration.before["ece"], calibration.after["ece"],
    )
    return calibration


def save_temperature(calibration: Calibration, path: Path) -> None:
    """Write the fitted temperature and its fit statistics as JSON."""
    payload: dict[str, Any] = {
        "target": ANY_ONSET,
        "temperature": calibration.temperature,
        "apply_as": "probability = sigmoid(logit / temperature)",
        "fitted_on": "validation patients with an observed label",
        "observed": calibration.observed,
        "positives": calibration.positives,
        "nll_before": calibration.before["nll"],
        "nll_after": calibration.after["nll"],
        "fold_temperatures": list(calibration.fold_temperatures),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    logger.info("Wrote %s (temperature %.4f)", path, calibration.temperature)


def load_temperature(path: Path) -> float:
    """Read the temperature written by ``save_temperature``."""
    return float(json.loads(path.read_text(encoding="utf-8"))["temperature"])


def plot_reliability(calibration: Calibration, path: Path, cfg: Config) -> None:
    """Save the reliability diagram of ``any_onset`` before and after temperature scaling.

    Args:
        calibration: ``calibrate`` output.
        path: Output PNG.
        cfg: Project configuration (``summary.figure_dpi``).
    """
    dpi = cfg.summary.figure_dpi
    px = _POINTS_PER_INCH / dpi
    series = [
        (calibration.bins_before, "before", _BEFORE_COLOR),
        (calibration.bins_after, f"after (T = {calibration.temperature:.2f})", _AFTER_COLOR),
    ]
    limit = 1.0  # probabilities: the full unit square
    with matplotlib.rc_context(chart_style(dpi)):
        fig = Figure(figsize=_FIG_SIZE_IN, layout="constrained")
        ax = fig.subplots()
        ax.plot([0, limit], [0, limit], color=_BASELINE, linewidth=px, linestyle=":")
        ax.annotate("perfect calibration", xy=(limit, limit), xytext=(-_LABEL_OFFSET_PT, 0),
                    textcoords="offset points", ha="right", va="top", color=_TEXT_SECONDARY)
        ends = [float(bins["observed_rate"].iloc[-1]) for bins, _, _ in series]
        for (bins, name, color), end, other in zip(series, ends, ends[::-1]):
            ax.plot(bins["mean_predicted"], bins["observed_rate"], color=color,
                    linewidth=_LINE_PX * px, marker="o", markersize=_MARKER_PX * px,
                    markeredgecolor=_SURFACE, markeredgewidth=_RING_PX * px, label=name)
            # The higher end label sits above its point and the other below, so the two
            # stay apart even when the curves coincide (ties: "before" above).
            above = end > other or (end == other and name == series[0][1])
            ax.annotate(name, xy=(bins["mean_predicted"].iloc[-1], end),
                        xytext=(_LABEL_OFFSET_PT, _LABEL_OFFSET_PT * (1 if above else -1)),
                        textcoords="offset points", ha="left",
                        va="bottom" if above else "top", color=_TEXT_SECONDARY)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal")
        ax.set_title(
            f"Reliability of {ANY_ONSET}, validation (n = {calibration.observed}, "
            f"{len(calibration.bins_before)} equal-count bins)"
        )
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed onset rate")
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.legend(frameon=False, loc="upper left")
        save_figure(fig, path)


def calibration_section(
    calibration: Calibration, figure: str, temperature_file: str, cfg: Config
) -> list[str]:
    """Markdown for the calibration result.

    Args:
        calibration: ``calibrate`` output.
        figure: The reliability figure's path relative to the report.
        temperature_file: Name of the saved temperature file.
        cfg: Project configuration.

    Returns:
        Lines to pass as ``train.write_training_summary(extra_sections=...)``.
    """
    before, after, cross = calibration.before, calibration.after, calibration.cross_fitted
    rows = [
        [name, f"{before[key]:.3f}", f"{after[key]:.3f}", f"{cross[key]:.3f}"]
        for name, key in (("AUROC", "auroc"), ("AUPRC", "auprc"), ("Brier score", "brier"),
                          ("Log loss", "nll"), ("ECE", "ece"),
                          ("Mean predicted probability", "mean_predicted"))
    ]
    prevalence = calibration.positives / calibration.observed
    folds = cfg.evaluation.calibration_cv_folds
    return [
        "## Calibration (temperature scaling)",
        "",
        f"`{ANY_ONSET}` probabilities are `sigmoid(logit / T)` with **T = "
        f"{calibration.temperature:.3f}**, fitted by minimizing the log loss on the "
        f"{calibration.observed:,} validation patients with an observed label "
        f"({calibration.positives:,} positives, rate {prevalence:.3f}); saved to "
        f"`{temperature_file}`. A temperature preserves the ranking, so AUROC and AUPRC "
        f"do not change. \"After\" is scored on the patients T was fitted on (optimistic); "
        f"\"cross-fitted\" scores each patient with a T fitted on the other {folds - 1} of "
        f"{folds} stratified folds (fold temperatures "
        f"{', '.join(f'{t:.2f}' for t in calibration.fold_temperatures)}); its AUROC and "
        "AUPRC pool folds scored with different temperatures, so they move slightly.",
        "",
        markdown_table(["Validation", "Before", "After", "After, cross-fitted"], rows, "lrrr"),
        "",
        f"ECE: expected calibration error over {cfg.evaluation.calibration_bins} equal-count "
        "bins. The model is trained with `pos_weight` = negatives / positives, which shifts "
        "its logits upward; a temperature only rescales logits around 0 and cannot remove "
        "such a shift, so compare the mean predicted probability with the observed rate.",
        "",
        f"![Reliability diagram]({figure})",
        "",
    ]


def _with_ci(value: float, low: float, high: float, signed: bool = False) -> str:
    """Format ``value (low to high)`` with 3 decimals, signed if asked."""
    fmt = "{:+.3f}" if signed else "{:.3f}"
    return f"{fmt.format(value)} ({fmt.format(low)} to {fmt.format(high)})"
