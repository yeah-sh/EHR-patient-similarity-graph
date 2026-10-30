"""Tests for ehrgraph.evaluate: bootstrap CIs, baselines and temperature scaling."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from ehrgraph.config import Config, load_config
from ehrgraph.evaluate import (
    BASELINE_COLUMNS,
    COMPARISON_COLUMNS,
    REFERENCE,
    baseline_inputs,
    bootstrap_metrics,
    calibrate,
    calibration_section,
    compare_models,
    fit_temperature,
    load_temperature,
    logistic_scores,
    percentile_interval,
    plot_reliability,
    probability_metrics,
    reliability_bins,
    run_baselines,
    save_temperature,
    sigmoid,
    validation_logits,
)
from ehrgraph.graph import normalize_adjacency
from ehrgraph.labels import ANY_ONSET, ONSET_PREFIX, SPLIT
from ehrgraph.train import TrainingData, fit, make_training_data

N, T, F_IN = 60, 3, 6
GROUPS = ("g1",)


def _data(val_flip: bool = False) -> TrainingData:
    """60 nodes on a ring: 36 train, 12 val, 12 test; the label follows feature 0."""
    rng = np.random.default_rng(0)
    features = [rng.random((N, F_IN)) for _ in range(T)]
    y = (features[-1][:, 0] + 0.3 * rng.standard_normal(N) > 0.6).astype(float)
    split = np.array(["train"] * 36 + ["val"] * 12 + ["test"] * 12)
    if val_flip:
        y[split != "train"] = 1 - y[split != "train"]
    ids = [f"p{i:02d}" for i in range(N)]
    labels = pd.DataFrame(
        {"patient_id": ids, ANY_ONSET: y, ONSET_PREFIX + "g1": y, SPLIT: split}
    )
    nodes = np.arange(N)
    upper = sp.csr_matrix((np.ones(N), (nodes, (nodes + 1) % N)), shape=(N, N))
    W = sp.csr_matrix(upper + upper.T)
    demographics = pd.DataFrame(
        {"age": rng.uniform(20, 80, N), "sex": (nodes % 2).astype(float)},
        index=pd.Index(ids, name="patient_id"),
    )
    return make_training_data(
        ids, (2020, 2021, 2022), [sp.csr_matrix(X) for X in features], [W] * T,
        [normalize_adjacency(W)] * T, labels, GROUPS, demographics,
    )


@pytest.fixture
def cfg() -> Config:
    base = load_config()
    return replace(
        base,
        model=replace(base.model, epochs=5, log_every_epochs=1000),
        evaluation=replace(base.evaluation, bootstrap_resamples=200),
    )


# --- Bootstrap -------------------------------------------------------------------------


def test_bootstrap_is_paired_reproducible_and_brackets_the_estimate() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(200) < 0.3).astype(float)
    good = y + rng.standard_normal(200)
    noise = rng.standard_normal(200)
    one = bootstrap_metrics(y, {"good": good, "same": good.copy(), "noise": noise}, 300, seed=42)
    two = bootstrap_metrics(y, {"good": good, "same": good.copy(), "noise": noise}, 300, seed=42)
    np.testing.assert_array_equal(one.auroc["good"], two.auroc["good"])
    assert one.drawn == 300 and one.used == 300
    np.testing.assert_array_equal(one.auroc["good"], one.auroc["same"])  # same resamples
    low, high = percentile_interval(one.auroc["good"], 0.95)
    assert low < roc_auc_score(y, good) < high
    assert percentile_interval(one.auroc["noise"], 0.95)[1] < low  # clearly worse


def test_bootstrap_skips_one_class_resamples_and_rejects_nan() -> None:
    y = np.array([1.0, 0.0, 0.0])
    boot = bootstrap_metrics(y, {"m": np.array([0.9, 0.1, 0.2])}, 100, seed=0)
    assert boot.used < boot.drawn and (boot.auroc["m"] == 1.0).all()
    with pytest.raises(ValueError, match="observed"):
        bootstrap_metrics(np.array([1.0, np.nan]), {"m": np.zeros(2)}, 10, seed=0)


def test_compare_models_reports_paired_differences(cfg: Config) -> None:
    rng = np.random.default_rng(2)
    y = (rng.random(300) < 0.25).astype(float)
    scores = {"ref": y + 0.5 * rng.standard_normal(300), "weak": rng.standard_normal(300)}
    table = compare_models(y, scores, "ref", cfg.evaluation, cfg.seed)
    assert list(table.columns) == COMPARISON_COLUMNS
    ref, weak = table.iloc[0], table.iloc[1]
    assert np.isnan(ref.delta_auroc)
    assert weak.delta_auroc == pytest.approx(ref.val_auroc - weak.val_auroc)
    assert 0 < weak.delta_auroc_low < weak.delta_auroc < weak.delta_auroc_high
    assert ref.val_auroc_low < ref.val_auroc < ref.val_auroc_high


# --- Baselines -------------------------------------------------------------------------


def test_baseline_inputs_shapes() -> None:
    data = _data()
    assert baseline_inputs(data, "last", False).shape == (N, F_IN)
    assert baseline_inputs(data, "all", True).shape == (N, T * F_IN + 2)
    np.testing.assert_array_equal(baseline_inputs(data, "last", False),
                                  data.features[-1].numpy())
    with pytest.raises(ValueError, match="windows"):
        baseline_inputs(data, "first", False)


def test_logistic_baseline_uses_training_labels_only(cfg: Config) -> None:
    X = baseline_inputs(_data(), "all", True)
    base = logistic_scores(X, _data(), cfg)
    flipped = logistic_scores(X, _data(val_flip=True), cfg)
    np.testing.assert_array_equal(base, flipped)


def test_run_baselines_scores_every_model_on_validation(cfg: Config) -> None:
    data = _data()
    results, val_scores = run_baselines(data, cfg)
    assert list(results.columns) == BASELINE_COLUMNS
    assert results["key"].tolist() == [
        "lr_last", "lr_all", "lr_last_demo", "lr_all_demo", REFERENCE, REFERENCE + "_demo",
    ]
    assert results["n_features"].tolist() == [F_IN, T * F_IN, F_IN + 2, T * F_IN + 2, F_IN, F_IN]
    assert (results["val_observed"] == 12).all()
    assert np.isnan(results.set_index("key").loc[REFERENCE, "delta_auroc"])
    assert len(val_scores) == 12 and val_scores["patient_id"].str.startswith("p").all()
    assert set(results["key"]) <= set(val_scores.columns)


# --- Calibration -----------------------------------------------------------------------


def test_fit_temperature_recovers_a_known_temperature() -> None:
    rng = np.random.default_rng(3)
    logits = 3.0 * rng.standard_normal(20_000)
    y = (rng.random(20_000) < sigmoid(logits / 2.5)).astype(float)
    assert fit_temperature(logits, y, (0.01, 100.0)) == pytest.approx(2.5, rel=0.06)


def test_fit_temperature_stays_in_bounds_when_the_logits_do_not_rank(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logits = np.array([2.0, 1.0, 2.0, 1.0])
    y = np.array([0.0, 1.0, 1.0, 0.0])  # AUROC 0.5: the best fit is "always 0.5"
    with caplog.at_level("WARNING", logger="ehrgraph.evaluate"):
        temperature = fit_temperature(logits, y, (0.01, 100.0))
    assert temperature == pytest.approx(100.0, rel=1e-2)
    assert "search bound" in caplog.text


def test_reliability_bins_and_metrics() -> None:
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1], dtype=float)
    p = np.linspace(0.05, 0.95, 10)
    bins = reliability_bins(y, p, 5)
    assert bins["n"].tolist() == [2] * 5
    np.testing.assert_allclose(bins["observed_rate"], [0, 0.5, 1, 0.5, 1])
    metrics = probability_metrics(y, p, 5)
    assert metrics["brier"] == pytest.approx(np.mean((p - y) ** 2))
    expected_ece = np.mean(np.abs(bins["mean_predicted"] - bins["observed_rate"]))
    assert metrics["ece"] == pytest.approx(expected_ece)


def test_calibrate_keeps_the_ranking_and_lowers_the_log_loss(cfg: Config, tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    logits = 4.0 * rng.standard_normal(400)  # overconfident: the truth is logits / 4
    y = (rng.random(400) < sigmoid(logits / 4.0)).astype(float)
    result = calibrate(logits, y, cfg.evaluation, cfg.seed)
    assert result.temperature > 1
    assert result.after["auroc"] == pytest.approx(result.before["auroc"])
    assert result.after["auprc"] == pytest.approx(result.before["auprc"])
    assert result.after["nll"] < result.before["nll"]
    assert result.after["brier"] < result.before["brier"]
    assert len(result.fold_temperatures) == cfg.evaluation.calibration_cv_folds
    assert result.cross_fitted["brier"] < result.before["brier"]
    assert len(result.bins_after) == cfg.evaluation.calibration_bins

    path = tmp_path / "temperature.json"
    save_temperature(result, path)
    assert load_temperature(path) == pytest.approx(result.temperature)
    figure = tmp_path / "reliability.png"
    plot_reliability(result, figure, cfg)
    assert figure.stat().st_size > 0
    text = "\n".join(calibration_section(result, "reliability.png", path.name, cfg))
    assert f"T = {result.temperature:.3f}" in text and "(reliability.png)" in text


def test_validation_logits_exclude_masked_and_other_splits(cfg: Config) -> None:
    data = _data()
    model = fit(data, cfg.model, cfg.seed, verbose=False).model
    logits, targets, ids = validation_logits(model, data)
    assert len(logits) == len(targets) == 12
    assert ids.tolist() == [f"p{i:02d}" for i in range(36, 48)]
    assert not np.isnan(targets).any()
