"""Tests for ehrgraph.evaluate: baselines and calibration (Phase 6) and Phase 7."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml
from sklearn.metrics import roc_auc_score

from ehrgraph.config import DEFAULT_CONFIG_PATH, Config, load_config
from ehrgraph.evaluate import (
    BASELINE_COLUMNS,
    COMPARISON_COLUMNS,
    DEMOGRAPHICS_MODEL,
    FIT_COLUMNS,
    FULL_MODEL,
    GROUP_RESULT_COLUMNS,
    REFERENCE,
    RESULT_COLUMNS,
    TEST_METRICS,
    UNLABELED,
    Calibrator,
    CrossValidation,
    PatientContext,
    analyse_embeddings,
    baseline_inputs,
    bin_labels,
    bootstrap_metrics,
    bootstrap_statistics,
    calibrate,
    calibration_section,
    categorize,
    chronic_jaccard,
    chronic_onsets,
    cluster_scores,
    compare_models,
    cross_validation_folds,
    cv_by_split,
    cv_fold_metrics,
    cv_group_results,
    cv_models,
    cv_results,
    evaluate_test,
    f1_threshold,
    fit_calibrator,
    fit_platt,
    fit_temperature,
    fold_split,
    fold_training_data,
    gcn_inputs,
    held_out_targets,
    load_calibration,
    load_cross_validation,
    logistic_inputs,
    logistic_scores,
    neighbour_agreement,
    percentile_interval,
    pick_patients,
    plot_cv_auroc,
    plot_embeddings,
    plot_reliability,
    prediction_column,
    probability_metrics,
    rank_agreement,
    reliability_bins,
    require_trained_model,
    rewire_graph,
    run_baselines,
    run_cross_validation,
    save_calibration,
    save_cross_validation,
    sigmoid,
    stratified_folds,
    threshold_metrics,
    validation_logits,
    write_evaluation_summary,
    write_results,
)
from ehrgraph.graph import normalize_adjacency
from ehrgraph.labels import ANY_ONSET, ONSET_PREFIX, PHENOTYPE, SPLIT
from ehrgraph.train import TEST, TRAIN, VAL, ModelData, TrainingData, fit, make_training_data

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


def test_bootstrap_statistics_uses_the_resamples_of_bootstrap_metrics() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(150) < 0.3).astype(float)
    p = np.clip(0.3 * y + 0.5 * rng.random(150), 0, 1)
    stats = bootstrap_statistics(y, p, lambda yy, pp: {"auroc": roc_auc_score(yy, pp)}, 100, 7)
    np.testing.assert_allclose(stats["auroc"], bootstrap_metrics(y, {"m": p}, 100, 7).auroc["m"])


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


def test_fit_platt_recovers_a_known_slope_and_intercept() -> None:
    rng = np.random.default_rng(5)
    logits = 2.0 * rng.standard_normal(40_000)
    y = (rng.random(40_000) < sigmoid(0.5 * logits - 1.0)).astype(float)
    slope, intercept = fit_platt(logits, y)
    assert slope == pytest.approx(0.5, abs=0.03)
    assert intercept == pytest.approx(-1.0, abs=0.05)


def test_calibrators_apply_one_formula(cfg: Config) -> None:
    rng = np.random.default_rng(6)
    logits = 3.0 * rng.standard_normal(5000)
    y = (rng.random(5000) < sigmoid(logits / 2.0)).astype(float)
    temperature = fit_calibrator(logits, y, "temperature", cfg.evaluation)
    assert temperature.intercept == 0.0 and 1.0 / temperature.slope == pytest.approx(2.0, rel=0.1)
    np.testing.assert_allclose(temperature.apply(logits), sigmoid(logits * temperature.slope))
    assert temperature.describe().startswith("T = ")
    platt = Calibrator("platt", 0.5, -1.0)
    np.testing.assert_allclose(platt.apply(np.array([2.0])), sigmoid(np.array([0.0])))
    with pytest.raises(ValueError, match="unknown calibration method"):
        fit_calibrator(logits, y, "isotonic", cfg.evaluation)


def test_platt_removes_a_shift_that_temperature_cannot(cfg: Config) -> None:
    rng = np.random.default_rng(7)
    true_logits = rng.standard_normal(3000) - 1.5
    y = (rng.random(3000) < sigmoid(true_logits)).astype(float)
    shifted = true_logits + 2.0  # as training with pos_weight does
    platt = calibrate(shifted, y, "platt", cfg.evaluation, cfg.seed)
    temperature = calibrate(shifted, y, "temperature", cfg.evaluation, cfg.seed)
    assert platt.calibrator.slope == pytest.approx(1.0, abs=0.15)
    assert platt.calibrator.intercept == pytest.approx(-2.0, abs=0.2)
    assert platt.after["mean_predicted"] == pytest.approx(y.mean(), abs=0.01)
    assert platt.after["nll"] < temperature.after["nll"]
    assert platt.cross_fitted["brier"] < platt.before["brier"]
    assert platt.cross_fitted["ece"] < temperature.cross_fitted["ece"]


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


@pytest.mark.parametrize("method", ["platt", "temperature"])
def test_calibrate_keeps_the_ranking_and_lowers_the_log_loss(
    cfg: Config, tmp_path: Path, method: str
) -> None:
    rng = np.random.default_rng(4)
    logits = 4.0 * rng.standard_normal(400)  # overconfident: the truth is logits / 4
    y = (rng.random(400) < sigmoid(logits / 4.0)).astype(float)
    result = calibrate(logits, y, method, cfg.evaluation, cfg.seed)
    assert 0 < result.calibrator.slope < 1
    assert result.after["auroc"] == pytest.approx(result.before["auroc"])
    assert result.after["auprc"] == pytest.approx(result.before["auprc"])
    assert result.after["nll"] < result.before["nll"]
    assert result.after["brier"] < result.before["brier"]
    assert len(result.fold_calibrators) == cfg.evaluation.calibration_cv_folds
    assert result.cross_fitted["brier"] < result.before["brier"]
    assert len(result.bins_after) == cfg.evaluation.calibration_bins

    path = tmp_path / "calibration.json"
    save_calibration(result, path)
    assert load_calibration(path) == result.calibrator
    figure = tmp_path / "reliability.png"
    plot_reliability(result, figure, cfg)
    assert figure.stat().st_size > 0
    text = "\n".join(calibration_section(result, "reliability.png", path.name, cfg))
    assert result.calibrator.describe() in text and "(reliability.png)" in text
    assert ("Platt scaling" in text) == (method == "platt")


def test_validation_logits_exclude_masked_and_other_splits(cfg: Config) -> None:
    data = _data()
    model = fit(data, cfg.model, cfg.seed, verbose=False).model
    logits, targets, ids = validation_logits(model, data)
    assert len(logits) == len(targets) == 12
    assert ids.tolist() == [f"p{i:02d}" for i in range(36, 48)]
    assert not np.isnan(targets).any()


# --- Phase 7: a small synthetic cohort ---------------------------------------------------

CV_N = 100
CV_GROUPS = ("g1", "g2")
YEARS = (2020, 2021, 2022)


def _model_data(seed: int = 0) -> ModelData:
    """100 nodes, 3 phenotypes, 6 masked; split 60/20/20; a ring with chords per window."""
    rng = np.random.default_rng(seed)
    ids = [f"p{i:03d}" for i in range(CV_N)]
    features = [rng.random((CV_N, F_IN)) for _ in YEARS]
    signal = features[-1][:, 0] + 0.3 * rng.standard_normal(CV_N)
    any_onset = (signal > np.quantile(signal, 0.7)).astype(float)
    any_onset[:6] = np.nan
    g2 = np.where(np.isnan(any_onset), np.nan, (features[-1][:, 1] > 0.6).astype(float))
    g2[10] = np.nan  # one masked group label of an observed patient
    split = np.array((["train"] * 6 + ["val"] * 2 + ["test"] * 2) * (CV_N // 10))
    labels = pd.DataFrame({
        "patient_id": pd.array(ids, dtype="string"),
        "node": np.arange(CV_N),
        ONSET_PREFIX + "g1": any_onset.copy(),
        ONSET_PREFIX + "g2": g2,
        ANY_ONSET: any_onset,
        PHENOTYPE: pd.array(np.arange(CV_N) % 3 + 1, dtype="Int64"),
        SPLIT: split,
    })
    nodes = np.arange(CV_N)
    weights = []
    for _ in YEARS:
        rows = np.concatenate([nodes, nodes])
        cols = np.concatenate([(nodes + 1) % CV_N, (nodes + 7) % CV_N])
        upper = sp.csr_matrix((rng.uniform(0.3, 1.0, len(rows)), (rows, cols)), shape=(CV_N, CV_N))
        weights.append(sp.csr_matrix(upper.maximum(upper.T)))
    demographics = pd.DataFrame(
        {"age": rng.uniform(5, 90, CV_N), "sex": (nodes % 2).astype(float)},
        index=pd.Index(ids, name="patient_id"),
    )
    return ModelData(
        patient_ids=tuple(ids), years=YEARS, matrices=[sp.csr_matrix(X) for X in features],
        weights=weights, adjacency=[normalize_adjacency(W) for W in weights], labels=labels,
        groups=CV_GROUPS, demographics=demographics,
    )


def _spectral(seed: int = 1) -> np.ndarray:
    return 0.02 * np.random.default_rng(seed).standard_normal((CV_N, 4))


def _cv_cfg(tmp_path: Path | None = None) -> Config:
    base = load_config()
    cfg = replace(
        base,
        model=replace(base.model, epochs=4, patience=2, log_every_epochs=1000),
        evaluation=replace(base.evaluation, bootstrap_resamples=60),
        cross_validation=replace(base.cross_validation, seeds=(42,), rewire_swaps_per_edge=5),
    )
    if tmp_path is None:
        return cfg
    reports = tmp_path / "reports"
    names = ("results_md", "results_csv", "evaluation_summary_md", "cv_predictions_csv",
             "cv_fits_csv", "test_predictions_csv", "cv_auroc_fig", "test_reliability_fig",
             "umap_phenotype_fig", "umap_age_fig", "umap_chronic_fig", "trajectory_umap_fig")
    moved = {name: (reports / "figures" if name.endswith("_fig") else reports)
             / getattr(cfg.paths, name).name for name in names}
    return replace(cfg, paths=replace(cfg.paths, figures_dir=reports / "figures", **moved))


@pytest.fixture(scope="module")
def cv_run() -> CrossValidation:
    return run_cross_validation(_model_data(), _spectral(), _cv_cfg())


def test_cv_models_number_the_eight_models() -> None:
    models = cv_models(YEARS, 4)
    assert [m.number for m in models] == list(range(1, 9))
    assert models[-1].key == FULL_MODEL
    assert len({m.key for m in models}) == 8


def test_rewire_graph_keeps_degrees_and_weights_but_moves_edges() -> None:
    W = _model_data().weights[0]
    one = rewire_graph(W, 10, np.random.default_rng(0))
    two = rewire_graph(W, 10, np.random.default_rng(0))
    assert (one != two).nnz == 0  # reproducible
    assert (one != one.T).nnz == 0 and one.diagonal().sum() == 0
    degree = lambda M: np.diff(sp.csr_matrix(M).indptr)  # noqa: E731
    np.testing.assert_array_equal(degree(one), degree(W))
    np.testing.assert_allclose(np.sort(one.data), np.sort(W.data))
    kept = sp.csr_matrix(W).multiply(one != 0).nnz
    assert kept < 0.3 * W.nnz  # most edges moved


def test_stratified_folds_spread_every_stratum() -> None:
    strata = pd.Series(["a"] * 13 + ["b"] * 2 + ["c"] * 7, index=[f"p{i}" for i in range(22)])
    folds = stratified_folds(strata, 5, seed=0)
    assert set(folds) == {1, 2, 3, 4, 5}
    sizes = folds.value_counts()
    assert sizes.max() - sizes.min() <= 1
    for _, part in folds.groupby(strata):
        counts = part.value_counts().reindex(range(1, 6), fill_value=0)
        assert counts.max() - counts.min() <= 1
    pd.testing.assert_series_equal(folds, stratified_folds(strata, 5, seed=0))


def test_fold_split_holds_out_one_fold_and_carves_an_inner_validation_set() -> None:
    data = _model_data()
    folds = cross_validation_folds(data.labels, 5, seed=42)
    assert len(folds) == CV_N - 6  # observed any_onset only
    split = fold_split(data.labels, folds, 2, 0.15, seed=42)
    ids = data.labels["patient_id"].astype(str)
    in_fold = set(folds.index[folds == 2].astype(str))
    assert set(ids[split == TEST]) == in_fold
    assert (split[:6] == UNLABELED).all()  # masked any_onset
    training = (split == TRAIN) | (split == VAL)
    assert training.sum() == len(folds) - len(in_fold)
    assert (split == VAL).sum() == pytest.approx(0.15 * training.sum(), abs=3)


def test_fold_training_data_hides_the_held_out_labels() -> None:
    data = _model_data()
    folds = cross_validation_folds(data.labels, 5, seed=42)
    split = fold_split(data.labels, folds, 1, 0.15, seed=42)
    inputs = gcn_inputs(data, 2, seed=0)
    fold = fold_training_data(data, inputs[FULL_MODEL], split)
    assert np.isnan(fold.any_onset.numpy()[split == TEST]).all()
    assert np.isnan(fold.onsets.numpy()[split == TEST]).all()
    inner = (split == TRAIN) | (split == VAL)
    assert not np.isnan(fold.any_onset.numpy()[inner]).any()
    train_age = data.demographics["age"].to_numpy()[split == TRAIN]
    assert fold.age_mean == pytest.approx(train_age.mean())


def test_gcn_inputs_build_each_ablation() -> None:
    data = _model_data()
    inputs = gcn_inputs(data, 5, seed=0)
    assert inputs["gcn_static"].years == (YEARS[-1],) and len(inputs["gcn_static"].matrices) == 1
    for A in inputs["temporal_no_graph"].adjacency:
        assert (A != sp.identity(CV_N, format="csr")).nnz == 0
    for W, rewired in zip(data.weights, inputs["tgcn_rewired"].weights):
        np.testing.assert_array_equal(np.diff(W.indptr), np.diff(rewired.indptr))
    assert inputs[FULL_MODEL].adjacency is data.adjacency


def test_spectral_inputs_are_standardized_on_the_training_patients() -> None:
    data = _model_data()
    folds = cross_validation_folds(data.labels, 5, seed=42)
    split = fold_split(data.labels, folds, 1, 0.15, seed=42)
    fold = fold_training_data(data, gcn_inputs(data, 2, 0)[FULL_MODEL], split)
    X = logistic_inputs("spectral_lr", fold, _spectral())
    assert X.shape == (CV_N, 4 + 2)
    np.testing.assert_allclose(X[split == TRAIN, :4].mean(axis=0), 0, atol=1e-9)
    assert logistic_inputs("age_sex", fold, _spectral()).shape == (CV_N, 2)
    with pytest.raises(ValueError, match="no logistic model"):
        logistic_inputs("gcn", fold, _spectral())


def test_cross_validation_predicts_every_labelled_patient_once(cv_run: CrossValidation) -> None:
    predictions = cv_run.predictions
    assert len(predictions) == CV_N - 6 and predictions["patient_id"].is_unique
    for model in cv_run.models:
        values = predictions[model.key].to_numpy()
        assert not np.isnan(values).any() and ((values > 0) & (values < 1)).all()
        for group in CV_GROUPS:
            assert not predictions[prediction_column(model.key, group)].isna().any()
    assert list(cv_run.fits.columns) == FIT_COLUMNS
    assert len(cv_run.fits) == 5 * 8  # one seed per GCN here
    assert (cv_run.fits.groupby("fold")["held_out"].first().sum()) == CV_N - 6


def test_held_out_fold_labels_do_not_change_its_predictions() -> None:
    cfg = _cv_cfg()
    base = _model_data()
    changed = _model_data()
    folds = cross_validation_folds(base.labels, cfg.cross_validation.folds, cfg.seed)
    fold_one = changed.labels["patient_id"].isin(folds.index[folds == 1]).to_numpy()
    flipped = changed.labels[ONSET_PREFIX + "g2"].to_numpy(copy=True)
    flipped[fold_one] = 1 - flipped[fold_one]  # strata use any_onset only: same folds
    changed.labels[ONSET_PREFIX + "g2"] = flipped
    first = run_cross_validation(base, _spectral(), cfg).predictions
    second = run_cross_validation(changed, _spectral(), cfg).predictions
    rows = (first["fold"] == 1).to_numpy()
    model_columns = [c for c in first.columns if "." in c or c in {m.key for m in cv_models(YEARS, 4)}]
    pd.testing.assert_frame_equal(first.loc[rows, model_columns], second.loc[rows, model_columns])
    assert not first.loc[~rows, model_columns].equals(second.loc[~rows, model_columns])


def test_cv_tables_and_figure(cv_run: CrossValidation, tmp_path: Path) -> None:
    cfg = _cv_cfg(tmp_path)
    table = cv_results(cv_run, cfg)
    assert list(table.columns) == RESULT_COLUMNS and len(table) == 8
    assert table["oof_auroc"].is_monotonic_decreasing
    full = table.set_index("key").loc[FULL_MODEL]
    assert np.isnan(full.delta_auroc) and np.isnan(full.folds_full_better)
    other = table[table["key"] != FULL_MODEL].iloc[0]
    assert other.delta_auroc == pytest.approx(full.oof_auroc - other.oof_auroc)
    folds = cv_fold_metrics(cv_run)
    assert len(folds) == 5 * 8
    mean = folds[folds["key"] == FULL_MODEL]["auroc"].mean()
    assert full.fold_auroc_mean == pytest.approx(mean)
    groups = cv_group_results(cv_run, cfg)
    assert list(groups.columns) == GROUP_RESULT_COLUMNS and len(groups) == 2 * 8
    assert (groups["auroc_low"] <= groups["auroc"]).all()
    plot_cv_auroc(table, cfg.paths.cv_auroc_fig, cfg)
    assert cfg.paths.cv_auroc_fig.stat().st_size > 0


def test_rank_agreement_and_results_by_original_split(cv_run: CrossValidation) -> None:
    agreement = rank_agreement(cv_run, DEMOGRAPHICS_MODEL)
    assert agreement.shape == (5, 8)
    np.testing.assert_allclose(agreement[DEMOGRAPHICS_MODEL], 1.0)
    fold = cv_run.predictions[cv_run.predictions["fold"] == 3]
    expected = fold[prediction_column(FULL_MODEL, "logit")].corr(
        fold[prediction_column(DEMOGRAPHICS_MODEL, "logit")], method="spearman")
    assert agreement.loc[3, FULL_MODEL] == pytest.approx(expected)
    table = cv_results(cv_run, _cv_cfg())
    row = table.set_index("key").loc[FULL_MODEL]
    assert row.age_sex_rank_corr_mean == pytest.approx(agreement[FULL_MODEL].mean())

    by_split = cv_by_split(cv_run, _model_data().labels)
    assert by_split[SPLIT].tolist() == [TRAIN, VAL, TEST]
    assert by_split["patients"].sum() == len(cv_run.predictions)
    test_rows = cv_run.predictions["node"].to_numpy() % 10 >= 8
    part = cv_run.predictions[test_rows]
    assert by_split.set_index(SPLIT).loc[TEST, FULL_MODEL] == pytest.approx(
        roc_auc_score(part[ANY_ONSET], part[FULL_MODEL]))


def test_cross_validation_round_trip(cv_run: CrossValidation, tmp_path: Path) -> None:
    cfg = _cv_cfg(tmp_path)
    save_cross_validation(cv_run, cfg)
    loaded = load_cross_validation(cfg, _model_data(), 4)
    for model in cv_run.models:
        np.testing.assert_allclose(loaded.predictions[model.key], cv_run.predictions[model.key])
    with pytest.raises(ValueError, match="without --reuse-cv"):
        load_cross_validation(replace(cfg, cross_validation=replace(cfg.cross_validation,
                                                                    folds=3)), _model_data(), 4)


# --- Phase 7: the held-out test set ----------------------------------------------------


def test_held_out_targets_are_the_test_split_only() -> None:
    data = _model_data()
    targets, onsets = held_out_targets(data)
    test = (data.labels[SPLIT] == TEST).to_numpy()
    assert np.isnan(targets[~test]).all() and np.isnan(onsets[~test]).all()
    np.testing.assert_array_equal(targets[test], data.labels[ANY_ONSET].to_numpy()[test])


def test_f1_threshold_and_threshold_metrics() -> None:
    y = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.65, 0.7, 0.9])
    threshold = f1_threshold(y, p)
    assert threshold == pytest.approx(0.4)
    metrics = threshold_metrics(y, p, threshold)
    assert metrics["recall"] == 1.0 and metrics["precision"] == pytest.approx(0.8)
    assert metrics["f1"] == pytest.approx(2 * 0.8 / 1.8)
    assert threshold_metrics(y, p, 2.0) == {"f1": 0.0, "precision": 0.0, "recall": 0.0}


def _test_evaluation(cfg: Config):  # noqa: ANN202
    data = _model_data()
    rng = np.random.default_rng(9)
    targets, onsets = held_out_targets(data)
    signal = np.nan_to_num(data.labels[ANY_ONSET].to_numpy(dtype=float), nan=0.0)
    logits = 2.0 * signal + rng.standard_normal(CV_N)
    onset_logits = np.column_stack([logits, rng.standard_normal(CV_N)])
    calibrator = Calibrator("platt", 0.8, -1.0)
    return evaluate_test(logits, onset_logits, targets, onsets, CV_GROUPS, calibrator, 0.4, cfg)


def test_evaluate_test_reports_intervals_and_groups() -> None:
    cfg = _cv_cfg()
    result = _test_evaluation(cfg)
    assert result.observed == 20 and 0 < result.positives < result.observed  # nodes 8, 9 mod 10
    for name in TEST_METRICS:
        low, high = result.intervals[name]
        assert low <= high
    assert result.intervals["auroc"][0] < result.calibrated["auroc"] < result.intervals["auroc"][1]
    assert result.raw["auroc"] == pytest.approx(result.calibrated["auroc"])
    assert result.groups["group"].tolist() == list(CV_GROUPS)
    assert len(result.bins_calibrated) == cfg.evaluation.calibration_bins


def test_require_trained_model_checks_the_model_section(tmp_path: Path) -> None:
    cfg = load_config()
    paths = replace(cfg.paths, model_pt=tmp_path / "model.pt",
                    config_used_yaml=tmp_path / "config_used.yaml",
                    calibration_json=tmp_path / "calibration.json")
    cfg = replace(cfg, paths=paths)
    with pytest.raises(ValueError, match="06_train.py"):
        require_trained_model(DEFAULT_CONFIG_PATH, cfg)
    paths.model_pt.write_bytes(b"")
    paths.config_used_yaml.write_text(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"),
                                      encoding="utf-8")
    paths.calibration_json.write_text('{"method": "platt", "slope": 0.9, "intercept": -1.2}',
                                      encoding="utf-8")
    assert require_trained_model(DEFAULT_CONFIG_PATH, cfg) == Calibrator("platt", 0.9, -1.2)
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["model"]["dropout"] = 0.1
    paths.config_used_yaml.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="another model section"):
        require_trained_model(DEFAULT_CONFIG_PATH, cfg)


# --- Phase 7: embedding analysis -------------------------------------------------------


def _context(data: ModelData) -> PatientContext:
    """Three blocks of chronic conditions that follow the phenotype; two onsets each."""
    rng = np.random.default_rng(11)
    phenotype = data.labels[PHENOTYPE].to_numpy(dtype=np.int64)
    codes = tuple(f"c{j}" for j in range(9))
    presence = np.zeros((CV_N, 9))
    for i, cluster in enumerate(phenotype):
        presence[i, 3 * (cluster - 1): 3 * cluster] = rng.random(3) < 0.8
    nonempty = np.ones((CV_N, len(YEARS)), dtype=bool)
    nonempty[::5, 0] = False
    onsets = pd.DataFrame({
        "node": np.repeat(np.arange(CV_N), 2),
        "year": np.tile([YEARS[0], YEARS[2]], CV_N),
        "code": [codes[3 * (phenotype[i] - 1) + k] for i in range(CV_N) for k in (0, 1)],
    })
    return PatientContext(
        patient_ids=data.patient_ids, years=YEARS, phenotype=phenotype,
        age=data.demographics["age"].to_numpy(), sex=data.demographics["sex"].to_numpy(),
        chronic=sp.csr_matrix(presence), codes=codes,
        names={code: f"Condition {code} with a fairly long name" for code in codes},
        nonempty=nonempty, onsets=onsets,
    )


def test_bin_labels_and_categorize() -> None:
    assert bin_labels((0, 18, 30, float("inf"))) == ["0-17", "18-29", "30+"]
    assert bin_labels((0, 1, 2, 4, float("inf"))) == ["0", "1", "2-3", "4+"]
    np.testing.assert_array_equal(categorize(np.array([0.0, 1.0, 3.0, 9.0]), (0, 1, 2, 4, np.inf)),
                                  np.array(["0", "1", "2-3", "4+"], dtype=object))


def test_pick_patients_prefers_different_phenotypes() -> None:
    phenotype = np.array([1, 1, 1, 2, 2, 3, 1, 1])
    chosen = pick_patients(np.ones(8, dtype=bool), phenotype, 3, seed=0)
    assert sorted(phenotype[chosen]) == [1, 2, 3]
    more = pick_patients(np.ones(8, dtype=bool), phenotype, 5, seed=0)
    assert len(set(more)) == 5
    np.testing.assert_array_equal(pick_patients(phenotype == 1, phenotype, 2, seed=0),
                                  pick_patients(phenotype == 1, phenotype, 2, seed=0))


def test_chronic_onsets_keep_first_chronic_events_in_the_windows() -> None:
    utc = lambda day: pd.Timestamp(day, tz="UTC")  # noqa: E731
    events = pd.DataFrame({
        "patient_id": ["a", "a", "a", "b", "b", "c"],
        "date": [utc("2019-03-01"), utc("2021-05-01"), utc("2015-01-01"), utc("2020-01-01"),
                 utc("2022-06-01"), utc("2020-02-02")],
        "stop": [pd.NaT, pd.NaT, pd.NaT, utc("2020-01-10"), pd.NaT, pd.NaT],
        "code": ["x", "x", "y", "z", "w", "x"],
    })
    onsets = chronic_onsets(events, ["x", "y", "z", "w"], (2019, 2020, 2021, 2022), ("a", "b"), 365)
    # a: x first in 2019 (its 2021 event is not an onset), y began before 2019;
    # b: z lasted 9 days (not chronic); c is not a node.
    assert onsets.to_dict("records") == [{"node": 0, "year": 2019, "code": "x"},
                                         {"node": 1, "year": 2022, "code": "w"}]


def test_cluster_scores_and_neighbour_agreement() -> None:
    data = _model_data()
    context = _context(data)
    rng = np.random.default_rng(2)
    separated = np.eye(3)[context.phenotype - 1] * 5 + 0.3 * rng.standard_normal((CV_N, 3))
    scores = cluster_scores({"separated": separated}, context.phenotype, 0, 8)
    assert scores["silhouette"].iloc[0] > 0.5
    assert abs(scores["silhouette"].iloc[-1]) < 0.1  # shuffled clusters
    pairs = np.array([0, 1]), np.array([3, 0])
    expected = [len(set(a) & set(b)) / len(set(a) | set(b)) if set(a) | set(b) else 0.0
                for a, b in ((context.chronic[0].indices, context.chronic[3].indices),
                             (context.chronic[1].indices, context.chronic[0].indices))]
    np.testing.assert_allclose(chronic_jaccard(context.chronic, *pairs), expected)
    agreement = neighbour_agreement({"separated": (separated, "euclidean")}, context, 5, 0, 8)
    assert agreement["same_phenotype"].iloc[0] > agreement["same_phenotype"].iloc[-1]
    assert agreement["neighbour_jaccard"].iloc[0] > agreement["neighbour_jaccard"].iloc[-1]


def test_phase_7_reports_end_to_end(cv_run: CrossValidation, tmp_path: Path) -> None:
    cfg = _cv_cfg(tmp_path)
    data = _model_data()
    context = _context(data)
    rng = np.random.default_rng(12)
    Z = rng.standard_normal((CV_N, len(YEARS), 8)) + context.phenotype[:, None, None]
    h = Z[:, -1, :] + 0.1 * rng.standard_normal((CV_N, 8))
    analysis = analyse_embeddings(Z, h, _spectral(), data.matrices[-1], context, cfg)
    assert analysis.trajectory_umap.shape == (CV_N, 2)
    assert np.isnan(analysis.window_umap[~context.nonempty]).all()
    assert not np.isnan(analysis.window_umap[context.nonempty]).any()
    assert len(analysis.trajectory_nodes) == cfg.embedding_analysis.trajectory_patients
    assert analysis.neighbours.shape == (cfg.embedding_analysis.neighbour_patients,
                                         cfg.embedding_analysis.n_neighbours)
    assert not (analysis.neighbours == analysis.query_nodes[:, None]).any()
    plot_embeddings(analysis, context, cfg)
    for name in ("umap_phenotype_fig", "umap_age_fig", "umap_chronic_fig", "trajectory_umap_fig"):
        assert getattr(cfg.paths, name).stat().st_size > 0

    assert analysis.pc_variance.sum() == pytest.approx(1.0)
    assert (np.diff(analysis.pc_variance) <= 1e-12).all() and analysis.median_distance > 0

    table = cv_results(cv_run, cfg)
    test = _test_evaluation(cfg)
    by_split = cv_by_split(cv_run, data.labels)
    results = write_results(table, cv_fold_metrics(cv_run), cv_group_results(cv_run, cfg),
                            cv_run, by_split, test, analysis, context, 0.7, cfg)
    summary = write_evaluation_summary(table, cv_group_results(cv_run, cfg), cv_run, by_split,
                                       test, analysis, 0.7, cfg)
    assert "## Ablations" in results and "## Held-out test set (single shot)" in results
    assert "### Out-of-fold AUROC by original split" in results
    assert pd.read_csv(cfg.paths.results_csv)["key"].tolist() == table["key"].tolist()
    for question in ("Does the graph help?", "Does time help?",
                     "Does the real similarity graph beat a random one?"):
        assert question in summary
    assert "Where the TemporalGCN does not beat a simpler model" in summary


# --- The saved project data ----------------------------------------------------------

_PROJECT = load_config()


@pytest.mark.skipif(not _PROJECT.paths.labels_parquet.exists(),
                    reason="run scripts/05_build_labels.py")
def test_project_cross_validation_folds_cover_the_observed_patients() -> None:
    labels = pd.read_parquet(_PROJECT.paths.labels_parquet)
    folds = cross_validation_folds(labels, _PROJECT.cross_validation.folds, _PROJECT.seed)
    assert len(folds) == int(labels[ANY_ONSET].notna().sum()) == 2221
    sizes = folds.value_counts()
    assert sizes.max() - sizes.min() <= 1
    split = fold_split(labels, folds, 1, _PROJECT.cross_validation.inner_val, _PROJECT.seed)
    assert (split == TEST).sum() == sizes[1]
    assert (split == UNLABELED).sum() == int(labels[ANY_ONSET].isna().sum())
    assert (split == VAL).sum() == pytest.approx(0.15 * (2221 - sizes[1]), abs=10)
