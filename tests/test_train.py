"""Tests for ehrgraph.train: data preparation, masking, training loop, saving, reports."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from ehrgraph.config import DEFAULT_CONFIG_PATH, Config, ModelConfig, load_config
from ehrgraph.data import DAYS_PER_YEAR, load_patients
from ehrgraph.graph import normalize_adjacency
from ehrgraph.labels import ANY_ONSET, ONSET_PREFIX, SPLIT
from ehrgraph.models import build_model, onset_loss, positive_weights
from ehrgraph.train import (
    REPORT_COLUMNS,
    TUNED_FIELDS,
    TUNING_RUN_COLUMNS,
    PosWeights,
    TrainingData,
    TrainOutputs,
    fit,
    load_trained_model,
    load_training_data,
    make_training_data,
    plot_training_curves,
    predict,
    save_training,
    select_setting,
    split_targets,
    summarize_tuning,
    train_step,
    tune,
    tuning_settings,
    validation_report,
    write_training_summary,
)

N, T, F_IN = 30, 3, 8
GROUPS = ("g1", "g2")
TRAIN_NODES, VAL_NODES, TEST_NODES = range(0, 18), range(18, 24), range(24, 30)
ISOLATED = N - 1  # a test node
NAN = np.nan


def _labels() -> pd.DataFrame:
    """Every third node positive; a few masked labels in each split."""
    nodes = np.arange(N)
    any_onset = (nodes % 3 == 0).astype(float)
    any_onset[[5, 20, 27]] = NAN
    g1 = (nodes % 4 == 0).astype(float)
    g1[[1, 19]] = NAN
    g2 = (nodes % 5 == 1).astype(float)
    g2[[2, 22]] = NAN
    split = np.array(["train"] * 18 + ["val"] * 6 + ["test"] * 6)
    return pd.DataFrame(
        {
            "patient_id": [f"p{i:02d}" for i in nodes],
            ANY_ONSET: any_onset,
            ONSET_PREFIX + "g1": g1,
            ONSET_PREFIX + "g2": g2,
            SPLIT: split,
        }
    )


def _ring() -> sp.csr_matrix:
    """A weighted ring over nodes 0..N-2; node N-1 is isolated."""
    nodes = np.arange(N - 1)
    upper = sp.csr_matrix(
        (np.linspace(0.3, 1.0, N - 1), (nodes, (nodes + 1) % (N - 1))), shape=(N, N)
    )
    return sp.csr_matrix(upper + upper.T)


def _demographics() -> pd.DataFrame:
    """Ages 20-79 and alternating sex, in node order."""
    ages = np.random.default_rng(1).uniform(20, 80, N)
    return pd.DataFrame(
        {"age": ages, "sex": (np.arange(N) % 2).astype(float)},
        index=pd.Index([f"p{i:02d}" for i in range(N)], name="patient_id"),
    )


def _data(
    labels: pd.DataFrame | None = None, demographics: pd.DataFrame | None = None
) -> TrainingData:
    rng = np.random.default_rng(0)
    W = _ring()
    return make_training_data(
        [f"p{i:02d}" for i in range(N)],
        (2020, 2021, 2022),
        [sp.csr_matrix(rng.random((N, F_IN))) for _ in range(T)],
        [W] * T,
        [normalize_adjacency(W)] * T,
        _labels() if labels is None else labels,
        GROUPS,
        _demographics() if demographics is None else demographics,
    )


@pytest.fixture
def model_cfg() -> ModelConfig:
    return replace(
        load_config().model, hidden_dim=16, embed_dim=8, dropout=0.0, epochs=40, patience=10,
        log_every_epochs=1000,
    )


def _cfg(tmp_path: Path) -> Config:
    base = load_config()
    return replace(base, paths=replace(base.paths, quick_dir=tmp_path / "quick"))


# --- Data and masks --------------------------------------------------------------------


def test_test_labels_are_dropped_on_loading() -> None:
    data = _data()
    test = list(TEST_NODES)
    assert torch.isnan(data.any_onset[test]).all()
    assert torch.isnan(data.onsets[test]).all()
    train = list(TRAIN_NODES)
    expected = _labels()[ANY_ONSET].to_numpy()[train]
    np.testing.assert_array_equal(data.any_onset[train].numpy(), expected.astype(np.float32))


def test_labels_must_be_in_node_order() -> None:
    with pytest.raises(ValueError, match="node order"):
        _data(_labels().iloc[::-1].reset_index(drop=True))


def test_age_is_standardized_with_training_statistics_only() -> None:
    data = _data()
    train = np.array([i in TRAIN_NODES for i in range(N)])
    ages = _demographics()["age"].to_numpy()
    assert data.age_mean == pytest.approx(ages[train].mean())
    assert data.age_sd == pytest.approx(ages[train].std())
    age = data.demographics[:, 0].numpy()
    assert age[train].mean() == pytest.approx(0.0, abs=1e-5)
    assert age[train].std() == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(age, (ages - data.age_mean) / data.age_sd, rtol=1e-5)
    np.testing.assert_array_equal(data.demographics[:, 1].numpy(), np.arange(N) % 2)
    # Validation and test ages do not move the statistics.
    shifted = _demographics()
    shifted.loc[~train, "age"] += 30
    assert _data(demographics=shifted).age_mean == pytest.approx(data.age_mean)


def test_demographics_must_be_in_node_order() -> None:
    with pytest.raises(ValueError, match="demographics are not in node order"):
        _data(demographics=_demographics().iloc[::-1])


def test_split_targets_mask_every_other_node() -> None:
    data = _data()
    any_onset, onsets = split_targets(data, "val")
    outside = [i for i in range(N) if i not in VAL_NODES]
    assert torch.isnan(any_onset[outside]).all() and torch.isnan(onsets[outside]).all()
    assert int((~torch.isnan(any_onset)).sum()) == 5  # node 20 is masked
    with pytest.raises(ValueError, match="test"):
        split_targets(data, "test")


def test_pos_weight_comes_from_the_training_split(model_cfg: ModelConfig) -> None:
    data = _data()
    result = fit(data, replace(model_cfg, epochs=1), seed=42)
    train = _labels().iloc[list(TRAIN_NODES)]
    y = train[ANY_ONSET].dropna()
    assert float(result.pos_weights.any_onset) == pytest.approx((y == 0).sum() / (y == 1).sum())
    g1 = train[ONSET_PREFIX + "g1"].dropna()
    assert float(result.pos_weights.onsets[0]) == pytest.approx((g1 == 0).sum() / (g1 == 1).sum())


# --- Training --------------------------------------------------------------------------


def _loss(model, data, targets, weights, model_cfg) -> float:
    output = predict(model, data)
    return float(onset_loss(output, *targets, *weights, model_cfg.secondary_weight).total)


def test_loss_decreases_on_tiny_overfit_fixture(model_cfg: ModelConfig) -> None:
    data = _data()
    torch.manual_seed(42)
    model = build_model(model_cfg, F_IN, len(GROUPS))
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.lr)
    targets = split_targets(data, "train")
    weights = PosWeights(*(positive_weights(t) for t in targets))
    start = _loss(model, data, targets, weights, model_cfg)
    for _ in range(300):
        train_step(model, optimizer, data, targets, weights, model_cfg)
    end = _loss(model, data, targets, weights, model_cfg)
    assert end < 0.2 * start


def test_held_out_and_masked_nodes_pass_messages(model_cfg: ModelConfig) -> None:
    data = _data()
    features = [X.clone().requires_grad_(True) for X in data.features]
    data = replace(data, features=features)
    torch.manual_seed(42)
    model = build_model(model_cfg, F_IN, len(GROUPS))
    targets = split_targets(data, "train")
    weights = PosWeights(*(positive_weights(t) for t in targets))
    output = model(data.features, data.adjacency, data.demographics)
    onset_loss(output, *targets, *weights, model_cfg.secondary_weight).total.backward()
    grad = sum(X.grad.abs().sum(dim=1) for X in features)
    assert grad[18] > 0  # a validation node next to training node 17
    assert grad[ISOLATED] == 0  # an isolated node reaches nobody
    assert grad[22] == 0  # more than two hops from every training node


def test_validation_and_test_labels_do_not_change_the_updates(model_cfg: ModelConfig) -> None:
    cfg = replace(model_cfg, epochs=15, patience=100)
    base = fit(_data(), cfg, seed=42)
    flipped = _labels()
    held_out = flipped[SPLIT] != "train"
    for column in (ANY_ONSET, ONSET_PREFIX + "g1", ONSET_PREFIX + "g2"):
        flipped.loc[held_out, column] = 1 - flipped.loc[held_out, column]
    other = fit(_data(flipped), cfg, seed=42)
    pd.testing.assert_series_equal(base.history["objective"], other.history["objective"])
    pd.testing.assert_series_equal(base.history["train_loss"], other.history["train_loss"])


def test_fit_keeps_the_best_validation_epoch(model_cfg: ModelConfig) -> None:
    data = _data()
    result = fit(data, model_cfg, seed=42)
    history = result.history
    assert result.best_epoch == int(history["epoch"][history["val_auroc"].idxmax()])
    report = validation_report(result.model, data, result.pos_weights)
    assert list(report.columns) == REPORT_COLUMNS
    assert report["target"].tolist() == [ANY_ONSET, *(ONSET_PREFIX + g for g in GROUPS)]
    best = history.iloc[result.best_epoch - 1]
    assert report.iloc[0]["val_auroc"] == pytest.approx(best["val_auroc"])
    assert report.iloc[0]["val_positives"] == 2 and report.iloc[0]["val_observed"] == 5


def test_early_stopping_waits_patience_epochs(model_cfg: ModelConfig) -> None:
    result = fit(_data(), replace(model_cfg, epochs=500, patience=3), seed=42)
    assert result.stopped_early
    assert result.epochs_run == result.best_epoch + 3


def test_fit_is_reproducible(model_cfg: ModelConfig) -> None:
    one = fit(_data(), replace(model_cfg, epochs=5), seed=42)
    two = fit(_data(), replace(model_cfg, epochs=5), seed=42)
    pd.testing.assert_frame_equal(one.history, two.history)


def test_smoothness_loss_is_off_by_default_and_runs_when_on(model_cfg: ModelConfig) -> None:
    assert load_config().model.smoothness_weight == 0
    off = fit(_data(), replace(model_cfg, epochs=3), seed=42)
    assert "smoothness" not in off.history
    on = fit(_data(), replace(model_cfg, epochs=3, smoothness_weight=0.5), seed=42)
    assert on.history["smoothness"].between(0, 2).all()
    assert (on.history["objective"] != off.history["objective"]).all()


def test_drop_edge_changes_training_and_is_reproducible(model_cfg: ModelConfig) -> None:
    cfg = replace(model_cfg, epochs=4, patience=100)
    off = fit(_data(), cfg, seed=42)
    one = fit(_data(), replace(cfg, drop_edge=0.4), seed=42)
    two = fit(_data(), replace(cfg, drop_edge=0.4), seed=42)
    pd.testing.assert_frame_equal(one.history, two.history)
    # The first objective already uses a thinned graph; the model starts the same.
    assert one.history["objective"].iloc[0] != off.history["objective"].iloc[0]


def test_use_demographics_switches_the_head_inputs(model_cfg: ModelConfig) -> None:
    data = _data()
    on = fit(data, replace(model_cfg, epochs=2), seed=42).model
    off = fit(data, replace(model_cfg, epochs=2, use_demographics=False), seed=42).model
    assert on.n_demographics == 2 and on.heads.primary.in_features == model_cfg.embed_dim + 2
    assert off.n_demographics == 0 and off.heads.primary.in_features == model_cfg.embed_dim
    other = replace(data, demographics=torch.zeros_like(data.demographics))
    assert torch.equal(predict(off, data).any_onset_logits, predict(off, other).any_onset_logits)
    assert not torch.equal(predict(on, data).any_onset_logits, predict(on, other).any_onset_logits)


# --- Tuning ----------------------------------------------------------------------------


def test_tuning_settings_cover_the_grid_and_subsample() -> None:
    tuning = load_config().tuning
    settings = tuning_settings(tuning, seed=42)
    assert len(settings) == 48 == tuning.max_settings
    assert all(set(s) == set(TUNED_FIELDS) for s in settings)
    assert len({tuple(s.values()) for s in settings}) == 48
    assert {(s["hidden_dim"], s["embed_dim"]) for s in settings} == {(64, 32), (32, 16)}
    subset = tuning_settings(replace(tuning, max_settings=20), seed=42)
    assert len(subset) == 20 and all(s in settings for s in subset)
    assert subset == tuning_settings(replace(tuning, max_settings=20), seed=42)


def _summary(rows: list[tuple[float, float, int, int, float]]) -> pd.DataFrame:
    """Settings as (auroc_mean, smoothness, hidden, embed, drop_edge)."""
    return pd.DataFrame(
        [
            {"setting": i + 1, "weight_decay": 5e-4, "dropout": 0.5, "hidden_dim": hidden,
             "embed_dim": embed, "smoothness_weight": smooth, "drop_edge": drop,
             "val_auroc_mean": auroc, "val_auroc_sd": 0.01}
            for i, (auroc, smooth, hidden, embed, drop) in enumerate(rows)
        ]
    ).sort_values("val_auroc_mean", ascending=False).reset_index(drop=True)


def test_select_setting_prefers_the_simpler_setting_within_the_tolerance() -> None:
    summary = _summary([
        (0.700, 0.1, 64, 32, 0.2),  # best, two optional parts on
        (0.697, 0.0, 64, 32, 0.0),  # within 0.005, none on
        (0.696, 0.0, 32, 16, 0.0),  # within 0.005, none on, smaller
        (0.690, 0.0, 32, 16, 0.0),  # outside the tolerance
    ])
    assert int(select_setting(summary, 0.005).setting) == 3
    assert int(select_setting(summary, 0.0035).setting) == 2  # setting 3 now outside
    assert int(select_setting(summary, 0.0).setting) == 1


def test_tune_runs_every_setting_per_seed_and_summarizes(model_cfg: ModelConfig) -> None:
    data = _data()
    settings = [
        {"weight_decay": 5e-4, "dropout": 0.0, "hidden_dim": 8, "embed_dim": 4,
         "smoothness_weight": 0.0, "drop_edge": 0.0},
        {"weight_decay": 1e-2, "dropout": 0.5, "hidden_dim": 8, "embed_dim": 4,
         "smoothness_weight": 0.1, "drop_edge": 0.2},
    ]
    seen: list[int] = []
    results = tune(data, replace(model_cfg, epochs=3), settings, (42, 43),
                   on_progress=lambda runs: seen.append(len(runs)))
    assert list(results.columns) == TUNING_RUN_COLUMNS
    assert len(results) == 4 and seen == [2, 4]
    assert results.groupby("setting")["seed"].apply(list).tolist() == [[42, 43], [42, 43]]
    summary = summarize_tuning(results)
    assert summary["val_auroc_mean"].is_monotonic_decreasing
    for row in summary.itertuples():
        runs = results[results["setting"] == row.setting]["val_auroc"]
        assert row.val_auroc_mean == pytest.approx(runs.mean())
        assert row.val_auroc_sd == pytest.approx(runs.std(ddof=1), nan_ok=True)


# --- Saving and reports ----------------------------------------------------------------


def test_quick_outputs_go_to_the_quick_dir(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    full, quick = TrainOutputs.from_config(cfg), TrainOutputs.from_config(cfg, quick=True)
    for f in fields(TrainOutputs):
        assert getattr(quick, f.name) == cfg.paths.quick_dir / getattr(full, f.name).name


def test_save_training_round_trip_and_reports(model_cfg: ModelConfig, tmp_path: Path) -> None:
    data = _data()
    cfg = replace(_cfg(tmp_path), model=model_cfg)
    result = fit(data, replace(model_cfg, epochs=8), seed=42)
    report = validation_report(result.model, data, result.pos_weights)
    outputs = TrainOutputs.from_config(cfg, quick=True)  # the real file names, in tmp_path
    save_training(result, report, data, cfg, outputs, DEFAULT_CONFIG_PATH, quick=False)
    plot_training_curves(result.history, result.best_epoch, outputs.training_curves_fig, cfg)
    text = write_training_summary(result, report, data, cfg, outputs, quick=False)

    loaded = load_trained_model(model_cfg, outputs.model_pt, F_IN, len(GROUPS))
    expected, actual = predict(result.model, data), predict(loaded, data)
    assert torch.equal(expected.any_onset_logits, actual.any_onset_logits)
    assert torch.equal(expected.onset_logits, actual.onset_logits)
    Z = np.load(outputs.embeddings_per_window_npy)
    h = np.load(outputs.trajectory_embeddings_npy)
    assert Z.shape == (N, T, model_cfg.embed_dim) and h.shape == (N, model_cfg.embed_dim)
    np.testing.assert_array_equal(h, actual.trajectory_embeddings.numpy())
    assert outputs.config_used_yaml.read_text(encoding="utf-8") == DEFAULT_CONFIG_PATH.read_text(
        encoding="utf-8"
    )
    log = pd.read_csv(outputs.train_log_csv)
    assert {"train_loss", "val_loss", "train_auroc", "val_auroc"} <= set(log.columns)
    assert len(log) == result.epochs_run
    assert outputs.training_curves_fig.stat().st_size > 0
    assert f"Best epoch **{result.best_epoch}**" in text
    assert f"({outputs.training_curves_fig.name})" in text  # linked relative to the report


# --- The saved project data ----------------------------------------------------------

_PROJECT = load_config()
_needs_labels = pytest.mark.skipif(
    not _PROJECT.paths.labels_parquet.exists(), reason="run scripts/05_build_labels.py"
)


@_needs_labels
def test_project_training_data_drops_test_labels() -> None:
    data = load_training_data(_PROJECT)
    assert data.years == tuple(range(2017, 2023))
    assert len(data.patient_ids) == 2581
    assert data.groups == tuple(_PROJECT.labels.onset_groups)
    test = torch.from_numpy(data.split == "test")
    assert torch.isnan(data.any_onset[test]).all() and torch.isnan(data.onsets[test]).all()
    assert (~torch.isnan(data.any_onset[~test])).sum() == 335 + 1556  # val + train observed


@_needs_labels
def test_project_age_is_taken_at_the_cutoff() -> None:
    data = load_training_data(_PROJECT)
    patients = load_patients(_PROJECT).set_index("patient_id")
    birth = patients["birth_date"].reindex(list(data.patient_ids))
    cutoff = pd.Timestamp(_PROJECT.labels.cutoff_date.isoformat(), tz="UTC")
    expected = ((cutoff - birth).dt.days / DAYS_PER_YEAR).to_numpy()
    age = data.demographics[:, 0].numpy().astype(np.float64) * data.age_sd + data.age_mean
    np.testing.assert_allclose(age, expected, atol=1e-4)
    # Patients who die after the cutoff are aged at the cutoff, not at death.
    death = patients["death_date"].reindex(list(data.patient_ids))
    later = (death > cutoff).to_numpy()
    assert later.any()
    at_death = ((death - birth).dt.days / DAYS_PER_YEAR).to_numpy()
    assert (age[later] < at_death[later]).all()
    sex = patients["gender"].reindex(list(data.patient_ids)).map({"F": 0.0, "M": 1.0})
    np.testing.assert_array_equal(data.demographics[:, 1].numpy(), sex.to_numpy())


@_needs_labels
def test_project_quick_subset_is_a_renormalized_subgraph() -> None:
    n = 50
    data = load_training_data(_PROJECT, n)
    assert len(data.patient_ids) == n and data.features[0].shape[0] == n
    for A in data.adjacency:
        dense = A.to_dense().numpy()  # tiny test
        np.testing.assert_allclose(dense, dense.T, atol=1e-6)
        assert (np.diag(dense) > 0).all()
        lonely = (dense != 0).sum(axis=1) == 1
        np.testing.assert_allclose(np.diag(dense)[lonely], 1.0)  # self-loop only
