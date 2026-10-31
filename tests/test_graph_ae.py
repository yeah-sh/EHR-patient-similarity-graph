"""Tests for the Phase 7b TemporalGraphAE: model, link sampling, targets, masking, training."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from ehrgraph.config import GraphAEConfig, load_config
from ehrgraph.models import (
    LinkSampler,
    OnsetModel,
    TemporalGraphAE,
    build_graph_ae,
    graph_ae_loss,
    link_loss,
    masked_bce,
    pc1_share,
    positive_weights,
)
from ehrgraph.train import (
    GRAPH_AE_RUN_COLUMNS,
    GraphAEData,
    fit_graph_ae,
    graph_ae_states,
    graph_ae_targets,
    load_graph_ae,
    make_graph_ae_data,
    next_window_targets,
    save_graph_ae,
    select_next_window_weight,
    tune_graph_ae,
    with_graphs,
)
from ehrgraph.trajectories import TrajectoryWindows

N, T, F_IN, C = 12, 3, 5, 4
YEARS = (2017, 2018, 2019)
TRAIN_NODES, VAL_NODES, TEST_NODES = range(0, 6), range(6, 9), range(9, 12)
ISOLATED = 11  # a test node with no edge
NAN = float("nan")


def _weights() -> sp.csr_matrix:
    """A ring 0..5, a path 6-7-8 and edges 2-9, 4-10, 7-9; node 11 isolated."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (6, 7), (7, 8), (2, 9), (4, 10),
             (7, 9)]
    rows, cols = zip(*edges)
    upper = sp.csr_matrix((np.linspace(0.5, 1.0, len(edges)), (rows, cols)), shape=(N, N))
    return sp.csr_matrix(upper + upper.T)


def _data(seed: int = 0) -> GraphAEData:
    rng = np.random.default_rng(seed)
    matrices = [sp.csr_matrix(rng.random((N, F_IN))) for _ in YEARS]
    targets = (rng.random((N, T - 1, C)) < 0.4).astype(np.float32)
    targets[3, 0] = np.nan  # a train node not at risk in the first pair
    split = np.array(["train"] * 6 + ["val"] * 3 + ["test"] * 3)
    return make_graph_ae_data([f"p{i}" for i in range(N)], YEARS, [f"c{j}" for j in range(C)],
                              matrices, [_weights()] * T, targets, split)


@pytest.fixture
def ae_cfg() -> GraphAEConfig:
    return replace(load_config().graph_ae, hidden_dim=8, embed_dim=4, dropout=0.0, epochs=12,
                   patience=3, log_every_epochs=2, tuning_next_window_weights=(0.5, 2.0),
                   tuning_seeds=(1, 2))


def _model(ae_cfg: GraphAEConfig, link_only: bool = False) -> TemporalGraphAE:
    torch.manual_seed(0)
    return build_graph_ae(ae_cfg, F_IN, C, link_only).eval()


# --- Model -----------------------------------------------------------------------------


def test_output_shapes(ae_cfg: GraphAEConfig) -> None:
    data = _data()
    out = _model(ae_cfg)(data.features, data.adjacency)
    d = ae_cfg.embed_dim
    assert out.window_embeddings.shape == (N, T, d)
    assert out.states.shape == (N, T, d)
    assert out.next_logits.shape == (N, T - 1, C)
    torch.testing.assert_close(out.trajectory_embeddings, out.states[:, -1])


def test_encoder_is_the_temporal_gcn(ae_cfg: GraphAEConfig) -> None:
    """The GraphAE's encoder and GRU load into the supervised model unchanged."""
    data = _data()
    ae = _model(ae_cfg)
    onset = OnsetModel(F_IN, 2, ae_cfg.hidden_dim, ae_cfg.embed_dim, ae_cfg.dropout).eval()
    onset.temporal.load_state_dict(ae.temporal.state_dict())
    with torch.no_grad():
        expected = ae(data.features, data.adjacency).trajectory_embeddings
        torch.testing.assert_close(onset(data.features, data.adjacency).trajectory_embeddings,
                                   expected)


def test_no_state_reads_a_later_window(ae_cfg: GraphAEConfig) -> None:
    data = _data()
    model = _model(ae_cfg)
    changed = list(data.features)
    changed[-1] = changed[-1] + 5.0
    with torch.no_grad():
        before = model(data.features, data.adjacency)
        after = model(changed, data.adjacency)
    torch.testing.assert_close(after.states[:, :-1], before.states[:, :-1])
    torch.testing.assert_close(after.next_logits, before.next_logits)  # from states 1..T-1
    assert not torch.allclose(after.states[:, -1], before.states[:, -1])


def test_link_only_cuts_the_next_window_gradient(ae_cfg: GraphAEConfig) -> None:
    data = _data()
    targets = graph_ae_targets(data, "train")
    pos_weight = positive_weights(targets.reshape(-1, C))
    for link_only, expect_gradient in ((False, True), (True, False)):
        model = _model(ae_cfg, link_only).train()
        out = model(data.features, data.adjacency)
        parts = graph_ae_loss(out, [], targets, pos_weight, 1.0)  # no link pairs: next term only
        parts.total.backward()
        encoder_grad = model.temporal.encoder.conv1.weight.grad
        has_gradient = encoder_grad is not None and bool(encoder_grad.abs().sum() > 0)
        assert has_gradient == expect_gradient
        assert model.next_window.weight.grad.abs().sum() > 0  # the readout always learns


# --- Link reconstruction ---------------------------------------------------------------


def test_link_sampler_positives_and_negatives() -> None:
    W = _weights()
    sampler = LinkSampler([W, W], seed=3)
    edges = {(int(a), int(b)) for a, b in zip(*sp.triu(W, k=1).nonzero())}
    connected = set(np.flatnonzero(np.diff(W.indptr) > 0))
    for positives, negatives in sampler.sample():
        assert {tuple(map(int, pair)) for pair in positives.T} == edges
        assert negatives.shape == positives.shape  # as many negatives as positives
        for a, b in negatives.T.tolist():
            assert a < b and (a, b) not in edges
            assert a in connected and b in connected  # isolated nodes are skipped
            assert ISOLATED not in (a, b)


def test_link_sampler_is_seeded_and_redraws() -> None:
    W = _weights()
    first = LinkSampler([W], seed=7).sample()[0][1]
    torch.testing.assert_close(LinkSampler([W], seed=7).sample()[0][1], first)
    sampler = LinkSampler([W], seed=7)
    assert not torch.equal(sampler.sample()[0][1], sampler.sample()[0][1])


def test_link_sampler_empty_graph_and_complete_graph() -> None:
    positives, negatives = LinkSampler([sp.csr_matrix((4, 4))], seed=0).sample()[0]
    assert positives.shape == (2, 0) and negatives.shape == (2, 0)
    complete = sp.csr_matrix(np.ones((3, 3)) - np.eye(3))
    with pytest.raises(ValueError, match="no non-edge"):
        LinkSampler([complete], seed=0)


def test_link_loss_matches_bce_formula() -> None:
    Z = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, -1.0]])
    positives, negatives = torch.tensor([[0], [1]]), torch.tensor([[0], [2]])
    loss = link_loss(Z[:, None, :], [(positives, negatives)])
    expected = (F.softplus(-torch.tensor(0.5)) + F.softplus(torch.tensor(0.0))) / 2
    torch.testing.assert_close(loss, expected)
    empty = torch.empty((2, 0), dtype=torch.long)
    assert float(link_loss(Z[:, None, :], [(empty, empty)])) == 0.0


# --- Next-window targets ---------------------------------------------------------------


def _windows() -> TrajectoryWindows:
    """Windows 2017-2021 for patients a, b, c; 2020-2021 are all ones (post-cutoff garbage)."""
    years = (2017, 2018, 2019, 2020, 2021)
    matrices = [sp.csr_matrix(np.full((3, 3), float(year >= 2020)) + np.eye(3)[[0, 1, 2]]
                              * (year < 2020)) for year in years]
    return TrajectoryWindows(years, matrices, ("a", "b", "c"), ("x", "y", "z"))


def _patients() -> pd.DataFrame:
    utc = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    return pd.DataFrame({
        "patient_id": ["a", "b", "c"],
        "birth_date": [utc("1960-01-01"), utc("1950-01-01"), utc("2018-05-01")],
        "death_date": [pd.NaT, utc("2018-06-30"), pd.NaT],
        "gender": ["F", "M", "F"],
    })


def test_next_window_targets_stop_at_the_cutoff() -> None:
    targets = next_window_targets(_windows(), ["a", "b", "c"], ["z", "x"], (2017, 2018, 2019),
                                  _patients(), date(2020, 1, 1))
    assert targets.shape == (3, 2, 2)  # pairs 2017->2018 and 2018->2019 only
    # Patient a: code x active in 2018 and 2019 (identity rows); later windows never read.
    np.testing.assert_array_equal(targets[0], [[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="close by"):
        next_window_targets(_windows(), ["a"], ["x"], (2018, 2019, 2020), _patients(),
                            date(2020, 1, 1))


def test_next_window_targets_mask_patients_not_at_risk() -> None:
    targets = next_window_targets(_windows(), ["a", "b", "c"], ["x", "y", "z"],
                                  (2017, 2018, 2019), _patients(), date(2020, 1, 1))
    # b dies 2018-06-30: alive on 2018-01-01 (pair 1 kept), dead by 2019-01-01 (pair 2 masked).
    assert not np.isnan(targets[1, 0]).any() and np.isnan(targets[1, 1]).all()
    # c is born 2018-05-01: not born by 2018-01-01 (pair 1 masked), born by 2019 (kept).
    assert np.isnan(targets[2, 0]).all() and not np.isnan(targets[2, 1]).any()


# --- Who enters which loss -------------------------------------------------------------


def test_split_targets_mask_every_other_split() -> None:
    data = _data()
    train = graph_ae_targets(data, "train")
    assert torch.isnan(train[list(VAL_NODES) + list(TEST_NODES)]).all()
    torch.testing.assert_close(train[list(TRAIN_NODES)], data.next_targets[list(TRAIN_NODES)],
                               equal_nan=True)
    assert torch.isnan(graph_ae_targets(data, "val")[list(TEST_NODES)]).all()
    with pytest.raises(ValueError, match="test"):
        graph_ae_targets(data, "test")


def test_test_targets_never_enter_training(ae_cfg: GraphAEConfig) -> None:
    """Changing the test patients' next-window targets changes nothing in training."""
    data = _data()
    flipped = data.next_targets.clone()
    flipped[list(TEST_NODES)] = 1.0 - torch.nan_to_num(flipped[list(TEST_NODES)])
    other = replace(data, next_targets=flipped)
    first, second = fit_graph_ae(data, ae_cfg, 5), fit_graph_ae(other, ae_cfg, 5)
    pd.testing.assert_frame_equal(first.history, second.history)
    for a, b in zip(first.model.state_dict().values(), second.model.state_dict().values()):
        torch.testing.assert_close(a, b)


def test_validation_targets_pick_the_epoch_but_are_not_trained_on(ae_cfg: GraphAEConfig) -> None:
    data = _data()
    flipped = data.next_targets.clone()
    flipped[list(VAL_NODES)] = 1.0 - torch.nan_to_num(flipped[list(VAL_NODES)])
    cfg = replace(ae_cfg, patience=ae_cfg.epochs)  # no early stop: compare every epoch
    first = fit_graph_ae(data, cfg, 5)
    second = fit_graph_ae(replace(data, next_targets=flipped), cfg, 5)
    trained = ["objective", "objective_link", "objective_next", "train_next_loss"]
    pd.testing.assert_frame_equal(first.history[trained], second.history[trained])
    assert not np.allclose(first.history["val_next_loss"], second.history["val_next_loss"])


def test_test_nodes_pass_messages(ae_cfg: GraphAEConfig) -> None:
    """A test node's features reach its training neighbours' embeddings (node 9 - node 2)."""
    data = _data()
    model = _model(ae_cfg)
    changed = [X.clone() for X in data.features]
    changed[0][9] += 3.0
    with torch.no_grad():
        before = model(data.features, data.adjacency).window_embeddings[2, 0]
        after = model(changed, data.adjacency).window_embeddings[2, 0]
    assert not torch.allclose(before, after)


def test_test_nodes_enter_the_link_term() -> None:
    sampler = LinkSampler([_weights()], seed=0)
    positives = sampler.sample()[0][0]
    assert 9 in positives.tolist()[0] + positives.tolist()[1]


# --- Training --------------------------------------------------------------------------


def test_early_stopping_on_validation_next_window_loss(ae_cfg: GraphAEConfig) -> None:
    data = _data()
    result = fit_graph_ae(data, replace(ae_cfg, epochs=60, lr=0.05), 1)
    losses = result.history["val_next_loss"].to_numpy()
    assert result.best_epoch == int(np.argmin(losses)) + 1  # the first lowest
    if result.stopped_early:
        assert result.epochs_run == result.best_epoch + ae_cfg.patience
    # The returned weights are the best epoch's: their validation loss is the minimum.
    targets = graph_ae_targets(data, "val")
    with torch.no_grad():
        out = result.model(data.features, data.adjacency)
    assert float(masked_bce(out.next_logits, targets, result.pos_weight)) == pytest.approx(
        losses.min(), rel=1e-5)


def test_history_logs_the_pc1_share(ae_cfg: GraphAEConfig) -> None:
    result = fit_graph_ae(_data(), replace(ae_cfg, patience=ae_cfg.epochs), 1, verbose=True)
    logged = result.history.loc[result.history["pc1_share"].notna(), "epoch"].tolist()
    assert logged == [1, *range(ae_cfg.log_every_epochs, ae_cfg.epochs + 1,
                                ae_cfg.log_every_epochs)]
    assert 0 < result.pc1_share <= 1


def test_fit_is_reproducible(ae_cfg: GraphAEConfig) -> None:
    first, second = fit_graph_ae(_data(), ae_cfg, 4), fit_graph_ae(_data(), ae_cfg, 4)
    pd.testing.assert_frame_equal(first.history, second.history)


def test_no_graph_variant_has_no_link_term(ae_cfg: GraphAEConfig) -> None:
    data = with_graphs(_data(), [sp.csr_matrix((N, N))] * T)
    result = fit_graph_ae(data, ae_cfg, 1)
    assert (result.history["objective_link"] == 0).all()
    assert np.isnan(result.link_auroc)


def test_link_only_variant_trains(ae_cfg: GraphAEConfig) -> None:
    result = fit_graph_ae(_data(), ae_cfg, 1, link_only=True)
    assert result.link_only and result.next_window_weight == 1.0
    assert result.model.detach_next_window


def test_tuning_and_selection(ae_cfg: GraphAEConfig) -> None:
    runs, fits = tune_graph_ae(_data(), ae_cfg)
    assert list(runs.columns) == GRAPH_AE_RUN_COLUMNS
    assert len(runs) == 4 and set(fits) == {(0.5, 1), (0.5, 2), (2.0, 1), (2.0, 2)}
    weight, summary = select_next_window_weight(runs)
    means = runs.groupby("next_window_weight")["val_next_loss"].mean()
    assert weight == means.idxmin()
    assert summary["val_next_loss_mean"].is_monotonic_increasing


def test_save_and_load_round_trip(ae_cfg: GraphAEConfig, tmp_path: Path) -> None:
    data = _data()
    result = fit_graph_ae(data, ae_cfg, 1)
    save_graph_ae(result, tmp_path, "graph_ae")
    model = load_graph_ae(ae_cfg, tmp_path / "graph_ae.pt", F_IN, C)
    np.testing.assert_allclose(graph_ae_states(model, data), graph_ae_states(result.model, data))
    assert (tmp_path / "graph_ae_log.csv").exists()


def test_pc1_share() -> None:
    line = np.outer(np.arange(10.0), [1.0, 2.0, -1.0])
    assert pc1_share(line) == pytest.approx(1.0)
    spread = np.random.default_rng(0).standard_normal((20_000, 4))
    assert pc1_share(spread) == pytest.approx(0.25, abs=0.02)
    assert np.isnan(pc1_share(np.ones((5, 3))))
