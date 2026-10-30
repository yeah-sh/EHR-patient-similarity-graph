"""Tests for ehrgraph.models: GCN layer, temporal GCN, heads, masked losses, smoothness."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from ehrgraph.config import ModelConfig, load_config
from ehrgraph.graph import normalize_adjacency
from ehrgraph.models import (
    EdgeDropper,
    GCNLayer,
    OnsetModel,
    build_model,
    graph_smoothness,
    masked_bce,
    onset_loss,
    positive_weights,
    smoothness_laplacian,
    sparse_tensor,
)
from ehrgraph.train import load_trained_model

N, T, F_IN, G = 10, 3, 5, 4
ISOLATED = 9
NAN = float("nan")


def _weights() -> sp.csr_matrix:
    """A path 0-1-2-3, a triangle 4-5-6, an edge 7-8 and the isolated node 9."""
    edges = [(0, 1, 0.9), (1, 2, 0.5), (2, 3, 0.7), (4, 5, 0.8), (5, 6, 0.6), (4, 6, 0.4),
             (7, 8, 1.0)]
    rows, cols, values = zip(*edges)
    upper = sp.csr_matrix((values, (rows, cols)), shape=(N, N))
    return sp.csr_matrix(upper + upper.T)


def _inputs(seed: int = 0) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """T windows of random features over the same tiny graph."""
    generator = torch.Generator().manual_seed(seed)
    features = [torch.rand(N, F_IN, generator=generator) for _ in range(T)]
    adjacency = [sparse_tensor(normalize_adjacency(_weights())) for _ in range(T)]
    return features, adjacency


@pytest.fixture
def model_cfg() -> ModelConfig:
    """The GCN mechanics, without demographics (tests that need them turn them on)."""
    return replace(
        load_config().model, hidden_dim=8, embed_dim=6, dropout=0.0, use_demographics=False
    )


def _model(model_cfg: ModelConfig, seed: int = 42) -> OnsetModel:
    torch.manual_seed(seed)
    return build_model(model_cfg, F_IN, G).eval()


# --- Layers and shapes -----------------------------------------------------------------


def test_gcn_layer_matches_dense_formula() -> None:
    torch.manual_seed(0)
    layer = GCNLayer(F_IN, 3)
    H = torch.rand(N, F_IN)
    A_hat = normalize_adjacency(_weights())
    dense = torch.from_numpy(A_hat.toarray().astype(np.float32))  # tiny test
    expected = dense @ (H @ layer.weight) + layer.bias
    torch.testing.assert_close(layer(H, sparse_tensor(A_hat)), expected)


def test_gcn_layer_xavier_initialization() -> None:
    torch.manual_seed(0)
    layer = GCNLayer(40, 60)
    bound = math.sqrt(6.0 / (40 + 60))
    assert layer.weight.abs().max() <= bound
    assert layer.weight.abs().max() > 0.9 * bound  # uniform over the full range
    assert torch.equal(layer.bias, torch.zeros(60))


def test_output_shapes(model_cfg: ModelConfig) -> None:
    out = _model(model_cfg)(*_inputs())
    d = model_cfg.embed_dim
    assert out.window_embeddings.shape == (N, T, d)
    assert out.trajectory_embeddings.shape == (N, d)
    assert out.any_onset_logits.shape == (N,)
    assert out.onset_logits.shape == (N, G)


def test_encoder_is_shared_across_windows(model_cfg: ModelConfig) -> None:
    model = _model(model_cfg)
    features, adjacency = _inputs()
    out = model(features, adjacency)
    with torch.no_grad():
        for t in range(T):
            expected = model.temporal.encoder(features[t], adjacency[t])
            torch.testing.assert_close(out.window_embeddings[:, t], expected)
    # The parameter count does not depend on the number of windows.
    assert len(list(model.temporal.encoder.parameters())) == 4  # two layers: W and b each


def test_trajectory_embedding_is_final_gru_state(model_cfg: ModelConfig) -> None:
    model = _model(model_cfg)
    out = model(*_inputs())
    with torch.no_grad():
        sequence, _ = model.temporal.gru(out.window_embeddings)
    torch.testing.assert_close(out.trajectory_embeddings, sequence[:, -1])


def test_message_passing_follows_the_graph(model_cfg: ModelConfig) -> None:
    model = _model(model_cfg)
    features, adjacency = _inputs()
    before = model(features, adjacency).window_embeddings
    changed = [X.clone() for X in features]
    for X in changed:
        X[0] += 1.0  # node 0: neighbour of 1, two hops from 2, three from 3
    after = model(changed, adjacency).window_embeddings
    moved = (after - before).abs().amax(dim=(1, 2))
    assert (moved[[0, 1, 2]] > 0).all()  # within two hops of a 2-layer GCN
    assert (moved[3:] == 0).all()  # beyond two hops, other components, the isolated node


def test_isolated_node_sees_only_itself(model_cfg: ModelConfig) -> None:
    model = _model(model_cfg)
    features, adjacency = _inputs()
    A_hat = normalize_adjacency(_weights())
    assert A_hat[ISOLATED].nnz == 1 and A_hat[ISOLATED, ISOLATED] == pytest.approx(1.0)
    alone = [X[[ISOLATED]] for X in features]
    self_loop = [sparse_tensor(sp.identity(1)) for _ in range(T)]
    full = model(features, adjacency).window_embeddings[ISOLATED]
    torch.testing.assert_close(full, model(alone, self_loop).window_embeddings[0])


def test_temporal_gcn_rejects_mismatched_windows(model_cfg: ModelConfig) -> None:
    features, adjacency = _inputs()
    with pytest.raises(ValueError, match="one adjacency per window"):
        _model(model_cfg)(features, adjacency[:-1], _demographics())


# --- Demographics ----------------------------------------------------------------------


def _demographics(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.cat([torch.randn(N, 1, generator=generator),
                      (torch.arange(N) % 2).float()[:, None]], dim=1)


def test_demographics_enter_the_heads_only(model_cfg: ModelConfig) -> None:
    model = _model(replace(model_cfg, use_demographics=True))
    assert model.n_demographics == 2
    assert model.heads.primary.in_features == model_cfg.embed_dim + 2
    one = model(*_inputs(), _demographics(0))
    two = model(*_inputs(), _demographics(1))
    assert torch.equal(one.window_embeddings, two.window_embeddings)
    assert torch.equal(one.trajectory_embeddings, two.trajectory_embeddings)
    assert not torch.equal(one.any_onset_logits, two.any_onset_logits)
    u = torch.cat([one.trajectory_embeddings, _demographics(0)], dim=1)
    with torch.no_grad():
        torch.testing.assert_close(one.any_onset_logits, model.heads.primary(u).squeeze(-1))


def test_model_without_demographics_ignores_them(model_cfg: ModelConfig) -> None:
    model = _model(model_cfg)
    assert model.n_demographics == 0
    assert model.heads.primary.in_features == model_cfg.embed_dim
    with_them = model(*_inputs(), _demographics())
    torch.testing.assert_close(with_them.any_onset_logits, model(*_inputs()).any_onset_logits)


def test_model_with_demographics_requires_them(model_cfg: ModelConfig) -> None:
    model = _model(replace(model_cfg, use_demographics=True))
    with pytest.raises(ValueError, match="demographics"):
        model(*_inputs())
    with pytest.raises(ValueError, match="demographics"):
        model(*_inputs(), _demographics()[:, :1])


# --- DropEdge --------------------------------------------------------------------------


def _dense(A: torch.Tensor) -> np.ndarray:
    return A.to_dense().numpy().astype(np.float64)  # tiny test


def test_edge_dropper_without_drops_is_the_normalized_adjacency() -> None:
    W = _weights()
    (A,) = EdgeDropper([W], 0.0, seed=0).sample()
    np.testing.assert_allclose(_dense(A), normalize_adjacency(W).toarray(), atol=1e-6)


def test_edge_dropper_drops_undirected_edges_and_renormalizes() -> None:
    W = _weights()
    dropper = EdgeDropper([W, W], 0.5, seed=3)
    samples = [dropper.sample() for _ in range(20)]
    kept_counts = []
    for windows in samples:
        for A in windows:
            dense = _dense(A)
            np.testing.assert_allclose(dense, dense.T, atol=1e-7)
            kept = (dense != 0) & ~np.eye(N, dtype=bool)
            assert not (kept & (W.toarray() == 0)).any()  # only existing edges survive
            thinned = sp.csr_matrix(np.where(kept, W.toarray(), 0.0))
            np.testing.assert_allclose(dense, normalize_adjacency(thinned).toarray(), atol=1e-6)
            assert dense[ISOLATED, ISOLATED] == pytest.approx(1.0)
            kept_counts.append(kept.sum() // 2)
    assert 0 < np.mean(kept_counts) < W.nnz // 2  # some, not all, of the 7 edges
    assert len(set(kept_counts)) > 1  # a fresh draw each time
    again = EdgeDropper([W, W], 0.5, seed=3).sample()
    np.testing.assert_array_equal(_dense(again[0]), _dense(samples[0][0]))


def test_edge_dropper_rejects_a_rate_of_one() -> None:
    with pytest.raises(ValueError, match="DropEdge"):
        EdgeDropper([_weights()], 1.0, seed=0)


# --- Losses ----------------------------------------------------------------------------


def test_masked_bce_uses_only_observed_entries() -> None:
    targets = torch.tensor([[1.0, NAN], [0.0, 1.0], [NAN, 0.0]])
    logits = torch.tensor([[0.3, 5.0], [-1.2, 0.7], [-9.0, 0.1]], requires_grad=True)
    pos_weight = torch.tensor([2.0, 3.0])
    loss = masked_bce(logits, targets, pos_weight)
    seen = ~torch.isnan(targets)
    weights = pos_weight.expand_as(targets)[seen]
    # pos_weight scales the positive term only: -(w y log s(x) + (1 - y) log(1 - s(x))).
    y, x = targets[seen], logits[seen]
    expected = -(weights * y * F.logsigmoid(x) + (1 - y) * F.logsigmoid(-x)).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert (logits.grad[~seen] == 0).all()  # masked entries get no gradient
    assert (logits.grad[seen] != 0).all()


def test_masked_bce_ignores_masked_logits_and_handles_all_masked() -> None:
    targets = torch.tensor([1.0, NAN, 0.0])
    one = masked_bce(torch.tensor([0.5, 100.0, -0.5]), targets, torch.tensor(1.0))
    two = masked_bce(torch.tensor([0.5, -100.0, -0.5]), targets, torch.tensor(1.0))
    assert one == two
    nothing = masked_bce(torch.zeros(3), torch.full((3,), NAN), torch.tensor(1.0))
    assert nothing == 0 and not torch.isnan(nothing)


def test_positive_weights_from_observed_labels() -> None:
    targets = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, NAN, 1.0], [0.0, 0.0, NAN], [NAN, 0.0, 1.0]]
    )
    torch.testing.assert_close(positive_weights(targets), torch.tensor([3.0, 1.0, 1.0]))
    torch.testing.assert_close(positive_weights(targets[:, 0]), torch.tensor(3.0))


def test_onset_loss_weights_the_secondary_term(model_cfg: ModelConfig) -> None:
    out = _model(model_cfg)(*_inputs())
    any_onset = torch.tensor([1.0, 0.0, NAN, 0.0, 1.0, NAN, 0.0, 0.0, 1.0, 0.0])
    onsets = torch.where(torch.rand(N, G) > 0.7, 1.0, 0.0)
    onsets[2] = NAN
    pos_any, pos_groups = torch.tensor(2.0), torch.full((G,), 3.0)
    parts = onset_loss(out, any_onset, onsets, pos_any, pos_groups, 0.5)
    torch.testing.assert_close(parts.primary, masked_bce(out.any_onset_logits, any_onset, pos_any))
    torch.testing.assert_close(parts.secondary, masked_bce(out.onset_logits, onsets, pos_groups))
    torch.testing.assert_close(parts.total, parts.primary + 0.5 * parts.secondary)


# --- Graph smoothness ------------------------------------------------------------------


def test_smoothness_laplacian_leaves_out_isolated_nodes() -> None:
    L = smoothness_laplacian(_weights())
    assert L[ISOLATED].nnz == 0 and L[:, ISOLATED].nnz == 0
    eigenvalues = np.linalg.eigvalsh(L.toarray())  # tiny test
    assert eigenvalues.min() > -1e-9 and eigenvalues.max() <= 2 + 1e-9


def test_graph_smoothness_is_zero_for_sqrt_degree_and_positive_otherwise() -> None:
    W = _weights()
    L = [sparse_tensor(smoothness_laplacian(W))]
    connected = [torch.from_numpy(np.diff(W.indptr) > 0)]
    degree = torch.from_numpy(np.asarray(W.sum(axis=1)).ravel().astype(np.float32))
    smooth = degree.sqrt()[:, None, None].expand(N, 1, 3).clone()
    smooth[ISOLATED] = 5.0  # an isolated node's embedding is not penalized
    assert graph_smoothness(smooth, L, connected) == pytest.approx(0.0, abs=1e-6)
    rough = torch.randn(N, 1, 3, generator=torch.Generator().manual_seed(0))
    value = float(graph_smoothness(rough, L, connected))
    assert 0 < value <= 2


# --- Saving ----------------------------------------------------------------------------


def test_state_dict_round_trip_gives_identical_outputs(
    model_cfg: ModelConfig, tmp_path: Path
) -> None:
    model = _model(model_cfg, seed=1)
    inputs = _inputs()
    expected = model(*inputs)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    loaded = load_trained_model(model_cfg, path, F_IN, G)  # built with other initial weights
    actual = loaded(*inputs)
    for name in ("window_embeddings", "trajectory_embeddings", "any_onset_logits",
                 "onset_logits"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name
