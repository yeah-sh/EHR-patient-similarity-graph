"""Tests for ehrgraph.graph on small hand-made feature matrices and graphs."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from ehrgraph.config import Config, feature_window_years, load_config
from ehrgraph.features import ModelInputs, load_model_inputs
from ehrgraph.graph import (
    WindowGraphs,
    build_graphs,
    cluster_phenotypes,
    drift_jumps,
    eigengap,
    final_window_spectrum,
    graph_path,
    graph_statistics,
    knn_graph,
    laplacian,
    load_graphs,
    neighbourhood_drift,
    neighbourhood_jaccard,
    normalize_adjacency,
    save_graphs,
    smallest_eigenpairs,
    spectral_clusters,
    spectral_embedding,
)

TOL = 1e-10
GROUP_SIZE = 4
ZERO_ROW = 3 * GROUP_SIZE  # the last row: nothing active yet


def _features(seed: int = 42) -> sp.csr_matrix:
    """Three groups of 4 rows near e1, e2 and e3 (5 dims), then one all-zero row."""
    rng = np.random.default_rng(seed)
    base = np.repeat(np.eye(5)[:3], GROUP_SIZE, axis=0)
    rows = base + 0.05 * rng.random(base.shape)
    return sp.csr_matrix(np.vstack([rows, np.zeros((1, 5))]))


def _cosine(x: np.ndarray, y: np.ndarray) -> float:
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def _edges(n: int, pairs: list[tuple[int, int]], weight: float = 1.0) -> sp.csr_matrix:
    rows = [a for a, b in pairs] + [b for a, b in pairs]
    cols = [b for a, b in pairs] + [a for a, b in pairs]
    return sp.csr_matrix((np.full(len(rows), weight), (rows, cols)), shape=(n, n))


def _planted(n_groups: int = 3, size: int = 5, bridge: float = 0.01) -> sp.csr_matrix:
    """Cliques of ``size`` nodes joined in a chain by single weak edges, then one isolated node."""
    n = n_groups * size + 1
    cliques = [
        (g * size + i, g * size + j)
        for g in range(n_groups)
        for i in range(size)
        for j in range(i + 1, size)
    ]
    bridges = [(g * size, (g + 1) * size) for g in range(n_groups - 1)]
    return _edges(n, cliques) + _edges(n, bridges, bridge)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """The real config on windows 2020-2023 with cutoff 2022-01-01 (training 2020-2021)."""
    base = load_config()
    return replace(
        base,
        paths=replace(
            base.paths,
            graphs_dir=tmp_path / "graphs",
            model_cohort_json=tmp_path / "model_cohort.json",
        ),
        trajectories=replace(base.trajectories, start_year=2020, end_year=2023),
        labels=replace(base.labels, cutoff_date=date(2022, 1, 1)),
        graph=replace(base.graph, k=3, spectral_dim=2, eigengap_n_eigenvalues=8),
    )


# --- k-NN graph ----------------------------------------------------------------------


def test_knn_graph_is_symmetric_with_cosine_weights() -> None:
    X = _features()
    W = knn_graph(X, 3, "cosine")
    dense = X.toarray()
    assert (W != W.T).nnz == 0
    assert W.diagonal().sum() == 0 and (W.data > 0).all()
    coo = sp.triu(W).tocoo()
    for i, j, weight in zip(coo.row, coo.col, coo.data):
        assert weight == pytest.approx(_cosine(dense[i], dense[j]))
    # Each node's 3 nearest neighbours are the rest of its group.
    groups = np.arange(ZERO_ROW) // GROUP_SIZE
    assert all(set(groups[W[i].indices]) == {groups[i]} for i in range(ZERO_ROW))


@pytest.mark.parametrize("k", [3, 50])
def test_zero_rows_are_isolated_and_nobodys_neighbour(k: int) -> None:
    W = knn_graph(_features(), k, "cosine", working_memory_mb=1)
    assert W[ZERO_ROW].nnz == 0 and W[:, ZERO_ROW].nnz == 0
    assert (np.diff(W.indptr)[:ZERO_ROW] > 0).all()


def test_knn_graph_drops_non_positive_weights() -> None:
    X = sp.csr_matrix(np.array([[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1]]))
    W = knn_graph(X, 2, "cosine")
    assert sorted(zip(*sp.triu(W).nonzero())) == [(0, 2)]  # row 1 is opposite to both


def test_knn_graph_rejects_other_metrics() -> None:
    with pytest.raises(ValueError, match="cosine"):
        knn_graph(_features(), 3, "euclidean")


# --- A_hat and the Laplacian --------------------------------------------------------


def _random_graph(n: int = 30, seed: int = 42) -> sp.csr_matrix:
    W = sp.random(n, n, density=0.2, random_state=seed, format="csr")
    W = sp.csr_matrix(W.maximum(W.T))
    W.setdiag(0)
    W.eliminate_zeros()
    return W


@pytest.mark.parametrize("make", [lambda: knn_graph(_features(), 3, "cosine"), _random_graph])
def test_a_hat_is_symmetric_with_self_loops_and_eigenvalues_in_range(make) -> None:
    W = make()
    A_hat = normalize_adjacency(W)
    assert abs(A_hat - A_hat.T).max() < TOL
    assert (A_hat.diagonal() > 0).all()  # every node has a self-loop
    eigenvalues = np.linalg.eigvalsh(A_hat.toarray())  # tiny fixture
    assert eigenvalues.min() >= -1 - TOL and eigenvalues.max() <= 1 + TOL


def test_isolated_node_has_only_its_self_loop() -> None:
    A_hat = normalize_adjacency(knn_graph(_features(), 3, "cosine"))
    row = A_hat[ZERO_ROW]
    assert row.indices.tolist() == [ZERO_ROW] and row.data.tolist() == [1.0]


def test_normalize_adjacency_handles_zero_degree() -> None:
    assert (normalize_adjacency(sp.csr_matrix((4, 4))) != sp.identity(4)).nnz == 0
    W = sp.csr_matrix(np.diag([-1.0, 0.0, 0.0]))  # node 0: W + I has degree 0
    A_hat = normalize_adjacency(W)
    assert np.isfinite(A_hat.toarray()).all()
    assert A_hat[0].nnz == 0 or A_hat[0].toarray().sum() == 0


@pytest.mark.parametrize("make", [lambda: knn_graph(_features(), 3, "cosine"), _random_graph])
def test_laplacian_is_positive_semi_definite(make) -> None:
    W = make()
    L = laplacian(W)
    assert abs(L - L.T).max() < TOL
    eigenvalues = np.linalg.eigvalsh(L.toarray())  # tiny fixture
    assert eigenvalues.min() >= -TOL and eigenvalues.max() <= 2 + TOL


def test_laplacian_zeros_count_components_and_isolated_rows_are_identity() -> None:
    L = laplacian(knn_graph(_features(), 3, "cosine"))
    eigenvalues = np.linalg.eigvalsh(L.toarray())
    assert int((np.abs(eigenvalues) < 1e-8).sum()) == 3  # three groups; the zero row is not one
    assert L[ZERO_ROW].indices.tolist() == [ZERO_ROW] and L[ZERO_ROW, ZERO_ROW] == 1.0


# --- Spectra -------------------------------------------------------------------------


def test_smallest_eigenpairs_match_dense_and_are_reproducible() -> None:
    L = laplacian(_planted())
    values, vectors = smallest_eigenpairs(L, 6, seed=42)
    np.testing.assert_allclose(values, np.linalg.eigvalsh(L.toarray())[:6], atol=1e-8)
    np.testing.assert_allclose(L @ vectors, vectors * values, atol=1e-8)
    again = smallest_eigenpairs(L, 6, seed=42)[1]
    np.testing.assert_allclose(again, vectors)


def test_spectral_embedding_skips_trivial_eigenvalues() -> None:
    L = laplacian(knn_graph(_features(), 3, "cosine"))  # three components and an isolated node
    values, embedding = spectral_embedding(L, 2, seed=42)
    dense = np.linalg.eigvalsh(L.toarray())
    np.testing.assert_allclose(values, dense[3:5], atol=1e-8)  # after the three zeros
    assert (values > 1e-8).all() and embedding.shape == (13, 2)
    np.testing.assert_allclose(L @ embedding, embedding * values, atol=1e-8)


def test_eigengap_picks_the_largest_gap_from_min_clusters() -> None:
    eigenvalues = np.array([0.0, 0.01, 0.02, 0.5, 0.55, 0.6])
    k, gaps = eigengap(eigenvalues, 2)
    assert k == 3 and gaps == pytest.approx(np.diff(eigenvalues))
    assert eigengap(np.array([0.0, 0.9, 1.0, 1.02]), 1)[0] == 1  # gaps 0.9, 0.1, 0.02
    assert eigengap(np.array([0.0, 0.9, 1.0, 1.02]), 2)[0] == 2


def test_eigengap_and_clusters_recover_planted_cliques() -> None:
    W = _planted()
    nodes = np.arange(15)  # the isolated node 15 is left out
    L = laplacian(W[nodes][:, nodes])
    values, vectors = smallest_eigenpairs(L, 8, seed=42)
    k, _ = eigengap(values, 2)
    assert k == 3
    labels = spectral_clusters(vectors[:, :k], k, n_init=10, seed=42)
    assert sorted(set(labels)) == [1, 2, 3]
    assert all(len(set(labels[g * 5 : (g + 1) * 5])) == 1 for g in range(3))
    assert len(set(labels[[0, 5, 10]])) == 3


def _window_graphs(weights: list[sp.csr_matrix], years: tuple[int, ...]) -> WindowGraphs:
    n = weights[0].shape[0]
    return WindowGraphs(
        years, tuple(f"p{i}" for i in range(n)), np.arange(n), weights,
        [normalize_adjacency(W) for W in weights],
    )


def test_final_window_spectrum_on_planted_cliques(cfg: Config) -> None:
    W = _planted()
    spectrum = final_window_spectrum(_window_graphs([W, W], (2020, 2021)), cfg)
    assert spectrum.year == 2021 and spectrum.suggested_k == 3
    assert spectrum.nodes.tolist() == list(range(15))  # largest component, not node 15
    assert spectrum.eigenvalues.shape == (8,) and spectrum.eigenvalues[0] == pytest.approx(0, abs=1e-8)
    assert spectrum.embedding.shape == (15, 2) and (spectrum.embedding_eigenvalues > 1e-8).all()
    assert np.bincount(spectrum.labels).tolist() == [0, 5, 5, 5]


def test_final_window_spectrum_requires_the_final_training_window(cfg: Config) -> None:
    W = _planted()
    with pytest.raises(ValueError, match="final window"):
        final_window_spectrum(_window_graphs([W], (2020,)), cfg)


# --- Statistics, drift and phenotypes -------------------------------------------------


def test_graph_statistics() -> None:
    # Triangle 0-1-2 with a pendant 6 on node 0, an edge 3-4, and isolated node 5.
    W = _edges(7, [(0, 1), (1, 2), (0, 2), (0, 6), (3, 4)], 0.5)
    stats = graph_statistics(W)
    assert (stats.nodes, stats.nodes_with_edges, stats.edges) == (7, 6, 5)
    assert stats.mean_degree == pytest.approx(10 / 6)
    assert stats.mean_weight == pytest.approx(0.5)
    assert (stats.components, stats.largest_component) == (2, 4)
    # Local clustering: node 0 has 1 triangle of 3 possible, nodes 1 and 2 are complete.
    assert stats.clustering == pytest.approx((1 / 3 + 1 + 1) / 3)


def test_neighbourhood_jaccard() -> None:
    before = _edges(4, [(0, 1), (0, 2)])
    after = _edges(4, [(0, 1), (0, 3)])
    jaccard = neighbourhood_jaccard(before, after)
    assert jaccard[0] == pytest.approx(1 / 3)  # {1, 2} vs {1, 3}
    assert jaccard[1] == 1.0
    assert np.isnan(jaccard[2]) and np.isnan(jaccard[3])  # no neighbours in one window


def test_drift_flags_a_jump(cfg: Config) -> None:
    ring = _edges(6, [(i, (i + 1) % 6) for i in range(6)])
    other = _edges(6, [(i, (i + 3) % 6) for i in range(3)] + [(0, 2), (1, 3), (2, 4), (3, 5)])
    graphs = _window_graphs([ring, ring, other, other], (2018, 2019, 2020, 2021))
    drift = neighbourhood_drift(graphs, cfg)
    assert drift["from_year"].tolist() == [2018, 2019, 2020]
    assert drift["mean"].tolist()[0] == 1.0 and drift["mean"].tolist()[2] == 1.0
    assert drift["jump"].tolist() == [False, True, False]
    assert drift["unchanged"].iloc[0] == 1.0


@pytest.mark.parametrize(
    ("means", "expected"),
    [
        ([0.405, 0.479, 0.520, 0.567, 0.604], [False] * 5),  # a steady trend is not a jump
        ([0.30, 0.45, 0.60, 0.75], [False] * 4),  # nor a steep one, step 0.15
        ([0.6, 0.3, 0.6], [False, True, False]),  # a dip
        ([0.6, 0.6, 0.3], [False, False, True]),  # a drop in the last pair
        ([0.3, 0.6, 0.6, 0.6], [True, False, False, False]),  # a low first pair
        ([0.6, 0.4], [False, True]),
        ([0.6], [False]),
    ],
)
def test_drift_jumps_flag_dips_not_trends(means: list[float], expected: list[bool]) -> None:
    assert drift_jumps(np.array(means), 0.1).tolist() == expected


def test_cluster_phenotypes_rank_by_prevalence_ratio() -> None:
    # Cluster 1: patients 0-3, cluster 2: patients 4-5. Codes x, y, z.
    presence = sp.csr_matrix(
        np.array([[1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1]])
    )
    labels = np.array([1, 1, 1, 1, 2, 2])
    table = cluster_phenotypes(presence, labels, ["x", "y", "z"], n_top=2, min_patients=2)
    first = table[table["cluster"] == 1]
    assert first["code"].tolist() == ["x", "z"]  # x: 1.0 / (4/6) = 1.5; z: 0.5 / 0.5 = 1.0
    assert first["ratio"].tolist() == pytest.approx([1.5, 1.0])
    second = table[table["cluster"] == 2]
    assert second["code"].tolist() == ["y"]  # z has only 1 patient there
    assert second["ratio"].iloc[0] == pytest.approx(1.0 / 0.5)
    assert (table["cluster_size"].tolist(), table["rank"].tolist()) == ([4, 4, 2], [1, 2, 1])


# --- Building, saving and the training-window guard ------------------------------------


def _inputs(years: tuple[int, ...]) -> ModelInputs:
    X = _features()
    n = X.shape[0]
    return ModelInputs(years, tuple(f"p{i}" for i in range(n)), np.arange(n), [X] * len(years))


def test_build_graphs_only_for_training_windows(cfg: Config) -> None:
    graphs = build_graphs(_inputs(feature_window_years(cfg)), cfg)
    assert graphs.years == (2020, 2021)
    for A_hat in graphs.adjacency:
        assert (A_hat.diagonal() > 0).all()
    with pytest.raises(ValueError, match="close by"):
        build_graphs(_inputs((2021, 2022)), cfg)  # 2022 is after the cutoff


def test_save_and_load_graphs(cfg: Config) -> None:
    W = _planted()
    graphs = _window_graphs([W, W], (2020, 2021))
    spectrum = final_window_spectrum(graphs, cfg)
    cfg.paths.model_cohort_json.write_text(
        json.dumps({"n_patients": 16, "patient_ids": list(graphs.patient_ids), "rows": list(range(16))}),
        encoding="utf-8",
    )
    save_graphs(graphs, spectrum, cfg)
    loaded = load_graphs(cfg)
    assert loaded.years == (2020, 2021) and loaded.patient_ids == graphs.patient_ids
    for a, b in zip(loaded.adjacency, graphs.adjacency):
        assert abs(a - b).max() < TOL
    later = replace(cfg, labels=replace(cfg.labels, cutoff_date=date(2023, 1, 1)))
    with pytest.raises(ValueError, match="re-run"):
        load_graphs(later)


# --- The saved project data ----------------------------------------------------------

_PROJECT = load_config()
_needs_graphs = pytest.mark.skipif(
    not (_PROJECT.paths.graphs_dir / "graphs.json").exists(), reason="run scripts/04_build_graphs.py"
)


@_needs_graphs
def test_saved_graphs_exist_only_for_training_windows() -> None:
    saved = sorted(int(p.stem.split("_")[-1]) for p in _PROJECT.paths.graphs_dir.glob("W_*.npz"))
    assert saved == list(range(2017, 2023))
    assert not list(_PROJECT.paths.graphs_dir.glob("*_2023.npz"))


@_needs_graphs
def test_saved_graphs_match_model_inputs() -> None:
    graphs = load_graphs(_PROJECT)
    inputs = load_model_inputs(_PROJECT)
    assert graphs.patient_ids == inputs.patient_ids
    for year, X, W, A_hat in zip(graphs.years, inputs.matrices, graphs.weights, graphs.adjacency):
        assert abs(A_hat - A_hat.T).max() < TOL, year
        assert (A_hat.diagonal() > 0).all(), year
        zero = np.diff(X.indptr) == 0
        assert (np.diff(W.indptr)[zero] == 0).all(), year  # all-zero rows are isolated
        assert W[:, np.flatnonzero(zero)].nnz == 0, year  # and nobody's neighbour
        assert graph_path(_PROJECT, "A_hat", year).exists()
