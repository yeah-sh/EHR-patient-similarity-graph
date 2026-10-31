"""Patient-similarity graphs per training-period window, their spectra and drift.

Pipeline (run end to end by ``scripts/04_build_graphs.py``)::

    inputs = load_model_inputs(cfg)  # model cohort, training-period windows only
    graphs = build_graphs(inputs, cfg)  # W and A_hat per window
    stats = [graph_statistics(w) for w in graphs.weights]
    spectrum = final_window_spectrum(graphs, cfg)  # eigengap, embedding, clusters
    phenotypes = cluster_phenotypes(presence, spectrum.labels, codes, ...)
    drift = neighbourhood_drift(graphs, cfg)
    save_graphs(graphs, spectrum, cfg)
    plot_eigenvalues(...); plot_drift(...); write_graph_summary(...)

Nodes are the model-cohort patients (``features.load_model_inputs``), in order.
Graphs exist only for the training-period windows: windows after the prediction
cutoff hold post-cutoff events and never get a graph. For window t:

* ``W_t``: k-NN graph of the window's feature rows; edge weight is the cosine
  similarity, symmetrized with ``max(W, W^T)``, non-positive weights dropped. A
  patient whose row is all zero (nothing active yet) is an isolated node: it is
  left out of the neighbour search, so it is nobody's neighbour.
* ``A_hat_t = D^(-1/2) (W_t + I) D^(-1/2)``: every node, isolated or not, has a
  self-loop.
* ``L_t = I - D^(-1/2) W_t D^(-1/2)``, the normalized Laplacian; an isolated
  node's row is that of I.

No dense N x N matrix is built: the k-NN search computes cosine distances a block
of rows at a time (``graph.knn_working_memory_mb``) and eigenpairs come from
``scipy.sparse.linalg.eigsh``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from matplotlib.figure import Figure
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from ehrgraph.config import Config, feature_window_years
from ehrgraph.data import chart_style, markdown_table, save_figure
from ehrgraph.features import ModelInputs

logger = logging.getLogger(__name__)

GRAPHS_JSON = "graphs.json"
PHENOTYPE_COLUMNS: list[str] = [
    "cluster", "cluster_size", "rank", "code", "patients", "prevalence",
    "overall_prevalence", "ratio",
]
DRIFT_COLUMNS: list[str] = [
    "from_year", "to_year", "patients", "mean", "median", "q25", "q75", "unchanged", "replaced",
    "jump",
]

# Chart tokens from the dataviz reference palette (as in data.py): one series on the
# light surface, secondary ink for annotations.
_SERIES_COLOR = "#2a78d6"
_TEXT_SECONDARY = "#52514e"
_POINTS_PER_INCH = 72
_FIG_SIZE_IN = (9.0, 4.5)
_LINE_PX = 2
_MARKER_PX = 8
_BAND_ALPHA = 0.2  # interquartile band, a light tint of the series colour
_QUARTILES = (25, 50, 75)
_X_MARGIN = 0.08  # air either side of the first and last window pair


@dataclass(frozen=True)
class WindowGraphs:
    """One patient-similarity graph per training-period window.

    Attributes:
        years: The windows, ascending; all close by the prediction cutoff.
        patient_ids: Node labels: the model cohort, in order.
        rows: Each node's row in patient_index.json.
        weights: ``W`` per window, CSR (n, n), symmetric, no self-loops.
        adjacency: ``A_hat`` per window, CSR (n, n), symmetric, with self-loops.
    """

    years: tuple[int, ...]
    patient_ids: tuple[str, ...]
    rows: np.ndarray
    weights: list[sp.csr_matrix]
    adjacency: list[sp.csr_matrix]


@dataclass(frozen=True)
class GraphStats:
    """Summary statistics of one graph ``W`` (see ``graph_statistics``)."""

    nodes: int
    nodes_with_edges: int
    edges: int
    mean_degree: float
    mean_weight: float
    components: int
    largest_component: int
    clustering: float


@dataclass(frozen=True)
class Spectrum:
    """Spectral analysis of the final training window's largest connected component.

    Attributes:
        year: The window.
        nodes: Graph nodes of the largest connected component, ascending.
        eigenvalues: The smallest eigenvalues of its normalized Laplacian, ascending,
            the trivial 0 first.
        gaps: ``eigenvalues[i + 1] - eigenvalues[i]``.
        suggested_k: Number of clusters before the largest eigengap.
        embedding_eigenvalues: The ``spectral_dim`` smallest non-trivial eigenvalues.
        embedding: Their eigenvectors, (len(nodes), spectral_dim).
        labels: Phenotype cluster per node of ``nodes``, 1 = largest cluster.
    """

    year: int
    nodes: np.ndarray
    eigenvalues: np.ndarray
    gaps: np.ndarray
    suggested_k: int
    embedding_eigenvalues: np.ndarray
    embedding: np.ndarray
    labels: np.ndarray


def knn_graph(
    X: sp.spmatrix, k: int, metric: str, working_memory_mb: int | None = None
) -> sp.csr_matrix:
    """Build a symmetric k-NN cosine-similarity graph over the rows of ``X``.

    Each non-zero row is linked to its k nearest other non-zero rows (fewer if
    there are not enough); the edge weight is their cosine similarity. The graph is
    symmetrized with ``W = max(W, W^T)`` and non-positive weights are dropped.
    All-zero rows are left out of the neighbour search: they become isolated nodes
    and are nobody's neighbour.

    Args:
        X: (n, f) sparse feature rows.
        k: Neighbours per node.
        metric: Must be "cosine".
        working_memory_mb: If given, the most memory (MB) sklearn may use per block
            of the distance computation, so no dense n x n matrix is built.

    Returns:
        CSR (n, n), symmetric, weights in (0, 1], no self-loops.

    Raises:
        ValueError: If ``metric`` is not "cosine".
    """
    if metric != "cosine":
        raise ValueError(f"graph.metric {metric!r} is not supported; use 'cosine'")
    X = sp.csr_matrix(X)
    n = X.shape[0]
    active = np.flatnonzero(_nonzero_rows(X))
    n_neighbors = min(k, len(active) - 1)
    if n_neighbors < 1:
        return sp.csr_matrix((n, n))
    search = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    search.fit(X[active])
    memory = (
        sklearn.config_context(working_memory=working_memory_mb)
        if working_memory_mb is not None
        else nullcontext()
    )
    with memory:
        distances, neighbours = search.kneighbors()  # no query point is its own neighbour
    similarity = np.clip(1.0 - distances.ravel(), -1.0, 1.0)
    directed = sp.csr_matrix(
        (similarity, (np.repeat(active, n_neighbors), active[neighbours.ravel()])), shape=(n, n)
    )
    weights = sp.csr_matrix(directed.maximum(directed.T))
    weights.data[weights.data <= 0] = 0
    weights.eliminate_zeros()
    return weights


def normalize_adjacency(W: sp.spmatrix) -> sp.csr_matrix:
    """Return ``A_hat = D^(-1/2) (W + I) D^(-1/2)``, D the degrees of ``W + I``.

    Every node gets a self-loop, so an isolated node has ``A_hat[i, i] = 1``. A
    zero degree (possible only with negative weights) gets ``D^(-1/2) = 0``.

    Args:
        W: Symmetric non-negative (n, n) adjacency without self-loops.

    Returns:
        CSR (n, n), symmetric, with eigenvalues in [-1, 1].
    """
    with_loops = sp.csr_matrix(W, dtype=np.float64) + sp.identity(W.shape[0], format="csr")
    return _symmetric_normalize(with_loops)


def laplacian(W: sp.spmatrix) -> sp.csr_matrix:
    """Return the normalized Laplacian ``L = I - D^(-1/2) W D^(-1/2)``.

    D is the degree of ``W``; an isolated node (degree 0) gets ``D^(-1/2) = 0``, so
    its row of L is that of I (eigenvalue 1). L is positive semi-definite with
    eigenvalues in [0, 2] and one 0 per connected component with an edge.

    Args:
        W: Symmetric non-negative (n, n) adjacency without self-loops.

    Returns:
        CSR (n, n).
    """
    n = W.shape[0]
    return sp.csr_matrix(sp.identity(n, format="csr") - _symmetric_normalize(W))


def smallest_eigenpairs(L: sp.spmatrix, m: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``m`` smallest eigenpairs of a symmetric sparse matrix, with eigsh.

    The start vector is drawn with ``seed`` and each eigenvector's sign is fixed
    (its largest-magnitude entry is positive), so results are reproducible.

    Args:
        L: Symmetric (n, n), e.g. ``laplacian(W)``.
        m: Number of eigenpairs, < n - 1.
        seed: Random seed for the start vector.

    Returns:
        Eigenvalues (m,), ascending, and eigenvectors (n, m) as columns.

    Raises:
        ValueError: If ``m`` is too large for ``L``.
    """
    n = L.shape[0]
    if not 0 < m < n - 1:
        raise ValueError(f"cannot compute {m} eigenpairs of a {n} x {n} matrix with eigsh")
    v0 = np.random.default_rng(seed).standard_normal(n)
    values, vectors = eigsh(sp.csr_matrix(L, dtype=np.float64), k=m, which="SA", v0=v0)
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    peaks = vectors[np.abs(vectors).argmax(axis=0), np.arange(m)]
    return values, vectors * np.where(peaks < 0, -1.0, 1.0)


def spectral_embedding(L: sp.spmatrix, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``d`` smallest non-trivial eigenpairs of a normalized Laplacian.

    The trivial eigenvalues are the 0s, one per connected component with an edge
    (read off L's off-diagonal pattern); they are computed and skipped.

    Args:
        L: Output of ``laplacian``.
        d: Embedding dimension.
        seed: Random seed (see ``smallest_eigenpairs``).

    Returns:
        Eigenvalues (d,), ascending, and the embedding (n, d).
    """
    off_diagonal = sp.csr_matrix(L - sp.diags(L.diagonal()))
    _, labels = connected_components(off_diagonal != 0, directed=False)
    n_trivial = int((np.bincount(labels) >= 2).sum())
    values, vectors = smallest_eigenpairs(L, d + n_trivial, seed)
    logger.info(
        "Spectral embedding: skipped %d trivial eigenvalues (max %.2e); next %d: %s",
        n_trivial,
        values[:n_trivial].max() if n_trivial else 0.0,
        d,
        np.round(values[n_trivial:], 4).tolist(),
    )
    return values[n_trivial:], vectors[:, n_trivial:]


def eigengap(eigenvalues: np.ndarray, min_clusters: int) -> tuple[int, np.ndarray]:
    """Suggest a number of clusters from the largest gap in a sorted spectrum.

    With ``g_i = eigenvalues[i] - eigenvalues[i - 1]`` (1-based i), the suggestion
    is the k >= ``min_clusters`` with the largest gap after the k-th eigenvalue.

    Args:
        eigenvalues: Smallest Laplacian eigenvalues, ascending.
        min_clusters: Smallest k considered.

    Returns:
        The suggested k and all gaps (len(eigenvalues) - 1,).

    Raises:
        ValueError: If there is no gap at or after ``min_clusters``.
    """
    gaps = np.diff(eigenvalues)
    if not 1 <= min_clusters <= len(gaps):
        raise ValueError(f"need more than {min_clusters} eigenvalues, got {len(eigenvalues)}")
    return min_clusters + int(np.argmax(gaps[min_clusters - 1 :])), gaps


def spectral_clusters(vectors: np.ndarray, k: int, n_init: int, seed: int) -> np.ndarray:
    """Cluster nodes by k-means on their rows of the first k eigenvectors.

    Rows are L2-normalized first (Ng, Jordan and Weiss). Clusters are numbered
    1 to k by size, largest first (ties by first member).

    Args:
        vectors: (n, k) eigenvectors of the normalized Laplacian, trivial one included.
        k: Number of clusters.
        n_init: k-means restarts.
        seed: Random seed.

    Returns:
        Cluster number per node, (n,).
    """
    raw = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(normalize(vectors))
    sizes = np.bincount(raw, minlength=k)
    first = np.array([np.flatnonzero(raw == c)[0] for c in range(k)])
    order = np.lexsort((first, -sizes))  # cluster ids, largest first
    renumber = np.empty(k, dtype=np.int64)
    renumber[order] = np.arange(1, k + 1)
    return renumber[raw]


def largest_component(W: sp.spmatrix) -> np.ndarray:
    """Return the nodes of the largest connected component of ``W``, ascending."""
    _, labels = connected_components(sp.csr_matrix(W) != 0, directed=False)
    return np.flatnonzero(labels == np.bincount(labels).argmax())


def graph_statistics(W: sp.spmatrix) -> GraphStats:
    """Summarize a graph: size, degree, components and clustering.

    Degrees count neighbours (unweighted); mean degree is over nodes with an edge.
    Components count those with an edge (isolated nodes are not components here).
    The clustering coefficient is the mean local clustering of nodes with degree
    >= 2, from the triangles ``diag(B^3) / 2`` of the binary adjacency B.

    Args:
        W: Symmetric adjacency without self-loops.

    Returns:
        The statistics.
    """
    binary = sp.csr_matrix((sp.csr_matrix(W) != 0).astype(np.float64))
    degree = np.diff(binary.indptr)
    with_edges = degree > 0
    _, labels = connected_components(binary, directed=False)
    sizes = np.bincount(labels)
    triangles = np.asarray((binary @ binary).multiply(binary).sum(axis=1)).ravel() / 2
    eligible = degree >= 2
    local = 2 * triangles[eligible] / (degree[eligible] * (degree[eligible] - 1))
    return GraphStats(
        nodes=W.shape[0],
        nodes_with_edges=int(with_edges.sum()),
        edges=binary.nnz // 2,
        mean_degree=float(degree[with_edges].mean()) if with_edges.any() else 0.0,
        mean_weight=float(sp.csr_matrix(W).data.mean()) if binary.nnz else 0.0,
        components=int((sizes >= 2).sum()),
        largest_component=int(sizes.max()),
        clustering=float(local.mean()) if eligible.any() else 0.0,
    )


def neighbourhood_jaccard(W_from: sp.spmatrix, W_to: sp.spmatrix) -> np.ndarray:
    """Return each node's neighbourhood Jaccard between two graphs on the same nodes.

    ``|N_from(i) & N_to(i)| / |N_from(i) | N_to(i)|``; 1 means the same neighbours,
    0 all new ones. NaN for nodes without neighbours in either graph.

    Args:
        W_from: Earlier graph.
        W_to: Later graph.

    Returns:
        (n,) floats.
    """
    a = sp.csr_matrix((sp.csr_matrix(W_from) != 0).astype(np.float64))
    b = sp.csr_matrix((sp.csr_matrix(W_to) != 0).astype(np.float64))
    common = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    size_a, size_b = np.diff(a.indptr), np.diff(b.indptr)
    both = (size_a > 0) & (size_b > 0)
    jaccard = np.full(a.shape[0], np.nan)
    jaccard[both] = common[both] / (size_a + size_b - common)[both]
    return jaccard


def neighbourhood_drift(graphs: WindowGraphs, cfg: Config) -> pd.DataFrame:
    """Summarize neighbourhood Jaccard between consecutive windows.

    Only patients with neighbours in both windows count. Jumps are flagged by
    ``drift_jumps`` on the pairs' mean Jaccard.

    Args:
        graphs: Output of ``build_graphs``.
        cfg: Project configuration.

    Returns:
        ``DRIFT_COLUMNS``, one row per consecutive pair; unchanged and replaced are
        the shares of patients with Jaccard 1 and 0.
    """
    rows = []
    for t in range(len(graphs.years) - 1):
        jaccard = neighbourhood_jaccard(graphs.weights[t], graphs.weights[t + 1])
        values = jaccard[~np.isnan(jaccard)]
        q25, median, q75 = np.percentile(values, _QUARTILES)
        rows.append(
            {
                "from_year": graphs.years[t],
                "to_year": graphs.years[t + 1],
                "patients": len(values),
                "mean": values.mean(),
                "median": median,
                "q25": q25,
                "q75": q75,
                "unchanged": float((values == 1).mean()),
                "replaced": float((values == 0).mean()),
            }
        )
    drift = pd.DataFrame(rows)
    drift["jump"] = drift_jumps(drift["mean"].to_numpy(), cfg.graph.drift_jump_threshold)
    return drift[DRIFT_COLUMNS]


def drift_jumps(means: np.ndarray, threshold: float) -> np.ndarray:
    """Flag consecutive-window pairs whose neighbourhoods change abruptly.

    A pair is a jump if its mean Jaccard is below that of each adjacent pair and more
    than ``threshold`` below what they predict on a straight line: the mean of its
    two neighbours, or at either end the extrapolation of the next two pairs (the
    single neighbour if there are only two pairs). A steady trend, however steep,
    flags nothing; a dip or a final drop does.

    Args:
        means: Mean neighbourhood Jaccard per consecutive pair, in time order.
        threshold: ``graph.drift_jump_threshold``.

    Returns:
        Boolean per pair; all False with a single pair.
    """
    n = len(means)
    jumps = np.zeros(n, dtype=bool)
    for t in range(n):
        adjacent = means[[s for s in (t - 1, t + 1) if 0 <= s < n]]
        if adjacent.size == 0:
            continue
        if 0 < t < n - 1:
            expected = adjacent.mean()
        elif n >= 3:
            near, far = (means[1], means[2]) if t == 0 else (means[-2], means[-3])
            expected = 2 * near - far
        else:
            expected = adjacent[0]
        jumps[t] = means[t] < adjacent.min() and means[t] < expected - threshold
    return jumps


def build_graphs(inputs: ModelInputs, cfg: Config) -> WindowGraphs:
    """Build ``W`` and ``A_hat`` for every training-period window of ``inputs``.

    Args:
        inputs: Output of ``features.load_model_inputs``.
        cfg: Project configuration (``graph`` section).

    Returns:
        One graph per window.

    Raises:
        ValueError: If a window does not close by the prediction cutoff.
    """
    allowed = feature_window_years(cfg)
    late = [year for year in inputs.years if year not in allowed]
    if late:
        raise ValueError(
            f"graphs are built only for windows that close by {cfg.labels.cutoff_date}; "
            f"got {late}"
        )
    graph_cfg = cfg.graph
    weights, adjacency = [], []
    for year, features in zip(inputs.years, inputs.matrices):
        W = knn_graph(features, graph_cfg.k, graph_cfg.metric, graph_cfg.knn_working_memory_mb)
        weights.append(W)
        adjacency.append(normalize_adjacency(W))
        logger.info(
            "Graph %d: %d nodes, %d with edges, %d edges",
            year,
            W.shape[0],
            int((np.diff(W.indptr) > 0).sum()),
            W.nnz // 2,
        )
    return WindowGraphs(inputs.years, inputs.patient_ids, inputs.rows, weights, adjacency)


def final_window_spectrum(graphs: WindowGraphs, cfg: Config) -> Spectrum:
    """Analyse the spectrum of the final training window's largest component.

    Computes the first ``eigengap_n_eigenvalues`` eigenvalues of its normalized
    Laplacian, the eigengap suggestion k, the ``spectral_dim``-dimensional spectral
    embedding, and k phenotype clusters (``spectral_clusters``).

    Args:
        graphs: Output of ``build_graphs``; its last window must be the final
            training-period window.
        cfg: Project configuration.

    Returns:
        The spectral analysis.

    Raises:
        ValueError: If the last graph is not the final training-period window.
    """
    final_year = feature_window_years(cfg)[-1]
    if graphs.years[-1] != final_year:
        raise ValueError(f"the last graph is {graphs.years[-1]}, not the final window {final_year}")
    graph_cfg = cfg.graph
    nodes = largest_component(graphs.weights[-1])
    L = laplacian(graphs.weights[-1][nodes][:, nodes])
    eigenvalues, vectors = smallest_eigenpairs(L, graph_cfg.eigengap_n_eigenvalues, cfg.seed)
    k, gaps = eigengap(eigenvalues, graph_cfg.eigengap_min_clusters)
    embedding_values, embedding = spectral_embedding(L, graph_cfg.spectral_dim, cfg.seed)
    labels = spectral_clusters(vectors[:, :k], k, graph_cfg.kmeans_n_init, cfg.seed)
    logger.info(
        "Spectrum %d: largest component %d of %d nodes; eigenvalues %s; suggested k = %d; "
        "cluster sizes %s",
        final_year,
        len(nodes),
        graphs.weights[-1].shape[0],
        np.round(eigenvalues, 4).tolist(),
        k,
        np.bincount(labels)[1:].tolist(),
    )
    return Spectrum(final_year, nodes, eigenvalues, gaps, k, embedding_values, embedding, labels)


def cluster_phenotypes(
    presence: sp.spmatrix,
    labels: np.ndarray,
    codes: Sequence[str],
    n_top: int,
    min_patients: int,
) -> pd.DataFrame:
    """List each cluster's most over-represented conditions.

    A condition's ratio in a cluster is its prevalence there over its prevalence
    among all clustered patients. Only conditions present in at least
    ``min_patients`` of the cluster's patients qualify; ties are broken by
    patients, then by code.

    Args:
        presence: Binary (n, n_codes), e.g. ``features.chronic_presence`` rows of
            the clustered patients.
        labels: Cluster per row of ``presence``.
        codes: Column labels.
        n_top: Conditions per cluster.
        min_patients: Minimum patients with the condition in the cluster.

    Returns:
        ``PHENOTYPE_COLUMNS``, by cluster then rank.
    """
    presence = sp.csr_matrix(presence)
    overall = np.asarray(presence.sum(axis=0)).ravel() / presence.shape[0]
    frames = []
    for cluster in np.unique(labels):
        members = labels == cluster
        counts = np.asarray(presence[members].sum(axis=0)).ravel()
        prevalence = counts / members.sum()
        ratio = np.divide(prevalence, overall, out=np.zeros_like(prevalence), where=overall > 0)
        frame = pd.DataFrame(
            {
                "code": list(codes),
                "patients": counts.astype(np.int64),
                "prevalence": prevalence,
                "overall_prevalence": overall,
                "ratio": ratio,
            }
        )
        top = (
            frame[frame["patients"] >= min_patients]
            .sort_values(["ratio", "patients", "code"], ascending=[False, False, True])
            .head(n_top)
            .assign(cluster=int(cluster), cluster_size=int(members.sum()))
        )
        top["rank"] = np.arange(1, len(top) + 1)
        frames.append(top)
    return pd.concat(frames, ignore_index=True)[PHENOTYPE_COLUMNS]


def graph_path(cfg: Config, name: str, year: int) -> Path:
    """Return the npz file of graph ``name`` ("W" or "A_hat") for the window ``year``."""
    return cfg.paths.graphs_dir / f"{name}_{year}.npz"


def save_graphs(graphs: WindowGraphs, spectrum: Spectrum, cfg: Config) -> None:
    """Write the graphs, their metadata, the final window's spectrum and its clusters.

    Files in ``cfg.paths.graphs_dir``: W_<year>.npz and A_hat_<year>.npz per
    window; graphs.json; spectral_<year>.npz (largest-component nodes, eigenvalues,
    embedding, cluster labels) and phenotype_clusters_<year>.csv for the final window.

    Args:
        graphs: Output of ``build_graphs``.
        spectrum: Output of ``final_window_spectrum``.
        cfg: Project configuration.
    """
    for year, W, A_hat in zip(graphs.years, graphs.weights, graphs.adjacency):
        _write_npz(graph_path(cfg, "W", year), W)
        _write_npz(graph_path(cfg, "A_hat", year), A_hat)
    graph_cfg = cfg.graph
    _write_json(
        cfg.paths.graphs_dir / GRAPHS_JSON,
        {
            "years": list(graphs.years),
            "cutoff_date": cfg.labels.cutoff_date.isoformat(),
            "n_nodes": len(graphs.patient_ids),
            "nodes": "the patient_ids of model_cohort.json, in order",
            "k": graph_cfg.k,
            "metric": graph_cfg.metric,
            "final_year": spectrum.year,
            "suggested_k": spectrum.suggested_k,
            "seed": cfg.seed,
        },
    )
    np.savez(
        cfg.paths.graphs_dir / f"spectral_{spectrum.year}.npz",
        nodes=spectrum.nodes,
        eigenvalues=spectrum.eigenvalues,
        embedding_eigenvalues=spectrum.embedding_eigenvalues,
        embedding=spectrum.embedding,
        labels=spectrum.labels,
    )
    clusters = pd.DataFrame(
        {
            "patient_id": [graphs.patient_ids[i] for i in spectrum.nodes],
            "cluster": spectrum.labels,
        }
    )
    path = phenotype_clusters_path(cfg, spectrum.year)
    clusters.to_csv(path, index=False)
    logger.info("Wrote %s", path)


def phenotype_clusters_path(cfg: Config, year: int) -> Path:
    """Return the csv of the phenotype cluster of each patient in the window ``year``."""
    return cfg.paths.graphs_dir / f"phenotype_clusters_{year}.csv"


def load_phenotype_clusters(cfg: Config) -> pd.Series:
    """Read the final training window's phenotype clusters written by ``save_graphs``.

    For analysis only: phenotypes are never a training target.

    Args:
        cfg: Project configuration.

    Returns:
        Cluster number (1 = largest) indexed by patient_id.
    """
    path = phenotype_clusters_path(cfg, feature_window_years(cfg)[-1])
    clusters = pd.read_csv(path, dtype={"patient_id": "string", "cluster": "int64"})
    return clusters.set_index("patient_id")["cluster"]


def load_graphs(cfg: Config) -> WindowGraphs:
    """Read the graphs written by ``save_graphs``.

    Args:
        cfg: Project configuration.

    Returns:
        The graphs of the training-period windows, nodes in model-cohort order.

    Raises:
        ValueError: If they were built for another cutoff, other windows or another
            model cohort.
    """
    meta = json.loads((cfg.paths.graphs_dir / GRAPHS_JSON).read_text(encoding="utf-8"))
    cohort = json.loads(cfg.paths.model_cohort_json.read_text(encoding="utf-8"))
    years = feature_window_years(cfg)
    if (
        meta["cutoff_date"] != cfg.labels.cutoff_date.isoformat()
        or meta["years"] != list(years)
        or meta["n_nodes"] != cohort["n_patients"]
    ):
        raise ValueError(
            f"{GRAPHS_JSON} does not match the cutoff, training windows and model cohort; "
            "re-run scripts/04_build_graphs.py"
        )
    return WindowGraphs(
        years=years,
        patient_ids=tuple(cohort["patient_ids"]),
        rows=np.asarray(cohort["rows"], dtype=np.int64),
        weights=[sp.load_npz(graph_path(cfg, "W", y)).tocsr() for y in years],
        adjacency=[sp.load_npz(graph_path(cfg, "A_hat", y)).tocsr() for y in years],
    )


def plot_eigenvalues(spectrum: Spectrum, path: Path, cfg: Config) -> None:
    """Save the smallest Laplacian eigenvalues, with the largest eigengap marked.

    Args:
        spectrum: Output of ``final_window_spectrum``.
        path: Output PNG.
        cfg: Project configuration (``summary`` section).
    """
    dpi = cfg.summary.figure_dpi
    px = _POINTS_PER_INCH / dpi
    index = np.arange(1, len(spectrum.eigenvalues) + 1)
    k = spectrum.suggested_k
    with matplotlib.rc_context(chart_style(dpi, cfg.summary.figure_save_dpi)):
        fig = Figure(figsize=_FIG_SIZE_IN, layout="constrained")
        ax = fig.subplots()
        ax.plot(
            index,
            spectrum.eigenvalues,
            color=_SERIES_COLOR,
            linewidth=_LINE_PX * px,
            marker="o",
            markersize=_MARKER_PX * px,
        )
        ax.axvline(k + 0.5, color=_TEXT_SECONDARY, linewidth=px, linestyle="--")
        ax.annotate(
            f"largest gap (k >= {cfg.graph.eigengap_min_clusters}): after "
            f"λ{k}, so k = {k}",
            xy=(k + 0.5, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            color=_TEXT_SECONDARY,
        )
        ax.set_title(
            f"Smallest eigenvalues of the normalized Laplacian, {spectrum.year} graph "
            f"(largest component, {len(spectrum.nodes):,} patients)"
        )
        ax.set_xlabel("Index i")
        ax.set_ylabel("Eigenvalue λ_i")
        ax.set_xticks(index)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        save_figure(fig, path)


def plot_drift(drift: pd.DataFrame, path: Path, cfg: Config) -> None:
    """Save the neighbourhood Jaccard between consecutive windows: mean and IQR band.

    Args:
        drift: Output of ``neighbourhood_drift``.
        path: Output PNG.
        cfg: Project configuration (``summary`` section).
    """
    dpi = cfg.summary.figure_dpi
    px = _POINTS_PER_INCH / dpi
    x = np.arange(len(drift))
    labels = [f"{a}→{b}" for a, b in zip(drift["from_year"], drift["to_year"])]
    with matplotlib.rc_context(chart_style(dpi, cfg.summary.figure_save_dpi)):
        fig = Figure(figsize=_FIG_SIZE_IN, layout="constrained")
        ax = fig.subplots()
        ax.fill_between(x, drift["q25"], drift["q75"], color=_SERIES_COLOR, alpha=_BAND_ALPHA,
                        linewidth=0)
        ax.plot(x, drift["mean"], color=_SERIES_COLOR, linewidth=_LINE_PX * px, marker="o",
                markersize=_MARKER_PX * px)
        ax.annotate("mean", xy=(x[-1], drift["mean"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=_TEXT_SECONDARY)
        ax.annotate("interquartile range", xy=(x[0], drift["q25"].iloc[0]), xytext=(4, -4),
                    textcoords="offset points", va="top", color=_TEXT_SECONDARY)
        ax.set_title("Neighbourhood overlap between consecutive windows (1 = same neighbours)")
        ax.set_ylabel("Neighbourhood Jaccard")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.margins(x=_X_MARGIN)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        save_figure(fig, path)


def write_graph_summary(
    graphs: WindowGraphs,
    stats: Sequence[GraphStats],
    spectrum: Spectrum,
    phenotypes: pd.DataFrame,
    presence: sp.spmatrix,
    drift: pd.DataFrame,
    descriptions: Mapping[str, str],
    cfg: Config,
) -> str:
    """Write the graph report to ``cfg.paths.graphs_summary_md``.

    Args:
        graphs: Output of ``build_graphs``.
        stats: ``graph_statistics`` per window.
        spectrum: Output of ``final_window_spectrum``.
        phenotypes: Output of ``cluster_phenotypes``.
        presence: The chronic presence rows of ``spectrum.nodes`` (as clustered).
        drift: Output of ``neighbourhood_drift``.
        descriptions: Code -> description.
        cfg: Project configuration.

    Returns:
        The markdown written.
    """
    graph_cfg, paths = cfg.graph, cfg.paths
    n = len(graphs.patient_ids)
    k = spectrum.suggested_k
    gaps = spectrum.gaps
    considered = np.arange(graph_cfg.eigengap_min_clusters, len(gaps) + 1)  # k values
    by_gap = considered[np.argsort(-gaps[considered - 1], kind="stable")]
    runner_up = int(by_gap[1]) if len(by_gap) > 1 else k

    def figure(path: Path) -> str:
        return path.relative_to(paths.graphs_summary_md.parent).as_posix()

    stats_rows = [
        [str(year), f"{s.nodes_with_edges:,}", f"{s.nodes - s.nodes_with_edges:,}", f"{s.edges:,}",
         f"{s.mean_degree:.1f}", f"{s.mean_weight:.3f}", f"{s.components:,}",
         f"{s.largest_component:,}", f"{s.clustering:.3f}"]
        for year, s in zip(graphs.years, stats)
    ]
    eigen_rows = [
        [str(i), f"{value:.4f}", f"{gap:.4f}" if i <= len(gaps) else ""]
        for i, (value, gap) in enumerate(zip(spectrum.eigenvalues, [*gaps, np.nan]), start=1)
    ]
    presence = sp.csr_matrix(presence)
    n_chronic = np.diff(presence.indptr)
    cluster_rows = []
    for cluster in range(1, k + 1):
        members = spectrum.labels == cluster
        cluster_rows.append(
            [str(cluster), f"{int(members.sum()):,}", f"{(n_chronic[members] > 0).mean():.0%}",
             f"{n_chronic[members].mean():.1f}"]
        )
    phenotype_rows = []
    for cluster in range(1, k + 1):
        rows = phenotypes[phenotypes["cluster"] == cluster]
        phenotype_rows += [
            [str(r.cluster), str(r.rank), descriptions[r.code], f"{r.patients:,}",
             f"{r.prevalence:.0%}", f"{r.overall_prevalence:.0%}", f"{r.ratio:.1f}"]
            for r in rows.itertuples(index=False)
        ] or [[str(cluster), "-", f"(no chronic condition in {graph_cfg.phenotype_min_patients} "
               "or more of its patients)", "", "", "", ""]]
    drift_rows = [
        [f"{r.from_year}→{r.to_year}", f"{r.patients:,}", f"{r.mean:.3f}", f"{r.median:.3f}",
         f"{r.q25:.3f}-{r.q75:.3f}", f"{r.unchanged:.1%}", f"{r.replaced:.1%}",
         "yes" if r.jump else "no"]
        for r in drift.itertuples(index=False)
    ]
    jumps = drift[drift["jump"]]
    means = drift["mean"].to_numpy()
    trend = (
        "rises every year" if (np.diff(means) > 0).all()
        else "falls every year" if (np.diff(means) < 0).all()
        else "moves"
    )
    jump_text = (
        "No jump: no pair's mean Jaccard is both below its adjacent pairs and more than "
        f"{graph_cfg.drift_jump_threshold:g} below the straight line through them. The mean "
        f"{trend}, from {means[0]:.3f} "
        f"({drift['from_year'].iloc[0]}→{drift['to_year'].iloc[0]}) to {means[-1]:.3f} "
        f"({drift['from_year'].iloc[-1]}→{drift['to_year'].iloc[-1]})."
        if jumps.empty
        else f"Jumps (mean Jaccard more than {graph_cfg.drift_jump_threshold:g} below the "
        "straight line through the adjacent pairs): "
        + ", ".join(f"{r.from_year}→{r.to_year} ({r.mean:.3f})" for r in jumps.itertuples())
        + "."
    )
    continuum = (
        f" The largest gap of all follows λ1 ({gaps[0]:.4f}), and the spectrum rises "
        f"smoothly from λ2 = {spectrum.eigenvalues[1]:.4f} to "
        f"λ{len(spectrum.eigenvalues)} = {spectrum.eigenvalues[-1]:.4f}: the graph is "
        f"one continuum without sharply separated communities, and the {k} clusters are a "
        "partition of it rather than disconnected groups."
        if gaps[0] > gaps[k - 1]
        else ""
    )

    lines = [
        "# Patient-similarity graphs",
        "",
        "Built by `scripts/04_build_graphs.py` from `load_model_inputs`: the "
        f"{n:,} model-cohort patients and the training-period windows "
        f"{graphs.years[0]}-{graphs.years[-1]} only; windows after the cutoff "
        f"({cfg.labels.cutoff_date:%Y-%m-%d}) get no graph. W is the {graph_cfg.k}-NN graph "
        f"of each window's feature rows, edge weight = {graph_cfg.metric} similarity, "
        "symmetrized with max(W, W^T), non-positive weights dropped. A patient whose row is "
        "all zero in a window is an isolated node there: left out of the neighbour search, "
        "with only the self-loop of A_hat = D^(-1/2) (W + I) D^(-1/2). "
        "L = I - D^(-1/2) W D^(-1/2).",
        "",
        "## Graph statistics per window",
        "",
        markdown_table(
            ["Year", "Nodes with edges", "Isolated", "Edges", "Mean degree", "Mean weight",
             "Components", "Largest component", "Clustering"],
            stats_rows,
            "rrrrrrrrr",
        ),
        "",
        "Mean degree is over nodes with an edge (k-NN symmetrized, so at least "
        f"{graph_cfg.k}); components count those with an edge; clustering is the mean "
        "local clustering coefficient of nodes with degree >= 2, computed exactly.",
        "",
        f"## Spectrum of the final window ({spectrum.year})",
        "",
        f"Normalized Laplacian of the largest connected component ({len(spectrum.nodes):,} "
        f"of {n:,} patients); the first {len(spectrum.eigenvalues)} eigenvalues, from eigsh.",
        "",
        f"![Smallest Laplacian eigenvalues]({figure(paths.eigenvalues_fig)})",
        "",
        markdown_table(["i", "Eigenvalue", "Gap to next"], eigen_rows, "rrr"),
        "",
        f"**Eigengap.** Among k >= {graph_cfg.eigengap_min_clusters}, the largest gap is "
        f"after λ{k} ({spectrum.eigenvalues[k - 1]:.4f} to "
        f"{spectrum.eigenvalues[k]:.4f}, gap {gaps[k - 1]:.4f}), so the eigengap heuristic "
        f"suggests {k} clusters. The next largest gap is after λ{runner_up} "
        f"({gaps[runner_up - 1]:.4f}); the largest is {gaps[k - 1] / gaps[runner_up - 1]:.1f} "
        f"times as large.{continuum}",
        "",
        f"Spectral embedding: the {len(spectrum.embedding_eigenvalues)} smallest non-trivial "
        "eigenvalues are "
        + ", ".join(f"{v:.4f}" for v in spectrum.embedding_eigenvalues)
        + ".",
        "",
        f"## Phenotypes ({k} clusters)",
        "",
        f"Spectral clustering of the {spectrum.year} graph: k-means ({graph_cfg.kmeans_n_init} "
        f"restarts) on the first {k} eigenvectors, rows L2-normalized. Chronic conditions are "
        "those with a chronic episode (no STOP or longer than "
        f"{cfg.features.chronic_min_duration_days} days) active in "
        f"{graphs.years[0]}-{graphs.years[-1]}.",
        "",
        markdown_table(
            ["Cluster", "Patients", "With a chronic condition", "Mean chronic conditions"],
            cluster_rows,
            "rrrr",
        ),
        "",
        f"Top {graph_cfg.phenotype_top_conditions} over-represented chronic conditions per "
        "cluster: ratio = prevalence in the cluster / prevalence among all clustered "
        f"patients, among conditions of at least {graph_cfg.phenotype_min_patients} of the "
        "cluster's patients.",
        "",
        markdown_table(
            ["Cluster", "Rank", "Condition", "Patients", "In cluster", "Overall", "Ratio"],
            phenotype_rows,
            "rrlrrrr",
        ),
        "",
        "## Graph drift",
        "",
        "Neighbourhood Jaccard between consecutive windows, for patients with neighbours in "
        "both: |N_t(i) & N_t+1(i)| / |N_t(i) | N_t+1(i)|. Unchanged: Jaccard 1; replaced: 0.",
        "",
        f"![Graph drift]({figure(paths.graph_drift_fig)})",
        "",
        markdown_table(
            ["Windows", "Patients", "Mean", "Median", "IQR", "Unchanged", "Replaced", "Jump"],
            drift_rows,
            "lrrrrrrl",
        ),
        "",
        jump_text,
        "",
    ]
    text = "\n".join(lines)
    paths.graphs_summary_md.parent.mkdir(parents=True, exist_ok=True)
    paths.graphs_summary_md.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", paths.graphs_summary_md)
    return text


def _symmetric_normalize(M: sp.spmatrix) -> sp.csr_matrix:
    """Return ``D^(-1/2) M D^(-1/2)``, D the row sums of M; 0 where a row sum is <= 0."""
    degree = np.asarray(M.sum(axis=1)).ravel()
    inverse_sqrt = np.zeros_like(degree, dtype=np.float64)
    positive = degree > 0
    inverse_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    scale = sp.diags(inverse_sqrt)
    return sp.csr_matrix(scale @ M @ scale)


def _nonzero_rows(matrix: sp.csr_matrix) -> np.ndarray:
    """Flag the rows of ``matrix`` with a non-zero value."""
    return np.asarray(abs(matrix).sum(axis=1)).ravel() > 0


def _write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` to ``path`` as indented UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)


def _write_npz(path: Path, matrix: sp.spmatrix) -> None:
    """Write a sparse matrix to ``path`` with ``scipy.sparse.save_npz``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, matrix)
    logger.info("Wrote %s, shape %s, %d non-zeros", path, matrix.shape, matrix.nnz)
