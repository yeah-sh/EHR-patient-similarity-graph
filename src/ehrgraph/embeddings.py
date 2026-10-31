"""Phase 7b: the self-supervised TemporalGraphAE embeddings against simpler ones.

Run end to end by ``scripts/07b_embeddings.py``::

    data = train.load_graph_ae_data(cfg)
    runs, fits = train.tune_graph_ae(data, cfg.graph_ae)  # validation next-window loss
    ...  # the final GraphAE and its three ablations (variant_graphs, train.fit_graph_ae)
    embeddings = {**window_pca_embeddings(...), "spectral": ..., ...}  # 8 embeddings
    task = next_year_task(...)  # new conditions 2021 -> 2022
    next_year = next_year_results(task, {key: next_year_scores(...)}, ...)
    concordance = concordance_results(...)
    clusters = evaluate.cluster_scores(...)
    probe = probe_results(linear_probe(...), ...)
    table = embedding_table(...)
    write_embedding_results(...); write_embedding_summary(...)

Every embedding is 32-dimensional and has two versions: ``final``, after the last
training window (2022), and ``previous``, after the one before (2021), which reads
no 2022 data. The 2021 version predicts the conditions that are new in 2022; the
2022 version is scored on neighbours, clusters, spread and the ``any_onset`` probe.

Unless noted, metrics are on test-split patients. Confidence intervals are
percentile bootstrap intervals over those patients; every embedding is scored on
the same resamples, so differences between embeddings are paired.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
import torch
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ehrgraph.config import (
    Config,
    EmbeddingEvalConfig,
    EvaluationConfig,
    feature_window_years,
)
from ehrgraph.data import chart_style, markdown_table, save_figure
from ehrgraph.evaluate import (
    FULL_MODEL,
    GraphInputs,
    PatientContext,
    baseline_inputs,
    bin_labels,
    categorize,
    chronic_jaccard,
    compare_models,
    cross_validation_folds,
    fit_calibrator,
    fit_logistic,
    fold_split,
    fold_training_data,
    percentile_interval,
    phenotype_names,
    plot_umap_facets,
    plot_window_paths,
)
from ehrgraph.graph import largest_component, laplacian, spectral_embedding
from ehrgraph.labels import ANY_ONSET
from ehrgraph.models import pc1_share
from ehrgraph.train import TEST, TRAIN, VAL, GraphAEData, ModelData, load_trained_model, split_targets

logger = logging.getLogger(__name__)

# Embedding keys; GRAPH_AE is the reference of every paired difference.
GRAPH_AE = "graph_ae"
NO_GRAPH, REWIRED, LINK_ONLY = "graph_ae_no_graph", "graph_ae_rewired", "graph_ae_link_only"
PCA_FINAL, PCA_WINDOWS, SPECTRAL, SUPERVISED = "pca_final", "pca_windows", "spectral", "tgcn_supervised"
# Phase 7 model 2 (logistic regression on the final window + age, sex): the probe's reference.
MODEL_2 = "lr_last"
# Next-year reference with no patient information (base_rate_scores).
BASE_RATE = "base_rate"
# (question, embedding compared with the GraphAE), in report order.
QUESTIONS: tuple[tuple[str, str], ...] = (
    ("Does the graph help?", NO_GRAPH),
    ("Is the real graph better than a random one?", REWIRED),
    ("Does the temporal (next-window) objective help?", LINK_ONLY),
    ("Does the GraphAE beat PCA of the final window?", PCA_FINAL),
    ("Does the GraphAE beat PCA of every window?", PCA_WINDOWS),
    ("Does the GraphAE beat the spectral embedding?", SPECTRAL),
    ("Does the GraphAE beat the supervised TemporalGCN's h?", SUPERVISED),
)
RESULT_COLUMNS: list[str] = [
    "number", "key", "name", "description", "graph", "signal", "label_informed", "dimensions",
    "next_micro_auroc", "next_micro_low", "next_micro_high", "next_macro_auroc",
    "next_macro_low", "next_macro_high", "next_codes",
    "next_micro_diff", "next_micro_diff_low", "next_micro_diff_high",
    "next_macro_diff", "next_macro_diff_low", "next_macro_diff_high",
    "concordance", "concordance_low", "concordance_high",
    "concordance_diff", "concordance_diff_low", "concordance_diff_high",
    "silhouette", "davies_bouldin", "pc1_share",
    "probe_auroc", "probe_low", "probe_high",
    "probe_vs_model2", "probe_vs_model2_low", "probe_vs_model2_high",
    "probe_diff", "probe_diff_low", "probe_diff_high", "probe_folds_flipped",
    "probe_fold_auroc_raw",
]

# Chart tokens from the dataviz reference palette (as in evaluate.py): series blue for
# the bars, accent for the GraphAE, secondary ink for values and whiskers.
_SERIES_COLOR = "#2a78d6"
_ACCENT_COLOR = "#eb6834"
_TEXT_SECONDARY = "#52514e"
_BASELINE = "#c3c2b7"
_POINTS_PER_INCH = 72
_MAX_BAR_PX = 24
_BAR_FILL = 0.5  # share of each row a bar fills; the rest is air
_CHROME_IN = 1.4  # titles, axis and margins around the bar rows
_HEADROOM_ROWS = 0.1  # extra rows above the top bar (beyond its half row) for a reference label
_LINE_PX = 2
_LABEL_OFFSET_PT = 4
_CHANCE_AUROC = 0.5
_SETTLING_RATIO = 2.0  # a first GRU step this many times the later mean is called settling
_SAME_PROBABILITY = 1e-6  # a refitted model 2 within this of Phase 7's reproduces it


@dataclass(frozen=True)
class EmbeddingSpec:
    """One compared embedding.

    Attributes:
        number: Roman numeral in the report, "i" to "viii".
        key: Column and row key.
        name: Display name.
        description: What it is.
        graph: The graph it uses: "real", "random", "none" or "real (2022 only)".
        signal: What trained it.
        label_informed: Whether ``any_onset`` labels of some patients shaped it (the
            supervised TemporalGCN): its ``any_onset`` probe is then optimistic.
    """

    number: str
    key: str
    name: str
    description: str
    graph: str
    signal: str
    label_informed: bool = False


@dataclass(frozen=True)
class Embedding:
    """One embedding of every node.

    Attributes:
        final: [N, d] after the last training window.
        previous: [N, d] after the window before it; reads no later window.
    """

    final: np.ndarray
    previous: np.ndarray


def embedding_specs(years: Sequence[int], dim: int) -> tuple[EmbeddingSpec, ...]:
    """Return the eight compared embeddings, in report order.

    Args:
        years: The training windows, oldest first.
        dim: Embedding width.

    Returns:
        Embeddings i to viii.
    """
    last, every = years[-1], f"{len(years)} windows {years[0]}-{years[-1]}"
    ae = "self-supervised: link + next-window"
    return (
        EmbeddingSpec("i", GRAPH_AE, "TemporalGraphAE",
                      f"final GRU state over {every}, on the similarity graphs", "real", ae),
        EmbeddingSpec("ii", PCA_FINAL, "PCA, final window",
                      f"{dim} principal components of the {last} input features", "none",
                      "none (unsupervised)"),
        EmbeddingSpec("iii", PCA_WINDOWS, "PCA, every window",
                      f"{dim} principal components of the {every} concatenated", "none",
                      "none (unsupervised)"),
        EmbeddingSpec("iv", SPECTRAL, "Spectral embedding",
                      f"first {dim} non-trivial normalized-Laplacian eigenvectors of the {last} "
                      "graph (largest component; other nodes 0)", f"real ({last} only)",
                      "none (unsupervised)"),
        EmbeddingSpec("v", SUPERVISED, "Supervised TemporalGCN h",
                      "the Phase 6/7 final model's GRU state", "real",
                      "supervised: any_onset (training split)", label_informed=True),
        EmbeddingSpec("vi", NO_GRAPH, "GraphAE, no graph",
                      "A_hat = I in every window, so no message passing and no edges for the "
                      "link term", "none", "self-supervised: next-window only"),
        EmbeddingSpec("vii", REWIRED, "GraphAE, rewired graphs",
                      "every window's graph rewired by degree-preserving double edge swaps "
                      "(Phase 7 model 7's graphs)", "random", ae),
        EmbeddingSpec("viii", LINK_ONLY, "GraphAE, link only",
                      "no next-window gradient: only the link term trains the encoder; the GRU "
                      "stays as initialized", "real", "self-supervised: link only"),
    )


# --- Building the embeddings -----------------------------------------------------------


def pca_embedding(X: sp.spmatrix | np.ndarray, dim: int, seed: int) -> np.ndarray:
    """The first ``dim`` principal-component scores of the rows of ``X`` (all nodes).

    Args:
        X: [N, F] features; densified (N x F, never N x N).
        dim: Components.
        seed: Random state of the solver.

    Returns:
        [N, dim].
    """
    dense = X.toarray() if sp.issparse(X) else np.asarray(X)
    return PCA(n_components=dim, random_state=seed).fit_transform(dense.astype(np.float64))


def window_pca_embeddings(
    matrices: Sequence[sp.spmatrix], dim: int, seed: int
) -> dict[str, Embedding]:
    """PCA of the final window (ii) and of every window concatenated (iii).

    The previous versions are fitted on the windows up to the one before the last
    (the 2021 window, and 2017-2021 concatenated), so they read no later window.

    Args:
        matrices: Sparse [N, F] features per window, oldest first.
        dim: Components.
        seed: Random state.

    Returns:
        ``PCA_FINAL`` and ``PCA_WINDOWS``.
    """
    return {
        PCA_FINAL: Embedding(pca_embedding(matrices[-1], dim, seed),
                             pca_embedding(matrices[-2], dim, seed)),
        PCA_WINDOWS: Embedding(pca_embedding(sp.hstack(matrices, format="csr"), dim, seed),
                               pca_embedding(sp.hstack(matrices[:-1], format="csr"), dim, seed)),
    }


def spectral_window_embedding(W: sp.spmatrix, dim: int, seed: int) -> np.ndarray:
    """The ``dim`` smallest non-trivial normalized-Laplacian eigenvectors of ``W``.

    As in Phase 4: computed on the largest connected component; other nodes get 0.

    Args:
        W: Symmetric (N, N) adjacency.
        dim: Eigenvectors.
        seed: Seed of the eigensolver's start vector.

    Returns:
        [N, dim].
    """
    nodes = largest_component(W)
    _, vectors = spectral_embedding(laplacian(sp.csr_matrix(W)[nodes][:, nodes]), dim, seed)
    embedding = np.zeros((W.shape[0], dim))
    embedding[nodes] = vectors
    return embedding


def states_embedding(states: np.ndarray) -> Embedding:
    """The GRU states after the last window and the one before, from [N, T, d] states."""
    states = np.asarray(states, dtype=np.float64)
    return Embedding(states[:, -1], states[:, -2])


def step_lengths(states: np.ndarray, nonempty: np.ndarray) -> np.ndarray:
    """Mean Euclidean step ``||h_{t+1} - h_t||`` per window pair, [T - 1].

    Over the patients with a non-empty feature row in every window, in the
    embedding's own space (not the UMAP).
    """
    rows = np.asarray(states, dtype=np.float64)[nonempty.all(axis=1)]
    return np.linalg.norm(np.diff(rows, axis=1), axis=2).mean(axis=0)


def supervised_states(cfg: Config, data: GraphAEData) -> np.ndarray:
    """The Phase 6 final model's GRU state after every window, [N, T, d] (eval mode).

    Args:
        cfg: Project configuration (``model`` section, ``paths.model_pt``).
        data: GraphAE data: the same features and graphs the model was trained on.

    Returns:
        [N, T, embed_dim].
    """
    model = load_trained_model(cfg.model, cfg.paths.model_pt, data.features[0].shape[1],
                               len(cfg.labels.onset_groups))
    with torch.no_grad():
        _, states = model.temporal.sequence(data.features, data.adjacency)
    return states.numpy().astype(np.float64)


# --- Next-year new conditions ----------------------------------------------------------


@dataclass(frozen=True)
class NextYearTask:
    """New conditions in ``year`` (the last training window), per node and code.

    Attributes:
        year: The target window.
        codes: Codes with at least ``embedding_eval.min_train_onsets`` training onsets.
        at_risk: Bool [N, C]: no event of the code before ``year``, born before it
            began and not dead before it began.
        onset: Bool [N, C]: at risk and the first event of the code falls in ``year``.
        train_onsets: [C] onsets among training-split nodes.
        test_onsets: [C] onsets among test-split nodes.
        test_at_risk: [C] test-split nodes at risk.
    """

    year: int
    codes: tuple[str, ...]
    at_risk: np.ndarray
    onset: np.ndarray
    train_onsets: np.ndarray
    test_onsets: np.ndarray
    test_at_risk: np.ndarray


def new_onsets(
    events: pd.DataFrame,
    patients: pd.DataFrame,
    patient_ids: Sequence[str],
    codes: Sequence[str],
    year: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Who is at risk of each code in ``year`` and whose first event of it falls in ``year``.

    Events on or after the end of ``year`` are dropped before anything is read.

    Args:
        events: ``Trajectories.events`` (patient_id, date, code).
        patients: ``data.load_patients`` output (birth_date, death_date).
        patient_ids: Node order.
        codes: Column order.
        year: The target window.

    Returns:
        ``at_risk`` and ``onset``, bool [N, C].
    """
    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
    events = events[events["date"] < end]
    events = events[events["code"].isin(codes) & events["patient_id"].isin(patient_ids)]
    first = events.groupby(["patient_id", "code"], as_index=False)["date"].min()
    rows = pd.Index(list(patient_ids)).get_indexer(first["patient_id"])
    cols = pd.Index(list(codes)).get_indexer(first["code"])
    shape = (len(patient_ids), len(codes))
    before, onset = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    earlier = (first["date"] < start).to_numpy()
    before[rows[earlier], cols[earlier]] = True
    onset[rows[~earlier], cols[~earlier]] = True
    table = patients.set_index("patient_id").reindex(pd.Index(list(patient_ids)))
    alive = ((table["birth_date"] < start) & ~(table["death_date"] < start)).to_numpy()
    at_risk = ~before & alive[:, None]
    return at_risk, onset & at_risk


def next_year_task(
    events: pd.DataFrame,
    patients: pd.DataFrame,
    patient_ids: Sequence[str],
    codes: Sequence[str],
    split: np.ndarray,
    year: int,
    cutoff: date,
    min_train_onsets: int,
) -> NextYearTask:
    """Build the next-year task: codes with enough training onsets in ``year``.

    Args:
        events: ``Trajectories.events``.
        patients: ``data.load_patients`` output.
        patient_ids: Node order.
        codes: Candidate codes (the feature vocabulary).
        split: Split per node.
        year: The target window; must close by the cutoff.
        cutoff: ``labels.cutoff_date``.
        min_train_onsets: ``embedding_eval.min_train_onsets``.

    Returns:
        The task.

    Raises:
        ValueError: If ``year`` does not close by the cutoff or no code qualifies.
    """
    if date(year + 1, 1, 1) > cutoff:
        raise ValueError(f"the target window {year} does not close by the cutoff {cutoff}")
    at_risk, onset = new_onsets(events, patients, patient_ids, codes, year)
    train, test = split == TRAIN, split == TEST
    train_onsets = onset[train].sum(axis=0)
    keep = np.flatnonzero(train_onsets >= min_train_onsets)
    if not len(keep):
        raise ValueError(f"no code has {min_train_onsets} training onsets in {year}")
    logger.info("Next-year task %d: %d of %d codes have >= %d training onsets; %d training and %d "
                "test onsets", year, len(keep), len(codes), min_train_onsets,
                int(train_onsets[keep].sum()), int(onset[test][:, keep].sum()))
    return NextYearTask(
        year=year, codes=tuple(codes[i] for i in keep), at_risk=at_risk[:, keep],
        onset=onset[:, keep], train_onsets=train_onsets[keep],
        test_onsets=onset[test][:, keep].sum(axis=0), test_at_risk=at_risk[test][:, keep].sum(axis=0),
    )


def next_year_scores(
    X: np.ndarray, task: NextYearTask, split: np.ndarray, settings: EmbeddingEvalConfig
) -> np.ndarray:
    """Fit one logistic regression per code on the training split; score every node.

    Inputs are standardized with the training-split nodes' mean and sd. Each code's
    model is fitted on the training-split nodes at risk of it.

    Args:
        X: [N, d] embedding (the version that reads no window of ``task.year``).
        task: ``next_year_task`` output.
        split: Split per node.
        settings: The ``embedding_eval`` section.

    Returns:
        [N, C] predicted probabilities; NaN where the node is not at risk.
    """
    train = split == TRAIN
    Z = StandardScaler().fit(X[train]).transform(X)
    scores = np.full(task.at_risk.shape, np.nan)
    for c, code in enumerate(task.codes):
        fit_rows = train & task.at_risk[:, c]
        model = LogisticRegression(
            C=settings.C, max_iter=settings.max_iter,
            class_weight="balanced" if settings.balanced_class_weight else None,
        ).fit(Z[fit_rows], task.onset[fit_rows, c].astype(int))
        if int(np.max(model.n_iter_)) >= settings.max_iter:
            logger.warning("Logistic regression for code %s did not converge", code)
        rows = task.at_risk[:, c]
        scores[rows, c] = model.predict_proba(Z[rows])[:, 1]
    return scores


def micro_auroc(task: NextYearTask, scores: np.ndarray, split: np.ndarray) -> float:
    """AUROC over every (test-split patient, code) pair at risk (``next_year_results``)."""
    test = split == TEST
    at_risk = task.at_risk[test]
    return float(roc_auc_score(task.onset[test][at_risk], scores[test][at_risk]))


def base_rate_scores(task: NextYearTask, split: np.ndarray) -> np.ndarray:
    """A score with no patient information: each code's onset rate among training patients at risk.

    The reference for the micro AUROC, which pools codes of different frequency: this
    score ranks only the codes, never the patients (its per-code AUROC is 0.5).

    Args:
        task: ``next_year_task`` output.
        split: Split per node.

    Returns:
        [N, C]; NaN where the node is not at risk.
    """
    train = split == TRAIN
    rate = task.onset[train].sum(axis=0) / np.maximum(task.at_risk[train].sum(axis=0), 1)
    return np.where(task.at_risk, rate[None, :], np.nan)


def patient_resamples(n: int, n_resamples: int, seed: int) -> np.ndarray:
    """How often each of ``n`` patients is drawn in each bootstrap resample, [B, n]."""
    draws = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))
    return np.stack([np.bincount(row, minlength=n) for row in draws]).astype(np.int32)


def weighted_auroc(y: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """AUROC of ``scores`` against binary ``y`` under each row of ``weights``, [B, n] -> [B].

    The weighted Mann-Whitney statistic, ties counting half: per row the same as
    ``roc_auc_score(y, scores, sample_weight=row)``, for every row at once. NaN for a
    row with no positive or no negative weight.
    """
    order = np.argsort(scores, kind="stable")
    y, sorted_scores = np.asarray(y, dtype=float)[order], np.asarray(scores)[order]
    W = np.asarray(weights, dtype=float)[:, order]
    starts = np.flatnonzero(np.r_[True, np.diff(sorted_scores) != 0])  # groups of tied scores
    positives = np.add.reduceat(W * y, starts, axis=1)
    negatives = np.add.reduceat(W * (1.0 - y), starts, axis=1)
    below = np.cumsum(negatives, axis=1) - negatives
    numerator = (positives * (below + 0.5 * negatives)).sum(axis=1)
    denominator = positives.sum(axis=1) * negatives.sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(W), np.nan),
                     where=denominator > 0)


@dataclass(frozen=True)
class NextYearResults:
    """Next-year new-condition prediction on the test split.

    Attributes:
        table: One row per scored key: key, pairs, positives, micro_auroc, micro_low,
            micro_high, macro_auroc, macro_low, macro_high, codes_scored, diff,
            diff_low, diff_high (the reference's micro AUROC minus this one's, paired)
            and macro_diff, macro_diff_low, macro_diff_high (the same for macro).
        per_code: One row per code: code, train_onsets, test_at_risk,
            test_onsets, then the AUROC of each key (NaN without both classes).
        resamples_used: Bootstrap resamples with both classes among the pooled pairs.
    """

    table: pd.DataFrame
    per_code: pd.DataFrame
    resamples_used: int


def next_year_results(
    task: NextYearTask,
    scores: Mapping[str, np.ndarray],
    split: np.ndarray,
    reference: str,
    evaluation: EvaluationConfig,
    seed: int,
) -> NextYearResults:
    """Micro and macro AUROC of each key's next-year scores on the test split.

    Micro: one AUROC over every (test patient, code) pair at risk; it rewards ranking
    the codes by frequency as well as the patients (see ``base_rate_scores``).
    Macro: the mean of the per-code AUROCs over the codes with a test onset, so it
    measures the ranking of patients within each code. The bootstrap resamples test
    patients (each with all their pairs); in a resample, a code without both classes
    is left out of the macro mean. Differences are paired.

    Args:
        task: ``next_year_task`` output.
        scores: Key -> ``next_year_scores`` output.
        split: Split per node.
        reference: Key the others are compared with.
        evaluation: Bootstrap settings.
        seed: Seed of the resamples.

    Returns:
        The results.
    """
    test = np.flatnonzero(split == TEST)
    at_risk, onset = task.at_risk[test], task.onset[test]
    counts = patient_resamples(len(test), evaluation.bootstrap_resamples, seed)
    patient, code = np.nonzero(at_risk)
    y = onset[patient, code].astype(float)
    micro = {key: weighted_auroc(y, s[test][patient, code], counts[:, patient])
             for key, s in scores.items()}
    per_code = pd.DataFrame({"code": task.codes, "train_onsets": task.train_onsets,
                             "test_at_risk": task.test_at_risk, "test_onsets": task.test_onsets})
    macro = {}
    for key, s in scores.items():
        points, boots = [], []
        for c in range(len(task.codes)):
            rows = np.flatnonzero(at_risk[:, c])
            y_c, s_c = onset[rows, c], s[test][rows, c]
            points.append(roc_auc_score(y_c, s_c) if 0 < y_c.sum() < len(y_c) else np.nan)
            boots.append(weighted_auroc(y_c, s_c, counts[:, rows]))
        per_code[key] = points
        with np.errstate(all="ignore"):
            macro[key] = (np.nanmean(points), np.nanmean(np.array(boots), axis=0))
    level = evaluation.ci_level
    rows = []
    for key, s in scores.items():
        compared = key != reference
        point = roc_auc_score(y, s[test][patient, code])
        reference_point = roc_auc_score(y, scores[reference][test][patient, code])
        rows.append({
            "key": key, "pairs": len(y), "positives": int(y.sum()), "micro_auroc": float(point),
            **dict(zip(("micro_low", "micro_high"), _interval(micro[key], level))),
            "macro_auroc": float(macro[key][0]),
            **dict(zip(("macro_low", "macro_high"), _interval(macro[key][1], level))),
            "codes_scored": int(np.isfinite(per_code[key]).sum()),
            "diff": reference_point - point if compared else np.nan,
            **dict(zip(("diff_low", "diff_high"),
                       _interval(micro[reference] - micro[key], level) if compared
                       else (np.nan, np.nan))),
            "macro_diff": macro[reference][0] - macro[key][0] if compared else np.nan,
            **dict(zip(("macro_diff_low", "macro_diff_high"),
                       _interval(macro[reference][1] - macro[key][1], level) if compared
                       else (np.nan, np.nan))),
        })
    return NextYearResults(pd.DataFrame(rows), per_code,
                           int(np.isfinite(micro[reference]).sum()))


def _interval(values: np.ndarray, level: float) -> tuple[float, float]:
    """``percentile_interval`` of the finite values (resamples where a metric is defined)."""
    values = np.asarray(values, dtype=float)
    return percentile_interval(values[np.isfinite(values)], level)


# --- Neighbour concordance -------------------------------------------------------------


def neighbour_jaccard(
    X: np.ndarray, chronic: sp.spmatrix, queries: np.ndarray, k: int, working_memory_mb: int
) -> np.ndarray:
    """Per query node, the mean chronic-condition Jaccard with its ``k`` nearest neighbours.

    Neighbours are searched among every node (cosine distance), never the node itself.

    Args:
        X: [N, d] representation.
        chronic: Binary [N, n_codes] chronic conditions.
        queries: Query nodes.
        k: Neighbours.
        working_memory_mb: Memory per block of the neighbour search.

    Returns:
        [len(queries)].
    """
    search = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(X)
    with sklearn.config_context(working_memory=working_memory_mb):
        neighbours = search.kneighbors(np.asarray(X)[queries], n_neighbors=k + 1,
                                       return_distance=False)
    # Drop the query itself (or, if it ties with a duplicate row, the last neighbour).
    others = np.array([row[row != q][:k] for row, q in zip(neighbours, queries)])
    jaccard = chronic_jaccard(chronic, np.repeat(queries, k), others.ravel())
    return jaccard.reshape(len(queries), k).mean(axis=1)


def random_jaccard(chronic: sp.spmatrix, queries: np.ndarray, k: int, seed: int) -> np.ndarray:
    """As ``neighbour_jaccard`` with ``k`` random other nodes per query (chance)."""
    n = chronic.shape[0]
    draws = np.random.default_rng(seed).integers(0, n - 1, size=(len(queries), k))
    others = draws + (draws >= queries[:, None])  # anyone but the query
    return chronic_jaccard(chronic, np.repeat(queries, k), others.ravel()).reshape(-1, k).mean(axis=1)


def concordance_results(
    representations: Mapping[str, np.ndarray],
    references: Mapping[str, np.ndarray],
    reference: str,
    evaluation: EvaluationConfig,
    seed: int,
) -> pd.DataFrame:
    """Mean neighbour concordance with bootstrap CIs and paired differences.

    Args:
        representations: Key -> per-query ``neighbour_jaccard``.
        references: Name -> per-query Jaccard of a reference (random patients,
            input features); CIs only.
        reference: Key the others are compared with.
        evaluation: Bootstrap settings.
        seed: Seed of the resamples (over query patients).

    Returns:
        key, patients, concordance, low, high, diff, diff_low, diff_high (the
        reference's concordance minus this one's); references last, without a diff.
    """
    entries = {**representations, **references}
    n = len(next(iter(entries.values())))
    counts = patient_resamples(n, evaluation.bootstrap_resamples, seed)
    boot = {key: counts @ values / n for key, values in entries.items()}
    level = evaluation.ci_level
    rows = []
    for key, values in entries.items():
        compared = key in representations and key != reference
        diff_ci = (percentile_interval(boot[reference] - boot[key], level) if compared
                   else (np.nan, np.nan))
        rows.append({"key": key, "patients": n, "concordance": float(values.mean()),
                     **dict(zip(("low", "high"), percentile_interval(boot[key], level))),
                     "diff": float(entries[reference].mean() - values.mean()) if compared else np.nan,
                     "diff_low": diff_ci[0], "diff_high": diff_ci[1]})
    return pd.DataFrame(rows)


# --- Linear probe of any_onset ---------------------------------------------------------


def linear_probe(
    finals: Mapping[str, np.ndarray], model_data: ModelData, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validated logistic regression of ``any_onset`` on each embedding + age, sex.

    The Phase 7 protocol and folds (``evaluate.cross_validation_folds``,
    ``fold_split``): per fold, standardize the embedding with the inner training
    patients, fit the ``baselines`` logistic regression on their labels, fit
    ``calibration.method`` on the inner validation patients and score the held-out
    fold. Phase 7 model 2 (``MODEL_2``: the final window's features + age, sex) is
    refitted the same way as the reference.

    Args:
        finals: Key -> [N, d] final embedding.
        model_data: ``train.load_model_data`` output (labels, demographics).
        cfg: Project configuration.

    Returns:
        The predictions, one row per patient with an observed ``any_onset``:
        patient_id, node, fold, ``ANY_ONSET``, then per key (and ``MODEL_2``) the
        calibrated out-of-fold probability (column key) and the uncalibrated logit
        (column "key.logit"); and the fits, one row per fold and key: fold, key,
        slope, intercept (the Platt calibration; a slope <= 0 reverses the fold's
        ranking).
    """
    settings = cfg.cross_validation
    labels = model_data.labels
    folds = cross_validation_folds(labels, settings.folds, cfg.seed)
    nodes = pd.Index(labels["patient_id"]).get_indexer(folds.index)
    row_of = np.full(len(labels), -1)
    row_of[nodes] = np.arange(len(nodes))
    inputs = GraphInputs(model_data.years, model_data.matrices, model_data.weights,
                         model_data.adjacency)
    probabilities = {key: np.full(len(nodes), np.nan) for key in (*finals, MODEL_2)}
    raw = {f"{key}.logit": np.full(len(nodes), np.nan) for key in probabilities}
    fits = []
    for fold in range(1, settings.folds + 1):
        split = fold_split(labels, folds, fold, settings.inner_val, cfg.seed)
        base = fold_training_data(model_data, inputs, split)
        train_y = split_targets(base, TRAIN)[0].numpy()
        val_y = split_targets(base, VAL)[0].numpy()
        val_seen = ~np.isnan(val_y)
        held_out = np.flatnonzero(split == TEST)
        demographics = base.demographics.numpy().astype(np.float64)
        designs = {key: np.hstack([StandardScaler().fit(X[split == TRAIN]).transform(X),
                                   demographics]) for key, X in finals.items()}
        designs[MODEL_2] = baseline_inputs(base, "last", True)
        for key, design in designs.items():
            logits = fit_logistic(design, train_y, cfg.baselines).decision_function(design)
            calibrator = fit_calibrator(logits[val_seen], val_y[val_seen], cfg.calibration.method,
                                        cfg.evaluation)
            probabilities[key][row_of[held_out]] = calibrator.apply(logits[held_out])
            raw[f"{key}.logit"][row_of[held_out]] = logits[held_out]
            fits.append({"fold": fold, "key": key, "slope": calibrator.slope,
                         "intercept": calibrator.intercept})
        logger.info("Probe fold %d/%d: held-out AUROC %s", fold, settings.folds,
                    {key: round(roc_auc_score(labels[ANY_ONSET].to_numpy()[held_out],
                                              probabilities[key][row_of[held_out]]), 3)
                     for key in designs})
    frame = pd.DataFrame({"patient_id": np.asarray(folds.index, dtype=str), "node": nodes,
                          "fold": folds.to_numpy(),
                          ANY_ONSET: labels[ANY_ONSET].to_numpy(dtype=float)[nodes]})
    predictions = pd.concat([frame, pd.DataFrame(probabilities), pd.DataFrame(raw)], axis=1)
    return predictions, pd.DataFrame(fits)


def check_model_2(predictions: pd.DataFrame, cfg: Config) -> float | None:
    """Largest absolute difference between the refitted model 2 and Phase 7's saved one.

    Returns None if Phase 7's out-of-fold predictions are not on disk.
    """
    path = cfg.paths.cv_predictions_csv
    if not path.exists():
        return None
    saved = pd.read_csv(path, dtype={"patient_id": str}).set_index("patient_id")[MODEL_2]
    return float(np.max(np.abs(saved.reindex(predictions["patient_id"]).to_numpy()
                               - predictions[MODEL_2].to_numpy())))


def probe_results(
    predictions: pd.DataFrame, fits: pd.DataFrame, keys: Sequence[str], cfg: Config
) -> pd.DataFrame:
    """Pooled out-of-fold AUROC of each probe, with CIs and paired differences.

    Two diagnostics of the calibration, which the pooled AUROC depends on: the
    folds whose Platt slope is <= 0 (the calibration reverses that fold's ranking:
    its inner validation patients ranked the other way), and the mean within-fold
    AUROC of the uncalibrated logits, which no calibration changes.

    Args:
        predictions: ``linear_probe`` predictions.
        fits: ``linear_probe`` fits.
        keys: Embedding keys, in report order (``GRAPH_AE`` included).
        cfg: Project configuration (bootstrap settings, seed).

    Returns:
        key, observed, positives, auroc, low, high, vs_model2, vs_model2_low,
        vs_model2_high (this probe minus ``MODEL_2``), diff, diff_low, diff_high
        (``GRAPH_AE`` minus this probe), folds_flipped, fold_auroc_raw; one row per
        key, then ``MODEL_2``.
    """
    y = predictions[ANY_ONSET].to_numpy(dtype=float)
    scores = {key: predictions[key].to_numpy(dtype=float) for key in (*keys, MODEL_2)}
    vs_model2 = compare_models(y, scores, MODEL_2, cfg.evaluation, cfg.seed).set_index("key")
    vs_ae = compare_models(y, scores, GRAPH_AE, cfg.evaluation, cfg.seed).set_index("key")
    flipped = fits.assign(flipped=fits["slope"] <= 0).groupby("key")["flipped"].sum()
    fold_raw = {
        key: float(np.mean([roc_auc_score(part[ANY_ONSET], part[f"{key}.logit"])
                            for _, part in predictions.groupby("fold")]))
        for key in scores
    }
    rows = []
    for key in scores:
        row, other = vs_model2.loc[key], vs_ae.loc[key]
        rows.append({
            "key": key, "observed": int(row.val_observed), "positives": int(row.val_positives),
            "auroc": row.val_auroc, "low": row.val_auroc_low, "high": row.val_auroc_high,
            "vs_model2": -row.delta_auroc, "vs_model2_low": -row.delta_auroc_high,
            "vs_model2_high": -row.delta_auroc_low,
            "diff": other.delta_auroc, "diff_low": other.delta_auroc_low,
            "diff_high": other.delta_auroc_high, "folds_flipped": int(flipped.get(key, 0)),
            "fold_auroc_raw": fold_raw[key],
        })
    return pd.DataFrame(rows)


# --- The results table -----------------------------------------------------------------


def embedding_table(
    specs: Sequence[EmbeddingSpec],
    embeddings: Mapping[str, Embedding],
    next_year: NextYearResults,
    concordance: pd.DataFrame,
    clusters: pd.DataFrame,
    probe: pd.DataFrame,
) -> pd.DataFrame:
    """Join every metric into ``RESULT_COLUMNS``, one row per embedding in report order.

    Args:
        specs: ``embedding_specs`` output.
        embeddings: Key -> embedding.
        next_year: ``next_year_results`` output.
        concordance: ``concordance_results`` output.
        clusters: ``evaluate.cluster_scores`` output, its representations named by key.
        probe: ``probe_results`` output.

    Returns:
        The table.
    """
    nxt = next_year.table.set_index("key")
    conc = concordance.set_index("key")
    clus = clusters.set_index("representation")
    prb = probe.set_index("key")
    rows = []
    for spec in specs:
        key = spec.key
        n, c, p = nxt.loc[key], conc.loc[key], prb.loc[key]
        rows.append({
            "number": spec.number, "key": key, "name": spec.name, "description": spec.description,
            "graph": spec.graph, "signal": spec.signal, "label_informed": spec.label_informed,
            "dimensions": embeddings[key].final.shape[1],
            "next_micro_auroc": n.micro_auroc, "next_micro_low": n.micro_low,
            "next_micro_high": n.micro_high, "next_macro_auroc": n.macro_auroc,
            "next_macro_low": n.macro_low, "next_macro_high": n.macro_high,
            "next_codes": int(n.codes_scored), "next_micro_diff": n["diff"],
            "next_micro_diff_low": n.diff_low, "next_micro_diff_high": n.diff_high,
            "next_macro_diff": n.macro_diff, "next_macro_diff_low": n.macro_diff_low,
            "next_macro_diff_high": n.macro_diff_high,
            "concordance": c.concordance, "concordance_low": c.low, "concordance_high": c.high,
            "concordance_diff": c["diff"], "concordance_diff_low": c.diff_low,
            "concordance_diff_high": c.diff_high,
            "silhouette": clus.loc[key, "silhouette"],
            "davies_bouldin": clus.loc[key, "davies_bouldin"],
            "pc1_share": pc1_share(embeddings[key].final),
            "probe_auroc": p.auroc, "probe_low": p.low, "probe_high": p.high,
            "probe_vs_model2": p.vs_model2, "probe_vs_model2_low": p.vs_model2_low,
            "probe_vs_model2_high": p.vs_model2_high, "probe_diff": p["diff"],
            "probe_diff_low": p.diff_low, "probe_diff_high": p.diff_high,
            "probe_folds_flipped": int(p.folds_flipped), "probe_fold_auroc_raw": p.fold_auroc_raw,
        })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


# --- Figures ---------------------------------------------------------------------------


def plot_ae_embeddings(
    final_umap: np.ndarray,
    window_umap: np.ndarray,
    trajectory_nodes: np.ndarray,
    context: PatientContext,
    cfg: Config,
) -> None:
    """Save the three UMAP small multiples of the GraphAE's h and its trajectory figure.

    Args:
        final_umap: [N, 2] UMAP of the final GraphAE embedding h.
        window_umap: [N, T, 2] UMAP of the GRU states after each window.
        trajectory_nodes: The trajectory figure's patients (Phase 7's).
        context: ``evaluate.load_patient_context`` output.
        cfg: Project configuration.
    """
    paths, settings = cfg.paths, cfg.embedding_analysis
    what = "UMAP of the TemporalGraphAE embeddings h (final GRU state)"
    clusters = sorted(int(c) for c in np.unique(context.phenotype) if c > 0)
    names = phenotype_names(context, cfg)
    plot_umap_facets(
        final_umap, context.phenotype.astype(object), clusters,
        {c: f"Cluster {c} (n = {int((context.phenotype == c).sum()):,})\n"
            f"{_cut(names[c], settings.onset_label_chars)}" for c in clusters},
        f"{what}, by Phase 4 phenotype cluster (top over-represented chronic condition)",
        paths.ae_umap_phenotype_fig, cfg,
    )
    age_bins = cfg.summary.age_bins
    ages = categorize(context.age, age_bins)
    plot_umap_facets(
        final_umap, ages, bin_labels(age_bins),
        {label: f"Age {label} (n = {int((ages == label).sum()):,})" for label in bin_labels(age_bins)},
        f"{what}, by age on {cfg.labels.cutoff_date:%Y-%m-%d}", paths.ae_umap_age_fig, cfg,
    )
    count_bins = settings.chronic_count_bins
    counts = categorize(np.diff(context.chronic.indptr).astype(float), count_bins)
    plot_umap_facets(
        final_umap, counts, bin_labels(count_bins),
        {label: f"{label} chronic condition{'' if label == '1' else 's'} "
                f"(n = {int((counts == label).sum()):,})" for label in bin_labels(count_bins)},
        f"{what}, by number of chronic conditions {context.years[0]}-{context.years[-1]}",
        paths.ae_umap_chronic_fig, cfg,
    )
    plot_window_paths(window_umap, trajectory_nodes, context, paths.ae_trajectory_fig, cfg, "h_t",
                      "TemporalGraphAE GRU states")


def plot_next_year_auroc(
    table: pd.DataFrame,
    specs: Sequence[EmbeddingSpec],
    year: int,
    base_rate_micro: float,
    path: Path,
    cfg: Config,
) -> None:
    """Save ranked bars of each embedding's next-year micro and macro AUROC, with CIs.

    Bars grow from chance (0.5). The micro panel marks what the codes' base rates
    alone reach (``base_rate_scores``). The GraphAE's bars are in the accent colour,
    the rest in the series colour.

    Args:
        table: ``embedding_table`` output.
        specs: ``embedding_specs`` output (for the labels).
        year: The target window.
        base_rate_micro: Micro AUROC of ``base_rate_scores``.
        path: Output PNG.
        cfg: Project configuration.
    """
    dpi = cfg.summary.figure_dpi
    px = _POINTS_PER_INCH / dpi
    level = f"{cfg.evaluation.ci_level:.0%}"
    rows = table.sort_values("next_micro_auroc", ascending=True, kind="stable").reset_index(drop=True)
    names = {spec.key: f"{spec.number}. {spec.name}" for spec in specs}
    colors = [_ACCENT_COLOR if key == GRAPH_AE else _SERIES_COLOR for key in rows["key"]]
    y = np.arange(len(rows))
    row_in = _MAX_BAR_PX / (_BAR_FILL * dpi)
    panels = [("Micro AUROC (every patient-code pair)", "next_micro", base_rate_micro,
               "code base rates only"),
              ("Macro AUROC (mean over codes)", "next_macro", None, None)]
    with matplotlib.rc_context(chart_style(dpi)):
        fig = Figure(figsize=(11.0, len(rows) * row_in + _CHROME_IN), layout="constrained")
        axes = fig.subplots(1, 2, sharey=True)
        for ax, (title, column, reference, reference_label) in zip(axes, panels):
            values = rows[f"{column}_auroc"].to_numpy(dtype=float)
            low, high = rows[f"{column}_low"].to_numpy(), rows[f"{column}_high"].to_numpy()
            ax.barh(y, values - _CHANCE_AUROC, left=_CHANCE_AUROC, height=_BAR_FILL, color=colors)
            ax.hlines(y, low, high, color=_TEXT_SECONDARY, linewidth=px * _LINE_PX / 2)
            for row, value, end in zip(y, values, high):
                ax.annotate(f"{value:.3f}", xy=(end, row), xytext=(_LABEL_OFFSET_PT, 0),
                            textcoords="offset points", va="center", color=_TEXT_SECONDARY)
            ax.axvline(_CHANCE_AUROC, color=_BASELINE, linewidth=px)
            if reference is not None:
                ax.axvline(reference, color=_TEXT_SECONDARY, linewidth=px, linestyle="--")
                ax.annotate(f"{reference_label} ({reference:.3f})", xy=(reference, 1),
                            xycoords=("data", "axes fraction"), xytext=(-_LABEL_OFFSET_PT, -2),
                            textcoords="offset points", ha="right", va="top",
                            color=_TEXT_SECONDARY)
            ax.set_ylim(-1 + _BAR_FILL, len(rows) + _HEADROOM_ROWS)  # room for the label above
            ax.set_title(f"{title}, {level} CI")
            ax.set_xlabel("AUROC (bars start at chance, 0.5)")
            ax.margins(x=0.15)
            ax.grid(axis="x")
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0, labelcolor=_TEXT_SECONDARY)
            ax.spines["left"].set_visible(False)
        axes[0].set_yticks(y, [names[key] for key in rows["key"]])
        fig.suptitle(f"New conditions in {year} predicted from the {year - 1} embedding "
                     "(test patients; logistic regression per code)", x=0.01, ha="left")
        save_figure(fig, path)


# --- Reports ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantFit:
    """Training summary of one GraphAE variant (from ``train.GraphAEFitResult``)."""

    key: str
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    val_next_loss: float
    link_auroc: float
    pc1_share: float
    seconds: float


def variant_fits_frame(fits: Sequence[VariantFit]) -> pd.DataFrame:
    """The variants' training summaries as a frame."""
    return pd.DataFrame([vars(fit) for fit in fits])


def write_embedding_results(
    table: pd.DataFrame,
    specs: Sequence[EmbeddingSpec],
    tuning_runs: pd.DataFrame,
    tuning_summary: pd.DataFrame,
    chosen_weight: float,
    variants: pd.DataFrame,
    pc1_history: pd.DataFrame,
    task: NextYearTask,
    next_year: NextYearResults,
    concordance: pd.DataFrame,
    clusters: pd.DataFrame,
    probe: pd.DataFrame,
    seed_sensitivity: pd.DataFrame,
    code_names: Mapping[str, str],
    n_target_codes: int,
    model_2_check: float | None,
    steps: Sequence[float],
    cfg: Config,
) -> str:
    """Write every Phase 7b number to ``paths.embedding_results_md`` and the table to the csv.

    Args:
        table: ``embedding_table`` output.
        specs: ``embedding_specs`` output.
        tuning_runs: ``train.tune_graph_ae`` runs.
        tuning_summary: ``train.select_next_window_weight`` summary.
        chosen_weight: The chosen next-window weight.
        variants: ``variant_fits_frame`` of the four GraphAE variants.
        pc1_history: epoch, pc1_share, link_auroc, val_next_loss of the final GraphAE's
            logged epochs.
        task: ``next_year_task`` output.
        next_year: ``next_year_results`` output.
        concordance: ``concordance_results`` output.
        clusters: ``evaluate.cluster_scores`` output.
        probe: ``probe_results`` output.
        seed_sensitivity: seed, micro_auroc of the chosen GraphAE setting per tuning seed.
        code_names: Code -> condition name.
        n_target_codes: Codes of the next-window target.
        model_2_check: ``check_model_2`` output.
        steps: ``step_lengths`` of the final GraphAE's states.
        cfg: Project configuration.

    Returns:
        The markdown written.
    """
    paths, ae, ev = cfg.paths, cfg.graph_ae, cfg.embedding_eval
    level = f"{cfg.evaluation.ci_level:.0%}"
    by_key = table.set_index("key")
    name = {spec.key: f"{spec.number}. {spec.name}" for spec in specs}
    n_test = int(next_year.table["pairs"].iloc[0])
    base = next_year.table.set_index("key").loc[BASE_RATE]
    figures = {fig: Path(os.path.relpath(fig, paths.embedding_results_md.parent)).as_posix()
               for fig in (paths.ae_umap_phenotype_fig, paths.ae_umap_age_fig,
                           paths.ae_umap_chronic_fig, paths.ae_trajectory_fig,
                           paths.next_year_auroc_fig)}

    main_rows = [[
        name[r.key], r.graph, r.signal + (" *" if r.label_informed else ""),
        _with_ci(r.next_micro_auroc, r.next_micro_low, r.next_micro_high),
        _with_ci(r.next_macro_auroc, r.next_macro_low, r.next_macro_high),
        _with_ci(r.concordance, r.concordance_low, r.concordance_high),
        f"{r.silhouette:.3f}", f"{r.davies_bouldin:.2f}", f"{r.pc1_share:.1%}",
        _with_ci(r.probe_auroc, r.probe_low, r.probe_high)
        + (" *" if r.label_informed else ""),
        _with_ci(r.probe_vs_model2, r.probe_vs_model2_low, r.probe_vs_model2_high, signed=True),
    ] for r in table.itertuples()]
    question_rows = []
    for question, key in QUESTIONS:
        r = by_key.loc[key]
        question_rows.append([
            question, f"i - {name[key]}",
            _with_ci(r.next_micro_diff, r.next_micro_diff_low, r.next_micro_diff_high, signed=True)
            + _mark(r.next_micro_diff_low, r.next_micro_diff_high),
            _with_ci(r.next_macro_diff, r.next_macro_diff_low, r.next_macro_diff_high, signed=True)
            + _mark(r.next_macro_diff_low, r.next_macro_diff_high),
            _with_ci(r.concordance_diff, r.concordance_diff_low, r.concordance_diff_high,
                     signed=True) + _mark(r.concordance_diff_low, r.concordance_diff_high),
            _with_ci(r.probe_diff, r.probe_diff_low, r.probe_diff_high, signed=True)
            + _mark(r.probe_diff_low, r.probe_diff_high),
        ])
    tuning_rows = [[f"{r.next_window_weight:g}", str(int(r.seed)), str(int(r.best_epoch)),
                    str(int(r.epochs_run)), f"{r.val_next_loss:.4f}", f"{r.train_next_loss:.4f}",
                    f"{r.link_auroc:.3f}", f"{r.pc1_share:.1%}"] for r in tuning_runs.itertuples()]
    summary_rows = [[f"{r.next_window_weight:g}"
                     + (" (chosen)" if r.next_window_weight == chosen_weight else ""),
                     f"{r.val_next_loss_mean:.4f} +/- {r.val_next_loss_sd:.4f}",
                     f"{r.best_epoch_mean:.0f}", f"{r.link_auroc_mean:.3f}",
                     f"{r.pc1_share_mean:.1%}"]
                    for r in tuning_summary.itertuples()]
    variant_rows = [[name[r.key], str(int(r.best_epoch)), str(int(r.epochs_run)),
                     f"{r.val_next_loss:.4f}", f"{r.link_auroc:.3f}" if np.isfinite(r.link_auroc)
                     else "n/a (no edges)", f"{r.pc1_share:.1%}", f"{r.seconds:.0f}"]
                    for r in variants.itertuples()]
    pc1_rows = [[str(int(r.epoch)), f"{r.pc1_share:.1%}", f"{r.link_auroc:.3f}",
                 f"{r.val_next_loss:.4f}"] for r in pc1_history.itertuples()]
    code_columns = [spec.key for spec in specs]
    code_rows = [[_cut(code_names.get(r.code, r.code), 45), str(int(r.train_onsets)),
                  f"{int(r.test_onsets)} / {int(r.test_at_risk)}",
                  *[("n/a" if not np.isfinite(getattr(r, key)) else f"{getattr(r, key):.2f}")
                    for key in code_columns]]
                 for r in next_year.per_code.sort_values("train_onsets", ascending=False).itertuples()]
    conc_names = {**name, **{key: key for key in concordance["key"] if key not in name}}
    conc_rows = [[conc_names[r.key], _with_ci(r.concordance, r.low, r.high),
                  "" if not np.isfinite(r.diff) else
                  _with_ci(r.diff, r.diff_low, r.diff_high, signed=True)
                  + _mark(r.diff_low, r.diff_high)]
                 for r in concordance.itertuples()]
    cluster_rows = [[name.get(r.representation,
                              r.representation.replace(GRAPH_AE, name[GRAPH_AE], 1)),
                     f"{r.silhouette:.3f}", f"{r.davies_bouldin:.2f}"]
                    for r in clusters.itertuples()]
    probe_names = {**name, MODEL_2: "Phase 7 model 2 (2022 features + age, sex)"}
    probe_rows = [[probe_names[r.key] + (" *" if by_key["label_informed"].get(r.key, False) else ""),
                   _with_ci(r.auroc, r.low, r.high),
                   "reference" if r.key == MODEL_2 else
                   _with_ci(r.vs_model2, r.vs_model2_low, r.vs_model2_high, signed=True)
                   + _mark(r.vs_model2_low, r.vs_model2_high),
                   f"{r.folds_flipped} of {cfg.cross_validation.folds}", f"{r.fold_auroc_raw:.3f}"]
                  for r in probe.itertuples()]
    seed_rows = [[str(int(r.seed)), f"{r.micro_auroc:.3f}", f"{r.pc1_share:.1%}"]
                 for r in seed_sensitivity.itertuples()]
    if model_2_check is None:
        model_2_note = "Phase 7's saved predictions were not found, so model 2 is not checked."
    elif model_2_check < _SAME_PROBABILITY:
        model_2_note = (f"Refitted here with the Phase 7 code and folds, model 2 reproduces "
                        f"Phase 7's saved out-of-fold probabilities (largest difference "
                        f"{model_2_check:.1e}).")
    else:
        model_2_note = (f"**Refitted model 2 differs from Phase 7's saved out-of-fold "
                        f"probabilities by up to {model_2_check:.3f}.**")
    probe_obs = probe.iloc[0]
    years = feature_window_years(cfg)
    trajectory_note = (
        f"Trajectories: the GRU state h_t after each window {years[0]}-{years[-1]} (one UMAP of "
        "every non-empty patient-window) for the 5 example patients of Phase 7. Mean step "
        "||h_{t+1} - h_t|| in the 32-d space, over patients with features in every window: "
        + ", ".join(f"{a}->{b} {step:.3f}" for a, b, step in zip(years, years[1:], steps)) + "."
    )
    if steps[0] > _SETTLING_RATIO * float(np.mean(steps[1:])):
        trajectory_note += (" The early steps are the longest: the state settles as the GRU "
                            "reads more windows (and the time-decayed cumulative features "
                            "change less each year), so the long early moves in the figure are "
                            "mostly that settling, not clinical change.")
    lines = [
        "# Embedding results (Phase 7b)",
        "",
        "Built by `scripts/07b_embeddings.py`. Plain-language findings: `embedding_summary.md`. "
        "Every number below is on test-split patients unless noted; intervals are "
        f"{level} percentile bootstrap intervals over those patients "
        f"({cfg.evaluation.bootstrap_resamples} resamples, seed {cfg.seed}), and every embedding "
        "is scored on the same resamples, so differences are paired. A difference marked "
        "**shown** has an interval that excludes 0.",
        "",
        "## Setup",
        "",
        f"- **TemporalGraphAE** (`models.TemporalGraphAE`): the TemporalGCN encoder (shared "
        f"2-layer GCN per window, {cfg.graph_ae.hidden_dim} -> {ae.embed_dim}, dropout "
        f"{ae.dropout}; GRU with {ae.embed_dim} units), no demographics, no onset labels. Loss "
        f"= link + {chosen_weight:g} x next-window. **Link:** per window, `sigmoid(z_i . z_j)` "
        "on Z_t for that window's k-NN edges against as many uniformly drawn non-edges between "
        "nodes with an edge (fresh each epoch), BCE, averaged over windows. **Next window:** a "
        "linear decoder from the GRU state after window t to the codes active in window t + 1 "
        f"(the {n_target_codes} codes of the feature code block), pairs "
        "2017->2018 to 2021->2022 only, BCE with `pos_weight` = negatives / "
        "positives per code on the training split; a patient not yet born or already dead at "
        "the start of window t + 1 is left out of that pair.",
        "- **Who enters which loss:** every patient is a node and passes messages, and every "
        "edge enters the link term, test patients included. Only training-split patients' "
        "next-window terms enter the loss; validation-split patients' next-window loss picks "
        "the epoch (early stopping, patience "
        f"{ae.patience}); test-split patients' next-window targets enter no loss, so the "
        "2021 -> 2022 evaluation on them is clean of that objective. Adam, lr "
        f"{ae.lr}, weight decay {ae.weight_decay}, gradient norm clipped at {ae.grad_clip_norm}, "
        f"at most {ae.epochs} epochs, seed {cfg.seed}.",
        f"- **Embeddings (all {ev.dim}-d):** " + "; ".join(
            f"**{spec.number}** {spec.name}: {spec.description}" for spec in specs) + ".",
        f"- **2021 versions:** for the next-year task each embedding is recomputed from the "
        f"windows up to 2021 only: the GRU state after 2021 (i, v-viii), PCA of the 2021 window "
        "(ii) or of 2017-2021 concatenated (iii), the spectral embedding of the 2021 graph (iv).",
        "",
        "## Choosing the next-window weight (validation only)",
        "",
        f"Each weight in {{{', '.join(f'{w:g}' for w in ae.tuning_next_window_weights)}}} trained "
        f"with seeds {', '.join(map(str, ae.tuning_seeds))}, embed_dim {ae.embed_dim}; chosen by "
        "the lowest mean validation next-window loss (the unweighted BCE, comparable across "
        f"weights). **Chosen: {chosen_weight:g}.** Test patients were not used.",
        "",
        markdown_table(["Next-window weight", "Val next-window loss, mean +/- sd",
                        "Best epoch (mean)", "Link AUROC (mean)", "PC1 share (mean)"],
                       summary_rows, "lrrrr"),
        "",
        markdown_table(["Weight", "Seed", "Best epoch", "Epochs run", "Val next loss",
                        "Train next loss", "Link AUROC", "PC1 share"], tuning_rows, "rrrrrrrr"),
        "",
        "## The four GraphAE variants (seed " f"{cfg.seed}, next-window weight {chosen_weight:g})",
        "",
        "Link AUROC: edges vs one fixed draw of non-edges, by `z_i . z_j`, mean over windows. "
        "PC1 share: the share of the final GRU states' variance on their first principal "
        f"component, all patients (the supervised h, v: {by_key.loc[SUPERVISED, 'pc1_share']:.1%}).",
        "",
        markdown_table(["Variant", "Best epoch", "Epochs run", "Val next loss", "Link AUROC",
                        "PC1 share", "Seconds"], variant_rows, "lrrrrrr"),
        "",
        "Anti-collapse check of the final GraphAE, logged during training (all patients):",
        "",
        markdown_table(["Epoch", "PC1 share of h", "Link AUROC", "Val next loss"], pc1_rows, "rrrr"),
        "",
        "## Main table",
        "",
        f"Next year: new conditions in {task.year} from the {task.year - 1} embedding, "
        f"{len(task.codes)} codes, {n_test:,} test patient-code pairs at risk "
        f"({int(next_year.table['positives'].iloc[0])} onsets); the codes' base rates alone "
        f"give a micro AUROC of {base.micro_auroc:.3f} (see below). Concordance: Jaccard of chronic "
        f"conditions with the {ev.n_neighbours} nearest neighbours (cosine). Silhouette and "
        "Davies-Bouldin: against the Phase 4 phenotypes (test patients in a cluster). PC1 share: "
        "all patients. Probe: pooled out-of-fold AUROC of `any_onset`, all "
        f"{int(probe_obs.observed):,} patients with a label ({cfg.cross_validation.folds} folds).",
        "",
        markdown_table(["Embedding", "Graph", "Trained by", f"Next-year micro AUROC ({level} CI)",
                        f"Macro AUROC ({level} CI)", f"Neighbour concordance ({level} CI)",
                        "Silhouette",
                        "Davies-Bouldin", "PC1 share", f"Probe AUROC ({level} CI)",
                        "Probe minus model 2"], main_rows, "lllrrrrrrrr"),
        "",
        "\\* v was trained on the `any_onset` labels of the training split, who are in the "
        "probe's held-out folds: its probe AUROC is optimistic. Its clean cross-validated "
        f"AUROC is Phase 7's model 8: {_phase7_auroc(cfg, FULL_MODEL)}.",
        "",
        "## The questions: paired differences (i minus the other)",
        "",
        "Positive: the GraphAE is better (for concordance and AUROC, higher is better).",
        "",
        markdown_table(["Question", "Comparison", "Next-year micro AUROC", "Next-year macro AUROC",
                        "Neighbour concordance", "Probe AUROC"], question_rows, "llllll"),
        "",
        "## Next-year new conditions",
        "",
        f"A code counts as new in {task.year} for a patient whose first-ever event of it falls in "
        f"{task.year}; patients with an earlier event of it, not yet born or already dead on "
        f"{task.year}-01-01 are not at risk of it. Codes with at least {ev.min_train_onsets} new "
        "onsets among training-split patients at risk are kept. One logistic regression per code "
        f"(C {ev.C}, {'balanced' if ev.balanced_class_weight else 'unweighted'} classes, inputs "
        "standardized on the training split) is fitted on training-split patients at risk and "
        "scored on test-split patients at risk. Micro AUROC pools every test (patient, code) "
        f"pair; macro averages the per-code AUROC over the codes with a test onset. Bootstrap "
        f"resamples with both classes: {next_year.resamples_used}.",
        "",
        f"**Read the micro AUROC against the codes' base rates.** Scoring every patient by "
        f"each code's training onset rate, with no patient information at all, already gives "
        f"a micro AUROC of {_with_ci(base.micro_auroc, base.micro_low, base.micro_high)} "
        "(macro 0.5 by construction): pooling codes rewards knowing which codes are common. "
        f"The GraphAE minus base rates: {_with_ci(base['diff'], base.diff_low, base.diff_high, True)}"
        f"{_mark(base.diff_low, base.diff_high)}. "
        "The macro AUROC, the ranking of patients within each code, is the measure of what "
        "an embedding knows about the patient.",
        "",
        markdown_table(["Condition", "Train onsets", "Test onsets / at risk",
                        *[spec.number for spec in specs]], code_rows,
                       "lrr" + "r" * len(specs)),
        "",
        "## Neighbour concordance",
        "",
        f"For each test patient with a chronic condition (2017-2022, as in Phase 7): the mean "
        f"Jaccard index of their chronic conditions with each of their {ev.n_neighbours} nearest "
        "neighbours by cosine distance, searched among all patients. References: random other "
        "patients, and the 2022 input features (the neighbours that build the graph).",
        "",
        markdown_table(["Representation", f"Concordance ({level} CI)", "i minus this"],
                       conc_rows, "lrr"),
        "",
        "## Phenotype separation",
        "",
        "Euclidean silhouette (higher is better, in [-1, 1]) and Davies-Bouldin index (lower "
        "is better) of the final embeddings against the 7 Phase 4 phenotype clusters, test "
        "patients in a cluster. The last row shuffles the clusters (chance). The phenotypes "
        "were found by k-means on the 2022 graph's Laplacian eigenvectors, so iv is scored "
        "against clusters derived from itself.",
        "",
        markdown_table(["Representation", "Silhouette", "Davies-Bouldin"], cluster_rows, "lrr"),
        "",
        "## Linear probe of any_onset",
        "",
        f"Phase 7's protocol and folds: {cfg.cross_validation.folds}-fold cross-validation over "
        "every patient with an `any_onset` label; per fold, logistic regression (the `baselines` "
        "settings) on the standardized embedding + age at the cutoff and sex, fitted on the "
        f"inner training patients and Platt-calibrated on the inner validation patients "
        f"({cfg.cross_validation.inner_val:.0%}). {model_2_note}",
        "",
        markdown_table(["Probe", f"Pooled OOF AUROC ({level} CI)", "Minus model 2",
                        "Folds with Platt slope <= 0", "Mean within-fold AUROC, uncalibrated"],
                       probe_rows, "lrrrr"),
        "",
        "A Platt slope <= 0 means the inner validation patients ranked the other way, so the "
        "calibration reverses that fold's ranking; the pooled AUROC (the Phase 7 protocol) "
        "keeps that reversal. The last column, the mean of the folds' AUROCs of the "
        "uncalibrated logits, is not affected by calibration; it is a check, not a replacement.",
        "",
        "## Seed sensitivity of the GraphAE",
        "",
        f"The chosen setting (weight {chosen_weight:g}) retrained with each tuning seed: "
        "next-year micro AUROC on the test patients and PC1 share. The bootstrap intervals above "
        "do not include this variation.",
        "",
        markdown_table(["Seed", "Next-year micro AUROC", "PC1 share"], seed_rows, "rrr"),
        "",
        "## Figures",
        "",
        f"![Next-year AUROC]({figures[paths.next_year_auroc_fig]})",
        "",
        f"![UMAP by phenotype]({figures[paths.ae_umap_phenotype_fig]})",
        "",
        f"![UMAP by age]({figures[paths.ae_umap_age_fig]})",
        "",
        f"![UMAP by chronic conditions]({figures[paths.ae_umap_chronic_fig]})",
        "",
        trajectory_note,
        "",
        f"![Trajectories]({figures[paths.ae_trajectory_fig]})",
        "",
    ]
    text = "\n".join(lines)
    paths.embedding_results_md.parent.mkdir(parents=True, exist_ok=True)
    paths.embedding_results_md.write_text(text, encoding="utf-8")
    table.to_csv(paths.embedding_results_csv, index=False)
    logger.info("Wrote %s and %s", paths.embedding_results_md, paths.embedding_results_csv)
    return text


def write_embedding_summary(
    table: pd.DataFrame,
    specs: Sequence[EmbeddingSpec],
    chosen_weight: float,
    task: NextYearTask,
    next_year: NextYearResults,
    concordance: pd.DataFrame,
    seed_sensitivity: pd.DataFrame,
    code_names: Mapping[str, str],
    cfg: Config,
) -> str:
    """Write the plain-language Phase 7b findings to ``paths.embedding_summary_md``.

    Every verdict follows from a paired interval: a difference counts as shown only
    when its interval excludes 0.

    Args:
        table: ``embedding_table`` output.
        specs: ``embedding_specs`` output.
        chosen_weight: The chosen next-window weight.
        task: ``next_year_task`` output.
        next_year: ``next_year_results`` output (for the base-rate reference).
        concordance: ``concordance_results`` output.
        seed_sensitivity: Seed sensitivity of the GraphAE.
        code_names: Code -> condition name.
        cfg: Project configuration.

    Returns:
        The markdown written.
    """
    level = f"{cfg.evaluation.ci_level:.0%}"
    folds = cfg.cross_validation.folds
    by_key = table.set_index("key")
    ae, supervised = by_key.loc[GRAPH_AE], by_key.loc[SUPERVISED]
    name = {spec.key: f"{spec.name} ({spec.number})" for spec in specs}
    conc = concordance.set_index("key")
    chance, features = [key for key in conc.index if key not in by_key.index]
    base = next_year.table.set_index("key").loc[BASE_RATE]
    seeds = seed_sensitivity["micro_auroc"]
    metrics = [("next-year micro AUROC", "next_micro_auroc", "next_micro_diff", "micro AUROC"),
               ("next-year macro AUROC", "next_macro_auroc", "next_macro_diff", "macro AUROC"),
               ("neighbour concordance", "concordance", "concordance_diff", "mean Jaccard"),
               ("`any_onset` probe", "probe_auroc", "probe_diff", "pooled AUROC")]

    def verdict(key: str) -> list[str]:
        r = by_key.loc[key]
        lines = [f"  - {label[0].upper() + label[1:]}: "
                 + _sentence(ae[value], r[value], r[diff], r[f"{diff}_low"], r[f"{diff}_high"],
                             metric, name[key], level)
                 for label, value, diff, metric in metrics]
        flipped = int(r.probe_folds_flipped)
        if flipped:
            lines.append(
                f"  - Caution on the probe: in {flipped} of {folds} folds the calibration of "
                f"{name[key]} has a slope <= 0 and reverses that fold's ranking, which pulls its "
                f"pooled AUROC down. Uncalibrated, the mean within-fold AUROC is "
                f"{ae.probe_fold_auroc_raw:.3f} for the GraphAE and {r.probe_fold_auroc_raw:.3f} "
                f"for {name[key]}.")
        return lines

    def tally(key: str) -> str:
        r = by_key.loc[key]
        groups: dict[str, list[str]] = {"better on": [], "worse on": [], "not shown different on": []}
        for label, _, diff, _ in metrics:
            low, high = r[f"{diff}_low"], r[f"{diff}_high"]
            word = "better on" if low > 0 else "worse on" if high < 0 else "not shown different on"
            groups[word].append(label)
        return "; ".join(f"{word} {', '.join(items)}" for word, items in groups.items() if items)

    above_chance = [name[r.key] for r in table.itertuples() if r.next_macro_low > _CHANCE_AUROC]
    above_base = [r.key for r in table.itertuples() if r.next_micro_auroc > base.micro_auroc]
    common = sorted(range(len(task.codes)), key=lambda c: -task.train_onsets[c])[:3]
    examples = ", ".join(code_names.get(task.codes[c], task.codes[c]) for c in common)
    versus_model_2 = _verdict_word(ae.probe_vs_model2_low, ae.probe_vs_model2_high,
                                   "better than", "worse than", "not distinguishable from")
    versus_supervised = _verdict_word(supervised.concordance_diff_low,
                                      supervised.concordance_diff_high, "a clear improvement",
                                      "a clear loss", "no clear change")
    lines = [
        "# Embedding summary (Phase 7b)",
        "",
        "Plain-language findings of `scripts/07b_embeddings.py`; every number, with its setup, "
        "is in `embedding_results.md`.",
        "",
        "**Why this phase.** Phase 7 found that the supervised TemporalGCN's trajectory "
        "embedding had collapsed onto almost a single direction, and that a patient's nearest "
        "neighbours in it shared few of their chronic conditions compared with the neighbours "
        "in the input features. Phase 7b trains the same encoder (graph convolutions per year, "
        "a GRU over the years) without any outcome labels: it learns to reconstruct each year's "
        "patient-similarity graph and to predict which conditions are active the next year. "
        "Its embeddings are compared with seven others on held-out (test) patients.",
        "",
        f"**How results are read.** {level} paired bootstrap intervals over test patients; a "
        "difference counts as shown only if its interval excludes 0. The data are synthetic "
        "(Synthea).",
        "",
        "**What the four measures mean.**",
        "",
        f"- *Next-year new conditions*: from a patient's embedding at the end of {task.year - 1}, "
        f"predict which conditions they are diagnosed with for the first time in {task.year}. "
        f"Only {len(task.codes)} conditions are common enough to score, mostly short acute "
        f"ones (most frequent: {examples}). The micro AUROC pools all conditions, and the "
        f"conditions' base rates alone already reach {base.micro_auroc:.3f} on it; the macro "
        "AUROC (mean over conditions) measures how well patients are ranked within a "
        "condition, where 0.5 is guessing.",
        "- *Neighbour concordance*: the share of chronic conditions a patient has in common "
        "with their 5 nearest neighbours in the embedding (Jaccard; random patients: "
        f"{conc.loc[chance, 'concordance']:.1%}, input features: "
        f"{conc.loc[features, 'concordance']:.1%}).",
        "- *`any_onset` probe*: the Phase 7 prediction task (a new chronic condition in "
        "2023-2025), by logistic regression on the embedding plus age and sex, cross-validated "
        "on Phase 7's folds; compared with Phase 7's model 2 (the 2022 features plus age and "
        "sex).",
        "",
        "## Did the collapse go away?",
        "",
        f"- The GraphAE's final embedding puts {ae.pc1_share:.1%} of its variance on the first "
        f"principal component (supervised h: {supervised.pc1_share:.1%}; the GraphAE across "
        f"the three tuning seeds: {seed_sensitivity['pc1_share'].min():.1%}-"
        f"{seed_sensitivity['pc1_share'].max():.1%}). "
        + ("It is no longer collapsed onto one line, but one direction still dominates; PCA "
           f"spreads its variance far more evenly ({by_key.loc[PCA_FINAL, 'pc1_share']:.1%})."
           if ae.pc1_share < 0.9 else "It is still close to collapsed."),
        f"- Its nearest neighbours share {ae.concordance:.1%} of their chronic conditions, "
        f"against {supervised.concordance:.1%} for the supervised h: "
        f"{versus_supervised[0].lower() + versus_supervised[1:]}. It is still below PCA "
        f"({by_key.loc[PCA_FINAL, 'concordance']:.1%}), the spectral embedding "
        f"({by_key.loc[SPECTRAL, 'concordance']:.1%}) and the input features "
        f"({conc.loc[features, 'concordance']:.1%}).",
        "",
        "## The questions",
        "",
    ]
    for question, key in QUESTIONS:
        lines += [f"**{question}** ({name[GRAPH_AE]} vs {name[key]})", "",
                  f"*In short, the GraphAE is {tally(key)}.*", "", *verdict(key), ""]
    lines += [
        "## Overall",
        "",
        f"- **Next-year prediction is weak for every embedding.** Within a condition (macro "
        f"AUROC) the embeddings rank patients at {table['next_macro_auroc'].min():.3f}-"
        f"{table['next_macro_auroc'].max():.3f}; {len(above_chance)} of {len(table)} have an "
        f"interval above 0.5 ({_join(above_chance) if above_chance else 'none'}), and the "
        f"GraphAE's is {ae.next_macro_low:.3f} to {ae.next_macro_high:.3f}. Pooled over "
        f"conditions (micro AUROC, {table['next_micro_auroc'].min():.3f}-"
        f"{table['next_micro_auroc'].max():.3f}), {len(above_base)} of {len(table)} score above "
        f"the {base.micro_auroc:.3f} that the conditions' base rates give with no patient "
        f"information; the GraphAE minus base rates is {base['diff']:+.3f} ({level} CI "
        f"{base.diff_low:+.3f} to {base.diff_high:+.3f}). Across the tuning seeds the GraphAE's "
        f"micro AUROC ranges {seeds.min():.3f}-{seeds.max():.3f}, a spread the intervals do not "
        "include.",
        f"- **`any_onset` probe:** the GraphAE + age, sex reaches {ae.probe_auroc:.3f}; minus "
        f"Phase 7 model 2: {ae.probe_vs_model2:+.3f} ({level} CI {ae.probe_vs_model2_low:+.3f} to "
        f"{ae.probe_vs_model2_high:+.3f}): {versus_model_2[0].lower() + versus_model_2[1:]} "
        "model 2.",
        "",
        "## Caveats",
        "",
        f"- **Next-window weight {chosen_weight:g}** was chosen on validation next-window loss; "
        "that criterion favours whichever setting weights the next-window term most, so it "
        "says nothing about the link term's value. No other hyperparameter was searched (the "
        "Phase 6 TemporalGCN setting is reused).",
        "- **Link-only (viii) has an untrained GRU.** With the next-window gradient cut, nothing "
        "trains the recurrent part: its final state is a fixed random recurrent map of the "
        "link-trained per-year embeddings. A linear readout trained alongside (on detached "
        "states) only picks its epoch.",
        "- **No graph (vi) also has no link term** (there are no edges to reconstruct), so i vs "
        "vi tests the graph as a whole, message passing and link supervision together.",
        "- **Probe calibration.** The probe follows Phase 7's protocol (Platt scaling on each "
        "fold's inner validation patients, then pooled). For embeddings with almost no signal "
        "the fitted slope can be negative and reverse a fold's ranking, which can push the "
        "pooled AUROC below 0.5; the uncalibrated within-fold AUROC is reported next to it.",
        "- **Transductive self-supervision.** Test patients' features and edges, including their "
        f"{task.year} graph, enter message passing and the link term, so the shared weights have "
        f"seen their {task.year} feature rows (never as next-window targets). The {task.year - 1} "
        f"embeddings used for the next-year task read no {task.year} data for any patient, but "
        f"the encoder weights were shaped partly by the {task.year} graph. The IDF weights and "
        f"the transition SVD of the features are fitted on 2017-{task.year} windows (population "
        "statistics, Phase 3).",
        "- **The supervised h (v)** was trained on training-split `any_onset` labels; its probe "
        "AUROC is optimistic (those patients are in the probe's held-out folds).",
        "- **The spectral embedding (iv)** is scored against phenotypes derived from the same "
        "graph's eigenvectors, so its silhouette is circular.",
        "- **What the intervals cover.** They resample test patients; they do not include "
        "retraining variation (seeds), which the seed check shows is not negligible.",
        f"- **Small numbers.** {int(task.test_onsets.sum())} test onsets over "
        f"{len(task.codes)} conditions, {int((task.test_onsets <= 3).sum())} of them with 3 or "
        "fewer: per-condition AUROCs, and so the macro AUROC, are noisy.",
        "",
    ]
    text = "\n".join(lines)
    path = cfg.paths.embedding_summary_md
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)
    return text


def _sentence(ae_value: float, other: float, diff: float, low: float, high: float, metric: str,
              other_name: str, level: str) -> str:
    """A verdict on the GraphAE's ``metric`` minus ``other_name``'s, with its interval."""
    numbers = (f"{metric} {ae_value:.3f} vs {other:.3f}, difference {diff:+.3f} ({level} CI "
               f"{low:+.3f} to {high:+.3f})")
    if low > 0:
        return f"**Yes**, the GraphAE is better: {numbers}."
    if high < 0:
        return f"**No, the reverse**: {other_name} is better: {numbers}."
    favours = "the GraphAE" if diff > 0 else other_name
    return f"**Not shown** (the interval includes 0; the point estimate favours {favours}): {numbers}."


def _phase7_auroc(cfg: Config, key: str) -> str:
    """Phase 7's pooled out-of-fold AUROC of model ``key`` with its CI, from results.csv."""
    if not cfg.paths.results_csv.exists():
        return "see reports/results.md"
    row = pd.read_csv(cfg.paths.results_csv).set_index("key").loc[key]
    return _with_ci(row.oof_auroc, row.oof_auroc_low, row.oof_auroc_high)


def _join(items: Sequence[str]) -> str:
    """["a"] -> "a"; ["a", "b"] -> "a and b"; ["a", "b", "c"] -> "a, b and c"."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _verdict_word(low: float, high: float, better: str, worse: str, tied: str) -> str:
    """``better`` if the interval is above 0, ``worse`` if below, else ``tied`` (capitalized)."""
    word = better if low > 0 else worse if high < 0 else tied
    return word[0].upper() + word[1:]


def _mark(low: float, high: float) -> str:
    """" **shown**" when the interval excludes 0, else ""."""
    return " **shown**" if low > 0 or high < 0 else ""


def _cut(text: str, max_chars: int) -> str:
    """``text`` cut to ``max_chars`` characters, an ellipsis marking a cut."""
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def _with_ci(value: float, low: float, high: float, signed: bool = False) -> str:
    """Format ``value (low to high)`` with 3 decimals, signed if asked."""
    if not np.isfinite(value):
        return "n/a"
    fmt = "{:+.3f}" if signed else "{:.3f}"
    return f"{fmt.format(value)} ({fmt.format(low)} to {fmt.format(high)})"

