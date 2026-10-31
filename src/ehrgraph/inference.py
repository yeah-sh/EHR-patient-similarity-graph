"""Phase 8: place a new patient among the reference patients and describe their neighbourhood.

Load once, then predict as often as needed (``scripts/08_predict_demo.py``)::

    predictor = Predictor.load()  # config, saved transforms, graphs, weights, reference tables
    prediction = predictor.predict(NewPatient(history=[(2018, "714628002"), ...],
                                              birth_date="1972-06-01", sex="M"))
    lifts = predictor.explain(prediction)

The reference patients are the 2,581 model-cohort patients of Phases 3-7b. Nothing is
refitted or retrained: the new patient's features use the saved IDF and transition SVD,
and every model keeps its saved ``state_dict``. A prediction runs these steps:

1. **Validation** (``Predictor.parse_history``): codes are merged (``data.code_merge``) and
   checked against the code vocabulary. Unknown codes are returned, never raised; redundant
   codes (``features.redundant_codes``) and events on or after the prediction cutoff are
   dropped and reported, so no later event can reach the features.
2. **Features** (``Predictor.history_features``): one row per window 2017-2022, built
   exactly as in Phase 3 (``features.build_features``) with the saved fitted transforms.
3. **Graph insertion** (``nearest_reference``, ``insert_node``): in each window the patient
   becomes node N + 1, linked to its ``graph.k`` most cosine-similar reference patients, and
   ``A_hat`` is re-normalized for the affected nodes only.
4. **Models** on the augmented graphs: the Phase 7b TemporalGraphAE's GRU state after each
   window (the trajectory) and the supervised TemporalGCN's ``any_onset`` logit, Platt-
   calibrated with ``calibration.json``. Only the new node's outputs are read.
5. **Spectral placement** (``nystrom_extension``) in the 2022 graph's spectral embedding,
   the best embedding by neighbour concordance in Phase 7b. There the patient gets a
   phenotype (the nearest k-means centroid of Phase 4), the most similar reference patients
   (cosine), and the onsets those neighbours had in the label horizon (observed labels only).

**Risk set.** ``any_onset`` is a NEW onset in any of the onset groups. A group the patient
already has cannot have a new onset, so each prediction reports the groups it is still
open in ("at risk for N of 9 groups", ``Prediction.at_risk_groups``). The neighbour outcomes
count only neighbours whose label is observed, i.e. who were still at risk of that group.

**Age band** (``Predictor.similar_patients``, ``Predictor.neighbour_outcomes``): optionally
the similar patients are post-filtered to reference patients of similar age, reading further
down the same spectral ranking. The graphs, the embeddings and the placement never change.

**The inductive approximation.** The reference graphs are k-NN graphs symmetrized with
``max(W, W^T)``: a reference patient is also linked to every patient that has it among its
own k nearest. Rebuilding the graph with the new patient would add such reverse edges and
could displace other patients' k-th neighbours. Here the reference graph stays fixed: the new
node gets only its own k out-edges (made symmetric), no reference edge is added or removed,
and only the degrees of the new node's neighbours change. The models were trained
transductively on the reference graphs; for a new node they are applied with frozen weights.
For a reference patient re-entered as new (its own node excluded from the search), the tests
compare this approximation with the transductive output.

No mortality output: mortality was never a model target (``labels``).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml
from sklearn.preprocessing import normalize

from ehrgraph.config import DEFAULT_CONFIG_PATH, Config, load_config
from ehrgraph.data import DAYS_PER_YEAR, load_patients
from ehrgraph.embeddings import GRAPH_AE, SPECTRAL, patient_resamples, weighted_auroc
from ehrgraph.evaluate import Calibrator, load_calibration, percentile_interval
from ehrgraph.features import (
    DEMOGRAPHICS,
    SEX_CODES,
    chronic_mask,
    chronic_presence,
    combine_features,
    load_feature_codes,
    load_model_inputs,
    patient_demographics,
    tfidf_transform,
    time_decay_cumulative,
    transition_blocks,
)
from ehrgraph.graph import cluster_phenotypes, laplacian, load_graphs, load_phenotype_clusters
from ehrgraph.labels import ANY_ONSET, ONSET_PREFIX, load_labels
from ehrgraph.models import OnsetModel, TemporalGraphAE, sparse_tensor
from ehrgraph.train import load_graph_ae, load_trained_model
from ehrgraph.trajectories import (
    TRANSITION_COLUMNS,
    Trajectories,
    active_in_window,
    extract_transitions,
    load_code_vocabulary,
    load_trajectories,
)

logger = logging.getLogger(__name__)

NEW_PATIENT = "new patient"  # patient_id of the new patient's events
EVENT_COLUMNS: list[str] = ["patient_id", "date", "stop", "active_until", "code"]
SIMILAR_COLUMNS: list[str] = [
    "rank", "patient_id", "similarity", "age", "sex", "phenotype", "chronic_conditions",
    "shared_conditions", "shared",
]
LIFT_COLUMNS: list[str] = [
    "code", "condition", "neighbours_with", "neighbours", "neighbourhood_share",
    "population_share", "lift",
]
DROPPED_COLUMNS: list[str] = ["code", "onset", "reason"]
# The risk set in one line, and why a healthier patient's neighbours can show a higher
# any_onset rate than a multimorbid patient's.
RISK_SET_RULE = "A new onset can only occur in a group the patient does not already have."
RISK_SET_EXPLANATION = (
    "A new onset can only occur in a group the patient does not already have, so a "
    "multimorbid patient has fewer groups left at risk, and so do their similar patients. "
    "A healthy patient's neighbours can therefore show a higher any_onset rate than a "
    "multimorbid patient's: more groups in which a new onset can occur, not more disease."
)
_SEMANTIC_TAG = re.compile(r"\s*\([^()]*\)$")  # " (disorder)", dropped from condition names


@dataclass(frozen=True)
class NewPatient:
    """A patient to place among the reference patients.

    Attributes:
        history: The conditions, each ``(onset, code)`` or ``(onset, code, stop)``, or a
            mapping code -> onset. Onset and stop are a year (int), a date or an ISO
            date string; a year means 1 January of it. A stop of None means still
            active. Without a stop, a code that is chronic in most reference episodes
            (``inference.chronic_default_share``) stays active through the last window,
            any other code is active in its onset year only.
        birth_date: Needed, with ``sex``, only for the supervised ``any_onset`` probability.
        sex: "F" or "M" (``features.SEX_CODES``).
    """

    history: Sequence[Sequence[Any]] | Mapping[str, Any]
    birth_date: date | str | None = None
    sex: str | None = None


@dataclass(frozen=True)
class ParsedHistory:
    """A validated history (``Predictor.parse_history``).

    Attributes:
        events: ``EVENT_COLUMNS``, the known codes' events before the cutoff, by date
            and code; ``active_until`` is the stop or, for a condition still active, the
            reference data's end.
        unknown_codes: Input codes not in the code vocabulary (after merging), in input
            order, each once.
        dropped: ``DROPPED_COLUMNS``: known entries not used, and why.
    """

    events: pd.DataFrame
    unknown_codes: tuple[str, ...]
    dropped: pd.DataFrame


@dataclass(frozen=True)
class Insertion:
    """The new node's edges in one window's graph (``nearest_reference``).

    Attributes:
        year: The window.
        neighbours: Reference nodes, most similar first; empty if the patient's row is
            all zero (an isolated node, as for reference patients).
        weights: Their cosine similarities, all positive.
    """

    year: int
    neighbours: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class Phenotype:
    """The nearest Phase 4 phenotype cluster.

    Attributes:
        cluster: Cluster number (1 = largest).
        patients: Reference patients in it.
        description: Its most over-represented chronic conditions (``graph.cluster_phenotypes``),
            with their prevalence in the cluster and overall.
    """

    cluster: int
    patients: int
    description: str


@dataclass(frozen=True)
class Prediction:
    """Everything ``Predictor.predict`` returns for one new patient.

    Attributes:
        unknown_codes: Input codes not in the code vocabulary; not used.
        dropped: Known entries not used (``DROPPED_COLUMNS``).
        conditions: code, condition, onset (first event), chronic: the conditions used.
        chronic_codes: Chronic conditions active in a feature window (feature vocabulary),
            as for the reference patients (``features.chronic_presence``).
        prior_groups: Onset groups the patient already had before the cutoff.
        at_risk_groups: The other onset groups, in config order: the only ones in which the
            patient can have a new onset, i.e. the risk set of ``any_onset``.
        age: Age in years at the cutoff, as for the reference patients (None without a
            birth date).
        excluded: Reference nodes that may not be neighbours (``predict(exclude=...)``).
        years: The feature windows.
        features: [T, F] the patient's feature rows.
        insertions: The graph insertion of each window.
        placed: Whether the patient has a 2022 graph neighbour; if not, they have no
            known condition in 2017-2022 (they would be outside the model cohort) and
            get no spectral placement, phenotype, neighbours or probability.
        window_states: [T, d] TemporalGraphAE GRU state after each window, in the space
            of ``Predictor.reference_states``.
        spectral: [d] Nystrom coordinates in the 2022 spectral embedding (NaN if not placed).
        phenotype: The nearest phenotype cluster (None if not placed).
        neighbours: Reference nodes by cosine similarity in the spectral embedding, most
            similar first, as many as any output needs.
        similarities: Their cosine similarities.
        similar: ``SIMILAR_COLUMNS``, the ``inference.n_similar`` most similar.
        outcomes: target, group, patient_had_it, then positives_<n> and observed_<n> per
            ``inference.outcome_neighbours`` n, and positives_all, observed_all over every
            reference patient: onsets in the label horizon, observed labels only.
        any_onset_logit: The supervised model's raw logit (None without demographics or
            placement).
        any_onset_probability: Its Platt-calibrated probability (None likewise).
        notes: Plain-language remarks on the input and on what could not be computed.
    """

    unknown_codes: tuple[str, ...]
    dropped: pd.DataFrame
    conditions: pd.DataFrame
    chronic_codes: tuple[str, ...]
    prior_groups: tuple[str, ...]
    at_risk_groups: tuple[str, ...]
    age: float | None
    excluded: tuple[int, ...]
    years: tuple[int, ...]
    features: np.ndarray
    insertions: tuple[Insertion, ...]
    placed: bool
    window_states: np.ndarray
    spectral: np.ndarray
    phenotype: Phenotype | None
    neighbours: np.ndarray
    similarities: np.ndarray
    similar: pd.DataFrame
    outcomes: pd.DataFrame
    any_onset_logit: float | None
    any_onset_probability: float | None
    notes: tuple[str, ...]

    @property
    def risk_set(self) -> str:
        """The risk set in words, e.g. "At risk for 5 of 9 groups" (``RISK_SET_EXPLANATION``)."""
        groups = len(self.at_risk_groups) + len(self.prior_groups)
        return f"At risk for {len(self.at_risk_groups)} of {groups} groups"


# --- Linear algebra of the insertion ---------------------------------------------------


def nearest_reference(
    reference: np.ndarray, x: np.ndarray, k: int, allowed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Find the ``k`` allowed reference rows most cosine-similar to ``x``.

    As ``graph.knn_graph``: all-zero rows are never neighbours, an all-zero ``x`` has
    none, and non-positive similarities are dropped. Ties keep the lower node first.

    Args:
        reference: [N, F] reference feature rows.
        x: [F] the new patient's row.
        k: Neighbours (``graph.k``).
        allowed: [N] bool; False rows are never chosen.

    Returns:
        The neighbours' rows, most similar first, and their cosine similarities.
    """
    norms = np.linalg.norm(reference, axis=1)
    x_norm = float(np.linalg.norm(x))
    candidates = np.flatnonzero(allowed & (norms > 0))
    if x_norm == 0 or len(candidates) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0)
    similarity = reference[candidates] @ x / (norms[candidates] * x_norm)
    order = np.argsort(-similarity, kind="stable")[:k]
    keep = similarity[order] > 0
    return candidates[order][keep], similarity[order][keep]


def insert_node(
    adjacency: sp.spmatrix, degrees: np.ndarray, neighbours: np.ndarray, weights: np.ndarray
) -> sp.csr_matrix:
    """Append a node to ``A_hat = D^(-1/2) (W + I) D^(-1/2)``, re-normalizing affected nodes only.

    The new node x has edges of weight ``w_j`` to its neighbours j. Their degrees grow to
    ``d'_j = d_j + w_j`` and x's degree is ``d_x = 1 + sum_j w_j`` (with its self-loop), so::

        A_hat' = [[R A_hat R,  a        ],
                  [a^T,        1 / d_x  ]],   R = diag(sqrt(d / d')),
                                              a_j = w_j / sqrt(d'_j d_x)

    R differs from I only at the neighbours, so only their rows and columns change; this
    equals ``graph.normalize_adjacency`` of the augmented W. With no neighbours the node is
    isolated and keeps only its self-loop.

    Args:
        adjacency: [N, N] ``A_hat`` of the reference graph.
        degrees: [N] degrees of ``W + I`` (1 + weighted degree of W).
        neighbours: The new node's neighbours (distinct reference nodes).
        weights: Their edge weights, positive.

    Returns:
        CSR [N + 1, N + 1]; the new node is the last.
    """
    n = adjacency.shape[0]
    grown = np.asarray(degrees, dtype=np.float64).copy()
    grown[neighbours] += weights
    d_x = 1.0 + float(np.sum(weights))
    R = sp.diags(np.sqrt(degrees / grown))
    column = sp.csr_matrix(
        (weights / np.sqrt(grown[neighbours] * d_x), (neighbours, np.zeros(len(neighbours), int))),
        shape=(n, 1),
    )
    return sp.bmat([[R @ sp.csr_matrix(adjacency) @ R, column],
                    [column.T, sp.csr_matrix([[1.0 / d_x]])]], format="csr")


def nystrom_extension(
    neighbours: np.ndarray,
    weights: np.ndarray,
    degrees: np.ndarray,
    vectors: np.ndarray,
    eigenvalues: np.ndarray,
) -> np.ndarray:
    """Place a new node in a normalized-Laplacian spectral embedding (Nystrom extension).

    The reference embedding holds eigenvectors u of ``L = I - D^(-1/2) W D^(-1/2)``:
    ``L u = lambda u``, i.e. ``D^(-1/2) W D^(-1/2) u = (1 - lambda) u``. Row i of that
    identity, ``u_i = 1 / (1 - lambda) * sum_j W_ij u_j / sqrt(d_i d_j)``, gives the
    out-of-sample formula for a new node x with edge weights ``w_xj``::

        u_x = 1 / (1 - lambda) * sum_j (w_xj / sqrt(d_x d_j)) u_j,   d_x = sum_j w_xj

    a degree-weighted average of its neighbours' eigenvector entries, divided by the
    eigenvalue term ``1 - lambda``. The reference degrees ``d_j`` stay those of the
    reference graph, whose eigenvectors these are; only x's own row is normalized by its
    degree. For a reference node with its own row of W the formula returns its
    eigenvector entries exactly; for a new node it is exact only if adding x left the
    reference graph unchanged (the approximation of ``insert_node``).

    Args:
        neighbours: The new node's neighbours (reference nodes).
        weights: Their edge weights, positive.
        degrees: [N] weighted degrees of the reference W (no self-loops).
        vectors: [N, m] eigenvectors, one per column.
        eigenvalues: [m] their eigenvalues, each below 1.

    Returns:
        [m] the new node's coordinates; NaN if it has no neighbour.
    """
    if len(neighbours) == 0:
        return np.full(vectors.shape[1], np.nan)
    d_x = float(np.sum(weights))
    coefficients = weights / np.sqrt(d_x * degrees[neighbours])
    return (coefficients @ vectors[neighbours]) / (1.0 - eigenvalues)


def cosine_ranking(
    embedding: np.ndarray, x: np.ndarray, allowed: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rank the allowed rows of ``embedding`` by cosine similarity to ``x``; keep the first ``n``.

    A zero row (or zero ``x``) has similarity 0. Ties keep the lower row first. Filtering
    with ``allowed`` and then ranking is the same as ranking every row and skipping the
    disallowed ones, so a post-filter reads further down one fixed ranking.

    Args:
        embedding: [N, d] reference rows.
        x: [d] the query.
        allowed: [N] bool; False rows are never returned.
        n: Rows to return (fewer if fewer are allowed).

    Returns:
        The rows, most similar first, and their cosine similarities.
    """
    norms = np.linalg.norm(embedding, axis=1) * np.linalg.norm(x)
    similarity = np.divide(embedding @ x, norms, out=np.zeros(len(norms)), where=norms > 0)
    candidates = np.flatnonzero(allowed)
    order = candidates[np.argsort(-similarity[candidates], kind="stable")][:n]
    return order, similarity[order]


def map_positions(
    states: np.ndarray, reference_states: np.ndarray, reference_positions: np.ndarray, k: int
) -> np.ndarray:
    """Place states on a fixed 2-D map of reference states (e.g. a UMAP), without refitting it.

    Each state goes to the mean map position of its ``k`` nearest reference states
    (Euclidean, in the state space). Reference rows without a position (NaN, e.g. empty
    windows left out of the map) are skipped.

    Args:
        states: [m, d] states to place.
        reference_states: [M, d] the mapped states.
        reference_positions: [M, 2] their map positions.
        k: Nearest reference states averaged (at most the number with a position).

    Returns:
        [m, 2] map positions.
    """
    mapped = ~np.isnan(reference_positions).any(axis=1)
    reference, positions = reference_states[mapped], reference_positions[mapped]
    k = min(k, len(reference))
    placed = np.empty((len(states), positions.shape[1]))
    for i, state in enumerate(np.asarray(states, dtype=np.float64)):
        distance = np.sum((reference - state) ** 2, axis=1)
        nearest = np.argpartition(distance, k - 1)[:k]
        placed[i] = positions[nearest].mean(axis=0)
    return placed


def outcome_counts(labels: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count positives and observed labels per target among ``nodes``.

    A masked (NaN) label is neither a positive nor a negative: it leaves the denominator.

    Args:
        labels: [N, G] 1.0, 0.0 or NaN.
        nodes: Rows to count.

    Returns:
        Positives [G] and observed labels [G], as integers.
    """
    rows = np.asarray(labels, dtype=float)[np.asarray(nodes, dtype=np.int64)]
    seen = ~np.isnan(rows)
    return (np.where(seen, rows, 0.0) == 1.0).sum(axis=0), seen.sum(axis=0)


def condition_lift(presence: sp.spmatrix, nodes: np.ndarray, columns: np.ndarray) -> pd.DataFrame:
    """Lift of conditions among ``nodes`` over all rows of ``presence``.

    ``lift = (share of nodes with the condition) / (share of all rows with it)``; 0 when
    no row has it.

    Args:
        presence: Binary [N, C] conditions.
        nodes: The neighbourhood's rows.
        columns: Condition columns to score.

    Returns:
        column, neighbours_with, neighbours, neighbourhood_share, population_share, lift.
    """
    presence = sp.csr_matrix(presence)
    columns = np.asarray(columns, dtype=np.int64)
    within = np.asarray(presence[np.asarray(nodes, dtype=np.int64)][:, columns].sum(axis=0)).ravel()
    population = np.asarray(presence[:, columns].sum(axis=0)).ravel() / presence.shape[0]
    share = within / max(len(nodes), 1)
    lift = np.divide(share, population, out=np.zeros_like(share), where=population > 0)
    return pd.DataFrame({"column": columns, "neighbours_with": within.astype(np.int64),
                         "neighbours": len(nodes), "neighbourhood_share": share,
                         "population_share": population, "lift": lift})


# --- The predictor -----------------------------------------------------------------------


@dataclass(frozen=True)
class Predictor:
    """Saved transforms, graphs, models and reference tables; see the module docstring.

    Build with ``Predictor.load``. Node order is the model cohort's.

    Attributes:
        cfg: Project configuration.
        years: The feature windows (2017-2022).
        patient_ids: The reference patients, in node order.
        vocabulary: Code -> description, the Phase 2 code vocabulary.
        feature_codes: The feature vocabulary (the first columns of the features).
        redundant_codes: Vocabulary codes dropped from the features (Phase 3).
        chronic_default: Codes taken as chronic when given without a stop.
        idf: Saved IDF per feature code.
        svd_components: Saved transition SVD components [n_components, n_codes ** 2].
        data_end: End of the reference data: a condition with no stop is active until then.
        reference: [N, F] float64 feature rows per window (the neighbour search).
        tensors: The same as float32 tensors (the models' input).
        adjacency: ``A_hat`` per window, CSR [N, N].
        degrees: Degrees of ``W + I`` per window.
        supervised: The Phase 6 TemporalGCN (``paths.model_pt``).
        calibrator: Its Platt scaling (``paths.calibration_json``).
        age_mean: Training-split mean age, for standardizing the head input.
        age_sd: Its standard deviation.
        any_onset_auroc: The supervised model's test AUROC and its bootstrap interval,
            from ``paths.test_predictions_csv``.
        graph_ae: The Phase 7b TemporalGraphAE (``inference.graph_ae_key``).
        reference_states: [N, T, d] its GRU state after each window for every reference
            patient (Phase 7b, transductive).
        spectral: [N, d] the 2022 spectral embedding (Phase 7b, ``embedding_eval.dim``).
        spectral_basis: [N, d + 1] the trivial eigenvector, then ``spectral``.
        spectral_values: [d + 1] their eigenvalues (Rayleigh quotients on the 2022 Laplacian).
        spectral_degrees: [N] weighted degrees of the 2022 W.
        clusters: [N] Phase 4 phenotype cluster (0 = unclustered).
        centroids: [k, k] k-means centroids of the L2-normalized rows of the first k
            eigenvectors, trivial one included (as ``graph.spectral_clusters``).
        phenotypes: Cluster -> its ``Phenotype``.
        chronic: CSR [N, n_feature_codes] chronic conditions in the feature windows.
        targets: Label columns of ``labels``: ``ANY_ONSET``, then each group's.
        labels: [N, len(targets)] 1.0, 0.0 or NaN (masked).
        demographics: Age at the cutoff and sex (``features.patient_demographics``).
        patients: patient_id-indexed birth_date and gender of the reference patients.
        events: The reference patients' trajectory events (``Trajectories.events``).
        names: Code -> condition name without its semantic tag.
    """

    cfg: Config
    years: tuple[int, ...]
    patient_ids: tuple[str, ...]
    vocabulary: Mapping[str, str]
    feature_codes: tuple[str, ...]
    redundant_codes: frozenset[str]
    chronic_default: frozenset[str]
    idf: np.ndarray
    svd_components: np.ndarray
    data_end: pd.Timestamp
    reference: list[np.ndarray]
    tensors: list[torch.Tensor]
    adjacency: list[sp.csr_matrix]
    degrees: list[np.ndarray]
    supervised: OnsetModel
    calibrator: Calibrator
    age_mean: float
    age_sd: float
    any_onset_auroc: tuple[float, float, float]
    graph_ae: TemporalGraphAE
    reference_states: np.ndarray
    spectral: np.ndarray
    spectral_basis: np.ndarray
    spectral_values: np.ndarray
    spectral_degrees: np.ndarray
    clusters: np.ndarray
    centroids: np.ndarray
    phenotypes: Mapping[int, Phenotype]
    chronic: sp.csr_matrix
    targets: tuple[str, ...]
    labels: np.ndarray
    demographics: pd.DataFrame
    patients: pd.DataFrame
    events: pd.DataFrame
    names: Mapping[str, str]

    @classmethod
    def load(cls, artifacts_dir: Path | None = None, config_path: Path | None = None) -> Predictor:
        """Load everything a prediction needs, once.

        Args:
            artifacts_dir: Where the fitted transforms and weights are (default
                ``paths.artifacts_dir``); each is read under its configured name relative
                to that directory. Reference features, graphs and labels come from
                ``paths.processed_dir``.
            config_path: Config YAML (default ``config/config.yaml``).

        Returns:
            The predictor.

        Raises:
            ValueError: If a saved model was trained with another config section, or the
                saved files disagree on the reference patients.
        """
        started = perf_counter()
        config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        cfg = load_config(config_path)
        paths, inf = cfg.paths, cfg.inference
        base = Path(artifacts_dir) if artifacts_dir is not None else paths.artifacts_dir

        def artifact(path: Path) -> Path:
            return base / path.relative_to(paths.artifacts_dir)

        current = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        _check_config_used(artifact(paths.config_used_yaml), current, ("model",))
        _check_config_used(artifact(paths.graph_ae_dir) / "config_used.yaml", current,
                           ("graph_ae", "embedding_eval", "seed"))

        inputs = load_model_inputs(cfg)
        graphs = load_graphs(cfg)
        if graphs.patient_ids != inputs.patient_ids:
            raise ValueError("the saved graphs and features disagree on the reference patients")
        patient_ids, years = inputs.patient_ids, inputs.years
        reference = [X.toarray().astype(np.float64) for X in inputs.matrices]
        in_dim = reference[0].shape[1]

        vocabulary = load_code_vocabulary(cfg)
        meta = json.loads((paths.features_dir / "features.json").read_text(encoding="utf-8"))
        feature_codes = load_feature_codes(cfg)
        trajectories = load_trajectories(cfg)
        episodes = trajectories.events.assign(
            chronic=chronic_mask(trajectories.events, cfg.features.chronic_min_duration_days))
        chronic_share = episodes.groupby("code")["chronic"].mean()

        supervised = load_trained_model(cfg.model, artifact(paths.model_pt), in_dim,
                                        len(cfg.labels.onset_groups))
        metrics = json.loads(artifact(paths.train_metrics_json).read_text(encoding="utf-8"))
        if tuple(metrics["patient_ids"]) != patient_ids:
            raise ValueError(f"{paths.model_pt.name} was trained on other reference patients")
        graph_ae = load_graph_ae(cfg.graph_ae, artifact(paths.graph_ae_dir) / f"{inf.graph_ae_key}.pt",
                                 in_dim, len(feature_codes))
        stored = np.load(artifact(paths.embeddings_npz))
        if tuple(stored["patient_ids"]) != patient_ids:
            raise ValueError(f"{paths.embeddings_npz.name} is for other reference patients")
        if inf.graph_ae_key != GRAPH_AE:
            raise ValueError(f"{paths.embeddings_npz.name} holds the per-window states of "
                             f"{GRAPH_AE!r} only, not {inf.graph_ae_key!r}")

        W_last = graphs.weights[-1]
        spectral = stored[f"{SPECTRAL}.final"]
        spectral_degrees = np.asarray(W_last.sum(axis=1)).ravel()
        if (spectral_degrees == 0).any():
            raise ValueError("the final window's graph has isolated nodes; the spectral "
                             "placement assumes one connected component (graphs_summary.md)")
        trivial = np.sqrt(spectral_degrees) / np.linalg.norm(np.sqrt(spectral_degrees))
        basis = np.column_stack([trivial, spectral])
        L = laplacian(W_last)
        values = np.einsum("ij,ij->j", basis, L @ basis) / np.einsum("ij,ij->j", basis, basis)

        clusters = (load_phenotype_clusters(cfg).reindex(pd.Index(patient_ids))
                    .fillna(0).to_numpy(dtype=np.int64))
        cluster_ids = np.unique(clusters[clusters > 0])
        rows = normalize(basis[:, :len(cluster_ids)])
        centroids = np.stack([rows[clusters == c].mean(axis=0) for c in cluster_ids])
        presence, chronic_codes = chronic_presence(trajectories, cfg)
        chronic = sp.csr_matrix(presence[inputs.rows])
        if tuple(chronic_codes) != feature_codes:
            raise ValueError("chronic presence and features use different code vocabularies")
        names = {code: _SEMANTIC_TAG.sub("", description)
                 for code, description in zip(vocabulary["code"], vocabulary["description"])}
        phenotypes = _phenotypes(chronic, clusters, feature_codes, names, cfg)

        labels = load_labels(cfg)
        if tuple(labels["patient_id"]) != patient_ids:
            raise ValueError("labels.parquet is not in node order")
        targets = (ANY_ONSET, *(ONSET_PREFIX + group for group in cfg.labels.onset_groups))
        patients = load_patients(cfg)
        cohort_patients = (patients.set_index("patient_id")
                           .reindex(pd.Index(patient_ids, name="patient_id"))[["birth_date", "gender"]])

        predictor = cls(
            cfg=cfg,
            years=years,
            patient_ids=patient_ids,
            vocabulary=dict(zip(vocabulary["code"], vocabulary["description"])),
            feature_codes=feature_codes,
            redundant_codes=frozenset(meta["redundant_codes"]),
            chronic_default=frozenset(chronic_share.index[chronic_share >= inf.chronic_default_share]),
            idf=np.load(artifact(paths.tfidf_idf_npy)),
            svd_components=np.load(artifact(paths.transition_svd_npz))["components"],
            data_end=trajectories.data_end,
            reference=reference,
            tensors=[torch.from_numpy(X.astype(np.float32)) for X in reference],
            adjacency=graphs.adjacency,
            degrees=[1.0 + np.asarray(W.sum(axis=1)).ravel() for W in graphs.weights],
            supervised=supervised,
            calibrator=load_calibration(artifact(paths.calibration_json)),
            age_mean=float(metrics["demographics"]["age_mean_train"]),
            age_sd=float(metrics["demographics"]["age_sd_train"]),
            any_onset_auroc=_test_auroc(artifact(paths.test_predictions_csv), cfg),
            graph_ae=graph_ae,
            reference_states=stored[f"{GRAPH_AE}.states"],
            spectral=spectral,
            spectral_basis=basis,
            spectral_values=values,
            spectral_degrees=spectral_degrees,
            clusters=clusters,
            centroids=centroids,
            phenotypes=phenotypes,
            chronic=chronic,
            targets=targets,
            labels=labels[list(targets)].to_numpy(dtype=np.float64),
            demographics=patient_demographics(patients, patient_ids, cfg.labels.cutoff_date),
            patients=cohort_patients,
            events=trajectories.events[trajectories.events["patient_id"].isin(patient_ids)],
            names=names,
        )
        logger.info("Predictor loaded in %.2f s: %d reference patients, windows %d-%d, %d "
                    "features, %d phenotypes, %d chronic-by-default codes; supervised model %s",
                    perf_counter() - started, len(patient_ids), years[0], years[-1], in_dim,
                    len(cluster_ids), len(predictor.chronic_default), predictor.model_label)
        return predictor

    @property
    def model_label(self) -> str:
        """The supervised model's test AUROC with its interval, e.g. "test AUROC 0.559 (0.483-0.632)"."""
        auroc, low, high = self.any_onset_auroc
        return f"test AUROC {auroc:.3f} ({low:.3f}-{high:.3f})"

    @property
    def node_index(self) -> dict[str, int]:
        """Reference patient_id -> node."""
        return {patient: i for i, patient in enumerate(self.patient_ids)}

    # --- Inputs ------------------------------------------------------------------------

    def parse_history(self, history: Sequence[Sequence[Any]] | Mapping[str, Any]) -> ParsedHistory:
        """Validate a history (see ``NewPatient.history``).

        Unknown codes are collected, not raised. Codes in ``data.code_merge`` are mapped
        to their canonical code first. Redundant codes and events on or after the
        prediction cutoff are dropped and listed with the reason.

        Args:
            history: ``NewPatient.history``.

        Returns:
            The parsed history.

        Raises:
            ValueError: If an entry is malformed: not 2 or 3 items, an unreadable date, or
                a stop on an earlier day than its onset.
        """
        entries = ([(onset, code) for code, onset in history.items()]
                   if isinstance(history, Mapping) else list(history))
        cutoff = _utc(self.cfg.labels.cutoff_date)
        merge = self.cfg.data.code_merge
        rows, unknown, dropped = [], [], []
        for entry in entries:
            if isinstance(entry, (str, bytes)) or len(entry) not in (2, 3):
                raise ValueError(f"a history entry is (onset, code) or (onset, code, stop), "
                                 f"got {entry!r}")
            raw = str(entry[1]).strip()
            code = merge.get(raw, raw)
            if code not in self.vocabulary:
                if raw not in unknown:
                    unknown.append(raw)
                continue
            onset = _timestamp(entry[0])
            if len(entry) == 3:
                stop = None if entry[2] is None else _timestamp(entry[2])
            else:
                stop = None if code in self.chronic_default else onset
            # By day: Synthea records some STOP times a few hours before the same day's START.
            if stop is not None and stop.normalize() < onset.normalize():
                raise ValueError(f"code {code}: stop {stop.date()} before onset {onset.date()}")
            if code in self.redundant_codes:
                dropped.append((code, onset, "redundant: recorded with a more specific code, "
                                             "dropped from the features as in Phase 3"))
            elif onset >= cutoff:
                dropped.append((code, onset, f"on or after the prediction cutoff "
                                             f"{self.cfg.labels.cutoff_date}"))
            else:
                rows.append((NEW_PATIENT, onset, stop, self.data_end if stop is None else stop,
                             code))
        events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
        for column in ("date", "stop", "active_until"):
            events[column] = pd.to_datetime(events[column], utc=True)
        events = events.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
        return ParsedHistory(events, tuple(unknown), pd.DataFrame(dropped, columns=DROPPED_COLUMNS))

    def history_features(self, events: pd.DataFrame) -> np.ndarray:
        """Build the patient's feature row of every window, exactly as Phase 3.

        ``[X_cum_t | transition_svd_t]`` row-L2-normalized (``features.combine_features``):
        the window's active codes weighted by the saved IDF and L2-normalized
        (``features.tfidf_transform``), summed with time decay
        (``features.time_decay_cumulative``), next to the cumulative transitions before
        each window's end and the cutoff (``features.transition_blocks``) projected on the
        saved SVD components (``TruncatedSVD.transform`` is ``X @ components.T``).

        Args:
            events: ``ParsedHistory.events``.

        Returns:
            [T, F] float64, one row per window; all zero where nothing is known yet.
        """
        feat_cfg = self.cfg.features
        codes = self.feature_codes
        events = events[events["code"].isin(codes)].reset_index(drop=True)
        columns = pd.Index(codes).get_indexer(events["code"])
        windows = []
        for year in self.years:
            active = active_in_window(events, year)
            window = sp.csr_matrix(
                (np.ones(int(active.sum())), (np.zeros(int(active.sum()), dtype=np.int64),
                                              columns[active])),
                shape=(1, len(codes)),
            )
            window.data[:] = 1  # a code with several episodes counts once
            windows.append(window)
        tfidf = [tfidf_transform(window, self.idf, feat_cfg.tfidf_norm) for window in windows]
        cumulative = time_decay_cumulative(tfidf, self.years, feat_cfg.time_decay_lambda)
        if events.empty:
            transitions = pd.DataFrame(columns=TRANSITION_COLUMNS)
        else:
            trajectories = Trajectories(events.assign(chronic=events["stop"].isna()),
                                        (NEW_PATIENT,), codes,
                                        tuple(self.vocabulary[c] for c in codes), self.data_end)
            transitions = extract_transitions(trajectories)
        blocks = transition_blocks(transitions, (NEW_PATIENT,), codes, self.years,
                                   self.cfg.labels.cutoff_date)
        embeddings = [np.asarray(block @ self.svd_components.T) for block in blocks]
        rows = combine_features(cumulative, embeddings, feat_cfg.transition_weight)
        return np.vstack([row.toarray() for row in rows])

    def reference_patient(self, patient_id: str) -> NewPatient:
        """Return a reference patient's own history as a ``NewPatient`` (for checks and examples).

        Each event keeps its onset and code; its stop is its ``active_until`` (the STOP
        date, or the death date if earlier), or None if it is active to the end of the
        data. The features are then those of Phase 3 exactly; for a patient who died, a
        condition open at death counts as stopping then when chronic conditions are
        listed.

        Args:
            patient_id: A reference patient.

        Returns:
            Their history, birth date and sex.

        Raises:
            KeyError: If the patient is not a reference patient.
        """
        if patient_id not in self.node_index:
            raise KeyError(f"{patient_id} is not a reference patient")
        events = self.events[self.events["patient_id"] == patient_id]
        history = [(row.date, str(row.code),
                    None if row.active_until >= self.data_end else row.active_until)
                   for row in events.itertuples()]
        patient = self.patients.loc[patient_id]
        return NewPatient(history, patient["birth_date"].date(), str(patient["gender"]))

    # --- Prediction ----------------------------------------------------------------------

    def predict(self, patient: NewPatient, exclude: Collection[str] = ()) -> Prediction:
        """Place a new patient and describe their neighbourhood (see the module docstring).

        Args:
            patient: The new patient.
            exclude: Reference patient_ids that may not be neighbours, for leave-one-out
                checks with a reference patient's own history. They stay nodes of the
                reference graphs (the spectral embedding is not recomputed).

        Returns:
            The prediction.

        Raises:
            ValueError: If the history is malformed, the sex is not in
                ``features.SEX_CODES``, or the birth date is after the cutoff.
        """
        inf = self.cfg.inference
        parsed = self.parse_history(patient.history)
        notes = []
        if parsed.unknown_codes:
            notes.append(f"{len(parsed.unknown_codes)} code(s) not in the vocabulary were "
                         f"ignored: {', '.join(parsed.unknown_codes)}")
        for row in parsed.dropped.itertuples():
            notes.append(f"{row.code} ({self.names[row.code]}, {row.onset.date()}) not used: "
                         f"{row.reason}")
        features = self.history_features(parsed.events)
        allowed = np.ones(len(self.patient_ids), dtype=bool)
        index = self.node_index
        allowed[[index[p] for p in exclude if p in index]] = False

        insertions = []
        adjacency, inputs = [], []
        for t, year in enumerate(self.years):
            neighbours, weights = nearest_reference(self.reference[t], features[t],
                                                    self.cfg.graph.k, allowed)
            insertions.append(Insertion(year, neighbours, weights))
            adjacency.append(sparse_tensor(insert_node(self.adjacency[t], self.degrees[t],
                                                       neighbours, weights)))
            row = torch.from_numpy(features[t].astype(np.float32))[None]
            inputs.append(torch.cat([self.tensors[t], row]))
        with torch.no_grad():
            _, states = self.graph_ae.temporal.sequence(inputs, adjacency)
            _, h = self.supervised.temporal(inputs, adjacency)
        window_states = states[-1].numpy().astype(np.float64)
        placed = len(insertions[-1].neighbours) > 0

        conditions = self._conditions(parsed.events)
        chronic_codes = self._chronic_codes(parsed.events)
        prior = tuple(group for group, codes in self.cfg.labels.onset_groups.items()
                      if parsed.events["code"].isin(codes).any())
        at_risk = tuple(group for group in self.cfg.labels.onset_groups if group not in prior)
        age = self._age(patient.birth_date)
        n_ranked = max(inf.n_similar, inf.explain_neighbours, *inf.outcome_neighbours)
        logit = probability = None
        if placed:
            last = insertions[-1]
            coordinates = nystrom_extension(last.neighbours, last.weights, self.spectral_degrees,
                                            self.spectral_basis, self.spectral_values)
            spectral = coordinates[1:]
            phenotype = self._phenotype(coordinates)
            neighbours, similarities = cosine_ranking(self.spectral, spectral, allowed, n_ranked)
            demographics = self._demographics(patient)
            if demographics is None:
                notes.append("No birth date or sex: the any_onset probability needs both.")
            else:
                logit = self._any_onset_logit(h[-1:], demographics)
                probability = float(self.calibrator.apply(np.array([logit]))[0])
        else:
            spectral = np.full(self.spectral.shape[1], np.nan)
            phenotype = None
            neighbours, similarities = np.empty(0, dtype=np.int64), np.empty(0)
            notes.append(f"No known condition is active in {self.years[0]}-{self.years[-1]} "
                         "and no transition precedes the cutoff: the patient would be outside "
                         "the model cohort, so they get no neighbours, phenotype or probability.")
        return Prediction(
            unknown_codes=parsed.unknown_codes,
            dropped=parsed.dropped,
            conditions=conditions,
            chronic_codes=chronic_codes,
            prior_groups=prior,
            at_risk_groups=at_risk,
            age=age,
            excluded=tuple(int(node) for node in np.flatnonzero(~allowed)),
            years=self.years,
            features=features,
            insertions=tuple(insertions),
            placed=placed,
            window_states=window_states,
            spectral=spectral,
            phenotype=phenotype,
            neighbours=neighbours,
            similarities=similarities,
            similar=self._similar(neighbours[:inf.n_similar], similarities[:inf.n_similar],
                                  chronic_codes),
            outcomes=self._outcomes(neighbours, prior),
            any_onset_logit=logit,
            any_onset_probability=probability,
            notes=tuple(notes),
        )

    def explain(self, prediction: Prediction) -> pd.DataFrame:
        """List the patient's chronic conditions most over-represented in their neighbourhood.

        For each of the patient's chronic conditions: its share among the
        ``inference.explain_neighbours`` nearest reference patients (in the spectral
        embedding) over its share among all reference patients. A high lift means the
        condition drew the patient toward this neighbourhood.

        Args:
            prediction: Output of ``predict``.

        Returns:
            ``LIFT_COLUMNS``, the ``inference.explain_top`` highest lifts; empty if the
            patient has no chronic condition or was not placed.
        """
        inf = self.cfg.inference
        if not prediction.placed or not prediction.chronic_codes:
            return pd.DataFrame(columns=LIFT_COLUMNS)
        columns = pd.Index(self.feature_codes).get_indexer(list(prediction.chronic_codes))
        lift = condition_lift(self.chronic, prediction.neighbours[:inf.explain_neighbours], columns)
        lift["code"] = [self.feature_codes[c] for c in lift["column"]]
        lift["condition"] = [self.names[code] for code in lift["code"]]
        return (lift.sort_values(["lift", "neighbours_with", "code"], ascending=[False, False, True])
                .head(inf.explain_top).reset_index(drop=True)[LIFT_COLUMNS])

    def similar_patients(self, prediction: Prediction, age_band: float | None = None) -> pd.DataFrame:
        """The ``inference.n_similar`` most similar reference patients, optionally of similar age.

        Without ``age_band`` this is ``prediction.similar``. With it, the reference patients
        are post-filtered to those within +/- ``age_band`` years of the patient's age at the
        cutoff (``Prediction.age``; reference ages as in ``features.patient_demographics``),
        reading further down the same spectral ranking, so ``n_similar`` are still returned
        while that many are in the band. The graphs, the embeddings and the placement never
        change; only which ranked reference patients are listed.

        Args:
            prediction: Output of ``predict``.
            age_band: Years either side of the patient's age (``inference.age_band_years``
                in the app), or None for no filter.

        Returns:
            ``SIMILAR_COLUMNS``; empty if the patient was not placed.

        Raises:
            ValueError: If ``age_band`` is not positive, or given for a patient without a
                birth date.
        """
        nodes, similarities = self._neighbours(prediction, self.cfg.inference.n_similar, age_band)
        return self._similar(nodes, similarities, prediction.chronic_codes)

    def neighbour_outcomes(self, prediction: Prediction, age_band: float | None = None) -> pd.DataFrame:
        """``Prediction.outcomes`` among the neighbours of ``similar_patients(prediction, age_band)``.

        Args:
            prediction: Output of ``predict``.
            age_band: As in ``similar_patients``.

        Returns:
            The layout of ``Prediction.outcomes``; without ``age_band`` the same table.

        Raises:
            ValueError: As ``similar_patients``.
        """
        nodes, _ = self._neighbours(prediction, max(self.cfg.inference.outcome_neighbours), age_band)
        return self._outcomes(nodes, prediction.prior_groups)

    def similar_to_reference(self, patient_id: str, age_band: float | None = None) -> pd.DataFrame:
        """The ``inference.n_similar`` reference patients most similar to a reference patient.

        Cosine similarity in the 2022 spectral embedding, as for a new patient; the patient
        themselves is excluded, and ``age_band`` filters as in ``similar_patients``.

        Args:
            patient_id: A reference patient.
            age_band: Years either side of their age at the cutoff, or None.

        Returns:
            ``SIMILAR_COLUMNS``; shared conditions are those shared with this patient.

        Raises:
            KeyError: If the patient is not a reference patient.
            ValueError: If ``age_band`` is not positive.
        """
        if patient_id not in self.node_index:
            raise KeyError(f"{patient_id} is not a reference patient")
        node = self.node_index[patient_id]
        allowed = self._within_age(float(self.demographics["age"].iloc[node]), age_band)
        allowed[node] = False
        nodes, similarities = cosine_ranking(self.spectral, self.spectral[node], allowed,
                                             self.cfg.inference.n_similar)
        codes = [self.feature_codes[c] for c in self.chronic[node].indices]
        return self._similar(nodes, similarities, codes)

    # --- Helpers ---------------------------------------------------------------------------

    def _neighbours(self, prediction: Prediction, n: int,
                    age_band: float | None) -> tuple[np.ndarray, np.ndarray]:
        """The ``n`` reference nodes most similar to a placed patient, within the age band."""
        if not prediction.placed:
            return np.empty(0, dtype=np.int64), np.empty(0)
        allowed = self._within_age(prediction.age, age_band)
        allowed[np.asarray(prediction.excluded, dtype=np.int64)] = False
        return cosine_ranking(self.spectral, prediction.spectral, allowed, n)

    def _within_age(self, age: float | None, age_band: float | None) -> np.ndarray:
        """[N] bool: reference patients within +/- ``age_band`` years of ``age`` (all if None)."""
        if age_band is None:
            return np.ones(len(self.patient_ids), dtype=bool)
        if age_band <= 0:
            raise ValueError(f"age_band must be positive, got {age_band}")
        if age is None:
            raise ValueError("an age band needs the patient's birth date")
        return np.abs(self.demographics["age"].to_numpy(dtype=float) - age) <= age_band

    def _age(self, birth_date: date | str | None) -> float | None:
        """Age in years (``data.DAYS_PER_YEAR``) at the cutoff, or None without a birth date."""
        if birth_date is None:
            return None
        days = (_utc(self.cfg.labels.cutoff_date) - _timestamp(birth_date)).days
        if days < 0:
            raise ValueError(f"born after the cutoff {self.cfg.labels.cutoff_date}")
        return days / DAYS_PER_YEAR

    def _conditions(self, events: pd.DataFrame) -> pd.DataFrame:
        """code, condition, onset (first event) and chronic (any chronic episode) per code."""
        flagged = events.assign(chronic=chronic_mask(events, self.cfg.features.chronic_min_duration_days))
        table = flagged.groupby("code", as_index=False).agg(onset=("date", "min"),
                                                            chronic=("chronic", "any"))
        table["condition"] = [self.names[code] for code in table["code"]]
        return table.sort_values(["onset", "code"]).reset_index(drop=True)[
            ["code", "condition", "onset", "chronic"]]

    def _chronic_codes(self, events: pd.DataFrame) -> tuple[str, ...]:
        """Feature codes with a chronic episode active in a feature window (``chronic_presence``)."""
        chronic = events[events["code"].isin(self.feature_codes)
                         & chronic_mask(events, self.cfg.features.chronic_min_duration_days)]
        active = np.logical_or.reduce([active_in_window(chronic, year) for year in self.years])
        return tuple(sorted(set(chronic.loc[active, "code"]))) if len(chronic) else ()

    def _demographics(self, patient: NewPatient) -> np.ndarray | None:
        """[len(DEMOGRAPHICS)] standardized age at the cutoff and sex, or None if missing."""
        if patient.birth_date is None or patient.sex is None:
            return None
        if patient.sex not in SEX_CODES:
            raise ValueError(f"sex must be one of {sorted(SEX_CODES)}, got {patient.sex!r}")
        values = {"age": (self._age(patient.birth_date) - self.age_mean) / self.age_sd,
                  "sex": SEX_CODES[patient.sex]}
        return np.array([values[name] for name in DEMOGRAPHICS], dtype=np.float32)

    def _any_onset_logit(self, h: torch.Tensor, demographics: np.ndarray) -> float:
        """The supervised heads' ``any_onset`` logit for one trajectory embedding."""
        u = h
        if self.supervised.n_demographics:
            u = torch.cat([h, torch.from_numpy(demographics)[None]], dim=1)
        with torch.no_grad():
            logit, _ = self.supervised.heads(u)
        return float(logit[0])

    def _phenotype(self, coordinates: np.ndarray) -> Phenotype:
        """The phenotype whose centroid is nearest the normalized first k coordinates."""
        k = self.centroids.shape[1]
        row = normalize(coordinates[None, :k])[0]
        cluster = int(np.argmin(np.linalg.norm(self.centroids - row, axis=1))) + 1
        return self.phenotypes[cluster]

    def _similar(self, nodes: np.ndarray, similarities: np.ndarray,
                 chronic_codes: Sequence[str]) -> pd.DataFrame:
        """``SIMILAR_COLUMNS`` for the given reference nodes."""
        mine = set(chronic_codes)
        rows = []
        sexes = {value: key for key, value in SEX_CODES.items()}
        for rank, (node, similarity) in enumerate(zip(nodes, similarities), start=1):
            theirs = [self.feature_codes[c] for c in self.chronic[node].indices]
            shared = sorted(mine.intersection(theirs))
            demographics = self.demographics.iloc[node]
            rows.append([rank, self.patient_ids[node], float(similarity),
                         float(demographics["age"]), sexes[float(demographics["sex"])],
                         int(self.clusters[node]), len(theirs), len(shared),
                         "; ".join(self.names[code] for code in shared)])
        return pd.DataFrame(rows, columns=SIMILAR_COLUMNS)

    def _outcomes(self, neighbours: np.ndarray, prior: Sequence[str]) -> pd.DataFrame:
        """Onsets in the label horizon among the nearest neighbours, observed labels only."""
        table = pd.DataFrame({
            "target": list(self.targets),
            "group": [None, *self.cfg.labels.onset_groups],
            "patient_had_it": [None, *(group in prior for group in self.cfg.labels.onset_groups)],
        })
        for n in self.cfg.inference.outcome_neighbours:
            table[f"positives_{n}"], table[f"observed_{n}"] = outcome_counts(self.labels,
                                                                             neighbours[:n])
        table["positives_all"], table["observed_all"] = outcome_counts(
            self.labels, np.arange(len(self.patient_ids)))
        return table


def format_share(positives: int, observed: int) -> str:
    """Render a count share, e.g. "3/9 (33%)", or "0/0" when nothing is observed."""
    if observed == 0:
        return "0/0"
    return f"{positives}/{observed} ({positives / observed:.0%})"


def _phenotypes(chronic: sp.csr_matrix, clusters: np.ndarray, codes: Sequence[str],
                names: Mapping[str, str], cfg: Config) -> dict[int, Phenotype]:
    """Each cluster's size and top over-represented chronic conditions, as in Phase 4."""
    clustered = clusters > 0
    top = cluster_phenotypes(chronic[clustered], clusters[clustered], codes,
                             cfg.graph.phenotype_top_conditions, cfg.graph.phenotype_min_patients)
    phenotypes = {}
    for cluster in np.unique(clusters[clustered]):
        rows = top[top["cluster"] == cluster]
        description = ("; ".join(f"{names[r.code]} ({r.prevalence:.0%} vs {r.overall_prevalence:.0%} "
                                 "overall)" for r in rows.itertuples())
                       or "no chronic condition common in the cluster")
        phenotypes[int(cluster)] = Phenotype(int(cluster), int((clusters == cluster).sum()),
                                             description)
    return phenotypes


def _test_auroc(path: Path, cfg: Config) -> tuple[float, float, float]:
    """The final model's test AUROC and its percentile bootstrap interval.

    The resamples are those of Phase 7 (``evaluate.bootstrap_metrics``: the global seed,
    ``evaluation.bootstrap_resamples``); resamples with one class are skipped.
    """
    frame = pd.read_csv(path)
    y, p = frame[ANY_ONSET].to_numpy(dtype=float), frame["probability"].to_numpy(dtype=float)
    auroc = float(weighted_auroc(y, p, np.ones((1, len(y))))[0])
    boot = weighted_auroc(y, p, patient_resamples(len(y), cfg.evaluation.bootstrap_resamples,
                                                  cfg.seed))
    low, high = percentile_interval(boot[~np.isnan(boot)], cfg.evaluation.ci_level)
    return auroc, low, high


def _check_config_used(path: Path, current: Mapping[str, Any], sections: Sequence[str]) -> None:
    """Raise if the config saved next to a model differs from ``current`` in ``sections``."""
    used = yaml.safe_load(path.read_text(encoding="utf-8"))
    stale = [section for section in sections if used.get(section) != current.get(section)]
    if stale:
        raise ValueError(f"the weights next to {path} were trained with other {stale} "
                         "sections than the current config")


def _timestamp(value: Any) -> pd.Timestamp:
    """A UTC timestamp from a year (1 January), a date or an ISO date string."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return pd.Timestamp(year=int(value), month=1, day=1, tz="UTC")
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"cannot read {value!r} as a year or date") from error
    if pd.isna(stamp):
        raise ValueError(f"cannot read {value!r} as a year or date")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _utc(day: date) -> pd.Timestamp:
    """Midnight UTC of ``day``."""
    return pd.Timestamp(day.isoformat(), tz="UTC")
