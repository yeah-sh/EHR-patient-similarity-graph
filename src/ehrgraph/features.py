"""Per-window patient features, the model cohort and the code comorbidity matrices.

Pipeline (run end to end by ``scripts/03_build_features.py``)::

    windows, trajectories = load_windows(cfg), load_trajectories(cfg)  # Phase 2
    co_recorded = co_recorded_codes(cohort_events, min_share, redundant_codes)
    transitions = feature_transitions(trajectories, cfg)  # without redundant codes
    features = build_features(windows, transitions, cfg)
    comorbidity = build_comorbidity(windows, cfg)  # all codes, patient-windows
    chronic = build_chronic_comorbidity(trajectories, cfg)  # chronic codes, patients
    save_features(features, comorbidity, chronic, cfg)  # + model_cohort.json
    plot_comorbidity_heatmap(chronic, descriptions, path, cfg, min_frequency)
    write_features_summary(features, comorbidity, chronic, co_recorded, descriptions, cfg)

The feature vocabulary is the Phase 2 vocabulary without ``features.redundant_codes``:
general codes that Synthea nearly always records together with a more specific
code (see ``co_recorded_codes``). Each window's feature matrix is
``[X_cum_t | transition_svd_t]``, row L2-normalized:

* ``X_t`` is the window's binary patient x code matrix weighted by TF-IDF. The IDF
  is fit on the training-period windows only (those that close by the prediction
  cutoff, ``config.feature_window_years``) and applied to every window.
* ``X_cum_t = sum over s <= t of exp(-lambda * (t - s)) * X_s``, with t - s in years.
* ``transition_svd_t`` is each patient's transition matrix, row-normalized per
  from-code and flattened, built from the transitions that end by the end of
  window t and before the cutoff, so it never sees later events. TruncatedSVD is
  fit once on the last training-period window's block and applied to every window.

Each block is row-L2-normalized before concatenation, so neither dominates by
scale. Windows after the cutoff get features too, but they hold post-cutoff
events: models must read features only through ``load_model_inputs``, which
returns the training-period windows for the model cohort (patients with a
non-zero feature row in at least one of them). Every patient-indexed matrix is
sparse except the (n_patients x n_components) SVD embeddings.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from ehrgraph.config import Config, feature_window_years
from ehrgraph.data import chart_style, markdown_table, save_figure
from ehrgraph.trajectories import (
    Trajectories,
    TrajectoryWindows,
    active_in_window,
    extract_transitions,
    pre_cutoff_transitions,
)

logger = logging.getLogger(__name__)

FEATURES_JSON = "features.json"
TOP_PAIR_COLUMNS: list[str] = [
    "code_a", "description_a", "code_b", "description_b",
    "frequency_a", "frequency_b", "count", "expected", "lift", "jaccard",
]
CO_RECORDED_COLUMNS: list[str] = ["code", "patients", "encounters", "coverage", "specific_codes"]

# Diverging pair from the dataviz reference palette: blue (less often together than
# expected) and red (more often) arms with equal steps, around a neutral gray at lift 1.
_DIVERGING = ["#184f95", "#6da7ec", "#f0efec", "#ee8a88", "#a8302f"]
_POINTS_PER_INCH = 72
_HEATMAP_SIZE_IN = (11.0, 11.0)
_CELL_GAP_PX = 2  # surface-coloured gap between adjacent cells
_COLORBAR_SHRINK = 0.5  # colorbar length as a share of the grid's height
_SEMANTIC_TAG = re.compile(r"\s*\([^()]*\)$")  # e.g. " (disorder)", dropped from labels


@dataclass(frozen=True)
class TransitionSVD:
    """Patients' cumulative transition vectors per window, reduced with TruncatedSVD.

    Attributes:
        embeddings: One (n_patients, n_components) array per window; zero for
            patients with no transition yet.
        components: (n_components, n_codes ** 2) right singular vectors; column
            ``i * n_codes + j`` is the transition codes[i] -> codes[j].
        explained_variance_ratio: Share of the fit block's variance per component.
        fit_year: The window whose block the SVD was fit on.
        n_transitions: Transitions in each window's block.
        n_patients_with_transitions: Patients with at least one, per window.
    """

    embeddings: list[np.ndarray]
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    fit_year: int
    n_transitions: tuple[int, ...]
    n_patients_with_transitions: tuple[int, ...]


@dataclass(frozen=True)
class Features:
    """Per-window feature matrices and the pieces they are built from.

    Attributes:
        years: Every window's year, ascending.
        feature_years: Windows that close by the prediction cutoff; the only ones
            that may be model inputs. The IDF is fit on them.
        patient_ids: Row labels (the Phase 2 cohort).
        codes: The feature vocabulary, labelling the first ``len(codes)`` columns.
        redundant_codes: Codes of the Phase 2 vocabulary dropped as redundant.
        idf: IDF weight per code.
        idf_rows: Non-empty patient-windows the IDF was fit on.
        tfidf: ``X_t`` per window, CSR (n_patients, n_codes).
        cumulative: ``X_cum_t`` per window, CSR (n_patients, n_codes).
        transitions: The transition blocks.
        matrices: Final features per window, CSR (n_patients, n_codes + n_components),
            each row of unit L2 norm or all zero.
        model_rows: Rows of the model cohort: patients with a non-zero row in at
            least one training-period window.
    """

    years: tuple[int, ...]
    feature_years: tuple[int, ...]
    patient_ids: tuple[str, ...]
    codes: tuple[str, ...]
    redundant_codes: tuple[str, ...]
    idf: np.ndarray
    idf_rows: int
    tfidf: list[sp.csr_matrix]
    cumulative: list[sp.csr_matrix]
    transitions: TransitionSVD
    matrices: list[sp.csr_matrix]
    model_rows: np.ndarray


@dataclass(frozen=True)
class Comorbidity:
    """Code co-occurrence over the training-period windows.

    Attributes:
        counts: Symmetric code x code counts: [a, b] counts units (patient-windows
            or patients) with a and b active together, the diagonal [a, a] units
            with a active.
        lift: ``counts[a, b] / expected[a, b]`` with ``expected[a, b] =
            counts[a, a] * counts[b, b] / population``; 1 means independent.
        population: Units counted (patient-windows or patients with an active
            condition in the training period).
        unit: "patient-windows" or "patients".
        name: What was counted, e.g. "all conditions" or "chronic conditions".
        years: The windows counted.
        codes: Row and column labels.
    """

    counts: sp.csr_matrix
    lift: sp.csr_matrix
    population: int
    unit: str
    name: str
    years: tuple[int, ...]
    codes: tuple[str, ...]


@dataclass(frozen=True)
class ModelInputs:
    """Training-period features of the model cohort; see ``load_model_inputs``.

    Attributes:
        years: The training-period windows, ascending; all close by the cutoff.
        patient_ids: The model cohort, in row order.
        rows: Their rows in patient_index.json and the saved feature matrices.
        matrices: One CSR (len(patient_ids), n_features) matrix per year.
    """

    years: tuple[int, ...]
    patient_ids: tuple[str, ...]
    rows: np.ndarray
    matrices: list[sp.csr_matrix]


def co_recorded_codes(
    events: pd.DataFrame, min_share: float, redundant: Collection[str] = ()
) -> pd.DataFrame:
    """Measure how often each code is recorded with a more specific code.

    Code s is a more specific form of code g if at least ``min_share`` of s's
    encounters also record g and s is recorded at no more encounters than g.
    Codes in ``redundant`` never count as specific forms, so of a 1:1 pair only
    the listed code is redundant. A code's coverage is the share of its encounters
    that also record one of its specific forms; at ``min_share`` or more it
    qualifies as a redundant general code. Clinical judgement still decides: a
    condition recorded with its complications also qualifies (see config.yaml).

    Args:
        events: Clinical events of the cohort (``data.EVENT_COLUMNS``).
        min_share: ``features.redundant_min_share``.
        redundant: ``features.redundant_codes``; a warning is logged for each one
            whose coverage is below ``min_share``.

    Returns:
        ``CO_RECORDED_COLUMNS``, one row per code, by coverage then encounters;
        specific_codes lists the specific forms, most recorded first.
    """
    redundant = set(redundant)
    encounters = events[["encounter_id", "code"]].drop_duplicates()
    n_encounters = encounters.groupby("code").size()
    pairs = encounters.merge(encounters, on="encounter_id", suffixes=("_g", "_s"))
    pairs = pairs[pairs["code_g"] != pairs["code_s"]]
    both = pairs.groupby(["code_g", "code_s"]).size().rename("both").reset_index()
    both["n_s"] = both["code_s"].map(n_encounters)
    specific = both[
        (both["both"] >= min_share * both["n_s"])
        & (both["n_s"] <= both["code_g"].map(n_encounters))
        & ~both["code_s"].isin(redundant)
    ].sort_values(["code_g", "n_s", "code_s"], ascending=[True, False, True])
    covered = (
        pairs.merge(specific[["code_g", "code_s"]], on=["code_g", "code_s"])
        .drop_duplicates(["code_g", "encounter_id"])
        .groupby("code_g")
        .size()
    )
    forms = specific.groupby("code_g")["code_s"].agg(tuple)
    stats = pd.DataFrame(
        {
            "patients": events.groupby("code")["patient_id"].nunique(),
            "encounters": n_encounters,
        }
    ).rename_axis("code")
    stats["coverage"] = covered.reindex(stats.index, fill_value=0) / stats["encounters"]
    stats["specific_codes"] = [forms.get(code, ()) for code in stats.index]
    stats = (
        stats.reset_index()
        .sort_values(["coverage", "encounters", "code"], ascending=[False, False, True])
        .reset_index(drop=True)[CO_RECORDED_COLUMNS]
    )
    coverage = dict(zip(stats["code"], stats["coverage"]))
    for code in sorted(redundant):
        if coverage.get(code, 0.0) < min_share:
            logger.warning(
                "Redundant code %s is recorded with a more specific code at only %.1f%% "
                "of its encounters (rule: %.0f%%)",
                code,
                100 * coverage.get(code, 0.0),
                100 * min_share,
            )
    return stats


def feature_vocabulary(codes: Sequence[str], cfg: Config) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split ``codes`` into the feature vocabulary and the redundant codes it drops.

    Args:
        codes: The Phase 2 vocabulary.
        cfg: Project configuration (``features.redundant_codes``).

    Returns:
        The kept codes and the dropped ones, both in the order of ``codes``. A
        configured code missing from ``codes`` is logged and ignored.
    """
    redundant = set(cfg.features.redundant_codes)
    missing = sorted(redundant - set(codes))
    if missing:
        logger.warning("Redundant codes not in the vocabulary, ignored: %s", missing)
    return (
        tuple(c for c in codes if c not in redundant),
        tuple(c for c in codes if c in redundant),
    )


def feature_transitions(trajectories: Trajectories, cfg: Config) -> pd.DataFrame:
    """Extract transitions as if the redundant codes had never been recorded.

    Rebuilding them (rather than dropping rows) links the codes before and after a
    redundant code that had an onset step of its own.

    Args:
        trajectories: ``trajectories.load_trajectories`` output (full range).
        cfg: Project configuration.

    Returns:
        ``trajectories.TRANSITION_COLUMNS`` over the full range; the feature
        builders keep only those ending before the cutoff.
    """
    codes, dropped = feature_vocabulary(trajectories.codes, cfg)
    wording = dict(zip(trajectories.codes, trajectories.descriptions))
    events = trajectories.events
    reduced = replace(
        trajectories,
        events=events[~events["code"].isin(dropped)].reset_index(drop=True),
        codes=codes,
        descriptions=tuple(wording[code] for code in codes),
    )
    return extract_transitions(reduced)


def training_windows(windows: TrajectoryWindows, cfg: Config) -> list[sp.csr_matrix]:
    """Return the windows that close by the prediction cutoff, in year order.

    Args:
        windows: Output of ``trajectories.load_windows``.
        cfg: Project configuration.

    Returns:
        One matrix per year of ``config.feature_window_years``.

    Raises:
        ValueError: If no window closes by the cutoff, or one is missing from ``windows``.
    """
    years = feature_window_years(cfg)
    missing = sorted(set(years) - set(windows.years))
    if not years or missing:
        raise ValueError(
            f"need the windows that close by {cfg.labels.cutoff_date}; feature years "
            f"{list(years)}, missing from the windows {missing}"
        )
    return [windows[windows.years.index(year)] for year in years]


def fit_idf(matrices: Sequence[sp.spmatrix]) -> tuple[np.ndarray, int]:
    """Fit IDF weights on the non-empty rows of ``matrices``.

    Each (patient, window) row with an active code is a document. With n documents,
    df_j of which hold code j, ``idf_j = ln((1 + n) / (1 + df_j)) + 1``
    (scikit-learn's smoothed IDF), which stays finite for codes never active here.

    Args:
        matrices: Binary patient x code matrices, normally ``training_windows``.

    Returns:
        The IDF per code and the number of documents n.
    """
    stacked = _nonempty_rows(sp.vstack(matrices, format="csr"))
    transformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True).fit(stacked)
    return transformer.idf_, stacked.shape[0]


def tfidf_transform(matrix: sp.spmatrix, idf: np.ndarray, norm: str) -> sp.csr_matrix:
    """Weight each code by its IDF, then normalize each row.

    Args:
        matrix: A binary patient x code window.
        idf: Output of ``fit_idf``.
        norm: Row norm for ``sklearn.preprocessing.normalize`` ("l1", "l2" or "max");
            empty rows stay empty.

    Returns:
        The TF-IDF matrix, CSR, same shape as ``matrix``.
    """
    weighted = sp.csr_matrix(matrix, dtype=np.float64) @ sp.diags(idf)
    return normalize(sp.csr_matrix(weighted), norm=norm)


def time_decay_cumulative(
    matrices: Sequence[sp.spmatrix], years: Sequence[int], decay_lambda: float
) -> list[sp.csr_matrix]:
    """Sum each window with the earlier ones, down-weighted by their age.

    ``X_cum_t = sum over s <= t of exp(-decay_lambda * (years[t] - years[s])) * X_s``,
    computed as ``X_cum_t = exp(-decay_lambda * gap) * X_cum_{t-1} + X_t``. With
    ``decay_lambda = 0`` it is the plain cumulative sum.

    Args:
        matrices: One matrix per window, all the same shape.
        years: Each window's year, strictly increasing.
        decay_lambda: Decay rate per year, >= 0.

    Returns:
        One CSR matrix per window.

    Raises:
        ValueError: If ``years`` does not match ``matrices`` or is not increasing.
    """
    if len(years) != len(matrices) or any(b <= a for a, b in zip(years, years[1:])):
        raise ValueError(f"need one strictly increasing year per matrix, got {list(years)}")
    cumulative: list[sp.csr_matrix] = []
    for t, matrix in enumerate(matrices):
        current = sp.csr_matrix(matrix, dtype=np.float64)
        if cumulative:
            decay = math.exp(-decay_lambda * (years[t] - years[t - 1]))
            current = sp.csr_matrix(decay * cumulative[-1] + current)
        cumulative.append(current)
    return cumulative


def patient_transition_matrix(
    transitions: pd.DataFrame, patient_ids: Sequence[str], codes: Sequence[str]
) -> sp.csr_matrix:
    """Flatten each patient's row-normalized transition matrix into one row.

    For patient p, ``T_p[a, b]`` counts p's transitions a -> b and is divided by
    p's transitions out of a, so each from-code's row sums to 1 (an empirical
    Markov transition matrix). Row p of the result is ``T_p`` flattened.

    Args:
        transitions: ``trajectories.TRANSITION_COLUMNS``, already filtered to the
            transitions the window may use (see ``transition_blocks``).
        patient_ids: Row labels.
        codes: The feature vocabulary.

    Returns:
        CSR (len(patient_ids), len(codes) ** 2); column ``i * len(codes) + j`` is
        codes[i] -> codes[j].

    Raises:
        ValueError: If a transition uses a patient or code outside the labels.
    """
    n_codes = len(codes)
    rows = pd.Index(patient_ids).get_indexer(transitions["patient_id"])
    code_index = pd.Index(codes)
    src = code_index.get_indexer(transitions["from_code"])
    dst = code_index.get_indexer(transitions["to_code"])
    if (rows < 0).any():
        raise ValueError("transitions use patients that are not in the patient index")
    if (src < 0).any() or (dst < 0).any():
        raise ValueError("transitions use codes that are not in the vocabulary")
    pairs = (
        pd.DataFrame({"row": rows, "src": src, "dst": dst})
        .value_counts()
        .rename("count")
        .reset_index()
    )
    out_degree = pairs.groupby(["row", "src"])["count"].transform("sum")
    columns = pairs["src"].to_numpy(np.int64) * n_codes + pairs["dst"].to_numpy(np.int64)
    return sp.csr_matrix(
        ((pairs["count"] / out_degree).to_numpy(np.float64), (pairs["row"].to_numpy(), columns)),
        shape=(len(patient_ids), n_codes * n_codes),
    )


def transition_blocks(
    transitions: pd.DataFrame,
    patient_ids: Sequence[str],
    codes: Sequence[str],
    years: Sequence[int],
    cutoff: date,
) -> list[sp.csr_matrix]:
    """Build each window's cumulative transition block.

    Window t uses the transitions that end by the end of ``years[t]`` and before
    ``cutoff``, whichever is earlier, so a window never sees a later transition and
    no window sees one after the cutoff. Windows after the cutoff repeat the block
    of the last window before it.

    Args:
        transitions: ``trajectories.TRANSITION_COLUMNS`` (full range).
        patient_ids: Row labels.
        codes: The feature vocabulary.
        years: Each window's year.
        cutoff: ``labels.cutoff_date``.

    Returns:
        One ``patient_transition_matrix`` per window.
    """
    return [
        patient_transition_matrix(
            pre_cutoff_transitions(transitions, min(date(year + 1, 1, 1), cutoff)),
            patient_ids,
            codes,
        )
        for year in years
    ]


def transition_features(
    transitions: pd.DataFrame,
    patient_ids: Sequence[str],
    codes: Sequence[str],
    years: Sequence[int],
    cfg: Config,
) -> TransitionSVD:
    """Build and reduce the cumulative transition block of every window.

    TruncatedSVD is fit once, on the block of the last training-period window (all
    transitions before the cutoff), and every window's block is projected on it.

    Args:
        transitions: Output of ``feature_transitions`` (full range).
        patient_ids: Row labels.
        codes: The feature vocabulary.
        years: Each window's year; must include the last training-period window.
        cfg: Project configuration (``features`` and ``labels`` sections, ``seed``).

    Returns:
        The reduced transition blocks.

    Raises:
        ValueError: If ``years`` lacks the last training-period window.
    """
    feature_years = feature_window_years(cfg)
    if not feature_years or feature_years[-1] not in years:
        raise ValueError(f"windows {list(years)} lack the last training-period window")
    fit_year = feature_years[-1]
    blocks = transition_blocks(transitions, patient_ids, codes, years, cfg.labels.cutoff_date)
    svd = TruncatedSVD(
        n_components=cfg.features.transition_svd_components,
        n_iter=cfg.features.transition_svd_n_iter,
        random_state=cfg.seed,
    ).fit(blocks[list(years).index(fit_year)])
    n_transitions = tuple(
        len(pre_cutoff_transitions(transitions, min(date(y + 1, 1, 1), cfg.labels.cutoff_date)))
        for y in years
    )
    logger.info(
        "Transition blocks: %s transitions per window; %d SVD components fit on %d explain "
        "%.1f%% of its variance",
        list(n_transitions),
        svd.n_components,
        fit_year,
        100 * svd.explained_variance_ratio_.sum(),
    )
    return TransitionSVD(
        embeddings=[svd.transform(block) for block in blocks],
        components=svd.components_,
        explained_variance_ratio=svd.explained_variance_ratio_,
        fit_year=fit_year,
        n_transitions=n_transitions,
        n_patients_with_transitions=tuple(int(_nonzero_rows(b).sum()) for b in blocks),
    )


def combine_features(
    cumulative: Sequence[sp.spmatrix],
    transition_embeddings: Sequence[np.ndarray],
    transition_weight: float,
) -> list[sp.csr_matrix]:
    """Concatenate ``[X_cum_t | transition block_t]`` per window, rows L2-normalized.

    Each block is row-L2-normalized first and the transition block scaled by
    ``transition_weight``, so it holds ``w^2 / (1 + w^2)`` of each row's squared
    norm when both blocks are non-zero. All-zero rows stay zero.

    Args:
        cumulative: Output of ``time_decay_cumulative``.
        transition_embeddings: ``TransitionSVD.embeddings``, one per window.
        transition_weight: Weight of the transition block.

    Returns:
        One CSR matrix per window, (n_patients, n_codes + n_components).
    """
    return [
        normalize(
            sp.hstack(
                [normalize(matrix), sp.csr_matrix(transition_weight * normalize(embedding))],
                format="csr",
            )
        )
        for matrix, embedding in zip(cumulative, transition_embeddings, strict=True)
    ]


def model_cohort_rows(
    matrices: Sequence[sp.spmatrix], years: Sequence[int], feature_years: Sequence[int]
) -> np.ndarray:
    """Return the rows that are non-zero in at least one training-period window.

    Args:
        matrices: Final feature matrices, one per year of ``years``.
        years: Each matrix's year.
        feature_years: The training-period windows.

    Returns:
        Row indices, ascending.
    """
    nonzero = np.logical_or.reduce(
        [_nonzero_rows(matrices[list(years).index(year)]) for year in feature_years]
    )
    return np.flatnonzero(nonzero)


def build_features(windows: TrajectoryWindows, transitions: pd.DataFrame, cfg: Config) -> Features:
    """Build every window's feature matrix and the model cohort (see the module docstring).

    Args:
        windows: Output of ``trajectories.load_windows`` (Phase 2 vocabulary).
        transitions: Output of ``feature_transitions`` (full range, no redundant codes).
        cfg: Project configuration.

    Returns:
        The features of every window; only ``feature_years`` may be model inputs.
    """
    feat_cfg = cfg.features
    codes, dropped = feature_vocabulary(windows.codes, cfg)
    keep = pd.Index(windows.codes).get_indexer(codes)
    reduced = TrajectoryWindows(
        windows.years, [m[:, keep] for m in windows.matrices], windows.patient_ids, codes
    )
    idf, idf_rows = fit_idf(training_windows(reduced, cfg))
    tfidf = [tfidf_transform(matrix, idf, feat_cfg.tfidf_norm) for matrix in reduced.matrices]
    cumulative = time_decay_cumulative(tfidf, windows.years, feat_cfg.time_decay_lambda)
    block = transition_features(transitions, windows.patient_ids, codes, windows.years, cfg)
    matrices = combine_features(cumulative, block.embeddings, feat_cfg.transition_weight)
    feature_years = feature_window_years(cfg)
    model_rows = model_cohort_rows(matrices, windows.years, feature_years)
    for year, matrix in zip(windows.years, matrices):
        logger.info(
            "Features %d: shape %s, %d non-zeros, %d all-zero rows",
            year,
            matrix.shape,
            matrix.nnz,
            int((~_nonzero_rows(matrix)).sum()),
        )
    logger.info(
        "Feature vocabulary: %d codes (%d redundant dropped); model cohort %d of %d patients",
        len(codes),
        len(dropped),
        len(model_rows),
        len(windows.patient_ids),
    )
    return Features(
        years=windows.years,
        feature_years=feature_years,
        patient_ids=windows.patient_ids,
        codes=codes,
        redundant_codes=dropped,
        idf=idf,
        idf_rows=idf_rows,
        tfidf=tfidf,
        cumulative=cumulative,
        transitions=block,
        matrices=matrices,
        model_rows=model_rows,
    )


def comorbidity_counts(matrices: Sequence[sp.spmatrix]) -> tuple[sp.csr_matrix, int]:
    """Count code co-occurrence, ``C = X^T X``, over the non-empty rows of ``matrices``.

    ``X`` is the matrices stacked and binarized. ``C`` is symmetric; ``C[a, b]``
    counts rows where a and b are both active and the diagonal holds each code's
    frequency (rows where it is active).

    Args:
        matrices: Patient x code windows.

    Returns:
        ``C`` as CSR (n_codes, n_codes) and the number of non-empty rows.
    """
    binary = (_nonempty_rows(sp.vstack(matrices, format="csr")) != 0).astype(np.float64)
    return sp.csr_matrix(binary.T @ binary), binary.shape[0]


def comorbidity_lift(counts: sp.spmatrix, population: int) -> sp.csr_matrix:
    """Divide each co-occurrence count by its count expected under independence.

    ``lift[a, b] = C[a, b] / (C[a, a] * C[b, b] / population)``: above 1 the codes
    are active together more often than chance, below 1 less often. The diagonal is
    ``population / C[a, a]`` and has no comorbidity meaning. ``exp(PMI) = lift``.

    Args:
        counts: Symmetric counts with frequencies on the diagonal.
        population: Units counted.

    Returns:
        CSR with the sparsity pattern of ``counts``.
    """
    frequency = counts.diagonal().astype(np.float64)
    inverse = np.divide(1.0, frequency, out=np.zeros_like(frequency), where=frequency > 0)
    scale = sp.diags(inverse)
    return sp.csr_matrix(population * (scale @ counts @ scale))


def build_comorbidity(windows: TrajectoryWindows, cfg: Config) -> Comorbidity:
    """Count co-occurrence of all codes in patient-windows of the training period.

    Uses the Phase 2 vocabulary, redundant codes included, so pairs of codes that
    are recorded together appear here.

    Args:
        windows: Output of ``trajectories.load_windows``.
        cfg: Project configuration.

    Returns:
        The comorbidity matrices, counted in patient-windows.
    """
    counts, n_rows = comorbidity_counts(training_windows(windows, cfg))
    years = feature_window_years(cfg)
    logger.info(
        "Comorbidity (all codes) over %d-%d: %d patient-windows, %d co-occurring pairs",
        years[0],
        years[-1],
        n_rows,
        sp.triu(counts, k=1).nnz,
    )
    return Comorbidity(
        counts, comorbidity_lift(counts, n_rows), n_rows, "patient-windows", "all conditions",
        years, windows.codes,
    )


def chronic_mask(events: pd.DataFrame, min_duration_days: int) -> pd.Series:
    """Flag chronic episodes: no STOP date, or a STOP more than ``min_duration_days`` later.

    Broader than ``Trajectories.events.chronic``, which means no STOP date only.

    Args:
        events: Events with date and stop.
        min_duration_days: ``features.chronic_min_duration_days``.

    Returns:
        Boolean mask aligned with ``events``.
    """
    duration = events["stop"] - events["date"]
    return events["stop"].isna() | (duration > pd.Timedelta(days=min_duration_days))


def build_chronic_comorbidity(trajectories: Trajectories, cfg: Config) -> Comorbidity:
    """Count patients with pairs of chronic conditions active together.

    Only chronic episodes (``chronic_mask``) of the feature vocabulary count. A
    patient has code a if a chronic episode of a is active in a training-period
    window, and pair (a, b) if chronic episodes of both are active in the same
    window. Each patient counts once per code and once per pair, however many
    windows. The population is the patients with any condition active in the
    training period.

    Args:
        trajectories: ``trajectories.load_trajectories`` output.
        cfg: Project configuration.

    Returns:
        The comorbidity matrices over the feature vocabulary, counted in patients.

    Raises:
        ValueError: If no window closes by the cutoff.
    """
    years = feature_window_years(cfg)
    if not years:
        raise ValueError(f"no window closes by the cutoff {cfg.labels.cutoff_date}")
    codes, _ = feature_vocabulary(trajectories.codes, cfg)
    events = trajectories.events
    in_period = np.logical_or.reduce([active_in_window(events, year) for year in years])
    population = int(events.loc[in_period, "patient_id"].nunique())

    chronic = events[
        events["code"].isin(codes) & chronic_mask(events, cfg.features.chronic_min_duration_days)
    ]
    active = (
        pd.concat(
            [
                chronic.loc[active_in_window(chronic, year), ["patient_id", "code"]].assign(year=year)
                for year in years
            ]
        )
        .drop_duplicates()
        .assign(col=lambda df: pd.Index(codes).get_indexer(df["code"]))
    )
    per_code = active.drop_duplicates(["patient_id", "col"]).groupby("col").size()
    pairs = active.merge(active, on=["patient_id", "year"], suffixes=("_a", "_b"))
    per_pair = (
        pairs.loc[pairs["col_a"] < pairs["col_b"], ["patient_id", "col_a", "col_b"]]
        .drop_duplicates()
        .groupby(["col_a", "col_b"])
        .size()
    )
    a = per_pair.index.get_level_values("col_a").to_numpy()
    b = per_pair.index.get_level_values("col_b").to_numpy()
    counts = sp.csr_matrix(
        (
            np.concatenate([per_code.to_numpy(), per_pair.to_numpy(), per_pair.to_numpy()]).astype(
                np.float64
            ),
            (
                np.concatenate([per_code.index.to_numpy(), a, b]),
                np.concatenate([per_code.index.to_numpy(), b, a]),
            ),
        ),
        shape=(len(codes), len(codes)),
    )
    logger.info(
        "Chronic comorbidity over %d-%d: %d chronic codes, %d patients with a pair, "
        "population %d patients",
        years[0],
        years[-1],
        len(per_code),
        pairs.loc[pairs["col_a"] < pairs["col_b"], "patient_id"].nunique(),
        population,
    )
    return Comorbidity(
        counts, comorbidity_lift(counts, population), population, "patients",
        "chronic conditions", years, codes,
    )


def top_comorbidity_pairs(
    comorbidity: Comorbidity,
    descriptions: Mapping[str, str],
    n: int,
    min_count: int,
    min_frequency: int = 0,
) -> pd.DataFrame:
    """Return the ``n`` code pairs with the highest lift.

    Only pairs active together in at least ``min_count`` units, of codes each
    active in at least ``min_frequency`` units, qualify, so rare codes seen
    together once or twice do not crowd the list. Ties are broken by count, then
    by code. Jaccard is ``count / (frequency_a + frequency_b - count)``.

    Args:
        comorbidity: Output of ``build_comorbidity`` or ``build_chronic_comorbidity``.
        descriptions: Code -> description.
        n: Number of pairs.
        min_count: Minimum co-occurrence count.
        min_frequency: Minimum frequency of each code.

    Returns:
        ``TOP_PAIR_COLUMNS``; code_a before code_b in vocabulary order.
    """
    upper = sp.triu(comorbidity.counts, k=1).tocoo()
    frequency = comorbidity.counts.diagonal()
    freq_a, freq_b = frequency[upper.row], frequency[upper.col]
    expected = freq_a * freq_b / comorbidity.population
    codes = np.asarray(comorbidity.codes, dtype=object)
    pairs = pd.DataFrame(
        {
            "code_a": codes[upper.row],
            "description_a": [descriptions[c] for c in codes[upper.row]],
            "code_b": codes[upper.col],
            "description_b": [descriptions[c] for c in codes[upper.col]],
            "frequency_a": freq_a.astype(np.int64),
            "frequency_b": freq_b.astype(np.int64),
            "count": upper.data.astype(np.int64),
            "expected": expected,
            "lift": upper.data / expected,
            "jaccard": upper.data / (freq_a + freq_b - upper.data),
        },
        columns=TOP_PAIR_COLUMNS,
    )
    qualifies = (
        (pairs["count"] >= min_count)
        & (pairs["frequency_a"] >= min_frequency)
        & (pairs["frequency_b"] >= min_frequency)
    )
    return (
        pairs[qualifies]
        .sort_values(["lift", "count", "code_a", "code_b"], ascending=[False, False, True, True])
        .head(n)
        .reset_index(drop=True)
    )


def most_common_codes(comorbidity: Comorbidity, n: int, min_frequency: int = 0) -> np.ndarray:
    """Return the indices of the ``n`` most frequent codes with at least ``min_frequency``.

    Ties are broken by code.
    """
    frequency = comorbidity.counts.diagonal()
    order = np.lexsort((np.asarray(comorbidity.codes), -frequency))
    return order[frequency[order] >= max(min_frequency, 1)][:n]


def plot_comorbidity_heatmap(
    comorbidity: Comorbidity,
    descriptions: Mapping[str, str],
    path: Path,
    cfg: Config,
    min_frequency: int = 0,
) -> None:
    """Save a heatmap of the lift between the most common codes to ``path``.

    Shows the ``cfg.summary.heatmap_codes`` most frequent codes with at least
    ``min_frequency``. Colour is log2 lift on a diverging scale centred on lift 1
    and clipped to ``cfg.summary.heatmap_log2_lift_limit``; pairs never active
    together sit at the blue end. The diagonal is left blank. Rows and columns are
    ordered by average-linkage clustering of the clipped log2 lift.

    Args:
        comorbidity: Output of ``build_comorbidity`` or ``build_chronic_comorbidity``.
        descriptions: Code -> description.
        path: Output PNG.
        cfg: Project configuration (``summary`` section).
        min_frequency: Minimum frequency of a code to be shown.
    """
    opts = cfg.summary
    limit = opts.heatmap_log2_lift_limit
    top = most_common_codes(comorbidity, opts.heatmap_codes, min_frequency)
    with np.errstate(divide="ignore"):
        values = np.clip(np.log2(comorbidity.lift[top][:, top].toarray()), -limit, limit)
    np.fill_diagonal(values, 0.0)
    order = leaves_list(linkage(values, method="average", optimal_ordering=True))
    values = values[np.ix_(order, order)]
    labels = [_SEMANTIC_TAG.sub("", descriptions[comorbidity.codes[i]]) for i in top[order]]
    n = len(labels)

    dpi = opts.figure_dpi
    # The blank diagonal is transparent, so it shows the surface.
    cmap = LinearSegmentedColormap.from_list("lift", _DIVERGING).with_extremes(bad=(0, 0, 0, 0))
    ticks = np.arange(-math.floor(limit), math.floor(limit) + 1)
    with matplotlib.rc_context(chart_style(dpi)):
        fig = Figure(figsize=_HEATMAP_SIZE_IN, layout="constrained")
        ax = fig.subplots()
        mesh = ax.pcolormesh(
            np.ma.masked_array(values, mask=np.eye(n, dtype=bool)),
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
            edgecolors=fig.get_facecolor(),
            linewidth=_CELL_GAP_PX * _POINTS_PER_INCH / dpi,
        )
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(n) + 0.5, labels, rotation=90)
        ax.set_yticks(np.arange(n) + 0.5, labels)
        ax.invert_yaxis()
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(
            f"Comorbidity lift, {n} most common {comorbidity.name}, "
            f"{comorbidity.years[0]}-{comorbidity.years[-1]} windows"
        )
        bar = fig.colorbar(mesh, ax=ax, extend="both", shrink=_COLORBAR_SHRINK)
        bar.set_ticks(ticks, labels=[_lift_label(t) for t in ticks])
        bar.set_label(f"Lift: observed / expected {comorbidity.unit} with both (log scale)")
        bar.outline.set_visible(False)
        save_figure(fig, path)


def feature_path(cfg: Config, year: int) -> Path:
    """Return the npz file holding the features of the window for ``year``."""
    return cfg.paths.features_dir / f"features_{year}.npz"


def save_features(
    features: Features, comorbidity: Comorbidity, chronic: Comorbidity, cfg: Config
) -> None:
    """Write the features, model cohort, fitted weights and comorbidity matrices.

    Files (paths from ``cfg.paths``): features_<year>.npz per window and
    features.json (years, which windows are model inputs, feature vocabulary,
    settings) in ``features_dir``; model_cohort.json; the IDF (tfidf_idf.npy) and
    the SVD components and explained variance (transition_svd.npz) as plain
    arrays; the counts and lift of both comorbidity matrices.

    Args:
        features: Output of ``build_features``.
        comorbidity: Output of ``build_comorbidity``.
        chronic: Output of ``build_chronic_comorbidity``.
        cfg: Project configuration.
    """
    paths, feat_cfg = cfg.paths, cfg.features
    cutoff = cfg.labels.cutoff_date.isoformat()
    for year, matrix in zip(features.years, features.matrices):
        _write_npz(feature_path(cfg, year), matrix)
    n_codes = len(features.codes)
    n_components = features.transitions.components.shape[0]
    _write_json(
        paths.features_dir / FEATURES_JSON,
        {
            "years": list(features.years),
            "feature_years": list(features.feature_years),
            "label_only_years": [y for y in features.years if y not in features.feature_years],
            "cutoff_date": cutoff,
            "n_patients": len(features.patient_ids),
            "n_codes": n_codes,
            "n_transition_components": n_components,
            "columns": (
                f"0-{n_codes - 1}: time-decayed TF-IDF per code of feature_codes; "
                f"{n_codes}-{n_codes + n_components - 1}: transition SVD components"
            ),
            "feature_codes": list(features.codes),
            "redundant_codes": list(features.redundant_codes),
            "idf_rows": features.idf_rows,
            "tfidf_norm": feat_cfg.tfidf_norm,
            "time_decay_lambda": feat_cfg.time_decay_lambda,
            "transition_svd_fit_year": features.transitions.fit_year,
            "transition_svd_n_iter": feat_cfg.transition_svd_n_iter,
            "transition_weight": feat_cfg.transition_weight,
            "seed": cfg.seed,
        },
    )
    rows = features.model_rows
    _write_json(
        paths.model_cohort_json,
        {
            "rule": "patients with a non-zero feature row in at least one training-period window",
            "feature_years": list(features.feature_years),
            "cutoff_date": cutoff,
            "n_patients": len(rows),
            "n_excluded": len(features.patient_ids) - len(rows),
            "rows": rows.tolist(),
            "patient_ids": [features.patient_ids[i] for i in rows],
        },
    )
    paths.tfidf_idf_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(paths.tfidf_idf_npy, features.idf)
    paths.transition_svd_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        paths.transition_svd_npz,
        components=features.transitions.components,
        explained_variance_ratio=features.transitions.explained_variance_ratio,
    )
    logger.info("Wrote %s and %s", paths.tfidf_idf_npy, paths.transition_svd_npz)
    _write_npz(paths.comorbidity_counts_npz, comorbidity.counts)
    _write_npz(paths.comorbidity_lift_npz, comorbidity.lift)
    _write_npz(paths.chronic_comorbidity_counts_npz, chronic.counts)
    _write_npz(paths.chronic_comorbidity_lift_npz, chronic.lift)


def load_model_inputs(cfg: Config) -> ModelInputs:
    """Load the training-period features of the model cohort: the only model inputs.

    The windows come from the cutoff (``config.feature_window_years``), never from
    the saved files, and each must close by the cutoff, so features of windows after
    it (2023-2025 for the project cutoff) are never read or returned.

    Args:
        cfg: Project configuration.

    Returns:
        One matrix per training-period window, rows restricted to the model cohort.

    Raises:
        ValueError: If no window closes by the cutoff, or the saved features or
            model cohort were built for another cutoff or other windows.
    """
    cutoff = cfg.labels.cutoff_date
    years = feature_window_years(cfg)
    if not years or any(date(year + 1, 1, 1) > cutoff for year in years):
        raise ValueError(f"windows {list(years)} do not all close by the cutoff {cutoff}")
    meta = json.loads((cfg.paths.features_dir / FEATURES_JSON).read_text(encoding="utf-8"))
    cohort = json.loads(cfg.paths.model_cohort_json.read_text(encoding="utf-8"))
    for name, saved in ((FEATURES_JSON, meta), (cfg.paths.model_cohort_json.name, cohort)):
        if saved["cutoff_date"] != cutoff.isoformat() or saved["feature_years"] != list(years):
            raise ValueError(
                f"{name} was built for cutoff {saved['cutoff_date']} and training windows "
                f"{saved['feature_years']}, not {cutoff} and {list(years)}; "
                "re-run scripts/03_build_features.py"
            )
    rows = np.asarray(cohort["rows"], dtype=np.int64)
    matrices = [sp.load_npz(feature_path(cfg, year)).tocsr()[rows] for year in years]
    return ModelInputs(years, tuple(cohort["patient_ids"]), rows, matrices)


def write_features_summary(
    features: Features,
    comorbidity: Comorbidity,
    chronic: Comorbidity,
    co_recorded: pd.DataFrame,
    descriptions: Mapping[str, str],
    cfg: Config,
) -> str:
    """Write the feature report to ``cfg.paths.features_summary_md``.

    Covers the settings, the redundant codes, each window's dimensions and
    sparsity, the model cohort, the top comorbidity pairs (all codes and chronic
    codes) and a link to the chronic heatmap.

    Args:
        features: Output of ``build_features``.
        comorbidity: Output of ``build_comorbidity``.
        chronic: Output of ``build_chronic_comorbidity``.
        co_recorded: Output of ``co_recorded_codes``.
        descriptions: Code -> description.
        cfg: Project configuration.

    Returns:
        The markdown written.
    """
    opts, feat_cfg = cfg.summary, cfg.features
    block = features.transitions
    n_patients, n_codes = len(features.patient_ids), len(features.codes)
    n_components = block.components.shape[0]
    first, last = features.feature_years[0], features.feature_years[-1]
    min_share = feat_cfg.redundant_min_share
    n_model = len(features.model_rows)

    def short(code: str) -> str:
        return _SEMANTIC_TAG.sub("", descriptions[code])

    def co_recorded_rows(frame: pd.DataFrame) -> list[list[str]]:
        return [
            [r.code, descriptions[r.code], f"{r.patients:,}", f"{r.encounters:,}",
             f"{r.coverage:.1%}", ", ".join(short(c) for c in r.specific_codes)]
            for r in frame.itertuples(index=False)
        ]

    configured = set(feat_cfg.redundant_codes)
    dropped = co_recorded[co_recorded["code"].isin(configured)]
    kept = co_recorded[(co_recorded["coverage"] >= min_share) & ~co_recorded["code"].isin(configured)]

    window_rows = []
    for t, year in enumerate(features.years):
        matrix, cumulative = features.matrices[t], features.cumulative[t]
        role = "model input" if year in features.feature_years else "label period only"
        window_rows.append(
            [
                str(year),
                role,
                f"{matrix.shape[0]:,} x {matrix.shape[1]:,}",
                f"{matrix.nnz:,}",
                f"{matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.1%}",
                f"{cumulative.nnz / (n_patients * n_codes):.1%}",
                f"{block.n_transitions[t]:,}",
                f"{block.n_patients_with_transitions[t]:,}",
                f"{int((~_nonzero_rows(matrix)).sum()):,}",
            ]
        )

    all_top = top_comorbidity_pairs(
        comorbidity, descriptions, opts.comorbidity_top_pairs, opts.comorbidity_min_count
    )
    chronic_top = top_comorbidity_pairs(
        chronic, descriptions, opts.comorbidity_top_pairs, opts.chronic_pair_min_patients,
        opts.chronic_min_patients,
    )
    n_chronic_codes = int((chronic.counts.diagonal() >= opts.chronic_min_patients).sum())
    figure = Path(
        os.path.relpath(cfg.paths.comorbidity_heatmap_fig, cfg.paths.features_summary_md.parent)
    ).as_posix()
    variance = block.explained_variance_ratio
    co_recorded_header = ["Code", "Condition", "Patients", "Encounters",
                          "With a more specific code", "More specific codes"]

    lines = [
        "# Patient features and comorbidity",
        "",
        "Built by `scripts/03_build_features.py` from the Phase 2 windows and trajectories. "
        "Each window's features are `[X_cum_t | transition_svd_t]`, row L2-normalized.",
        "",
        "## Settings",
        "",
        markdown_table(
            ["Setting", "Value"],
            [
                ["Patients x feature codes", f"{n_patients:,} x {n_codes:,} "
                 f"({len(features.redundant_codes)} redundant codes dropped)"],
                ["Windows", f"{features.years[0]}-{features.years[-1]}"],
                ["Training-period windows (model inputs, IDF fit, comorbidity)",
                 f"{first}-{last}, before the cutoff {cfg.labels.cutoff_date:%Y-%m-%d}"],
                ["IDF documents (non-empty patient-windows in the training period)",
                 f"{features.idf_rows:,}"],
                ["TF-IDF row norm", feat_cfg.tfidf_norm],
                ["Time decay lambda (weight of a window t - s years old)",
                 f"{feat_cfg.time_decay_lambda:g} (exp(-lambda) = "
                 f"{math.exp(-feat_cfg.time_decay_lambda):.3f} per year)"],
                ["Transition SVD", f"fit on the {block.fit_year} block: {n_codes:,}^2 = "
                 f"{n_codes * n_codes:,} -> {n_components} components, {variance.sum():.1%} of "
                 f"the variance (first {variance[0]:.1%})"],
                ["Transition block weight", f"{feat_cfg.transition_weight:g}"],
            ],
            "lr",
        ),
        "",
        "Each block is row-L2-normalized before concatenation and the transition block "
        "scaled by its weight, so with both present it holds "
        f"{feat_cfg.transition_weight**2 / (1 + feat_cfg.transition_weight**2):.0%} of each "
        "row's squared norm. Window t's transition block uses only transitions that end by "
        "the end of year t and before the cutoff; the SVD basis is fit once, on the "
        f"{block.fit_year} block, like the IDF on the whole training period.",
        "",
        "## Redundant general codes",
        "",
        "Dropped from the feature vocabulary (code block, transitions, chronic comorbidity): "
        f"at {min_share:.0%} or more of their encounters a more specific code is also recorded. "
        "A code is more specific if it is recorded with the general one at "
        f"{min_share:.0%} or more of its own encounters and is no more common. Patients are "
        "cohort patients with the code at any date.",
        "",
        markdown_table(co_recorded_header, co_recorded_rows(dropped), "llrrrl"),
        "",
        "These also meet the rule but are kept, as they are not general forms of the codes "
        "recorded with them (see config.yaml):",
        "",
        markdown_table(co_recorded_header, co_recorded_rows(kept), "llrrrl"),
        "",
        "## Feature matrices per window",
        "",
        f"Each matrix has {n_codes:,} code columns (time-decayed TF-IDF) and {n_components} "
        "transition columns. Transitions counts those in the window's cumulative block. "
        "Windows after the cutoff contain post-cutoff events; models read features only "
        "through `load_model_inputs`, which never returns them.",
        "",
        markdown_table(
            ["Year", "Use", "Shape", "Non-zeros", "Density", "Code-block density",
             "Transitions", "Patients with transitions", "All-zero rows"],
            window_rows,
            "llrrrrrrr",
        ),
        "",
        "## Model cohort",
        "",
        f"{n_model:,} of {n_patients:,} patients have a non-zero feature row in at least one "
        f"training-period window ({first}-{last}); the other {n_patients - n_model:,} have no "
        "condition active in those windows and no transition before the cutoff, and are "
        f"excluded. Saved to `{cfg.paths.model_cohort_json.name}`; everything from Phase 4 on "
        "uses this cohort.",
        "",
        "## Comorbidity",
        "",
        "### All pairs (includes co-recorded codes)",
        "",
        f"C = X^T X over the binary patient x code rows of windows {first}-{last} stacked "
        f"(Phase 2 vocabulary, redundant codes included), keeping the "
        f"{comorbidity.population:,} patient-windows with an active condition. C[a, b] counts "
        "patient-windows where a and b are both active. Lift = C[a, b] / (C[a, a] C[b, b] / n): "
        "above 1, more often together than independence predicts. Top "
        f"{len(all_top)} pairs by lift, at least {opts.comorbidity_min_count} patient-windows "
        "together:",
        "",
        markdown_table(
            ["Rank", "Condition A", "Condition B", "Together", "Expected", "Lift"],
            [[str(i), r.description_a, r.description_b, f"{r.count:,}", f"{r.expected:.1f}",
              f"{r.lift:.1f}"] for i, r in enumerate(all_top.itertuples(index=False), start=1)],
            "rllrrr",
        ),
        "",
        "### Chronic comorbidity pairs",
        "",
        "Chronic episodes have no STOP date or last more than "
        f"{feat_cfg.chronic_min_duration_days} days; redundant codes are left out. Counted in "
        "patients: a patient has a pair if chronic episodes of both are active in the same "
        f"window of {first}-{last}. The population is the {chronic.population:,} patients with "
        f"a condition active in those windows. Codes chronic in at least "
        f"{opts.chronic_min_patients} patients ({n_chronic_codes}), pairs together in at least "
        f"{opts.chronic_pair_min_patients}; Jaccard = together / (patients with A or B).",
        "",
        markdown_table(
            ["Rank", "Condition A", "Condition B", "Patients A", "Patients B", "Together",
             "Expected", "Lift", "Jaccard"],
            [[str(i), r.description_a, r.description_b, f"{r.frequency_a:,}",
              f"{r.frequency_b:,}", f"{r.count:,}", f"{r.expected:.1f}", f"{r.lift:.1f}",
              f"{r.jaccard:.2f}"]
             for i, r in enumerate(chronic_top.itertuples(index=False), start=1)],
            "rllrrrrrr",
        ),
        "",
        f"![Chronic comorbidity lift heatmap]({figure})",
        "",
        f"Lift between the {opts.heatmap_codes} chronic codes present in most patients (each "
        f"in at least {opts.chronic_min_patients}), ordered by clustering. The colour scale "
        f"is clipped at lift {_lift_label(-math.floor(opts.heatmap_log2_lift_limit))} and "
        f"{_lift_label(math.floor(opts.heatmap_log2_lift_limit))}; pairs never active "
        "together sit at the blue end.",
        "",
    ]
    text = "\n".join(lines)
    path = cfg.paths.features_summary_md
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)
    return text


def _nonempty_rows(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Drop the rows of ``matrix`` that have no stored entry."""
    return matrix[np.diff(matrix.indptr) > 0]


def _nonzero_rows(matrix: sp.spmatrix) -> np.ndarray:
    """Flag the rows of ``matrix`` with a non-zero value."""
    return np.asarray(abs(matrix).sum(axis=1)).ravel() > 0


def _lift_label(log2_lift: float) -> str:
    """Label a power of two as a lift, e.g. -2 -> "1/4" and 3 -> "8"."""
    return f"{2.0 ** log2_lift:g}" if log2_lift >= 0 else f"1/{2.0 ** -log2_lift:g}"


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
