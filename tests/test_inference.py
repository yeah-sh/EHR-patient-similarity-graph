"""Tests for Phase 8 inference: graph insertion, Nystrom placement, outcomes and the Predictor.

The unit tests use tiny synthetic graphs. The Predictor tests load the saved artifacts
(skipped if they are missing) and re-enter reference patients as new patients, their own
node excluded from every neighbour search, to compare the inductive approximation with
the transductive outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from ehrgraph.config import load_config
from ehrgraph.graph import laplacian, load_graphs, normalize_adjacency
from ehrgraph.inference import (
    NewPatient,
    Prediction,
    Predictor,
    condition_lift,
    format_share,
    insert_node,
    nearest_reference,
    nystrom_extension,
    outcome_counts,
)
from ehrgraph.labels import MORTALITY, load_labels

logger = logging.getLogger(__name__)
NAN = float("nan")
LEAVE_ONE_OUT_PATIENTS = 40  # test-split patients re-entered as new patients


def _ring_with_chords(n: int = 12) -> sp.csr_matrix:
    """A weighted ring 0..n-1 with chords i -> i + 3: connected, every degree 4."""
    rows = list(range(n)) * 2
    cols = [(i + 1) % n for i in range(n)] + [(i + 3) % n for i in range(n)]
    weights = np.linspace(0.3, 1.0, len(rows))
    upper = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    return sp.csr_matrix(upper + upper.T)


# --- Insertion into A_hat ----------------------------------------------------------------


def test_insert_node_equals_full_renormalization() -> None:
    W = _ring_with_chords()
    neighbours, weights = np.array([1, 4, 6]), np.array([0.5, 0.2, 0.9])
    n = W.shape[0]
    column = sp.csr_matrix((weights, (neighbours, np.zeros(3, int))), shape=(n, 1))
    augmented = sp.bmat([[W, column], [column.T, None]], format="csr")
    degrees = 1.0 + np.asarray(W.sum(axis=1)).ravel()
    inserted = insert_node(normalize_adjacency(W), degrees, neighbours, weights)
    np.testing.assert_allclose(inserted.toarray(), normalize_adjacency(augmented).toarray(),
                               atol=1e-12)


def test_insert_node_changes_only_the_neighbours_rows_and_columns() -> None:
    W = _ring_with_chords()
    A = normalize_adjacency(W)
    neighbours = np.array([2, 7])
    inserted = insert_node(A, 1.0 + np.asarray(W.sum(axis=1)).ravel(), neighbours,
                           np.array([0.4, 0.8])).toarray()
    others = np.setdiff1d(np.arange(W.shape[0]), neighbours)
    np.testing.assert_array_equal(inserted[np.ix_(others, others)], A.toarray()[np.ix_(others, others)])
    assert not np.allclose(inserted[np.ix_(neighbours, others)], A.toarray()[np.ix_(neighbours, others)])


def test_a_node_without_neighbours_keeps_only_its_self_loop() -> None:
    W = _ring_with_chords()
    A = normalize_adjacency(W)
    inserted = insert_node(A, 1.0 + np.asarray(W.sum(axis=1)).ravel(), np.empty(0, int),
                           np.empty(0)).toarray()
    assert inserted[-1, -1] == 1.0 and inserted[-1, :-1].sum() == 0 == inserted[:-1, -1].sum()
    np.testing.assert_array_equal(inserted[:-1, :-1], A.toarray())


# --- Nystrom extension -----------------------------------------------------------------


def test_nystrom_is_exact_for_a_reference_node_with_its_own_row() -> None:
    W = _ring_with_chords()
    values, vectors = np.linalg.eigh(laplacian(W).toarray())  # tiny test: dense is fine
    keep = values < 1  # the formula divides by 1 - lambda
    degrees = np.asarray(W.sum(axis=1)).ravel()
    for i in range(W.shape[0]):
        row = W[i]
        placed = nystrom_extension(row.indices, row.data, degrees, vectors[:, keep], values[keep])
        np.testing.assert_allclose(placed, vectors[i, keep], atol=1e-10)


def test_nystrom_without_neighbours_is_nan() -> None:
    placed = nystrom_extension(np.empty(0, int), np.empty(0), np.ones(4), np.eye(4)[:, :2],
                               np.array([0.0, 0.5]))
    assert np.isnan(placed).all()


# --- Neighbour search, outcomes and lift -----------------------------------------------


def test_nearest_reference_respects_allowed_zero_rows_and_k() -> None:
    reference = np.array([[1.0, 0.0], [0.0, 0.0], [0.9, 0.1], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
    allowed = np.array([True, True, True, True, False, True])
    nodes, sims = nearest_reference(reference, np.array([1.0, 0.0]), 3, allowed)
    # Row 4 is not allowed, row 1 is all zero, rows 3 and 5 have similarity <= 0.
    assert nodes.tolist() == [0, 2]
    np.testing.assert_allclose(sims, [1.0, 0.9 / np.hypot(0.9, 0.1)])
    nodes, _ = nearest_reference(reference, np.zeros(2), 3, allowed)
    assert len(nodes) == 0


def test_outcome_counts_never_count_masked_labels() -> None:
    labels = np.array([[1.0, NAN], [0.0, NAN], [NAN, 1.0], [1.0, 0.0]])
    positives, observed = outcome_counts(labels, np.array([0, 1, 2]))
    assert positives.tolist() == [1, 1] and observed.tolist() == [2, 1]
    positives, observed = outcome_counts(labels, np.array([1]))
    assert positives.tolist() == [0, 0] and observed.tolist() == [1, 0]


def test_condition_lift() -> None:
    presence = sp.csr_matrix(np.array([[1, 0], [1, 1], [0, 1], [0, 0]]))
    lift = condition_lift(presence, np.array([0, 1]), np.array([0, 1]))
    assert lift["neighbours_with"].tolist() == [2, 1]
    np.testing.assert_allclose(lift["lift"], [1.0 / 0.5, 0.5 / 0.5])


def test_format_share() -> None:
    assert format_share(3, 9) == "3/9 (33%)" and format_share(0, 0) == "0/0"


# --- The Predictor on the saved artifacts -------------------------------------------------


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    try:
        return Predictor.load()
    except FileNotFoundError as error:
        pytest.skip(f"saved artifacts missing: {error}")


@dataclass(frozen=True)
class LeaveOneOut:
    """Test-split patients re-entered as new patients, their own node excluded."""

    nodes: np.ndarray
    transductive: np.ndarray  # calibrated any_onset probability of the final model
    predictions: list[Prediction]


@pytest.fixture(scope="module")
def leave_one_out(predictor: Predictor) -> LeaveOneOut:
    cfg = load_config()
    test = pd.read_csv(cfg.paths.test_predictions_csv).sample(LEAVE_ONE_OUT_PATIENTS,
                                                              random_state=cfg.seed)
    index = predictor.node_index
    return LeaveOneOut(
        nodes=np.array([index[p] for p in test["patient_id"]]),
        transductive=test["probability"].to_numpy(),
        predictions=[predictor.predict(predictor.reference_patient(p), exclude=[p])
                     for p in test["patient_id"]],
    )


def _demo(predictor: Predictor, number: int) -> NewPatient:
    demo = predictor.cfg.inference.demo_patients[number]
    return NewPatient(list(demo.history), demo.birth_date, demo.sex)


def test_load_reproduces_the_phase_4_spectrum_and_phenotypes(predictor: Predictor) -> None:
    saved = np.load(predictor.cfg.paths.graphs_dir / "spectral_2022.npz")
    n = len(saved["embedding_eigenvalues"])
    np.testing.assert_allclose(predictor.spectral_values[1:n + 1], saved["embedding_eigenvalues"],
                               atol=1e-8)
    k = predictor.centroids.shape[1]
    rows = predictor.spectral_basis[:, :k] / np.linalg.norm(predictor.spectral_basis[:, :k],
                                                            axis=1, keepdims=True)
    nearest = np.argmin(((rows[:, None] - predictor.centroids[None]) ** 2).sum(axis=2), axis=1) + 1
    assert (nearest == predictor.clusters).all()


def test_features_reproduce_phase_3_for_reference_patients(predictor: Predictor) -> None:
    rng = np.random.default_rng(predictor.cfg.seed)
    for node in rng.choice(len(predictor.patient_ids), 50, replace=False):
        patient = predictor.reference_patient(predictor.patient_ids[node])
        features = predictor.history_features(predictor.parse_history(patient.history).events)
        saved = np.stack([reference[node] for reference in predictor.reference])
        np.testing.assert_allclose(features, saved, atol=1e-12)


def test_neighbour_search_matches_the_saved_graphs(predictor: Predictor) -> None:
    graphs = load_graphs(predictor.cfg)
    rng = np.random.default_rng(predictor.cfg.seed)
    k = predictor.cfg.graph.k
    for t, W in enumerate(graphs.weights):
        for node in rng.choice(np.flatnonzero(np.diff(W.indptr) > 0), 30, replace=False):
            allowed = np.ones(len(predictor.patient_ids), dtype=bool)
            allowed[node] = False
            nodes, sims = nearest_reference(predictor.reference[t], predictor.reference[t][node],
                                            k, allowed)
            row = W[node].toarray().ravel()
            in_graph = row[nodes] > 0
            np.testing.assert_allclose(sims[in_graph], row[nodes][in_graph], atol=1e-9)
            # A neighbour missing from the saved graph can only be a tie at the k-th place.
            np.testing.assert_allclose(sims[~in_graph], sims[-1], atol=1e-12)


def test_reference_patient_probability_is_close_to_the_transductive_output(
    leave_one_out: LeaveOneOut,
) -> None:
    inductive = np.array([p.any_onset_probability for p in leave_one_out.predictions])
    difference = np.abs(inductive - leave_one_out.transductive)
    correlation = np.corrcoef(inductive, leave_one_out.transductive)[0, 1]
    logger.info("Inductive vs transductive any_onset probability, %d test patients: median "
                "|difference| %.4f, max %.4f, correlation %.4f", len(difference),
                np.median(difference), difference.max(), correlation)
    assert np.median(difference) < 0.02 and difference.max() < 0.1 and correlation > 0.95


def test_reference_patient_trajectory_is_close_to_the_transductive_states(
    predictor: Predictor, leave_one_out: LeaveOneOut,
) -> None:
    cosines = []
    for node, prediction in zip(leave_one_out.nodes, leave_one_out.predictions):
        true = predictor.reference_states[node]
        cosines.append(np.sum(prediction.window_states * true, axis=1)
                       / (np.linalg.norm(prediction.window_states, axis=1)
                          * np.linalg.norm(true, axis=1)))
    logger.info("GraphAE per-window states, inductive vs transductive: median cosine %.4f, "
                "min %.4f", np.median(cosines), np.min(cosines))
    assert np.median(cosines) > 0.99


def test_nystrom_places_reference_patients_near_their_spectral_coordinates(
    predictor: Predictor, leave_one_out: LeaveOneOut,
) -> None:
    true = predictor.spectral[leave_one_out.nodes]
    placed = np.stack([p.spectral for p in leave_one_out.predictions])
    relative = np.linalg.norm(placed - true, axis=1) / np.linalg.norm(true, axis=1)
    cosine = np.sum(placed * true, axis=1) / (np.linalg.norm(placed, axis=1)
                                              * np.linalg.norm(true, axis=1))
    same_phenotype = np.mean([p.phenotype.cluster == predictor.clusters[node]
                              for node, p in zip(leave_one_out.nodes, leave_one_out.predictions)])
    logger.info("Nystrom placement of %d reference patients (own node excluded): median relative "
                "error %.4f (90th percentile %.4f), median cosine %.4f, same phenotype %.1f%%",
                len(relative), np.median(relative), np.quantile(relative, 0.9), np.median(cosine),
                100 * same_phenotype)
    assert np.median(relative) < 0.2 and np.median(cosine) > 0.99 and same_phenotype >= 0.9


def test_excluded_patients_are_never_neighbours(predictor: Predictor,
                                                leave_one_out: LeaveOneOut) -> None:
    for node, prediction in zip(leave_one_out.nodes, leave_one_out.predictions):
        assert node not in prediction.neighbours
        assert all(node not in insertion.neighbours for insertion in prediction.insertions)
        assert predictor.patient_ids[node] not in set(prediction.similar["patient_id"])


def test_unknown_codes_are_reported_not_fatal(predictor: Predictor) -> None:
    history = [(2019, "714628002"), (2020, "not-a-code"), (2021, "99999999"), (2021, "99999999")]
    prediction = predictor.predict(NewPatient(history, "1980-01-01", "F"))
    assert prediction.unknown_codes == ("not-a-code", "99999999")
    assert prediction.placed and prediction.any_onset_probability is not None
    assert prediction.conditions["code"].tolist() == ["714628002"]

    nothing = predictor.predict(NewPatient([(2020, "not-a-code")], "1980-01-01", "F"))
    assert nothing.unknown_codes == ("not-a-code",) and not nothing.placed
    assert nothing.any_onset_probability is None and nothing.phenotype is None
    assert nothing.similar.empty and predictor.explain(nothing).empty and nothing.notes


def test_malformed_entries_raise(predictor: Predictor) -> None:
    with pytest.raises(ValueError, match="history entry"):
        predictor.parse_history([(2019,)])
    with pytest.raises(ValueError, match="year or date"):
        predictor.parse_history([("soon", "714628002")])
    with pytest.raises(ValueError, match="before onset"):
        predictor.parse_history([(2019, "714628002", 2018)])


def test_history_formats_and_code_merge(predictor: Predictor) -> None:
    as_pairs = predictor.parse_history([(2019, "714628002"), ("2020-05-03", "59621000")])
    as_mapping = predictor.parse_history({"714628002": 2019, "59621000": "2020-05-03"})
    pd.testing.assert_frame_equal(as_pairs.events, as_mapping.events)
    merged = predictor.parse_history([(2019, "15777000")])  # data.code_merge
    assert merged.events["code"].tolist() == ["714628002"] and not merged.unknown_codes
    explicit = predictor.parse_history([(2018, "444814009", 2020), (2019, "59621000", None)])
    assert explicit.events["stop"].isna().tolist() == [False, True]
    # Without a stop: chronic codes stay active, acute ones end at their onset.
    defaults = predictor.parse_history([(2018, "59621000"), (2018, "444814009")]).events
    assert defaults.set_index("code")["stop"].isna().to_dict() == {"59621000": True,
                                                                   "444814009": False}


def test_events_after_the_cutoff_change_nothing(predictor: Predictor) -> None:
    patient = _demo(predictor, 1)
    later = NewPatient([*patient.history, (2024, "44054006"), ("2023-01-01", "431855005")],
                       patient.birth_date, patient.sex)
    before, after = predictor.predict(patient), predictor.predict(later)
    np.testing.assert_array_equal(before.features, after.features)
    np.testing.assert_array_equal(before.window_states, after.window_states)
    assert before.any_onset_probability == after.any_onset_probability
    assert set(after.dropped["code"]) == {"44054006", "431855005"}


def test_prediction_is_deterministic(predictor: Predictor) -> None:
    patient = _demo(predictor, 2)
    first, second = predictor.predict(patient), predictor.predict(patient)
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.window_states, second.window_states)
    np.testing.assert_array_equal(first.spectral, second.spectral)
    assert first.any_onset_probability == second.any_onset_probability
    pd.testing.assert_frame_equal(first.similar, second.similar)
    pd.testing.assert_frame_equal(first.outcomes, second.outcomes)
    pd.testing.assert_frame_equal(predictor.explain(first), predictor.explain(second))


def test_neighbour_outcomes_use_observed_labels_only(predictor: Predictor) -> None:
    labels = load_labels(predictor.cfg).set_index("patient_id")
    for number in range(len(predictor.cfg.inference.demo_patients)):
        prediction = predictor.predict(_demo(predictor, number))
        assert MORTALITY not in set(prediction.outcomes["target"])
        for n in predictor.cfg.inference.outcome_neighbours:
            ids = [predictor.patient_ids[node] for node in prediction.neighbours[:n]]
            for row in prediction.outcomes.itertuples():
                values = labels.loc[ids, row.target]
                assert getattr(row, f"observed_{n}") == values.notna().sum()
                assert getattr(row, f"positives_{n}") == (values == 1.0).sum()
                masked = values.isna().sum()
                assert getattr(row, f"observed_{n}") + masked == n


def test_demographics_are_needed_only_for_the_probability(predictor: Predictor) -> None:
    patient = _demo(predictor, 1)
    without = predictor.predict(NewPatient(patient.history))
    assert without.any_onset_probability is None and without.placed
    assert not without.similar.empty and without.phenotype is not None
    with pytest.raises(ValueError, match="sex"):
        predictor.predict(NewPatient(patient.history, patient.birth_date, "X"))


def test_the_demo_patients_differ(predictor: Predictor) -> None:
    predictions = [predictor.predict(_demo(predictor, i))
                   for i in range(len(predictor.cfg.inference.demo_patients))]
    assert len({p.phenotype.cluster for p in predictions}) == len(predictions)
    assert all(p.placed and p.any_onset_probability is not None for p in predictions)
