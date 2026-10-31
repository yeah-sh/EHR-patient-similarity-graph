"""Tests for ehrgraph.embeddings: the Phase 7b embeddings, metrics and figures."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from ehrgraph.config import load_config
from ehrgraph.embeddings import (
    BASE_RATE,
    GRAPH_AE,
    MODEL_2,
    QUESTIONS,
    RESULT_COLUMNS,
    Embedding,
    NextYearTask,
    base_rate_scores,
    concordance_results,
    embedding_specs,
    micro_auroc,
    neighbour_jaccard,
    new_onsets,
    next_year_results,
    next_year_scores,
    next_year_task,
    pca_embedding,
    plot_next_year_auroc,
    probe_results,
    random_jaccard,
    spectral_window_embedding,
    states_embedding,
    weighted_auroc,
    window_pca_embeddings,
)
from ehrgraph.labels import ANY_ONSET

UTC = "UTC"


def _ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=UTC)


# --- Specs and embeddings --------------------------------------------------------------


def test_eight_embeddings_in_report_order() -> None:
    specs = embedding_specs((2017, 2018, 2019, 2020, 2021, 2022), 32)
    assert [s.number for s in specs] == ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
    assert specs[0].key == GRAPH_AE and len({s.key for s in specs}) == 8
    assert [s.key for s in specs if s.label_informed] == ["tgcn_supervised"]
    assert {key for _, key in QUESTIONS} <= {s.key for s in specs}


def test_states_embedding_previous_is_the_second_to_last_state() -> None:
    states = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    emb = states_embedding(states)
    np.testing.assert_array_equal(emb.final, states[:, -1])
    np.testing.assert_array_equal(emb.previous, states[:, -2])


def test_window_pca_previous_reads_no_last_window() -> None:
    rng = np.random.default_rng(0)
    matrices = [sp.csr_matrix(rng.random((30, 6))) for _ in range(3)]
    first = window_pca_embeddings(matrices, 4, 42)
    changed = matrices[:-1] + [sp.csr_matrix(rng.random((30, 6)))]
    second = window_pca_embeddings(changed, 4, 42)
    for key in first:
        assert first[key].final.shape == (30, 4)
        np.testing.assert_allclose(first[key].previous, second[key].previous)
        assert not np.allclose(first[key].final, second[key].final)
    np.testing.assert_allclose(first["pca_final"].final, pca_embedding(matrices[-1], 4, 42))


def test_spectral_embedding_zero_outside_the_largest_component() -> None:
    ring = [(i, (i + 1) % 8) for i in range(8)]
    rows, cols = zip(*(ring + [(8, 9)]))
    upper = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(11, 11))
    embedding = spectral_window_embedding(upper + upper.T, 3, 42)
    assert embedding.shape == (11, 3)
    assert np.all(embedding[8:] == 0) and np.abs(embedding[:8]).sum() > 0


# --- Next-year new conditions ----------------------------------------------------------


def _events() -> pd.DataFrame:
    rows = [
        ("a", "2021-03-01", "x"), ("a", "2022-02-01", "x"),  # x before 2022: not at risk
        ("a", "2022-05-01", "y"),  # y first in 2022: onset
        ("a", "2023-02-01", "z"),  # z only after the window: at risk, no onset
        ("b", "2022-04-01", "y"),  # b died in 2021: not at risk
        ("c", "2022-06-01", "y"),  # c born in 2022: not at risk
        ("d", "2022-12-31", "z"),
    ]
    return pd.DataFrame([(p, _ts(d), c) for p, d, c in rows],
                        columns=["patient_id", "date", "code"])


def _patients() -> pd.DataFrame:
    return pd.DataFrame({
        "patient_id": ["a", "b", "c", "d"],
        "birth_date": [_ts("1970-01-01"), _ts("1950-01-01"), _ts("2022-02-01"), _ts("1980-01-01")],
        "death_date": [pd.NaT, _ts("2021-07-01"), pd.NaT, pd.NaT],
        "gender": ["F", "M", "F", "M"],
    })


def test_new_onsets_first_event_in_the_year() -> None:
    at_risk, onset = new_onsets(_events(), _patients(), ["a", "b", "c", "d"], ["x", "y", "z"], 2022)
    np.testing.assert_array_equal(at_risk, [[False, True, True], [False] * 3, [False] * 3,
                                            [True, True, True]])
    np.testing.assert_array_equal(onset, [[False, True, False], [False] * 3, [False] * 3,
                                          [False, False, True]])


def test_next_year_task_needs_a_window_before_the_cutoff() -> None:
    split = np.array(["train", "train", "test", "test"])
    with pytest.raises(ValueError, match="close by"):
        next_year_task(_events(), _patients(), ["a", "b", "c", "d"], ["x", "y", "z"], split,
                       2023, date(2023, 1, 1), 1)
    task = next_year_task(_events(), _patients(), ["a", "b", "c", "d"], ["x", "y", "z"], split,
                          2022, date(2023, 1, 1), 1)
    assert task.codes == ("y",)  # the only code with a training onset
    assert task.test_onsets.tolist() == [0] and task.test_at_risk.tolist() == [1]


def _task(n: int = 60, seed: int = 0) -> tuple[NextYearTask, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    onset = np.column_stack([X[:, 0] + 0.3 * rng.standard_normal(n) > 0.5,
                             X[:, 1] + 0.3 * rng.standard_normal(n) > 0.8])
    at_risk = np.ones_like(onset)
    at_risk[::7, 1] = False
    onset &= at_risk
    split = np.array(["train", "test"] * (n // 2))
    task = NextYearTask(2022, ("c1", "c2"), at_risk, onset, onset[split == "train"].sum(axis=0),
                        onset[split == "test"].sum(axis=0), at_risk[split == "test"].sum(axis=0))
    return task, X, split


def test_next_year_scores_use_training_labels_only() -> None:
    task, X, split = _task()
    settings = load_config().embedding_eval
    scores = next_year_scores(X, task, split, settings)
    assert np.isnan(scores[~task.at_risk]).all() and not np.isnan(scores[task.at_risk]).any()
    flipped = task.onset.copy()
    flipped[split == "test"] = ~flipped[split == "test"] & task.at_risk[split == "test"]
    other = next_year_scores(X, replace(task, onset=flipped), split, settings)
    np.testing.assert_allclose(scores, other, equal_nan=True)
    assert micro_auroc(task, scores, split) > 0.8


def test_next_year_results_paired_differences() -> None:
    task, X, split = _task()
    settings = load_config().embedding_eval
    evaluation = replace(load_config().evaluation, bootstrap_resamples=50)
    good = next_year_scores(X, task, split, settings)
    noise = next_year_scores(np.random.default_rng(1).standard_normal(X.shape), task, split,
                             settings)
    results = next_year_results(task, {GRAPH_AE: good, "noise": noise}, split, GRAPH_AE,
                                evaluation, 42)
    table = results.table.set_index("key")
    assert np.isnan(table.loc[GRAPH_AE, "diff"])
    assert table.loc["noise", "diff"] == pytest.approx(table.loc[GRAPH_AE, "micro_auroc"]
                                                       - table.loc["noise", "micro_auroc"])
    assert table.loc["noise", "diff_low"] > 0  # the informative embedding is clearly better
    assert table.loc[GRAPH_AE, "micro_auroc"] == pytest.approx(micro_auroc(task, good, split))
    assert list(results.per_code["code"]) == ["c1", "c2"]
    assert table.loc[GRAPH_AE, "macro_auroc"] == pytest.approx(
        results.per_code[GRAPH_AE].mean())
    assert table.loc[GRAPH_AE, "macro_low"] <= table.loc[GRAPH_AE, "macro_auroc"]         <= table.loc[GRAPH_AE, "macro_high"]
    assert table.loc["noise", "macro_diff"] == pytest.approx(
        table.loc[GRAPH_AE, "macro_auroc"] - table.loc["noise", "macro_auroc"])


def test_base_rate_scores_rank_codes_not_patients() -> None:
    task, X, split = _task()
    scores = base_rate_scores(task, split)
    assert np.isnan(scores[~task.at_risk]).all()
    train = split == "train"
    expected = task.onset[train].sum(axis=0) / task.at_risk[train].sum(axis=0)
    for c in range(2):
        np.testing.assert_allclose(scores[task.at_risk[:, c], c], expected[c])
    evaluation = replace(load_config().evaluation, bootstrap_resamples=20)
    table = next_year_results(task, {BASE_RATE: scores}, split, BASE_RATE, evaluation, 0).table
    assert table.loc[0, "macro_auroc"] == pytest.approx(0.5)  # constant within each code


def test_weighted_auroc_matches_sklearn() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(80) < 0.3).astype(float)
    scores = np.round(rng.random(80) + 0.5 * y, 1)  # rounding makes ties
    weights = rng.integers(0, 3, size=(6, 80))
    expected = [roc_auc_score(y, scores, sample_weight=w) for w in weights]
    np.testing.assert_allclose(weighted_auroc(y, scores, weights), expected)
    no_positive = np.where(y == 1, 0, 1)[None, :]
    assert np.isnan(weighted_auroc(y, scores, no_positive))[0]


# --- Neighbour concordance -------------------------------------------------------------


def _chronic() -> sp.csr_matrix:
    return sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 1]],
                                  dtype=float))


def test_neighbour_jaccard_excludes_the_patient() -> None:
    X = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.7, 0.7]])
    queries = np.array([0, 2])
    values = neighbour_jaccard(X, _chronic(), queries, 1, 8)
    np.testing.assert_allclose(values, [1.0, 1.0])  # 0 -> 1 and 2 -> 3 share everything
    duplicate = np.vstack([X[:1], X[:1], X[2:]])  # node 1 duplicates node 0
    np.testing.assert_allclose(neighbour_jaccard(duplicate, _chronic(), np.array([0]), 1, 8),
                               [1.0])


def test_random_jaccard_never_picks_the_patient() -> None:
    queries = np.arange(5)
    values = random_jaccard(sp.identity(5, format="csr"), queries, 4, 0)
    np.testing.assert_allclose(values, 0.0)  # identity: only the patient itself would match


def test_concordance_results_differences() -> None:
    evaluation = replace(load_config().evaluation, bootstrap_resamples=50)
    frame = concordance_results(
        {GRAPH_AE: np.array([0.5, 0.6, 0.7]), "other": np.array([0.1, 0.2, 0.3])},
        {"chance": np.array([0.0, 0.1, 0.0])}, GRAPH_AE, evaluation, 0,
    ).set_index("key")
    assert frame.loc["other", "diff"] == pytest.approx(0.4)
    assert np.isnan(frame.loc[GRAPH_AE, "diff"]) and np.isnan(frame.loc["chance", "diff"])
    assert frame.loc["other", "diff_low"] == pytest.approx(0.4)  # constant paired difference


# --- Probe -----------------------------------------------------------------------------


def test_probe_results_sign_conventions() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(float)
    scores = {GRAPH_AE: y + rng.normal(0, 0.6, 400), "weak": y + rng.normal(0, 3.0, 400),
              MODEL_2: y + rng.normal(0, 1.5, 400)}
    predictions = pd.DataFrame({ANY_ONSET: y, "fold": np.arange(400) % 4 + 1, **scores,
                                **{f"{key}.logit": value for key, value in scores.items()}})
    fits = pd.DataFrame({"fold": [1, 2, 1, 1], "key": [GRAPH_AE, "weak", "weak", MODEL_2],
                         "slope": [0.5, -0.1, -0.2, 1.0], "intercept": 0.0})
    cfg = load_config()
    cfg = replace(cfg, evaluation=replace(cfg.evaluation, bootstrap_resamples=100))
    table = probe_results(predictions, fits, [GRAPH_AE, "weak"], cfg).set_index("key")
    assert table["folds_flipped"].to_dict() == {GRAPH_AE: 0, "weak": 2, MODEL_2: 0}
    folds = predictions.groupby("fold")
    assert table.loc["weak", "fold_auroc_raw"] == pytest.approx(
        np.mean([roc_auc_score(part[ANY_ONSET], part["weak.logit"]) for _, part in folds]))
    assert table.loc[GRAPH_AE, "vs_model2"] > 0 > table.loc["weak", "vs_model2"]
    assert table.loc["weak", "diff"] > 0  # the GraphAE minus a weaker probe
    assert table.loc[GRAPH_AE, "vs_model2_low"] <= table.loc[GRAPH_AE, "vs_model2"]


# --- Figure ----------------------------------------------------------------------------


def test_plot_next_year_auroc(tmp_path: Path) -> None:
    specs = embedding_specs((2021, 2022), 4)
    table = pd.DataFrame({column: np.nan for column in RESULT_COLUMNS}, index=range(len(specs)))
    table["key"] = [s.key for s in specs]
    table["next_micro_auroc"] = np.linspace(0.55, 0.7, len(specs))
    table["next_micro_low"] = table["next_micro_auroc"] - 0.02
    table["next_micro_high"] = table["next_micro_auroc"] + 0.02
    table["next_macro_auroc"] = table["next_micro_auroc"] - 0.03
    table["next_macro_low"] = table["next_macro_auroc"] - 0.04
    table["next_macro_high"] = table["next_macro_auroc"] + 0.04
    path = tmp_path / "next_year.png"
    plot_next_year_auroc(table, specs, 2022, 0.6, path, load_config())
    assert path.stat().st_size > 0


def test_embedding_is_frozen() -> None:
    emb = Embedding(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(AttributeError):
        emb.final = np.ones((2, 2))  # type: ignore[misc]
