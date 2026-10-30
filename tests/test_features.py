"""Tests for ehrgraph.features on hand-made windows: 4 patients, 3 codes, 2020-2023."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from ehrgraph.config import Config, feature_window_years, load_config
from ehrgraph.data import DATE_DTYPE
from ehrgraph.features import (
    Features,
    build_chronic_comorbidity,
    build_comorbidity,
    build_features,
    chronic_presence,
    co_recorded_codes,
    comorbidity_counts,
    comorbidity_lift,
    feature_path,
    feature_transitions,
    feature_vocabulary,
    load_model_inputs,
    patient_demographics,
    patient_transition_matrix,
    plot_comorbidity_heatmap,
    save_features,
    time_decay_cumulative,
    top_comorbidity_pairs,
    transition_blocks,
    transition_features,
    write_features_summary,
)
from ehrgraph.trajectories import (
    TRAJECTORY_COLUMNS,
    TRANSITION_COLUMNS,
    Trajectories,
    TrajectoryWindows,
    load_trajectories,
    load_windows,
    pre_cutoff_transitions,
)

PATIENTS = ("p1", "p2", "p3", "p4")
CODES = ("c1", "c2", "c3")
DESCRIPTIONS = {"c1": "Code one (disorder)", "c2": "Code two (finding)", "c3": "Code three"}
YEARS = (2020, 2021, 2022, 2023)
CUTOFF = date(2022, 1, 1)  # windows 2020-2021 are the training period, 2022-2023 labels only
UTC = "UTC"
# Active codes per patient in each window.
ACTIVE = {
    2020: {"p1": ["c1", "c2"], "p2": ["c1"], "p4": ["c3"]},
    2021: {"p1": ["c1", "c2"], "p2": ["c2"], "p3": ["c1", "c3"]},
    2022: {"p1": ["c3"], "p2": ["c1", "c2", "c3"], "p3": ["c1"], "p4": ["c2"]},
    2023: {"p1": ["c1"], "p3": ["c2", "c3"]},
}
# (patient, from_code, from_date, to_code, to_date)
TRANSITION_ROWS = [
    ("p1", "c1", "2020-03-01", "c2", "2020-06-01"),
    ("p1", "c2", "2020-06-01", "c3", "2022-02-01"),  # ends after the cutoff
    ("p2", "c2", "2019-05-01", "c1", "2021-05-01"),  # c1 and c3 share an onset,
    ("p2", "c2", "2019-05-01", "c3", "2021-05-01"),  # so both follow c2
    ("p3", "c3", "2022-06-01", "c1", "2023-04-01"),  # p3 has only post-cutoff transitions
    ("p4", "c3", "2020-01-10", "c1", "2021-12-31T23:59:59Z"),  # ends just before the cutoff
    ("p4", "c1", "2021-12-31T23:59:59Z", "c2", "2022-01-01"),  # ends exactly at the cutoff
]
# Trajectory events for the chronic comorbidity: (patient, date, stop, code).
CHRONIC_ROWS = [
    ("p1", "2019-01-01", None, "c1"),  # p1 has c1 and c2 together in 2020 and 2021
    ("p1", "2020-02-01", None, "c2"),
    ("p2", "2020-01-01", "2021-06-01", "c1"),  # 517 days: chronic by duration
    ("p2", "2021-01-01", None, "c2"),
    ("p3", "2020-03-01", "2020-04-01", "c1"),  # 31 days: acute, so p3 has no pair
    ("p3", "2020-03-01", None, "c2"),
    ("p4", "2020-01-01", None, "c3"),
    ("p4", "2023-01-01", None, "c2"),  # starts after the training period
]
DATA_END = pd.Timestamp("2025-12-31", tz=UTC)


def _window(active: dict[str, list[str]]) -> sp.csr_matrix:
    rows = [PATIENTS.index(p) for p, codes in active.items() for _ in codes]
    cols = [CODES.index(c) for codes in active.values() for c in codes]
    return sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(PATIENTS), len(CODES))
    )


def _windows(active: dict[int, dict[str, list[str]]] = ACTIVE) -> TrajectoryWindows:
    return TrajectoryWindows(YEARS, [_window(active[y]) for y in YEARS], PATIENTS, CODES)


def _transitions(rows: list[tuple[str, ...]] = TRANSITION_ROWS) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["patient_id", "from_code", "from_date", "to_code", "to_date"])
    for name in ("from_date", "to_date"):
        df[name] = pd.to_datetime(df[name], utc=True, format="ISO8601").astype(DATE_DTYPE)
    for name in ("patient_id", "from_code", "to_code"):
        df[name] = df[name].astype("string")
    return df[TRANSITION_COLUMNS]


def _trajectories(
    rows: list[tuple[str, str, str | None, str]] = CHRONIC_ROWS,
    codes: tuple[str, ...] = CODES,
) -> Trajectories:
    """Trajectories with the dtypes of trajectories.parquet; no death, so active to STOP."""
    df = pd.DataFrame(rows, columns=["patient_id", "date", "stop", "code"])
    for name in ("date", "stop"):
        df[name] = pd.to_datetime(df[name], utc=True, format="ISO8601").astype(DATE_DTYPE)
    df["active_until"] = df["stop"].fillna(DATA_END).astype(DATE_DTYPE)
    df["chronic"] = df["stop"].isna()
    for name in ("patient_id", "code"):
        df[name] = df[name].astype("string")
    events = df[TRAJECTORY_COLUMNS].sort_values(["patient_id", "date", "code"]).reset_index(drop=True)
    patient_ids = tuple(sorted(df["patient_id"].unique()))
    return Trajectories(events, patient_ids, codes, tuple(f"{c} (disorder)" for c in codes), DATA_END)


def _with_redundant(cfg: Config, codes: tuple[str, ...]) -> Config:
    return replace(cfg, features=replace(cfg.features, redundant_codes=codes))


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """The real config on the fixture's years, with every Phase 3 output in ``tmp_path``."""
    base = load_config()
    processed = tmp_path / "processed"
    paths = replace(
        base.paths,
        features_dir=processed / "features",
        comorbidity_counts_npz=processed / "comorbidity_counts.npz",
        comorbidity_lift_npz=processed / "comorbidity_lift.npz",
        chronic_comorbidity_counts_npz=processed / "chronic_comorbidity_counts.npz",
        chronic_comorbidity_lift_npz=processed / "chronic_comorbidity_lift.npz",
        model_cohort_json=processed / "model_cohort.json",
        tfidf_idf_npy=tmp_path / "artifacts" / "tfidf_idf.npy",
        transition_svd_npz=tmp_path / "artifacts" / "transition_svd.npz",
        features_summary_md=tmp_path / "reports" / "features_summary.md",
        comorbidity_heatmap_fig=tmp_path / "reports" / "figures" / "heatmap.png",
    )
    return replace(
        base,
        paths=paths,
        trajectories=replace(base.trajectories, start_year=YEARS[0], end_year=YEARS[-1]),
        labels=replace(base.labels, cutoff_date=CUTOFF),
        features=replace(base.features, transition_svd_components=2, redundant_codes=()),
        summary=replace(
            base.summary,
            comorbidity_min_count=1,
            chronic_min_patients=1,
            chronic_pair_min_patients=1,
            heatmap_codes=3,
        ),
    )


def _save(cfg: Config, active: dict[int, dict[str, list[str]]] = ACTIVE) -> Features:
    """Build and save the fixture's features; return them."""
    windows = _windows(active)
    features = build_features(windows, _transitions(), cfg)
    chronic = build_chronic_comorbidity(_trajectories(), cfg)
    save_features(features, build_comorbidity(windows, cfg), chronic, cfg)
    return features


def test_fixture_training_period_is_2020_2021(cfg: Config) -> None:
    assert feature_window_years(cfg) == (2020, 2021)


# --- Redundant general codes -------------------------------------------------------


def _encounter_events() -> pd.DataFrame:
    """g is general (always with s1 or s2); n and w are a 1:1 pair; x only half the time."""
    rows = [
        ("q1", "e1", "g"), ("q1", "e1", "s1"),
        ("q2", "e2", "g"), ("q2", "e2", "s2"),
        ("q3", "e3", "g"), ("q3", "e3", "s1"),
        ("q1", "e4", "g"), ("q1", "e4", "s2"),
        ("q2", "e5", "n"), ("q2", "e5", "w"),
        ("q3", "e6", "n"), ("q3", "e6", "w"),
        ("q1", "e7", "x"),
        ("q2", "e8", "x"), ("q2", "e8", "y"),
    ]
    return pd.DataFrame(rows, columns=["patient_id", "encounter_id", "code"]).astype("string")


def test_co_recorded_codes_finds_general_codes() -> None:
    stats = co_recorded_codes(_encounter_events(), 0.95, redundant=("n",)).set_index("code")
    assert stats.loc["g", "coverage"] == 1.0
    assert set(stats.loc["g", "specific_codes"]) == {"s1", "s2"}
    assert (stats.loc["g", "patients"], stats.loc["g", "encounters"]) == (3, 4)
    assert stats.loc["n", "coverage"] == 1.0  # whiplash-like 1:1 pair: n is the listed one,
    assert stats.loc["w", "coverage"] == 0.0  # so w does not count n as its specific form
    assert stats.loc["x", "coverage"] == 0.5
    assert stats.loc["s1", "coverage"] == 0.0  # g is more common, so not a specific form of s1


def test_co_recorded_codes_warns_about_listed_codes_below_the_rule(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="ehrgraph.features"):
        co_recorded_codes(_encounter_events(), 0.95, redundant=("g", "x"))
    assert "Redundant code x" in caplog.text
    assert "Redundant code g" not in caplog.text


def test_redundant_codes_are_dropped_from_the_feature_vocabulary(cfg: Config) -> None:
    dropped_c3 = _with_redundant(cfg, ("c3", "not-in-vocabulary"))
    assert feature_vocabulary(CODES, dropped_c3) == (("c1", "c2"), ("c3",))
    transitions = _transitions()
    without_c3 = transitions[(transitions["from_code"] != "c3") & (transitions["to_code"] != "c3")]
    features = build_features(_windows(), without_c3, dropped_c3)
    assert features.codes == ("c1", "c2") and features.redundant_codes == ("c3",)
    assert all(m.shape == (4, 2 + 2) for m in features.matrices)
    assert features.idf.shape == (2,)
    with pytest.raises(ValueError, match="vocabulary"):  # transitions must be rebuilt first
        build_features(_windows(), transitions, dropped_c3)


def test_feature_transitions_relink_around_redundant_codes(cfg: Config) -> None:
    rows = [
        ("p1", "2019-01-01", None, "c1"),
        ("p1", "2020-01-01", None, "g"),  # g recorded with c2: c1 -> c2 survives
        ("p1", "2020-01-01", None, "c2"),
        ("p2", "2019-01-01", None, "c1"),
        ("p2", "2020-01-01", None, "g"),  # g alone at its step: c1 -> g -> c2 becomes c1 -> c2
        ("p2", "2021-01-01", None, "c2"),
    ]
    trajectories = _trajectories(rows, codes=("c1", "c2", "g"))
    kept = feature_transitions(trajectories, _with_redundant(cfg, ("g",)))
    assert kept[["patient_id", "from_code", "to_code"]].values.tolist() == [
        ["p1", "c1", "c2"],
        ["p2", "c1", "c2"],
    ]
    with_g = feature_transitions(trajectories, cfg)
    assert {"g"} <= set(with_g["from_code"]) | set(with_g["to_code"])


# --- Comorbidity -------------------------------------------------------------------


def test_comorbidity_is_symmetric_with_frequency_diagonal(cfg: Config) -> None:
    windows = _windows()
    counts, n_rows = comorbidity_counts(windows.matrices[:2])
    assert (counts != counts.T).nnz == 0
    # 2020 and 2021 each have one empty row (p3, then p4): 6 patient-windows remain.
    assert n_rows == 6
    stacked = sp.vstack(windows.matrices[:2])
    frequency = np.asarray(stacked.sum(axis=0)).ravel()  # patient-windows per code
    np.testing.assert_array_equal(counts.diagonal(), frequency)
    np.testing.assert_array_equal(counts.diagonal(), [4, 3, 2])
    assert counts[0, 1] == 2  # c1 and c2: p1 in 2020 and 2021
    assert counts[0, 2] == 1  # c1 and c3: p3 in 2021
    assert counts[1, 2] == 0


def test_comorbidity_lift_is_observed_over_expected() -> None:
    counts, n_rows = comorbidity_counts(_windows().matrices[:2])
    lift = comorbidity_lift(counts, n_rows)
    assert (lift != lift.T).nnz == 0
    assert lift[0, 1] == pytest.approx(2 / (4 * 3 / 6))  # 1.0: independent
    assert lift[0, 2] == pytest.approx(1 / (4 * 2 / 6))  # 0.75
    assert lift[1, 2] == 0
    assert lift.diagonal() == pytest.approx([6 / 4, 6 / 3, 6 / 2])


def test_comorbidity_uses_training_windows_only(cfg: Config) -> None:
    base = build_comorbidity(_windows(), cfg)
    after = build_comorbidity(_windows({**ACTIVE, 2023: {p: list(CODES) for p in PATIENTS}}), cfg)
    assert base.years == (2020, 2021) and base.unit == "patient-windows"
    assert (base.counts != after.counts).nnz == 0


def test_top_pairs_rank_by_lift(cfg: Config) -> None:
    comorbidity = build_comorbidity(_windows(), cfg)
    top = top_comorbidity_pairs(comorbidity, DESCRIPTIONS, n=5, min_count=1)
    assert top[["code_a", "code_b"]].values.tolist() == [["c1", "c2"], ["c1", "c3"]]
    assert top["lift"].tolist() == pytest.approx([1.0, 0.75])
    assert top["jaccard"].tolist() == pytest.approx([2 / (4 + 3 - 2), 1 / (4 + 2 - 1)])
    assert top_comorbidity_pairs(comorbidity, DESCRIPTIONS, 5, min_count=2)["code_b"].tolist() == ["c2"]
    by_frequency = top_comorbidity_pairs(comorbidity, DESCRIPTIONS, 5, 1, min_frequency=3)
    assert by_frequency["code_b"].tolist() == ["c2"]  # c3 has 2
    assert top_comorbidity_pairs(comorbidity, DESCRIPTIONS, 5, 1, min_frequency=4).empty  # c2: 3


def test_chronic_comorbidity_counts_patients_not_patient_years(cfg: Config) -> None:
    chronic = build_chronic_comorbidity(_trajectories(), cfg)
    assert chronic.unit == "patients" and chronic.population == 4
    np.testing.assert_array_equal(chronic.counts.diagonal(), [2, 3, 1])  # c2 of p4 starts 2023
    assert (chronic.counts != chronic.counts.T).nnz == 0
    assert chronic.counts[0, 1] == 2  # p1 (in 2020 and 2021, once) and p2; p3's c1 is acute
    assert chronic.counts[1, 2] == 0
    top = top_comorbidity_pairs(chronic, DESCRIPTIONS, n=5, min_count=1)
    assert top[["code_a", "code_b", "count"]].values.tolist() == [["c1", "c2", 2]]
    assert top["lift"].iloc[0] == pytest.approx(2 * 4 / (2 * 3))
    assert top["jaccard"].iloc[0] == pytest.approx(2 / (2 + 3 - 2))


def test_chronic_presence_flags_each_patients_chronic_conditions(cfg: Config) -> None:
    presence, codes = chronic_presence(_trajectories(), cfg)
    assert codes == CODES
    assert presence.toarray().tolist() == [
        [1, 1, 0],  # p1
        [1, 1, 0],  # p2: c1 chronic by duration
        [0, 1, 0],  # p3: c1 is acute
        [0, 0, 1],  # p4: c2 starts after the training period
    ]
    # Column sums are the diagonal of the chronic comorbidity counts.
    chronic = build_chronic_comorbidity(_trajectories(), cfg)
    np.testing.assert_array_equal(np.asarray(presence.sum(axis=0)).ravel(), chronic.counts.diagonal())


def test_chronic_comorbidity_leaves_out_redundant_codes(cfg: Config) -> None:
    chronic = build_chronic_comorbidity(_trajectories(), _with_redundant(cfg, ("c3",)))
    assert chronic.codes == ("c1", "c2")
    assert chronic.population == 4  # p4 still has a condition active in the period


# --- TF-IDF and time decay ---------------------------------------------------------


def test_idf_is_fit_on_training_windows_only(cfg: Config) -> None:
    features = build_features(_windows(), _transitions(), cfg)
    # 6 documents (non-empty 2020-2021 patient-windows); df = 4, 3, 2.
    expected = [math.log(7 / (1 + df)) + 1 for df in (4, 3, 2)]
    np.testing.assert_allclose(features.idf, expected)
    assert features.idf_rows == 6

    label_changed = {**ACTIVE, 2022: {p: ["c3"] for p in PATIENTS}}
    np.testing.assert_allclose(build_features(_windows(label_changed), _transitions(), cfg).idf, expected)
    train_changed = {**ACTIVE, 2021: {p: ["c3"] for p in PATIENTS}}
    assert not np.allclose(build_features(_windows(train_changed), _transitions(), cfg).idf, expected)


def test_tfidf_rows_are_idf_weighted_and_unit_norm(cfg: Config) -> None:
    features = build_features(_windows(), _transitions(), cfg)
    x = features.tfidf[0].toarray()  # 2020
    norms = np.linalg.norm(x, axis=1)
    np.testing.assert_allclose(norms, [1, 1, 0, 1])  # p3 has nothing active in 2020
    idf = features.idf
    np.testing.assert_allclose(x[0], np.array([idf[0], idf[1], 0]) / math.hypot(idf[0], idf[1]))


def test_time_decay_with_zero_lambda_is_cumulative_sum() -> None:
    rng = np.random.default_rng(42)
    matrices = [sp.random(5, 4, density=0.4, format="csr", random_state=rng) for _ in YEARS]
    cumulative = time_decay_cumulative(matrices, YEARS, 0.0)
    running = np.zeros((5, 4))
    for matrix, result in zip(matrices, cumulative):
        running += matrix.toarray()
        assert isinstance(result, sp.csr_matrix)
        np.testing.assert_allclose(result.toarray(), running)


def test_time_decay_matches_formula_with_year_gaps() -> None:
    rng = np.random.default_rng(42)
    years = (2017, 2018, 2020, 2021)  # 2019 missing: a gap of 2 years
    matrices = [sp.random(5, 4, density=0.4, format="csr", random_state=rng) for _ in years]
    decay = 0.3
    cumulative = time_decay_cumulative(matrices, years, decay)
    for t, year in enumerate(years):
        expected = sum(
            math.exp(-decay * (year - years[s])) * matrices[s].toarray() for s in range(t + 1)
        )
        np.testing.assert_allclose(cumulative[t].toarray(), expected)


def test_time_decay_rejects_unordered_years() -> None:
    matrices = [sp.csr_matrix((2, 2)) for _ in range(2)]
    with pytest.raises(ValueError, match="increasing"):
        time_decay_cumulative(matrices, (2021, 2020), 0.1)


# --- Transitions and leakage -------------------------------------------------------


def test_pre_cutoff_transitions_end_strictly_before_cutoff() -> None:
    kept = pre_cutoff_transitions(_transitions(), CUTOFF)
    assert (kept["to_date"] < pd.Timestamp("2022-01-01", tz=UTC)).all()
    assert kept[["patient_id", "from_code", "to_code"]].values.tolist() == [
        ["p1", "c1", "c2"],
        ["p2", "c2", "c1"],
        ["p2", "c2", "c3"],
        ["p4", "c3", "c1"],  # the transition ending exactly at the cutoff is dropped
    ]


def test_patient_transition_matrix_is_row_normalized_per_from_code() -> None:
    matrix = patient_transition_matrix(pre_cutoff_transitions(_transitions(), CUTOFF), PATIENTS, CODES)
    assert matrix.shape == (4, 9)
    dense = matrix.toarray().reshape(4, 3, 3)  # patient, from, to
    assert dense[0, 0, 1] == 1.0  # p1: c1 -> c2
    assert dense[1, 1].tolist() == [0.5, 0.0, 0.5]  # p2: c2 -> c1 and c2 -> c3
    assert not dense[2].any()  # p3: nothing before the cutoff
    row_sums = dense.sum(axis=2)
    assert set(np.round(row_sums[row_sums > 0], 12)) == {1.0}


def test_patient_transition_matrix_rejects_unknown_patient() -> None:
    with pytest.raises(ValueError, match="patient index"):
        patient_transition_matrix(_transitions().assign(patient_id="nobody"), PATIENTS, CODES)


def test_transition_blocks_are_cumulative_and_stop_at_the_cutoff() -> None:
    blocks = transition_blocks(_transitions(), PATIENTS, CODES, YEARS, CUTOFF)
    with_any = [set(np.flatnonzero(np.diff(b.indptr))) for b in blocks]
    assert with_any[0] == {0}  # 2020: only p1's c1 -> c2 has ended
    assert with_any[1] == {0, 1, 3}  # 2021: p2's and p4's too
    # After the cutoff nothing is added: not p4's transition ending exactly at the cutoff,
    # nor p1's (2022-02) or p3's (2023).
    assert (blocks[2] != blocks[1]).nnz == 0 and (blocks[3] != blocks[1]).nnz == 0


def test_first_window_transition_block_ignores_later_transitions(cfg: Config) -> None:
    full = _transitions()
    early = full[full["to_date"] < pd.Timestamp("2021-01-01", tz=UTC)]  # dated 2020 only
    blocks_full = transition_blocks(full, PATIENTS, CODES, YEARS, CUTOFF)
    blocks_early = transition_blocks(early, PATIENTS, CODES, YEARS, CUTOFF)
    assert (blocks_full[0] != blocks_early[0]).nnz == 0
    assert (blocks_full[1] != blocks_early[1]).nnz > 0  # the later window does use them
    # Projected on the SVD basis (fit once, on the last training window), the first
    # window's transition features are the same without the later transitions.
    svd = transition_features(full, PATIENTS, CODES, YEARS, cfg)
    assert svd.fit_year == 2021
    np.testing.assert_allclose(svd.embeddings[0], blocks_early[0] @ svd.components.T)


def test_transition_features_use_only_pre_cutoff_transitions(cfg: Config) -> None:
    full = transition_features(_transitions(), PATIENTS, CODES, YEARS, cfg)
    pre_only = transition_features(
        pre_cutoff_transitions(_transitions(), CUTOFF), PATIENTS, CODES, YEARS, cfg
    )
    for a, b in zip(full.embeddings, pre_only.embeddings):
        np.testing.assert_array_equal(a, b)
    assert full.n_transitions == (1, 4, 4, 4)
    assert not any(e[PATIENTS.index("p3")].any() for e in full.embeddings)  # post-cutoff only

    # The post-cutoff rows would change the features if they were not dropped.
    later = replace(cfg, labels=replace(cfg.labels, cutoff_date=date(2030, 1, 1)))
    leaky = transition_features(_transitions(), PATIENTS, CODES, YEARS, later)
    assert leaky.n_transitions[-1] == len(TRANSITION_ROWS)
    assert not np.allclose(np.abs(leaky.embeddings[-1]), np.abs(full.embeddings[-1]))


def test_final_features_ignore_post_cutoff_transitions(cfg: Config) -> None:
    full = build_features(_windows(), _transitions(), cfg)
    pre_only = build_features(_windows(), pre_cutoff_transitions(_transitions(), CUTOFF), cfg)
    for a, b in zip(full.matrices, pre_only.matrices):
        np.testing.assert_array_equal(a.toarray(), b.toarray())


# --- Final features and the model cohort ---------------------------------------------


def test_final_features_have_unit_or_zero_rows(cfg: Config) -> None:
    features = build_features(_windows(), _transitions(), cfg)
    assert features.years == YEARS and features.feature_years == (2020, 2021)
    for matrix in features.matrices:
        assert isinstance(matrix, sp.csr_matrix) and matrix.shape == (4, 3 + 2)
        norms = np.linalg.norm(matrix.toarray(), axis=1)
        assert np.all(np.isclose(norms, 1.0) | (norms == 0.0))
    # p3 has nothing active in 2020 and no transition yet: an all-zero row.
    assert features.matrices[0][PATIENTS.index("p3")].nnz == 0
    # The code block equals the normalized X_cum_t, scaled into the unit row.
    x_cum = features.cumulative[1].toarray()
    code_block = features.matrices[1].toarray()[:, :3]
    p1 = PATIENTS.index("p1")
    np.testing.assert_allclose(
        code_block[p1] / np.linalg.norm(code_block[p1]), x_cum[p1] / np.linalg.norm(x_cum[p1])
    )


def test_model_cohort_excludes_patients_with_no_training_features(cfg: Config) -> None:
    assert build_features(_windows(), _transitions(), cfg).model_rows.tolist() == [0, 1, 2, 3]
    # Without its 2021 conditions p3 has nothing in 2020-2021 and only post-cutoff transitions.
    no_p3 = {**ACTIVE, 2021: {p: c for p, c in ACTIVE[2021].items() if p != "p3"}}
    features = build_features(_windows(no_p3), _transitions(), cfg)
    assert features.model_rows.tolist() == [0, 1, 3]
    assert features.matrices[YEARS.index(2022)][PATIENTS.index("p3")].nnz > 0  # active later


def test_model_inputs_are_the_training_windows_of_the_model_cohort(cfg: Config) -> None:
    no_p3 = {**ACTIVE, 2021: {p: c for p, c in ACTIVE[2021].items() if p != "p3"}}
    features = _save(cfg, no_p3)
    cohort = json.loads(cfg.paths.model_cohort_json.read_text(encoding="utf-8"))
    assert (cohort["n_patients"], cohort["n_excluded"], cohort["rows"]) == (3, 1, [0, 1, 3])
    assert cohort["patient_ids"] == ["p1", "p2", "p4"]

    for year in (2022, 2023):  # the loader never opens label-period files
        feature_path(cfg, year).unlink()
    inputs = load_model_inputs(cfg)
    assert inputs.years == (2020, 2021)
    assert inputs.patient_ids == ("p1", "p2", "p4")
    for year, matrix in zip(inputs.years, inputs.matrices):
        expected = features.matrices[YEARS.index(year)][[0, 1, 3]]
        assert matrix.shape == (3, 5) and (matrix != expected).nnz == 0


@pytest.mark.parametrize("saved_file", ["features_json", "model_cohort_json"])
def test_model_inputs_never_return_label_period_windows(cfg: Config, saved_file: str) -> None:
    _save(cfg)
    path = (
        cfg.paths.features_dir / "features.json"
        if saved_file == "features_json"
        else cfg.paths.model_cohort_json
    )
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["feature_years"] = list(YEARS)  # claims 2022 and 2023 are model inputs
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="re-run"):
        load_model_inputs(cfg)


def test_model_inputs_reject_features_built_for_another_cutoff(cfg: Config) -> None:
    _save(cfg)
    later = replace(cfg, labels=replace(cfg.labels, cutoff_date=date(2024, 1, 1)))
    with pytest.raises(ValueError, match="re-run"):
        load_model_inputs(later)


# --- Saving and the report ---------------------------------------------------------


def _patients() -> pd.DataFrame:
    """Three patients; b dies after the 2023-01-01 cutoff, c is born just before it."""
    def day(text: str | None) -> pd.Timestamp:
        return pd.Timestamp(text, tz="UTC") if text else pd.NaT

    return pd.DataFrame(
        {
            "patient_id": pd.array(["a", "b", "c"], dtype="string"),
            "birth_date": pd.Series([day("2000-01-01"), day("1950-07-01"), day("2022-12-31")],
                                    dtype=DATE_DTYPE),
            "death_date": pd.Series([day(None), day("2024-06-01"), day(None)], dtype=DATE_DTYPE),
            "gender": pd.array(["F", "M", "F"], dtype="string"),
        }
    )


def test_age_is_computed_at_the_cutoff_not_later() -> None:
    cutoff = date(2023, 1, 1)
    demographics = patient_demographics(_patients(), ["b", "a", "c"], cutoff)
    assert demographics.index.tolist() == ["b", "a", "c"]
    assert list(demographics.columns) == ["age", "sex"]
    expected = [
        (cutoff - date(1950, 7, 1)).days / 365.25,  # not at death (2024-06-01)
        (cutoff - date(2000, 1, 1)).days / 365.25,
        1 / 365.25,
    ]
    np.testing.assert_allclose(demographics["age"], expected)
    np.testing.assert_array_equal(demographics["sex"], [1.0, 0.0, 0.0])
    # The age moves with the reference date, so it is the cutoff's age, not today's.
    later = patient_demographics(_patients(), ["b", "a", "c"], date(2024, 1, 1))
    np.testing.assert_allclose(later["age"] - demographics["age"], 365 / 365.25)
    # A death after the cutoff is never read.
    alive = _patients().assign(death_date=pd.Series([pd.NaT] * 3, dtype=DATE_DTYPE))
    pd.testing.assert_frame_equal(patient_demographics(alive, ["b", "a", "c"], cutoff),
                                  demographics)


def test_demographics_reject_bad_patients() -> None:
    with pytest.raises(ValueError, match="born after"):
        patient_demographics(_patients(), ["c"], date(2022, 6, 1))
    with pytest.raises(ValueError, match="no birth date"):
        patient_demographics(_patients(), ["a", "zz"], date(2023, 1, 1))
    other = _patients().assign(gender=pd.array(["F", "M", "U"], dtype="string"))
    with pytest.raises(ValueError, match="genders"):
        patient_demographics(other, ["c"], date(2023, 1, 1))


def test_save_report_and_heatmap(cfg: Config) -> None:
    features = _save(cfg)
    meta = json.loads((cfg.paths.features_dir / "features.json").read_text(encoding="utf-8"))
    assert meta["feature_years"] == [2020, 2021] and meta["label_only_years"] == [2022, 2023]
    assert (meta["n_patients"], meta["n_codes"], meta["n_transition_components"]) == (4, 3, 2)
    assert meta["feature_codes"] == list(CODES) and meta["transition_svd_fit_year"] == 2021
    for year, built in zip(YEARS, features.matrices):
        assert (sp.load_npz(feature_path(cfg, year)) != built).nnz == 0
    np.testing.assert_array_equal(np.load(cfg.paths.tfidf_idf_npy), features.idf)

    chronic = build_chronic_comorbidity(_trajectories(), cfg)
    assert (sp.load_npz(cfg.paths.chronic_comorbidity_counts_npz) != chronic.counts).nnz == 0
    plot_comorbidity_heatmap(chronic, DESCRIPTIONS, cfg.paths.comorbidity_heatmap_fig, cfg, 1)
    assert cfg.paths.comorbidity_heatmap_fig.stat().st_size > 0

    events = pd.DataFrame(
        [("p1", "e1", "c1"), ("p1", "e1", "c2"), ("p2", "e2", "c3")],
        columns=["patient_id", "encounter_id", "code"],
    )
    co_recorded = co_recorded_codes(events, 0.95)
    comorbidity = build_comorbidity(_windows(), cfg)
    text = write_features_summary(features, comorbidity, chronic, co_recorded, DESCRIPTIONS, cfg)
    assert cfg.paths.features_summary_md.read_text(encoding="utf-8") == text
    assert "### All pairs (includes co-recorded codes)" in text
    assert "| 1 | Code one (disorder) | Code two (finding) | 2 | 2.0 | 1.0 |" in text
    assert "| 1 | Code one (disorder) | Code two (finding) | 2 | 3 | 2 | 1.5 | 1.3 | 0.67 |" in text
    assert "| 2022 | label period only | 4 x 5 |" in text
    assert "4 of 4 patients have a non-zero feature row" in text
    assert "](figures/heatmap.png)" in text


# --- The saved project data --------------------------------------------------------

_PROJECT = load_config()
_needs_features = pytest.mark.skipif(
    not _PROJECT.paths.model_cohort_json.exists(), reason="run scripts/03_build_features.py"
)


@_needs_features
def test_saved_comorbidity_is_symmetric_with_frequency_diagonal() -> None:
    counts = sp.load_npz(_PROJECT.paths.comorbidity_counts_npz)
    assert (counts != counts.T).nnz == 0
    windows = load_windows(_PROJECT)
    train = [windows[windows.years.index(y)] for y in feature_window_years(_PROJECT)]
    frequency = np.asarray(sp.vstack(train).sum(axis=0)).ravel()
    np.testing.assert_array_equal(counts.diagonal(), frequency)
    chronic = sp.load_npz(_PROJECT.paths.chronic_comorbidity_counts_npz)
    assert (chronic != chronic.T).nnz == 0


@_needs_features
def test_saved_model_inputs_stop_at_2022() -> None:
    inputs = load_model_inputs(_PROJECT)
    assert inputs.years == tuple(range(2017, 2023))
    cohort = json.loads(_PROJECT.paths.model_cohort_json.read_text(encoding="utf-8"))
    assert all(m.shape[0] == cohort["n_patients"] for m in inputs.matrices)
    nonzero = np.logical_or.reduce([np.diff(m.indptr) > 0 for m in inputs.matrices])
    assert nonzero.all()  # every model-cohort patient has a feature in some window


@_needs_features
def test_saved_2017_transition_block_ignores_transitions_from_2018() -> None:
    trajectories = load_trajectories(_PROJECT)
    transitions = feature_transitions(trajectories, _PROJECT)
    codes, _ = feature_vocabulary(trajectories.codes, _PROJECT)
    early = transitions[transitions["to_date"] < pd.Timestamp("2018-01-01", tz=UTC)]
    years = (2017,)
    cutoff = _PROJECT.labels.cutoff_date
    full_block = transition_blocks(transitions, trajectories.patient_ids, codes, years, cutoff)[0]
    early_block = transition_blocks(early, trajectories.patient_ids, codes, years, cutoff)[0]
    assert len(early) < len(transitions)
    assert (full_block != early_block).nnz == 0
