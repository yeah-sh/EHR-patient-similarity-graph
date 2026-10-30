"""Tests for ehrgraph.trajectories on a hand-made cohort of 4 patients."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from ehrgraph.config import Config, load_config
from ehrgraph.data import DATE_DTYPE, EVENT_COLUMNS, active_in_year
from ehrgraph.trajectories import (
    TRAJECTORY_COLUMNS,
    TRANSITION_COLUMNS,
    build_trajectories,
    extract_transitions,
    load_transitions,
    load_trajectories,
    load_windows,
    pre_cutoff_transitions,
    save_trajectories,
    transition_matrix,
    window_trajectories,
    write_trajectory_summary,
)

DIABETES, HYPERTENSION = "44054006", "59621000"
BRONCHITIS, PHARYNGITIS, SINUSITIS = "10509002", "195662009", "444814009"
DESCRIPTIONS = {
    DIABETES: "Diabetes mellitus type 2 (disorder)",
    HYPERTENSION: "Essential hypertension (disorder)",
    BRONCHITIS: "Acute bronchitis (disorder)",
    PHARYNGITIS: "Acute viral pharyngitis (disorder)",
    SINUSITIS: "Viral sinusitis (disorder)",
}
PD_DEATH = pd.Timestamp("2016-05-10", tz="UTC")
DATA_END = pd.Timestamp("2020-07-15", tz="UTC")  # pC's last STOP
# (patient, date, stop, code, encounter)
EVENT_ROWS = [
    # A develops diabetes, then hypertension; both chronic.
    ("pA", "2012-03-01T09:00:00Z", None, DIABETES, "a1"),
    ("pA", "2015-06-01T09:00:00Z", None, HYPERTENSION, "a2"),
    # B develops the same two conditions in the reverse order.
    ("pB", "2012-03-01T09:00:00Z", None, HYPERTENSION, "b1"),
    ("pB", "2015-06-01T09:00:00Z", None, DIABETES, "b2"),
    # C has one distinct code (two episodes), so is not in the cohort.
    ("pC", "2019-05-01T10:00:00Z", "2019-05-20", SINUSITIS, "c1"),
    ("pC", "2020-07-01T10:00:00Z", "2020-07-15", SINUSITIS, "c2"),
    # D dies on 2016-05-10: two bronchitis episodes in 2013, pharyngitis and sinusitis
    # at one encounter, and a last bronchitis episode whose STOP is after death.
    ("pD", "2011-01-05T08:00:00Z", None, DIABETES, "d1"),
    ("pD", "2013-02-01T08:00:00Z", "2013-02-20", BRONCHITIS, "d2"),
    ("pD", "2013-09-01T08:00:00Z", "2013-09-15", BRONCHITIS, "d3"),
    ("pD", "2014-04-04T08:00:00Z", "2014-04-20", PHARYNGITIS, "d4"),
    ("pD", "2014-04-04T08:00:00Z", "2014-04-25", SINUSITIS, "d4"),
    ("pD", "2015-01-01T08:00:00Z", None, HYPERTENSION, "d5"),
    ("pD", "2016-04-01T08:00:00Z", "2016-08-01", BRONCHITIS, "d6"),
    # E has two distinct codes, both resolved before the first window (2010), so E
    # has no active condition in any window and is dropped from the cohort.
    ("pE", "2005-03-01T09:00:00Z", "2005-03-20", BRONCHITIS, "e1"),
    ("pE", "2007-06-01T09:00:00Z", "2007-06-10", PHARYNGITIS, "e2"),
]


def _events() -> pd.DataFrame:
    """The fixture rows as an events table with the dtypes of events_clinical.parquet."""
    df = pd.DataFrame(EVENT_ROWS, columns=["patient_id", "date", "stop", "code", "encounter_id"])
    df["description"] = df["code"].map(DESCRIPTIONS)
    for name in ("date", "stop"):
        df[name] = pd.to_datetime(df[name], utc=True, format="ISO8601").astype(DATE_DTYPE)
    for name in ("patient_id", "code", "description", "encounter_id"):
        df[name] = df[name].astype("string")
    return df[EVENT_COLUMNS].sort_values(["patient_id", "date", "code"]).reset_index(drop=True)


def _patients() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "patient_id": ["pA", "pB", "pC", "pD", "pE"],
            "birth_date": ["1960-01-01", "1962-01-01", "1990-01-01", "1940-01-01", "1985-01-01"],
            "death_date": [None, None, None, "2016-05-10", None],
            "gender": ["F", "M", "F", "M", "F"],
        }
    )
    for name in ("birth_date", "death_date"):
        df[name] = pd.to_datetime(df[name], utc=True).astype(DATE_DTYPE)
    for name in ("patient_id", "gender"):
        df[name] = df[name].astype("string")
    return df


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """The real config with windows 2010-2020 and every Phase 2 output in ``tmp_path``."""
    base = load_config()
    processed = tmp_path / "processed"
    paths = replace(
        base.paths,
        trajectories_parquet=processed / "trajectories.parquet",
        patient_index_json=processed / "patient_index.json",
        code_vocabulary_json=processed / "code_vocabulary.json",
        windows_npz=processed / "windows.npz",
        windows_json=processed / "windows.json",
        transitions_npz=processed / "transitions.npz",
        transitions_parquet=processed / "transitions.parquet",
        trajectories_summary_md=tmp_path / "reports" / "trajectories_summary.md",
    )
    trajectories = replace(base.trajectories, start_year=2010, end_year=2020)
    return replace(base, paths=paths, trajectories=trajectories)


def _active_codes(windows, year: int, patient_id: str) -> set[str]:
    """Codes active for ``patient_id`` in the window for ``year``."""
    row = windows[windows.years.index(year)][windows.patient_ids.index(patient_id)]
    return {windows.codes[j] for j in row.indices}


def test_build_trajectories_keeps_multi_code_patients_in_time_order(cfg: Config) -> None:
    traj = build_trajectories(_events(), _patients(), cfg)
    assert traj.patient_ids == ("pA", "pB", "pD")  # pC has one code, pE nothing in 2010-2020
    assert traj.codes == tuple(sorted(DESCRIPTIONS))
    assert traj.descriptions == tuple(DESCRIPTIONS[c] for c in traj.codes)
    assert list(traj.events.columns) == TRAJECTORY_COLUMNS
    assert traj.data_end == DATA_END
    sequences = traj.sequences()
    assert sequences["pA"] == [
        (pd.Timestamp("2012-03-01T09:00:00Z"), DIABETES),
        (pd.Timestamp("2015-06-01T09:00:00Z"), HYPERTENSION),
    ]
    assert [code for _, code in sequences["pD"]] == [
        DIABETES, BRONCHITIS, BRONCHITIS, PHARYNGITIS, SINUSITIS, HYPERTENSION, BRONCHITIS
    ]


def test_active_until_ends_at_stop_death_or_data_end(cfg: Config) -> None:
    events = build_trajectories(_events(), _patients(), cfg).events
    row = events.set_index(["patient_id", "code", "date"])
    a_diabetes = row.loc[("pA", DIABETES, pd.Timestamp("2012-03-01T09:00:00Z"))]
    d_diabetes = row.loc[("pD", DIABETES, pd.Timestamp("2011-01-05T08:00:00Z"))]
    d_pharyngitis = row.loc[("pD", PHARYNGITIS, pd.Timestamp("2014-04-04T08:00:00Z"))]
    d_last_bronchitis = row.loc[("pD", BRONCHITIS, pd.Timestamp("2016-04-01T08:00:00Z"))]
    assert (a_diabetes["active_until"], a_diabetes["chronic"]) == (DATA_END, True)
    assert (d_diabetes["active_until"], d_diabetes["chronic"]) == (PD_DEATH, True)
    assert d_pharyngitis["active_until"] == pd.Timestamp("2014-04-20", tz="UTC")
    assert not d_pharyngitis["chronic"]
    assert d_last_bronchitis["active_until"] == PD_DEATH  # STOP 2016-08-01 is after death
    assert events["active_until"].notna().all()


def test_windows_hold_the_set_of_active_conditions(cfg: Config) -> None:
    traj = build_trajectories(_events(), _patients(), cfg)
    windows = window_trajectories(traj, cfg)
    assert windows.years == tuple(range(2010, 2021))
    assert len(windows) == 11
    assert all(isinstance(m, sp.csr_matrix) and m.shape == (3, 5) for m in windows.matrices)
    assert all(set(m.data) <= {1.0} for m in windows.matrices)
    assert _active_codes(windows, 2010, "pD") == set()
    assert _active_codes(windows, 2011, "pD") == {DIABETES}
    assert _active_codes(windows, 2013, "pD") == {DIABETES, BRONCHITIS}  # two episodes, once
    assert _active_codes(windows, 2014, "pD") == {DIABETES, PHARYNGITIS, SINUSITIS}
    assert _active_codes(windows, 2016, "pD") == {DIABETES, HYPERTENSION, BRONCHITIS}
    assert _active_codes(windows, 2017, "pD") == set()  # died in 2016
    assert _active_codes(windows, 2014, "pA") == {DIABETES}  # chronic, carried forward
    assert _active_codes(windows, 2020, "pA") == {DIABETES, HYPERTENSION}


def test_windows_agree_with_active_in_year(cfg: Config) -> None:
    events, patients = _events(), _patients()
    traj = build_trajectories(events, patients, cfg)
    windows = window_trajectories(traj, cfg)
    cohort = events[events["patient_id"].isin(traj.patient_ids)]
    for w, year in enumerate(windows.years):
        mask = active_in_year(cohort, patients, year)
        expected = set(zip(cohort.loc[mask, "patient_id"], cohort.loc[mask, "code"]))
        coo = windows[w].tocoo()
        got = {(traj.patient_ids[i], traj.codes[j]) for i, j in zip(coo.row, coo.col)}
        assert got == expected, year


def test_windows_reject_unsupported_window(cfg: Config) -> None:
    traj = build_trajectories(_events(), _patients(), cfg)
    monthly = replace(cfg, trajectories=replace(cfg.trajectories, window="monthly"))
    with pytest.raises(ValueError, match="monthly"):
        window_trajectories(traj, monthly)


def test_patients_with_no_active_window_are_dropped_everywhere(cfg: Config) -> None:
    traj = build_trajectories(_events(), _patients(), cfg)
    assert "pE" not in traj.patient_ids
    assert "pE" not in set(traj.events["patient_id"])
    assert "pE" not in set(extract_transitions(traj)["patient_id"])
    assert window_trajectories(traj, cfg).patient_ids == traj.patient_ids


def test_patients_whose_conditions_begin_after_the_last_window_are_dropped(cfg: Config) -> None:
    early = replace(cfg, trajectories=replace(cfg.trajectories, start_year=2010, end_year=2011))
    traj = build_trajectories(_events(), _patients(), early)
    assert traj.patient_ids == ("pD",)  # pA and pB start in 2012; pD has diabetes from 2011
    assert set(traj.events["patient_id"]) == {"pD"}


def test_no_cohort_patient_is_empty_in_every_window(cfg: Config) -> None:
    windows = window_trajectories(build_trajectories(_events(), _patients(), cfg), cfg)
    active_windows = sum((np.diff(m.indptr) > 0).astype(int) for m in windows.matrices)
    assert (active_windows > 0).all()


@pytest.mark.skipif(
    not load_config().paths.windows_npz.exists(), reason="run scripts/02_build_trajectories.py"
)
def test_saved_cohort_has_no_patient_empty_in_every_window() -> None:
    windows = load_windows(load_config())
    active_windows = sum((np.diff(m.indptr) > 0).astype(int) for m in windows.matrices)
    empty = int((active_windows == 0).sum())
    assert empty == 0, f"{empty} saved patients have no active condition in any window"


def test_pre_cutoff_transitions_are_those_of_pre_cutoff_events(cfg: Config) -> None:
    traj = build_trajectories(_events(), _patients(), cfg)
    cutoff = date(2014, 6, 1)
    boundary = pd.Timestamp("2014-06-01", tz="UTC")
    kept = pre_cutoff_transitions(extract_transitions(traj), cutoff)
    assert (kept["to_date"] < boundary).all() and (kept["from_date"] < boundary).all()
    assert set(map(tuple, kept[["patient_id", "from_code", "to_code"]].values.tolist())) == {
        ("pD", DIABETES, BRONCHITIS),
        ("pD", BRONCHITIS, PHARYNGITIS),
        ("pD", BRONCHITIS, SINUSITIS),
    }
    truncated = replace(traj, events=traj.events[traj.events["date"] < boundary].reset_index(drop=True))
    pd.testing.assert_frame_equal(kept, extract_transitions(truncated))


def test_reverse_order_changes_transitions_but_not_final_window(cfg: Config) -> None:
    traj = build_trajectories(_events(), _patients(), cfg)
    transitions = extract_transitions(traj)
    per_patient = {
        p: transitions[transitions["patient_id"] == p] for p in ("pA", "pB")
    }
    assert per_patient["pA"][["from_code", "to_code"]].values.tolist() == [[DIABETES, HYPERTENSION]]
    assert per_patient["pB"][["from_code", "to_code"]].values.tolist() == [[HYPERTENSION, DIABETES]]

    a = transition_matrix(per_patient["pA"], traj.codes)
    b = transition_matrix(per_patient["pB"], traj.codes)
    d, h = traj.codes.index(DIABETES), traj.codes.index(HYPERTENSION)
    assert (a[d, h], a[h, d]) == (1, 0)
    assert (b[d, h], b[h, d]) == (0, 1)
    assert (a[d] != b[d]).nnz > 0  # the diabetes rows differ
    assert (a[h] != b[h]).nnz > 0  # and so do the hypertension rows

    windows = window_trajectories(traj, cfg)
    final = windows[len(windows) - 1]
    row_a, row_b = traj.patient_ids.index("pA"), traj.patient_ids.index("pB")
    assert set(final[row_a].indices) == set(final[row_b].indices) == {d, h}


def test_simultaneous_onsets_share_a_step_and_repeats_add_nothing(cfg: Config) -> None:
    traj = build_trajectories(_events(), _patients(), cfg)
    transitions = extract_transitions(traj)
    assert list(transitions.columns) == TRANSITION_COLUMNS
    d_pairs = transitions.loc[transitions["patient_id"] == "pD", ["from_code", "to_code"]]
    assert set(map(tuple, d_pairs.values.tolist())) == {
        (DIABETES, BRONCHITIS),
        (BRONCHITIS, PHARYNGITIS),  # pharyngitis and sinusitis share an onset, so both
        (BRONCHITIS, SINUSITIS),  # follow bronchitis, with nothing between them
        (PHARYNGITIS, HYPERTENSION),
        (SINUSITIS, HYPERTENSION),
    }
    first = transitions[(transitions["patient_id"] == "pD") & (transitions["to_code"] == BRONCHITIS)]
    assert first["to_date"].tolist() == [pd.Timestamp("2013-02-01T08:00:00Z")]  # first episode

    counts = transition_matrix(transitions, traj.codes)
    assert counts.shape == (5, 5)
    assert counts.sum() == 7  # 1 for pA, 1 for pB, 5 for pD


def test_transition_matrix_rejects_unknown_code(cfg: Config) -> None:
    traj = build_trajectories(_events(), _patients(), cfg)
    transitions = extract_transitions(traj).assign(to_code="not-a-code")
    with pytest.raises(ValueError, match="vocabulary"):
        transition_matrix(transitions, traj.codes)


def test_save_and_load_round_trip(cfg: Config) -> None:
    traj = build_trajectories(_events(), _patients(), cfg)
    windows = window_trajectories(traj, cfg)
    transitions = extract_transitions(traj)
    save_trajectories(traj, windows, transitions, cfg)

    loaded = load_windows(cfg)
    assert (loaded.years, loaded.patient_ids, loaded.codes) == (
        windows.years, windows.patient_ids, windows.codes
    )
    assert all((x != y).nnz == 0 for x, y in zip(loaded.matrices, windows.matrices))
    pd.testing.assert_frame_equal(load_transitions(cfg), transitions)
    counts = transition_matrix(transitions, traj.codes)
    assert (sp.load_npz(cfg.paths.transitions_npz) != counts).nnz == 0
    vocabulary = json.loads(cfg.paths.code_vocabulary_json.read_text(encoding="utf-8"))
    assert vocabulary[traj.codes.index(DIABETES)] == {
        "code": DIABETES, "description": DESCRIPTIONS[DIABETES]
    }
    meta = json.loads(cfg.paths.windows_json.read_text(encoding="utf-8"))
    assert (meta["n_patients"], meta["n_codes"], meta["years"][0]) == (3, 5, 2010)
    loaded_traj = load_trajectories(cfg)
    pd.testing.assert_frame_equal(loaded_traj.events, traj.events)
    assert (loaded_traj.patient_ids, loaded_traj.codes, loaded_traj.descriptions) == (
        traj.patient_ids, traj.codes, traj.descriptions
    )
    assert loaded_traj.data_end == traj.data_end


def test_write_trajectory_summary(cfg: Config) -> None:
    events, patients = _events(), _patients()
    traj = build_trajectories(events, patients, cfg)
    windows = window_trajectories(traj, cfg)
    transitions = extract_transitions(traj)
    text = write_trajectory_summary(events, patients, traj, windows, transitions, cfg)
    assert cfg.paths.trajectories_summary_md.read_text(encoding="utf-8") == text
    assert "| Patients with at least 2 distinct codes | 4 |" in text
    assert "| Of those, with no condition active in any window 2010-2020 (dropped) | 1 |" in text
    assert "| Patients in the cohort | 3 |" in text
    assert "7 of them end before the prediction cutoff (2023-01-01)" in text
    assert "| 2010 | 0 | 0.0% | n/a | n/a | n/a | 0 |" in text  # nobody active yet
    assert "| 2016 | 3 | 100.0% | 2.33 | 2 | 3 | 1 |" in text
    assert "| 2020 | 2 | 66.7% | 2.00 | 2 | 2 | 0 |" in text  # pD died in 2016
    assert "7 transitions from 3 patients, 7 distinct code pairs." in text
    assert text.count("### Patient ") == 3
    assert "| 2016-04-01 | Acute bronchitis (disorder) | 10509002 | 2016-05-10 (death) | 2016 |" in text
    assert "| 2012-03-01 | Diabetes mellitus type 2 (disorder) | 44054006 | ongoing | 2012-2020 |" in text
