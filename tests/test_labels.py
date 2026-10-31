"""Tests for ehrgraph.labels: onset, any_onset and mortality labels, censoring, leakage, split."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ehrgraph.config import Config, feature_window_years, horizon_end, load_config
from ehrgraph.data import DATE_DTYPE, EVENT_COLUMNS
from ehrgraph.features import build_features, feature_transitions, load_model_cohort
from ehrgraph.labels import (
    ANY_ONSET,
    DIED_BEFORE_CUTOFF,
    DIED_WITHOUT_ONSET,
    LOST_AFTER_ONSET,
    LOST_TO_FOLLOW_UP,
    MORTALITY,
    NEGATIVE,
    PHENOTYPE,
    POSITIVE,
    PREVALENT,
    PRIMARY_TARGET,
    SELECTED_BY_FUTURE,
    SPLIT,
    SPLITS,
    STATUS_PREFIX,
    attach_phenotypes,
    build_labels,
    label_columns,
    label_summary,
    load_labels,
    load_splits,
    observed,
    onset_columns,
    pooled_onset_status,
    save_labels,
    split_fractions,
    split_strata,
    stratified_split,
    target_columns,
    write_labels_summary,
)
from ehrgraph.trajectories import build_trajectories, window_trajectories

NAN = np.nan
GROUPS = {"g1": ("A1", "A2"), "g2": ("B",)}
# (patient, date, code). X and Y are codes outside every group.
EVENT_ROWS = [
    ("pPrev", "2019-01-01", "X"),
    ("pPrev", "2020-05-01", "A1"),  # g1 before the cutoff: masked
    ("pPrev", "2024-03-01", "B"),  # g2 in the horizon: 1
    ("pNeg", "2019-01-01", "X"),
    ("pNeg", "2020-01-01", "Y"),
    ("pLost", "2019-01-01", "X"),
    ("pLost", "2020-01-01", "Y"),
    ("pLost", "2023-05-01", "A2"),  # seen, but the patient is lost to follow-up later
    ("pDied", "2019-01-01", "X"),
    ("pDied", "2020-01-01", "Y"),
    ("pDied", "2023-03-01", "A1"),  # g1 before death: 1; g2 never: masked by rule (b)
    ("pEarly", "2019-01-01", "X"),
    ("pEarly", "2020-01-01", "Y"),
    ("pLate", "2019-01-01", "X"),
    ("pLate", "2020-01-01", "Y"),
    ("pLate", "2026-02-01", "B"),  # after the horizon: g2 is 0
    ("pEdge", "2019-01-01", "X"),
    ("pEdge", "2020-01-01", "Y"),
    ("pEdge", "2025-12-31T23:00:00Z", "A2"),  # the last hour of the horizon: g1 is 1
    ("pFuture", "2021-01-01", "X"),  # one code before the cutoff: every label masked
    ("pFuture", "2024-01-01", "A1"),
]
DEATHS = {"pDied": "2024-02-01", "pEarly": "2022-06-01", "pLate": "2026-05-01"}
# (patient, encounter start)
ENCOUNTER_ROWS = [
    ("pPrev", "2025-06-01"),
    ("pNeg", "2025-03-01"),
    ("pLost", "2024-06-01"),  # last encounter before 2025-01-01, alive: rule (a)
    ("pDied", "2024-01-15"),  # also before 2025-01-01, but died: rule (b) only
    ("pEarly", "2022-05-01"),
    ("pLate", "2025-12-01"),
    ("pLate", "2026-03-01"),
    ("pEdge", "2025-01-01"),  # exactly the follow-up date: followed
    ("pFuture", "2025-05-01"),
]
PATIENT_IDS = ("pPrev", "pNeg", "pLost", "pDied", "pEarly", "pLate", "pEdge", "pFuture")
EXPECTED = {  # patient: (any_onset, g1, g2, mortality)
    "pPrev": (1.0, NAN, 1.0, 0.0),
    "pNeg": (0.0, 0.0, 0.0, 0.0),
    "pLost": (NAN, NAN, NAN, 0.0),
    "pDied": (1.0, 1.0, NAN, 1.0),
    "pEarly": (NAN, NAN, NAN, NAN),
    "pLate": (0.0, 0.0, 0.0, 0.0),
    "pEdge": (1.0, 1.0, 0.0, 0.0),
    "pFuture": (NAN, NAN, NAN, NAN),
}


def _dates(values: list[str | None]) -> pd.Series:
    return pd.to_datetime(pd.Series(values), utc=True, format="ISO8601").astype(DATE_DTYPE)


def _events(rows: list[tuple[str, str, str]] = EVENT_ROWS) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["patient_id", "date", "code"])
    df["date"] = _dates(df["date"].tolist())
    return df.astype({"patient_id": "string", "code": "string"})


def _patients(deaths: dict[str, str] = DEATHS) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": pd.array(PATIENT_IDS, dtype="string"),
            "death_date": _dates([deaths.get(p) for p in PATIENT_IDS]),
        }
    )


def _encounters(rows: list[tuple[str, str]] = ENCOUNTER_ROWS) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["patient_id", "encounter_start"])
    df["encounter_start"] = _dates(df["encounter_start"].tolist())
    return df.astype({"patient_id": "string"})


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """The real config (cutoff 2023-01-01, horizon to 2026-01-01) with two onset groups."""
    base = load_config()
    return replace(
        base,
        paths=replace(
            base.paths,
            labels_parquet=tmp_path / "labels.parquet",
            splits_json=tmp_path / "splits.json",
        ),
        labels=replace(
            base.labels, onset_groups=GROUPS, borderline_groups=(), near_threshold_groups=()
        ),
    )


def _labels(cfg: Config, **overrides) -> pd.DataFrame:
    parts = {"events": _events(), "patients": _patients(), "encounters": _encounters()} | overrides
    return build_labels(
        parts["events"], parts["patients"], parts["encounters"], PATIENT_IDS, cfg
    ).set_index("patient_id")


def _statuses(labels: pd.DataFrame, cfg: Config, patient: str) -> list[str]:
    """``patient``'s statuses in ``label_columns`` order: any_onset, g1, g2, mortality."""
    return labels.loc[patient, [STATUS_PREFIX + c for c in label_columns(cfg)]].astype(str).tolist()


# --- Labels and censoring --------------------------------------------------------------


def test_fixture_horizon(cfg: Config) -> None:
    assert (cfg.labels.cutoff_date, horizon_end(cfg)) == (date(2023, 1, 1), date(2026, 1, 1))
    assert cfg.labels.min_last_encounter_date == date(2025, 1, 1)
    assert onset_columns(cfg) == ["onset_g1", "onset_g2"]
    assert target_columns(cfg) == [ANY_ONSET, "onset_g1", "onset_g2"]
    assert label_columns(cfg) == [ANY_ONSET, "onset_g1", "onset_g2", MORTALITY]
    assert PRIMARY_TARGET == ANY_ONSET


@pytest.mark.parametrize("patient", PATIENT_IDS)
def test_labels_follow_the_rules(cfg: Config, patient: str) -> None:
    labels = _labels(cfg)
    got = tuple(labels.loc[patient, label_columns(cfg)])
    np.testing.assert_array_equal(np.array(got, dtype=float), np.array(EXPECTED[patient]))


def test_censoring_statuses(cfg: Config) -> None:  # columns: any_onset, g1, g2, mortality
    status = _labels(cfg)[[STATUS_PREFIX + c for c in label_columns(cfg)]].astype(str)
    assert status.loc["pPrev"].tolist() == [POSITIVE, PREVALENT, POSITIVE, NEGATIVE]
    assert status.loc["pLost"].tolist() == [
        LOST_AFTER_ONSET, LOST_AFTER_ONSET, LOST_TO_FOLLOW_UP, NEGATIVE
    ]
    assert status.loc["pDied"].tolist() == [POSITIVE, POSITIVE, DIED_WITHOUT_ONSET, POSITIVE]
    assert status.loc["pEarly"].tolist() == [DIED_BEFORE_CUTOFF] * 4
    assert status.loc["pFuture"].tolist() == [SELECTED_BY_FUTURE] * 4


def test_censoring_flags(cfg: Config) -> None:
    labels = _labels(cfg)
    assert labels.index[labels["lost_to_follow_up"]].tolist() == ["pLost"]  # not pDied
    assert labels.index[labels["died_in_horizon"]].tolist() == ["pDied"]  # not pLate (2026)
    assert labels.index[labels["died_before_cutoff"]].tolist() == ["pEarly"]
    assert labels.index[labels["selected_by_future"]].tolist() == ["pFuture"]


def test_selected_by_future_patients_stay_nodes_with_every_label_masked(cfg: Config) -> None:
    labels = _labels(cfg)
    assert labels.index.tolist() == list(PATIENT_IDS)  # still a row, and so a graph node
    assert labels.loc["pFuture", "node"] == PATIENT_IDS.index("pFuture")
    assert labels.loc["pFuture", label_columns(cfg)].isna().all()  # its g1 onset is not a 1
    # With a second code before the cutoff, the same onset counts.
    rescued = _labels(cfg, events=_events(EVENT_ROWS + [("pFuture", "2022-01-01", "Y")]))
    assert not rescued.loc["pFuture", "selected_by_future"]
    assert rescued.loc["pFuture", label_columns(cfg)].tolist() == [1.0, 1.0, 0.0, 0.0]
    # Its stratum is masked, and it still gets a split.
    strata = split_strata(attach_phenotypes(labels.reset_index(), pd.Series(dtype="int64")))
    assert strata["pFuture"] == "none|masked"
    assert stratified_split(strata, split_fractions(cfg), cfg.seed).notna().all()


# --- Onsets held before the cutoff ------------------------------------------------------


def test_a_group_held_before_the_cutoff_stays_masked_when_another_of_its_codes_follows(
    cfg: Config,
) -> None:
    labels = _labels(cfg, events=_events(EVENT_ROWS + [
        ("pNeg", "2021-06-01", "A2"),  # g1's onset is its first event of A1 or A2 ...
        ("pNeg", "2024-02-01", "A1"),  # ... so a first A1 in the horizon is no new onset
        ("pPrev", "2024-06-01", "A2"),  # nor is a recurrence of the same group
    ]))
    assert _statuses(labels, cfg, "pNeg") == [NEGATIVE, PREVALENT, NEGATIVE, NEGATIVE]
    assert _statuses(labels, cfg, "pPrev") == [POSITIVE, PREVALENT, POSITIVE, NEGATIVE]


@pytest.mark.parametrize(
    ("when", "status"),
    [
        ("2022-12-31T23:59:59Z", PREVALENT),  # the last second before the cutoff
        ("2023-01-01T00:00:00Z", POSITIVE),  # the cutoff itself opens the horizon
    ],
)
def test_onset_on_either_side_of_the_cutoff(cfg: Config, when: str, status: str) -> None:
    labels = _labels(cfg, events=_events(EVENT_ROWS + [("pNeg", when, "B")]))
    assert labels.loc["pNeg", STATUS_PREFIX + "onset_g2"] == status


def test_prevalence_precedes_both_censoring_rules(cfg: Config) -> None:
    labels = _labels(cfg, events=_events(EVENT_ROWS + [
        ("pLost", "2021-01-01", "B"),  # lost to follow-up (rule a), but already had g2
        ("pDied", "2021-01-01", "B"),  # died in the horizon (rule b), but already had g2
    ]))
    assert _statuses(labels, cfg, "pLost") == [
        LOST_AFTER_ONSET, LOST_AFTER_ONSET, PREVALENT, NEGATIVE
    ]
    assert _statuses(labels, cfg, "pDied") == [POSITIVE, POSITIVE, PREVALENT, POSITIVE]


def test_every_group_held_before_the_cutoff_masks_any_onset(cfg: Config) -> None:
    labels = _labels(cfg, events=_events(EVENT_ROWS + [
        ("pNeg", "2021-01-01", "A1"), ("pNeg", "2021-01-01", "B"),
    ]))
    assert _statuses(labels, cfg, "pNeg") == [PREVALENT, PREVALENT, PREVALENT, NEGATIVE]
    assert labels.loc["pNeg", target_columns(cfg)].isna().all()
    assert labels.loc["pNeg", MORTALITY] == 0.0  # still observed: alive and followed up


# --- Censoring rules (a) and (b) ---------------------------------------------------------


@pytest.mark.parametrize(
    ("death", "died_before", "statuses"),
    [
        ("2022-12-31T23:59:59Z", True, [DIED_BEFORE_CUTOFF] * 4),
        # On the cutoff: a death in the horizon, so rule (b) masks every group without
        # an onset and mortality is 1.
        ("2023-01-01T00:00:00Z", False, [DIED_WITHOUT_ONSET] * 3 + [POSITIVE]),
    ],
)
def test_death_on_either_side_of_the_cutoff(
    cfg: Config, death: str, died_before: bool, statuses: list[str]
) -> None:
    encounters = [r for r in ENCOUNTER_ROWS if r[0] != "pNeg"] + [("pNeg", "2022-12-01")]
    labels = _labels(cfg, patients=_patients(DEATHS | {"pNeg": death}),
                     encounters=_encounters(encounters))
    assert _statuses(labels, cfg, "pNeg") == statuses
    assert bool(labels.loc["pNeg", "died_before_cutoff"]) is died_before
    assert bool(labels.loc["pNeg", "died_in_horizon"]) is not died_before
    assert not labels.loc["pNeg", "lost_to_follow_up"]  # rule (a) never judges the dead


def test_a_death_after_the_horizon_is_unseen_so_rule_a_applies(cfg: Config) -> None:
    # pLate dies 2026-05-01, after the horizon: alive through it as far as labels can see,
    # so a last encounter before 2025-01-01 masks every onset label (rule a).
    encounters = [r for r in ENCOUNTER_ROWS if r[0] != "pLate"] + [("pLate", "2024-03-01")]
    labels = _labels(cfg, encounters=_encounters(encounters))
    assert labels.loc["pLate", "lost_to_follow_up"]
    assert not labels.loc["pLate", "died_in_horizon"]
    assert _statuses(labels, cfg, "pLate") == [LOST_TO_FOLLOW_UP] * 3 + [NEGATIVE]


# --- Selected by post-cutoff codes ---------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "selected"),
    [
        (("pFuture", "2022-03-01", "X"), True),  # the same code again: one distinct code
        (("pFuture", "2023-01-01", "Y"), True),  # a second code on the cutoff is too late
        (("pFuture", "2022-12-31T23:59:59Z", "Y"), False),  # just before it: two codes
    ],
)
def test_selected_by_future_counts_distinct_codes_before_the_cutoff(
    cfg: Config, extra: tuple[str, str, str], selected: bool
) -> None:
    labels = _labels(cfg, events=_events(EVENT_ROWS + [extra]))
    assert bool(labels.loc["pFuture", "selected_by_future"]) is selected
    assert bool(labels.loc["pFuture", label_columns(cfg)].isna().all()) is selected


def test_selected_by_future_threshold_is_the_configured_minimum(cfg: Config) -> None:
    one = replace(cfg, data=replace(cfg.data, min_conditions_per_patient=1))
    labels = _labels(one)
    assert not labels["selected_by_future"].any()
    assert labels.loc["pFuture", label_columns(one)].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_selected_by_future_ranks_below_early_death_and_above_prevalence(cfg: Config) -> None:
    died = _labels(cfg, patients=_patients(DEATHS | {"pFuture": "2022-06-01"}))
    assert died.loc["pFuture", "selected_by_future"]  # still flagged
    assert _statuses(died, cfg, "pFuture") == [DIED_BEFORE_CUTOFF] * 4
    # Its one pre-cutoff code is g2's: selected by post-cutoff codes, not "had it before".
    rows = [r for r in EVENT_ROWS if r[:2] != ("pFuture", "2021-01-01")]
    held = _labels(cfg, events=_events(rows + [("pFuture", "2021-01-01", "B")]))
    assert _statuses(held, cfg, "pFuture") == [SELECTED_BY_FUTURE] * 4
    # A death in the horizon does not turn its masked mortality into a 1.
    dead_later = _labels(cfg, patients=_patients(DEATHS | {"pFuture": "2024-06-01"}))
    assert dead_later.loc["pFuture", "died_in_horizon"]
    assert _statuses(dead_later, cfg, "pFuture") == [SELECTED_BY_FUTURE] * 4


# --- The primary target: any_onset -----------------------------------------------------


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ((POSITIVE, NEGATIVE), POSITIVE),
        ((PREVALENT, POSITIVE), POSITIVE),
        ((POSITIVE, DIED_WITHOUT_ONSET), POSITIVE),  # an onset before death counts
        ((NEGATIVE, NEGATIVE), NEGATIVE),
        ((PREVALENT, NEGATIVE), NEGATIVE),  # every observed group is 0
        ((PREVALENT, PREVALENT), PREVALENT),  # every group masked
        ((PREVALENT, DIED_WITHOUT_ONSET), DIED_WITHOUT_ONSET),
        ((DIED_WITHOUT_ONSET, DIED_WITHOUT_ONSET), DIED_WITHOUT_ONSET),
        ((PREVALENT, LOST_TO_FOLLOW_UP), LOST_TO_FOLLOW_UP),
        ((LOST_AFTER_ONSET, LOST_TO_FOLLOW_UP), LOST_AFTER_ONSET),
        ((SELECTED_BY_FUTURE, SELECTED_BY_FUTURE), SELECTED_BY_FUTURE),
        ((DIED_BEFORE_CUTOFF, DIED_BEFORE_CUTOFF), DIED_BEFORE_CUTOFF),
    ],
)
def test_pooled_onset_status(statuses: tuple[str, str], expected: str) -> None:
    assert pooled_onset_status(np.array([statuses])).tolist() == [expected]


def test_any_onset_pools_the_group_labels(cfg: Config) -> None:
    labels = _labels(cfg)
    groups = labels[onset_columns(cfg)]
    pooled = labels[ANY_ONSET]
    assert pooled.isna().equals(groups.isna().all(axis=1))  # masked iff every group is
    assert (pooled == 1).equals((groups == 1).any(axis=1))
    assert (pooled == 0).equals(groups.notna().any(axis=1) & ~(groups == 1).any(axis=1))


def test_pooled_onset_status_is_row_wise_over_any_number_of_groups() -> None:
    statuses = np.array([
        [PREVALENT, NEGATIVE, POSITIVE],
        [PREVALENT, PREVALENT, NEGATIVE],
        [LOST_TO_FOLLOW_UP, PREVALENT, LOST_TO_FOLLOW_UP],
        [PREVALENT, PREVALENT, PREVALENT],
    ])
    assert pooled_onset_status(statuses).tolist() == [
        POSITIVE, NEGATIVE, LOST_TO_FOLLOW_UP, PREVALENT
    ]
    assert pooled_onset_status(statuses[:, :1]).tolist() == [
        PREVALENT, PREVALENT, LOST_TO_FOLLOW_UP, PREVALENT
    ]


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ((SELECTED_BY_FUTURE, DIED_BEFORE_CUTOFF), DIED_BEFORE_CUTOFF),
        ((LOST_AFTER_ONSET, SELECTED_BY_FUTURE), SELECTED_BY_FUTURE),
        ((DIED_WITHOUT_ONSET, LOST_TO_FOLLOW_UP), LOST_TO_FOLLOW_UP),
        ((DIED_WITHOUT_ONSET, LOST_AFTER_ONSET), LOST_AFTER_ONSET),
    ],
)
def test_pooled_masked_reason_follows_the_documented_order(
    statuses: tuple[str, str], expected: str
) -> None:
    assert pooled_onset_status(np.array([statuses])).tolist() == [expected]


def test_any_onset_counts_a_patient_with_several_new_onsets_once(cfg: Config) -> None:
    labels = _labels(cfg, events=_events(EVENT_ROWS + [
        ("pNeg", "2023-06-01", "A1"), ("pNeg", "2024-06-01", "B"),
    ]))
    assert labels.loc["pNeg", target_columns(cfg)].tolist() == [1.0, 1.0, 1.0]
    summary = label_summary(labels, [ANY_ONSET]).iloc[0]
    assert (summary.observed, summary.positives) == (5, 4)  # pNeg is one positive patient


def test_lost_to_follow_up_needs_an_encounter_on_or_after_the_date(cfg: Config) -> None:
    encounters = [r for r in ENCOUNTER_ROWS if r[0] != "pEdge"] + [("pEdge", "2024-12-31T23:59:59Z")]
    labels = _labels(cfg, encounters=_encounters(encounters))
    assert labels.loc["pEdge", "lost_to_follow_up"]
    assert np.isnan(labels.loc["pEdge", "onset_g1"])  # its onset is masked too (rule a)
    no_encounters = _labels(cfg, encounters=_encounters([r for r in ENCOUNTER_ROWS if r[0] != "pNeg"]))
    assert no_encounters.loc["pNeg", "lost_to_follow_up"]


def test_no_label_uses_data_after_the_horizon(cfg: Config) -> None:
    before = _labels(cfg)
    after = _labels(
        cfg,
        events=_events(EVENT_ROWS + [
            ("pNeg", "2026-01-01", "A1"),  # exactly at the horizon end
            ("pLate", "2027-01-01", "A2"),
            ("pLost", "2026-06-01", "B"),
        ]),
        encounters=_encounters(ENCOUNTER_ROWS + [("pLost", "2026-02-01")]),  # would rescue pLost
        patients=_patients(DEATHS | {"pNeg": "2026-01-01", "pFuture": "2027-03-01"}),
    )
    pd.testing.assert_frame_equal(before, after)


# --- Masked entries --------------------------------------------------------------------


def test_masked_entries_are_nan_and_never_counted(cfg: Config) -> None:
    labels = _labels(cfg)
    for column in label_columns(cfg):
        status = labels[STATUS_PREFIX + column].astype(str)
        assert labels[column].isna().equals(~status.isin([POSITIVE, NEGATIVE])), column
    summary = label_summary(labels, label_columns(cfg)).set_index("label")
    # g1: pNeg 0, pDied 1, pLate 0, pEdge 1; pPrev, pLost, pEarly and pFuture masked.
    assert summary.loc["onset_g1", ["observed", "positives", "negatives", "masked"]].tolist() == [
        4, 2, 2, 4
    ]
    assert summary.loc["onset_g1", "rate"] == pytest.approx(2 / 4)
    assert observed(labels, "onset_g1").sum() == 4
    # any_onset: pPrev 1, pNeg 0, pDied 1, pLate 0, pEdge 1; pLost, pEarly and pFuture masked.
    assert summary.loc[ANY_ONSET, ["observed", "positives", "negatives", "masked"]].tolist() == [
        5, 3, 2, 3
    ]
    # Every rate is positives / (positives + negatives) by status: masked entries never count.
    for column in label_columns(cfg):
        status = labels[STATUS_PREFIX + column].astype(str)
        positives, negatives = int((status == POSITIVE).sum()), int((status == NEGATIVE).sum())
        assert summary.loc[column, "rate"] == pytest.approx(positives / (positives + negatives))
        assert summary.loc[column, "observed"] + summary.loc[column, "masked"] == len(labels)


def test_label_summary_by_split_adds_up(cfg: Config) -> None:
    labels = _labels(cfg).reset_index()
    labels[SPLIT] = ["train", "train", "val", "test", "train", "val", "test", "train"]
    by_split = label_summary(labels, label_columns(cfg), by=SPLIT)
    overall = label_summary(labels, label_columns(cfg)).set_index("label")
    totals = by_split.groupby("label", sort=False)[["observed", "positives", "masked"]].sum()
    pd.testing.assert_frame_equal(totals, overall[["observed", "positives", "masked"]])


# --- No input feature uses data after the cutoff ----------------------------------------

_FEATURE_EVENTS = [  # (patient, date, stop, code): every patient has >= 2 codes by 2022
    ("q1", "2019-03-01", None, "c1"),
    ("q1", "2020-05-01", "2020-06-01", "c2"),
    ("q1", "2021-02-01", None, "c3"),
    ("q2", "2020-01-10", None, "c2"),
    ("q2", "2021-07-01", "2021-08-01", "c4"),
    ("q2", "2022-03-01", None, "c1"),
    ("q3", "2018-04-01", None, "c3"),
    ("q3", "2020-09-01", None, "c4"),
    ("q4", "2020-02-01", "2020-03-01", "c1"),
    ("q4", "2021-05-01", None, "c2"),
    ("q4", "2022-08-01", "2022-09-01", "c4"),
]
_POST_CUTOFF = [  # new episodes and new conditions, all on or after 2023-01-01
    ("q1", "2023-04-01", None, "c4"),
    ("q2", "2024-05-01", "2024-06-01", "c2"),
    ("q3", "2024-01-01", None, "c1"),
    ("q3", "2025-06-01", None, "c2"),
    ("q4", "2023-02-01", None, "c3"),
]


def _clinical_events(rows: list[tuple[str, str, str | None, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["patient_id", "date", "stop", "code"])
    df["date"], df["stop"] = _dates(df["date"].tolist()), _dates(df["stop"].tolist())
    df["description"] = df["code"] + " (disorder)"
    df["encounter_id"] = [f"e{i}" for i in range(len(df))]
    for name in ("patient_id", "code", "description", "encounter_id"):
        df[name] = df[name].astype("string")
    return df[EVENT_COLUMNS]


def test_no_input_feature_uses_data_after_the_cutoff() -> None:
    base = load_config()
    cfg = replace(
        base,
        trajectories=replace(base.trajectories, start_year=2020, end_year=2025),
        features=replace(base.features, redundant_codes=(), transition_svd_components=2),
    )
    patients = pd.DataFrame(
        {
            "patient_id": pd.array(["q1", "q2", "q3", "q4"], dtype="string"),
            "birth_date": _dates(["1960-01-01"] * 4),
            "death_date": _dates([None] * 4),
            "gender": pd.array(["F", "M", "F", "M"], dtype="string"),
        }
    )

    def build(rows: list[tuple[str, str, str | None, str]]):
        trajectories = build_trajectories(_clinical_events(rows), patients, cfg)
        windows = window_trajectories(trajectories, cfg)
        return build_features(windows, feature_transitions(trajectories, cfg), cfg)

    before, after = build(_FEATURE_EVENTS), build(_FEATURE_EVENTS + _POST_CUTOFF)
    assert before.feature_years == feature_window_years(cfg) == (2020, 2021, 2022)
    assert (before.patient_ids, before.codes) == (after.patient_ids, after.codes)
    for year in before.feature_years:
        t = before.years.index(year)
        np.testing.assert_array_equal(before.matrices[t].toarray(), after.matrices[t].toarray())
    t = before.years.index(2023)  # the label period does see the new events
    assert not np.array_equal(before.matrices[t].toarray(), after.matrices[t].toarray())


# --- Phenotypes and the split -----------------------------------------------------------


def test_split_strata_combine_phenotype_and_any_onset(cfg: Config) -> None:
    labels = _labels(cfg).reset_index()
    clusters = pd.Series([1, 1, 2, 2, 3, 3, 1], index=list(PATIENT_IDS[:7]))  # pFuture unclustered
    strata = split_strata(attach_phenotypes(labels, clusters))
    assert strata["pPrev"] == "1|1" and strata["pNeg"] == "1|0"
    assert strata["pDied"] == "2|1"  # any_onset 1 (mortality is not used)
    assert strata["pLost"] == "2|masked" and strata["pEarly"] == "3|masked"
    assert strata["pFuture"] == "none|masked"
    assert attach_phenotypes(labels, clusters)[PHENOTYPE].isna().sum() == 1


def test_stratified_split_divides_each_stratum() -> None:
    ids = [f"p{i:03d}" for i in range(125)]
    strata = pd.Series(["a"] * 100 + ["b"] * 20 + ["c"] * 5, index=ids)
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    split = stratified_split(strata, fractions, seed=42)
    assert split.notna().all() and split.index.equals(strata.index)
    counts = pd.crosstab(strata, split)[list(SPLITS)]
    assert counts.loc["a"].tolist() == [70, 15, 15]
    assert counts.loc["b"].tolist() == [14, 3, 3]
    assert counts.loc["c"].tolist() == [3, 1, 1]  # quotas 3.5 / 0.75 / 0.75
    assert split.equals(stratified_split(strata, fractions, seed=42))
    assert not split.equals(stratified_split(strata, fractions, seed=7))


def test_saved_splits_do_not_overlap(cfg: Config) -> None:
    labels = attach_phenotypes(_labels(cfg).reset_index(), pd.Series(dtype="int64"))
    split = stratified_split(split_strata(labels), split_fractions(cfg), cfg.seed)
    save_labels(labels, split, cfg)
    splits = load_splits(cfg)
    members = [p for name in SPLITS for p in splits[name]]
    assert sorted(members) == sorted(PATIENT_IDS) and len(members) == len(set(members))
    saved = load_labels(cfg)
    assert saved[SPLIT].tolist() == split.tolist()
    payload = json.loads(cfg.paths.splits_json.read_text(encoding="utf-8"))
    payload["val"].append(payload["train"][0])
    cfg.paths.splits_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="more than one split"):
        load_splits(cfg)


@pytest.mark.parametrize(
    ("size", "counts"),
    [  # train / val / test for fractions 0.7 / 0.15 / 0.15
        (1, [1, 0, 0]),
        (2, [2, 0, 0]),  # remainders 0.4 / 0.3 / 0.3
        (3, [2, 1, 0]),  # remainders 0.1 / 0.45 / 0.45: the tie goes to val, listed first
        (4, [3, 1, 0]),  # remainders 0.8 / 0.6 / 0.6
        (6, [4, 1, 1]),  # remainders 0.2 / 0.9 / 0.9
    ],
)
def test_stratified_split_gives_leftovers_to_the_largest_remainders(
    size: int, counts: list[int]
) -> None:
    strata = pd.Series("s", index=[f"p{i}" for i in range(size)])
    split = stratified_split(strata, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    assert split.value_counts().reindex(list(SPLITS), fill_value=0).tolist() == counts


def test_stratified_split_breaks_ties_in_the_order_of_the_fractions() -> None:
    strata = pd.Series("s", index=["p0", "p1", "p2"])  # quotas 2.1 / 0.45 / 0.45
    val_first = stratified_split(strata, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    test_first = stratified_split(strata, {"train": 0.7, "test": 0.15, "val": 0.15}, seed=42)
    assert sorted(val_first) == ["train", "train", "val"]
    assert sorted(test_first) == ["test", "train", "train"]


@pytest.mark.parametrize("seed", [42, 7])
def test_stratified_split_counts_stay_within_one_patient_of_each_quota(seed: int) -> None:
    sizes = np.random.default_rng(seed).integers(1, 60, size=15)
    strata = pd.Series(np.repeat([f"s{i:02d}" for i in range(len(sizes))], sizes),
                       index=[f"p{i:04d}" for i in range(sizes.sum())])
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    split = stratified_split(strata, fractions, seed)
    assert split.notna().all()
    counts = pd.crosstab(strata, split).reindex(columns=list(SPLITS), fill_value=0)
    assert counts.sum(axis=1).tolist() == sizes.tolist()
    quotas = np.outer(sizes, [fractions[name] for name in SPLITS])
    assert (np.abs(counts.to_numpy() - quotas) < 1).all()


def test_stratified_split_does_not_depend_on_the_order_of_the_strata() -> None:
    a, b = [f"a{i}" for i in range(7)], [f"b{i}" for i in range(7)]
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    blocks = stratified_split(pd.Series(["a"] * 7 + ["b"] * 7, index=a + b), fractions, 42)
    flipped = stratified_split(pd.Series(["b"] * 7 + ["a"] * 7, index=b + a), fractions, 42)
    mixed_ids = [p for pair in zip(a, b) for p in pair]
    mixed = stratified_split(pd.Series([p[0] for p in mixed_ids], index=mixed_ids), fractions, 42)
    assert blocks.equals(flipped.loc[blocks.index])
    assert blocks.equals(mixed.loc[blocks.index])


def test_save_labels_needs_every_patient_in_exactly_one_split(cfg: Config) -> None:
    labels = attach_phenotypes(_labels(cfg).reset_index(), pd.Series(dtype="int64"))
    split = stratified_split(split_strata(labels), split_fractions(cfg), cfg.seed)
    unassigned = split.copy()
    unassigned.iloc[0] = pd.NA
    for bad in (unassigned, split.iloc[::-1], split.iloc[1:]):
        with pytest.raises(ValueError, match="exactly one split"):
            save_labels(labels, bad, cfg)
    assert not cfg.paths.labels_parquet.exists() and not cfg.paths.splits_json.exists()


# --- The labels report -------------------------------------------------------------------

DESCRIPTIONS = {
    "A1": "Alpha one (disorder)", "A2": "Alpha two (finding)", "B": "Beta (disorder)",
    "X": "Ex (disorder)", "Y": "Why",
}


def _report(cfg: Config, tmp_path: Path) -> str:
    """``write_labels_summary`` of the fixture, g2 borderline and g1 near the threshold."""
    cfg = replace(
        cfg,
        paths=replace(cfg.paths, labels_summary_md=tmp_path / "labels_summary.md"),
        labels=replace(cfg.labels, borderline_groups=("g2",), near_threshold_groups=("g1",)),
    )
    clusters = pd.Series([1, 1, 2, 2, 3, 3, 1], index=list(PATIENT_IDS[:7]))  # pFuture unclustered
    labels = attach_phenotypes(_labels(cfg).reset_index(), clusters)
    labels[SPLIT] = ["train", "train", "val", "test", "train", "val", "test", "train"]
    phenotypes = pd.DataFrame({"cluster": [1, 1, 2], "rank": [2, 1, 1], "code": ["Y", "X", "A1"]})
    text = write_labels_summary(labels, phenotypes, DESCRIPTIONS, cfg)
    assert cfg.paths.labels_summary_md.read_text(encoding="utf-8") == text
    return text


def test_report_counts_the_censoring_rules(cfg: Config, tmp_path: Path) -> None:
    text = _report(cfg, tmp_path)
    assert "| Died before the cutoff: every label masked | 1 | all |" in text
    assert ("| Selected by post-cutoff codes (fewer than 2 distinct codes before the cutoff): "
            "every label masked | 1 | all |") in text
    # (a) pLost: g1 seen in the horizon and then lost, g2 never seen: 2 entries, 1 an onset.
    assert ("| (a) Alive through the horizon, last encounter before 2025-01-01: every onset "
            "label masked | 1 | 2 |") in text
    assert "Rule (a) also masks 1 onsets that fall in the horizon" in text
    # (b) pDied: g1 before death kept as a 1, g2 masked.
    assert "| (b) Died in the horizon: onset labels without that onset masked | 1 | 1 |" in text
    assert "rule (b) keeps the 1 onsets seen before death as positives" in text


def test_report_group_table_and_review_flags(cfg: Config, tmp_path: Path) -> None:
    text = _report(cfg, tmp_path)
    assert "| g1 | Alpha one (disorder) (A1); Alpha two (finding) (A2) |" in text
    # Had it before, new onsets, negatives, rate, (a) lost (after onset), (b) died,
    # died before, selected by future, review (min_onsets 30).
    assert "| g1 | 1 | 2 | 2 | 50.0% | 1 (1) | 0 | 1 | 1 | keep, **< 30 onsets** |" in text
    assert ("| g2 | 0 | 1 | 3 | 25.0% | 1 (0) | 1 | 1 | 1 | "
            "keep, **borderline**, **< 30 onsets** |") in text
    assert "`g1` (2 onsets) is also near the threshold." in text


def test_report_any_onset_mortality_split_and_phenotypes(cfg: Config, tmp_path: Path) -> None:
    text = _report(cfg, tmp_path)
    assert "Positives / observed: 3 / 5 (60.0%); masked: 3. 0 of the positives" in text
    for row in ("| positive | 1 | 3 |", "| negative | 0 | 2 |",
                "| died before cutoff | masked | 1 |",
                "| selected by post-cutoff codes | masked | 1 |",
                "| lost to follow-up after onset | masked | 1 |"):
        assert row in text
    assert "had every group before cutoff" not in text  # no such patient: no row
    assert "With mortality masked too, so every label: 2" in text  # pEarly and pFuture
    assert "Died 2023-01-01 to 2026-01-01 (exclusive): 1 (16.7%). Masked: 2." in text
    assert "| Patients | 4 | 2 | 2 |" in text
    assert "| any_onset (primary) | 1 / 2 (50.0%) | 0 / 1 (0.0%) | 2 / 2 (100.0%) |" in text
    assert "| mortality (descriptive only) | 0 / 2 (0.0%) | 0 / 2 (0.0%) | 1 / 2 (50.0%) |" in text
    assert "| 1 | 3 | Ex, Why |" in text  # in rank order, semantic tags dropped
    assert "| 2 | 2 | Alpha one |" in text
    assert "| 3 | 2 | (no chronic condition common enough to list) |" in text


# --- The saved project data ---------------------------------------------------------------

_PROJECT = load_config()
_needs_labels = pytest.mark.skipif(
    not _PROJECT.paths.labels_parquet.exists(), reason="run scripts/05_build_labels.py"
)


@_needs_labels
def test_saved_labels_cover_the_model_cohort_once() -> None:
    labels = load_labels(_PROJECT)
    patient_ids, _ = load_model_cohort(_PROJECT)
    assert labels["patient_id"].tolist() == list(patient_ids)
    splits = load_splits(_PROJECT)
    members = [p for name in SPLITS for p in splits[name]]
    assert sorted(members) == sorted(patient_ids) and len(members) == len(set(members))
    for column in label_columns(_PROJECT):
        status = labels[STATUS_PREFIX + column].astype(str)
        assert labels[column].isna().equals(~status.isin([POSITIVE, NEGATIVE])), column


@_needs_labels
def test_saved_labels_mask_selected_by_future_and_pool_any_onset() -> None:
    labels = load_labels(_PROJECT)
    future = labels["selected_by_future"].to_numpy()
    assert future.any()
    assert labels.loc[future, label_columns(_PROJECT)].isna().all().all()
    groups = labels[onset_columns(_PROJECT)]
    assert labels[ANY_ONSET].isna().equals(groups.isna().all(axis=1))
    assert (labels[ANY_ONSET] == 1).equals((groups == 1).any(axis=1))
