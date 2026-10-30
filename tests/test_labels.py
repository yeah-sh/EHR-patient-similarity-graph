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
        labels=replace(base.labels, onset_groups=GROUPS, borderline_groups=()),
    )


def _labels(cfg: Config, **overrides) -> pd.DataFrame:
    parts = {"events": _events(), "patients": _patients(), "encounters": _encounters()} | overrides
    return build_labels(
        parts["events"], parts["patients"], parts["encounters"], PATIENT_IDS, cfg
    ).set_index("patient_id")


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
