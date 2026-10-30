"""Tests for ehrgraph.data on a tiny hand-made Synthea extract of 3 patients."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from ehrgraph.config import CodeFilterConfig, Config, load_config
from ehrgraph.data import (
    EVENT_COLUMNS,
    HARD_CHECKS,
    SOFT_CHECKS,
    active_in_year,
    active_patients_per_year,
    alive_patients_per_year,
    build_code_dictionary,
    build_events,
    classify_codes,
    filter_clinical_events,
    load_raw,
    merge_codes,
    patient_ages,
    save_code_dictionary,
    save_events,
    top_conditions,
    validate,
    write_clinical_summary,
    write_data_summary,
)

# Older-Synthea layout: no MIDDLE, FIPS or INCOME in patients and no SYSTEM in
# conditions. FIRST, LAST, RACE and ENCOUNTERCLASS are present but never read.
PATIENTS_CSV = """\
Id,BIRTHDATE,DEATHDATE,FIRST,LAST,GENDER,RACE
p1,1980-05-01,,Ana,Diaz,F,white
p2,1950-01-15,2020-06-01,Ben,Okafor,M,black
p3,2001-11-30,,Cai,Lee,F,asian
"""
# e4 falls on p2's day of death, which is allowed.
ENCOUNTERS_CSV = """\
Id,START,STOP,PATIENT,ENCOUNTERCLASS
e0,1996-03-15T09:00:00Z,1996-03-15T09:20:00Z,p2,wellness
e6,1997-05-01T11:00:00Z,1997-05-01T11:15:00Z,p1,ambulatory
e1,2015-03-01T10:00:00Z,2015-03-01T10:30:00Z,p1,wellness
e2,2018-07-15T09:30:00Z,2018-07-15T10:00:00Z,p1,ambulatory
e3,2019-01-10T08:00:00Z,2019-01-10T08:15:00Z,p2,ambulatory
e4,2020-06-01T14:00:00Z,2020-06-01T15:00:00Z,p2,emergency
e5,2020-02-02T12:00:00Z,2020-02-02T12:20:00Z,p3,wellness
"""
# Row 2 is an exact duplicate of row 1. Encounter e9 does not exist. Code 38341003
# has two wordings. p2's prediabetes (untagged, whitelisted) is chronic: diagnosed
# in 1996 with no STOP. p1's pharyngitis resolved in 1998. Employment is social.
CONDITIONS_CSV = """\
START,STOP,PATIENT,ENCOUNTER,CODE,DESCRIPTION
2015-03-01,,p1,e1,44054006,Diabetes mellitus type 2 (disorder)
2015-03-01,,p1,e1,44054006,Diabetes mellitus type 2 (disorder)
1997-05-01,1998-02-01,p1,e6,195662009,Acute viral pharyngitis (disorder)
2018-07-15,,p1,e2,38341003,Hypertension
2018-07-15,,p1,e2,160903007,Full-time employment (finding)
1996-03-15,,p2,e0,15777000,Prediabetes
2019-01-10,,p2,e3,38341003,Essential hypertension (disorder)
2019-01-09,2019-01-20,p2,e9,10509002,Acute bronchitis (disorder)
2020-06-01,,p2,e4,10509002,Acute bronchitis (disorder)
2020-02-02,,p3,e5,38341003,Essential hypertension (disorder)
"""
NOW = pd.Timestamp("2026-01-01", tz="UTC")
N_EVENTS = 9  # 10 condition rows minus the exact duplicate


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """The real config with every input and output path moved into ``tmp_path``."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name, text in [
        ("patients", PATIENTS_CSV),
        ("conditions", CONDITIONS_CSV),
        ("encounters", ENCOUNTERS_CSV),
    ]:
        (raw_dir / f"{name}.csv").write_text(text, encoding="utf-8")
    base = load_config()
    paths = replace(
        base.paths,
        raw_dir=raw_dir,
        processed_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "reports" / "figures",
        patients_csv=raw_dir / "patients.csv",
        conditions_csv=raw_dir / "conditions.csv",
        encounters_csv=raw_dir / "encounters.csv",
        events_parquet=tmp_path / "processed" / "events.parquet",
        events_clinical_parquet=tmp_path / "processed" / "events_clinical.parquet",
        code_dictionary_csv=tmp_path / "processed" / "code_dictionary.csv",
        data_summary_md=tmp_path / "reports" / "data_summary.md",
        clinical_summary_md=tmp_path / "reports" / "clinical_summary.md",
        events_per_patient_fig=tmp_path / "reports" / "figures" / "events.png",
        top_conditions_fig=tmp_path / "reports" / "figures" / "top.png",
    )
    return replace(base, paths=paths)


def test_load_raw_reads_only_needed_columns(cfg: Config) -> None:
    raw = load_raw(cfg)
    assert list(raw.patients.columns) == ["patient_id", "birth_date", "death_date", "gender"]
    assert list(raw.conditions.columns) == [
        "start", "stop", "patient_id", "encounter_id", "code", "description"
    ]
    assert list(raw.encounters.columns) == ["encounter_id", "encounter_start"]
    assert str(raw.encounters["encounter_start"].dt.tz) == "UTC"
    assert str(raw.conditions["stop"].dt.tz) == "UTC"
    assert raw.patients["death_date"].notna().sum() == 1
    assert raw.conditions["code"].iloc[0] == "44054006"  # codes stay strings


def test_load_raw_rejects_missing_required_column(cfg: Config) -> None:
    cfg.paths.conditions_csv.write_text(
        CONDITIONS_CSV.replace("CODE", "SNOMED"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="CODE"):
        load_raw(cfg)


def test_build_events_dates_by_encounter_with_fallback(
    cfg: Config, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="ehrgraph.data"):
        events = build_events(load_raw(cfg))
    assert list(events.columns) == EVENT_COLUMNS
    date = events.set_index("encounter_id")["date"]
    assert date["e1"] == pd.Timestamp("2015-03-01T10:00:00Z")
    assert date["e9"] == pd.Timestamp("2019-01-09", tz="UTC")  # unknown encounter: START
    assert "1 of 10 conditions had no matching encounter" in caplog.text


def test_build_events_drops_exact_duplicates(cfg: Config) -> None:
    events = build_events(load_raw(cfg))
    assert len(events) == N_EVENTS
    assert not events.duplicated().any()
    assert events["patient_id"].is_monotonic_increasing


def test_code_dictionary_uses_most_frequent_description(cfg: Config) -> None:
    dictionary = build_code_dictionary(build_events(load_raw(cfg)))
    assert dict(zip(dictionary["code"], dictionary["description"])) == {
        "10509002": "Acute bronchitis (disorder)",
        "15777000": "Prediabetes",
        "160903007": "Full-time employment (finding)",
        "195662009": "Acute viral pharyngitis (disorder)",
        "38341003": "Essential hypertension (disorder)",
        "44054006": "Diabetes mellitus type 2 (disorder)",
    }
    save_code_dictionary(dictionary, cfg.paths.code_dictionary_csv)
    back = pd.read_csv(cfg.paths.code_dictionary_csv, dtype=str)
    assert back.to_dict("list") == dictionary.astype(str).to_dict("list")


def test_validate_passes_on_clean_fixture(cfg: Config) -> None:
    raw = load_raw(cfg)
    checks = validate(build_events(raw), raw.patients, reference_date=NOW)
    assert checks["patients"] == 3
    assert checks["events"] == N_EVENTS
    # Includes e4, which is on p2's day of death and so is not "after death".
    assert all(checks[name] == 0 for name in HARD_CHECKS + SOFT_CHECKS)


def test_validate_raises_on_duplicate_patient_ids(cfg: Config) -> None:
    raw = load_raw(cfg)
    patients = pd.concat([raw.patients, raw.patients.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate_patient_ids"):
        validate(build_events(raw), patients, reference_date=NOW)


def test_validate_raises_on_unknown_patient(cfg: Config) -> None:
    raw = load_raw(cfg)
    patients = raw.patients[raw.patients["patient_id"] != "p3"]
    with pytest.raises(ValueError, match="events_unknown_patient"):
        validate(build_events(raw), patients, reference_date=NOW)


def test_validate_logs_but_keeps_soft_issues(
    cfg: Config, caplog: pytest.LogCaptureFixture
) -> None:
    raw = load_raw(cfg)
    events = build_events(raw)
    after_death = events.iloc[[0]].assign(
        patient_id="p2", date=pd.Timestamp("2020-06-02", tz="UTC")
    )
    in_future = events.iloc[[0]].assign(date=NOW + pd.Timedelta(days=1))
    events = pd.concat([events, after_death, in_future], ignore_index=True)
    with caplog.at_level(logging.WARNING, logger="ehrgraph.data"):
        checks = validate(events, raw.patients, reference_date=NOW)
    assert checks["events_after_death"] == 1
    assert checks["events_in_future"] == 1
    assert "events_after_death: 1" in caplog.text
    assert "events_in_future: 1" in caplog.text


def test_validate_default_reference_is_latest_event_not_today(cfg: Config) -> None:
    raw = load_raw(cfg)
    events = build_events(raw)
    # Synthea can date events a few days past the run date; that is not an error.
    past_today = events.iloc[[0]].assign(
        date=pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=4)
    )
    events = pd.concat([events, past_today], ignore_index=True)
    assert validate(events, raw.patients)["events_in_future"] == 0


def test_save_events_round_trip(cfg: Config) -> None:
    events = build_events(load_raw(cfg))
    save_events(events, cfg.paths.events_parquet)
    pd.testing.assert_frame_equal(pd.read_parquet(cfg.paths.events_parquet), events)


def test_patient_ages_use_death_date_for_deceased(cfg: Config) -> None:
    raw = load_raw(cfg)
    ages = patient_ages(raw.patients, pd.Timestamp("2020-06-01", tz="UTC"))
    assert ages.tolist() == [40, 70, 18]


def test_top_conditions_rank_by_patients_then_events(cfg: Config) -> None:
    events = build_events(load_raw(cfg))
    top = top_conditions(events, build_code_dictionary(events), n=2)
    assert top["code"].tolist() == ["38341003", "10509002"]
    assert top["patients"].tolist() == [3, 1]
    assert top["events"].tolist() == [3, 2]


def test_write_data_summary(cfg: Config) -> None:
    raw = load_raw(cfg)
    events = build_events(raw)
    checks = validate(events, raw.patients, reference_date=NOW)
    text = write_data_summary(events, raw.patients, build_code_dictionary(events), checks, cfg)
    assert cfg.paths.data_summary_md.read_text(encoding="utf-8") == text
    assert cfg.paths.events_per_patient_fig.stat().st_size > 0
    assert cfg.paths.top_conditions_fig.stat().st_size > 0
    assert "| Deceased | 1 | 33.3% |" in text
    assert "Essential hypertension (disorder)" in text
    assert "(figures/events.png)" in text


# --- Clinical code filter -------------------------------------------------------

FILTER = CodeFilterConfig(
    keep_suffixes=("(disorder)",),
    keep_descriptions=("prediabetes (finding)", "Not in the data"),
)
# Real Synthea descriptions the project config must drop or keep.
SOCIAL_AND_ADMIN = [
    "Stress (finding)",
    "Full-time employment (finding)",
    "Part-time employment (finding)",
    "Not in labor force (finding)",
    "Unemployed (finding)",
    "Received higher education (finding)",
    "Educated to high school level (finding)",
    "Social isolation (finding)",
    "Limited social contact (finding)",
    "Victim of intimate partner abuse (finding)",
    "Reports of violence in the environment (finding)",
    "Risk activity involvement (finding)",
    "Medication review due (situation)",
    "Housing unsatisfactory (finding)",
    "Homeless (finding)",
    "Lack of access to transportation (finding)",
    "Transport problem (finding)",
    "Has a criminal record (finding)",
    "Refugee (person)",
]
CLINICAL = [
    "Gingivitis (disorder)",
    "Metabolic syndrome X (disorder)",
    "Prediabetes (finding)",
    "Prediabetes",
    "Body mass index 30+ - obesity (finding)",
    "Body mass index 40+ - severely obese (finding)",
    "Hypertension",
    "Hyperlipidemia",
    "Coronary Heart Disease",
    # Added back by the clinical review: substance use and major cardiac history.
    "Smokes tobacco daily (finding)",  # its untagged wording shares the code
    "Unhealthy alcohol drinking behavior (finding)",
    "Misuses drugs (finding)",
    "History of myocardial infarction (situation)",
    "History of coronary artery bypass grafting (situation)",
    "History of cardiac arrest (situation)",
    "History of aortic valve replacement (situation)",
]
# Kept dropped by the clinical review.
REVIEWED_DROPS = [
    "Normal pregnancy (finding)",
    "Abnormal findings diagnostic imaging heart+coronary circulat (finding)",
    "Fever (finding)",
    "Cough (finding)",
    "Fatigue (finding)",
    "Fractured dental filling (finding)",
    "Leaking dental filling (finding)",
    "At risk for suicide (finding)",
    "History of appendectomy (situation)",  # history-of, but not cardiac
]


def _wordings(*rows: tuple[str, str]) -> pd.DataFrame:
    """A minimal events table: one row per (code, description) for patient p1."""
    return pd.DataFrame(rows, columns=["code", "description"]).assign(patient_id="p1")


def test_classify_codes_keeps_disorders_and_whitelist(caplog: pytest.LogCaptureFixture) -> None:
    events = _wordings(
        ("1", "Gingivitis (disorder)"),
        ("2", "Prediabetes (FINDING)"),  # whitelisted, case-insensitive
        ("3", "Stress (finding)"),  # social
        ("4", "Stroke"),  # untagged wording of ...
        ("4", "Cerebrovascular accident (disorder)"),  # ... a disorder code
        ("5", "Fever (finding)"),  # symptom
    )
    with caplog.at_level(logging.WARNING, logger="ehrgraph.data"):
        codes = classify_codes(events, FILTER)
    assert dict(zip(codes["code"], codes["rule"])) == {
        "1": "suffix", "2": "whitelist", "3": "dropped", "4": "suffix", "5": "dropped"
    }
    assert codes["kept"].tolist() == [True, True, False, True, False]
    assert "Not in the data" in caplog.text  # unmatched whitelist entry is reported


def test_config_code_filter_drops_social_and_keeps_clinical() -> None:
    descriptions = SOCIAL_AND_ADMIN + REVIEWED_DROPS + CLINICAL
    events = _wordings(*[(str(i), d) for i, d in enumerate(descriptions)])
    kept = classify_codes(events, load_config().data.code_filter).set_index("code")["kept"]
    by_description = {d: bool(kept[str(i)]) for i, d in enumerate(descriptions)}
    assert [d for d in SOCIAL_AND_ADMIN + REVIEWED_DROPS if by_description[d]] == []
    assert [d for d in CLINICAL if not by_description[d]] == []


def test_merge_codes_renames_prediabetes_on_fixture(cfg: Config) -> None:
    events = build_events(load_raw(cfg))
    merged = merge_codes(events, cfg.data.code_merge)
    assert list(merged.columns) == EVENT_COLUMNS
    assert len(merged) == N_EVENTS
    assert "15777000" not in set(merged["code"])
    prediabetes = merged[merged["code"] == "714628002"]
    assert prediabetes["patient_id"].tolist() == ["p2"]
    assert prediabetes["description"].tolist() == ["Prediabetes"]  # keeps its own wording
    assert merged["patient_id"].is_monotonic_increasing


def test_merge_codes_drops_renamed_repeat_of_canonical_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def event(code: str, description: str, encounter: str) -> dict[str, object]:
        return {"patient_id": "p1", "date": pd.Timestamp("2020-01-01", tz="UTC"),
                "stop": pd.NaT, "code": code, "description": description,
                "encounter_id": encounter}

    events = pd.DataFrame(
        [
            event("old", "Old wording", "e1"),  # repeats the canonical event below
            event("new", "New wording (finding)", "e1"),
            event("old", "Old wording", "e2"),  # a separate encounter, so it stays
        ],
        columns=EVENT_COLUMNS,
    )
    with caplog.at_level(logging.WARNING, logger="ehrgraph.data"):
        merged = merge_codes(events, {"old": "new", "typo": "new"})
    assert merged[["code", "description", "encounter_id"]].values.tolist() == [
        ["new", "New wording (finding)", "e1"],
        ["new", "Old wording", "e2"],
    ]
    assert "typo" in caplog.text  # a merge code matching no event is reported


def test_merge_codes_warns_but_does_not_fail_when_old_code_absent(
    cfg: Config, caplog: pytest.LogCaptureFixture
) -> None:
    # Data from a single newer Synthea version has no 15777000 to merge.
    events = build_events(load_raw(cfg))
    events = events[events["code"] != "15777000"].reset_index(drop=True)
    with caplog.at_level(logging.WARNING, logger="ehrgraph.data"):
        merged = merge_codes(events, cfg.data.code_merge)
    pd.testing.assert_frame_equal(merged, events)
    assert "15777000" in caplog.text
    assert all(r.levelno == logging.WARNING for r in caplog.records)


def test_filter_clinical_events_on_fixture(cfg: Config) -> None:
    events = build_events(load_raw(cfg))
    clinical = filter_clinical_events(events, cfg.data.code_filter)
    assert list(clinical.columns) == EVENT_COLUMNS
    assert len(clinical) == N_EVENTS - 1
    assert "160903007" not in set(clinical["code"])  # Full-time employment (finding)
    assert "15777000" in set(clinical["code"])  # untagged "Prediabetes", whitelisted


def test_write_clinical_summary(cfg: Config) -> None:
    raw = load_raw(cfg)
    events = build_events(raw)
    merged = merge_codes(events, cfg.data.code_merge)
    clinical = filter_clinical_events(merged, cfg.data.code_filter)
    text = write_clinical_summary(
        events, clinical, raw.patients, build_code_dictionary(events), cfg
    )
    assert cfg.paths.clinical_summary_md.read_text(encoding="utf-8") == text
    assert "| 160903007 | Full-time employment (finding) | 1 | 1 |" in text  # dropped list
    assert "| Events | 8 | 1 | 9 | 88.9% |" in text
    # The fixture has only the old prediabetes code, so the canonical one has no description.
    assert "| 15777000 | Prediabetes | 714628002 | not in data | 1 | 1 |" in text
    assert "| Codes | 5 | 1 | 6 | 83.3% |" in text  # the renamed code still counts


# --- Active conditions per year -------------------------------------------------


def test_chronic_condition_from_before_2000_without_stop_is_active_in_2000(
    cfg: Config,
) -> None:
    raw = load_raw(cfg)
    events = build_events(raw)
    prediabetes = events["code"] == "15777000"  # p2, diagnosed 1996, no STOP
    assert events.loc[prediabetes, "date"].dt.year.tolist() == [1996]
    assert events.loc[prediabetes, "stop"].isna().all()
    assert active_in_year(events, raw.patients, 2000)[prediabetes].all()


def test_active_in_year_ends_at_stop_and_at_death(cfg: Config) -> None:
    raw = load_raw(cfg)
    events = build_events(raw)
    pharyngitis = events["code"] == "195662009"  # p1, 1997-05-01 to 1998-02-01
    prediabetes = events["code"] == "15777000"  # p2, no STOP; p2 died 2020-06-01
    active = {y: active_in_year(events, raw.patients, y) for y in range(1996, 2022)}
    assert [active[y][pharyngitis].item() for y in (1996, 1997, 1998, 1999)] == [
        False, True, True, False
    ]
    assert active[2020][prediabetes].item()  # died during 2020
    assert not active[2021][prediabetes].item()  # dead before 2021 began


def test_active_and_alive_patients_per_year(cfg: Config) -> None:
    raw = load_raw(cfg)
    events = build_events(raw)
    active = active_patients_per_year(events, raw.patients, 1995, 2021)
    assert active.index.tolist() == list(range(1995, 2022))
    assert {y: active[y] for y in (1995, 1997, 1999, 2015, 2020, 2021)} == {
        1995: 0, 1997: 2, 1999: 1, 2015: 2, 2020: 3, 2021: 2
    }
    alive = alive_patients_per_year(raw.patients, 2000, 2021)
    assert {y: alive[y] for y in (2000, 2001, 2020, 2021)} == {
        2000: 2, 2001: 3, 2020: 3, 2021: 2
    }
