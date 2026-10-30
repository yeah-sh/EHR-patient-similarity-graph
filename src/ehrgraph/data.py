"""Load the raw Synthea CSVs into a tidy, validated table of condition events.

Pipeline (run end to end by ``scripts/01_prepare_data.py``)::

    raw = load_raw(cfg)                  # three CSVs, only the columns we need
    events = build_events(raw)           # one row per condition, dated by its encounter
    checks = validate(events, raw.patients)
    code_dictionary = build_code_dictionary(events)
    merged = merge_codes(events, cfg.data.code_merge)
    clinical = filter_clinical_events(merged, cfg.data.code_filter)
    write_data_summary(events, raw.patients, code_dictionary, checks, cfg)
    write_clinical_summary(events, clinical, raw.patients, code_dictionary, cfg)

``active_in_year`` defines when a condition counts as active in a yearly window.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ehrgraph.config import CodeFilterConfig, Config

logger = logging.getLogger(__name__)

# Raw column -> tidy name, for the only columns we read. Everything else, including
# MIDDLE, FIPS, INCOME and SYSTEM (absent in older Synthea exports), is never read.
PATIENT_COLUMNS: dict[str, str] = {
    "Id": "patient_id",
    "BIRTHDATE": "birth_date",
    "DEATHDATE": "death_date",
    "GENDER": "gender",
}
CONDITION_COLUMNS: dict[str, str] = {
    "START": "start",
    "STOP": "stop",
    "PATIENT": "patient_id",
    "ENCOUNTER": "encounter_id",
    "CODE": "code",
    "DESCRIPTION": "description",
}
ENCOUNTER_COLUMNS: dict[str, str] = {"Id": "encounter_id", "START": "encounter_start"}
# For follow-up: who was seen when (labels.last_encounters).
ENCOUNTER_DATE_COLUMNS: dict[str, str] = {"PATIENT": "patient_id", "START": "encounter_start"}
DATE_COLUMNS: frozenset[str] = frozenset(
    {"birth_date", "death_date", "start", "stop", "encounter_start"}
)
DATE_DTYPE = "datetime64[us, UTC]"

# stop is conditions.STOP: when the condition resolved (missing if it never did).
EVENT_COLUMNS: list[str] = ["patient_id", "date", "stop", "code", "description", "encounter_id"]

# validate() raises when a hard check is non-zero; soft checks are only logged.
HARD_CHECKS: tuple[str, ...] = (
    "duplicate_patient_ids",
    "events_missing_fields",
    "events_unknown_patient",
)
SOFT_CHECKS: tuple[str, ...] = ("events_after_death", "events_in_future")

DAYS_PER_YEAR = 365.25

# Chart tokens from the dataviz reference palette: one series on the light surface.
_SERIES_COLOR = "#2a78d6"
_SURFACE = "#fcfcfb"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_TEXT_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
_POINTS_PER_INCH = 72
_FIG_WIDTH_IN = 9.0
_HIST_HEIGHT_IN = 4.5
_BAR_CHROME_IN = 0.9  # title and margins around the bar rows
_BAR_FILL = 0.5  # share of each row a bar fills; the rest is air
_MAX_BAR_PX = 24
_SURFACE_GAP_PX = 2  # surface-coloured gap between adjacent histogram bars
_LINE_PX = 2
_VALUE_LABEL_MARGIN = 0.1  # x headroom so value labels at the bar tips are not clipped


@dataclass(frozen=True)
class RawTables:
    """The three Synthea tables, reduced to the columns we use, with tidy names.

    Attributes:
        patients: patient_id, birth_date, death_date, gender.
        conditions: start, stop, patient_id, encounter_id, code, description.
        encounters: encounter_id, encounter_start.
    """

    patients: pd.DataFrame
    conditions: pd.DataFrame
    encounters: pd.DataFrame


def load_raw(cfg: Config) -> RawTables:
    """Read patients, conditions and encounters from the paths in ``cfg``.

    Every column is read as a string (codes and IDs stay text); dates are then
    parsed as UTC timestamps. Columns we do not need are never read, so files
    without the optional MIDDLE, FIPS, INCOME or SYSTEM columns load fine.

    Args:
        cfg: Project configuration.

    Returns:
        The three tables with tidy column names (see ``RawTables``).

    Raises:
        ValueError: If a file lacks a required column or a date is not ISO 8601.
    """
    paths = cfg.paths
    raw = RawTables(
        patients=load_patients(cfg),
        conditions=_read_csv(paths.conditions_csv, CONDITION_COLUMNS),
        encounters=_read_csv(paths.encounters_csv, ENCOUNTER_COLUMNS),
    )
    logger.info(
        "Loaded %d patients, %d conditions and %d encounters",
        len(raw.patients),
        len(raw.conditions),
        len(raw.encounters),
    )
    return raw


def load_patients(cfg: Config) -> pd.DataFrame:
    """Read the patients table alone (see ``load_raw``).

    Args:
        cfg: Project configuration.

    Returns:
        Columns patient_id, birth_date, death_date and gender.
    """
    return _read_csv(cfg.paths.patients_csv, PATIENT_COLUMNS)


def load_encounter_dates(cfg: Config) -> pd.DataFrame:
    """Read the patient and start date of every encounter, of any type.

    Args:
        cfg: Project configuration.

    Returns:
        Columns patient_id and encounter_start (UTC), one row per encounter.
    """
    return _read_csv(cfg.paths.encounters_csv, ENCOUNTER_DATE_COLUMNS)


def _read_csv(path: Path, columns: dict[str, str]) -> pd.DataFrame:
    """Read ``columns`` from a Synthea CSV as strings, rename them and parse dates."""
    df = pd.read_csv(
        path,
        usecols=lambda name: name in columns,
        dtype={name: "string" for name in columns},
        keep_default_na=False,  # only empty fields are missing; "NA" stays text
        na_values=[""],
    )
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")
    df = df[list(columns)].rename(columns=columns)
    for name in DATE_COLUMNS.intersection(df.columns):
        df[name] = pd.to_datetime(df[name], utc=True, format="ISO8601").astype(DATE_DTYPE)
    return df


def build_events(raw: RawTables) -> pd.DataFrame:
    """Turn raw conditions into a tidy events table dated by their encounter.

    Each condition takes its date from ``encounters.START`` via ``ENCOUNTER = Id``.
    Conditions whose encounter is missing fall back to ``conditions.START``, and
    the number of fallbacks is logged. ``stop`` is ``conditions.STOP``. Exact
    duplicate rows are dropped.

    Args:
        raw: Output of ``load_raw``.

    Returns:
        Columns ``EVENT_COLUMNS``, sorted by patient_id, date and code.

    Raises:
        pandas.errors.MergeError: If encounter IDs are not unique.
    """
    merged = raw.conditions.merge(
        raw.encounters, on="encounter_id", how="left", validate="many_to_one"
    )
    n_fallback = int(merged["encounter_start"].isna().sum())
    log = logger.warning if n_fallback else logger.info
    log(
        "%d of %d conditions had no matching encounter; dated them by conditions.START",
        n_fallback,
        len(merged),
    )
    merged["date"] = merged["encounter_start"].fillna(merged["start"])

    events = merged[EVENT_COLUMNS]
    n_rows = len(events)
    events = (
        events.drop_duplicates()
        .sort_values(["patient_id", "date", "code"])
        .reset_index(drop=True)
    )
    logger.info(
        "Dropped %d exact duplicate events; %d events remain for %d patients and %d codes",
        n_rows - len(events),
        len(events),
        events["patient_id"].nunique(),
        events["code"].nunique(),
    )
    return events


def build_code_dictionary(events: pd.DataFrame) -> pd.DataFrame:
    """Map each condition code to a single description.

    Synthea versions word some codes differently (for example with or without a
    "(disorder)" suffix). Each code gets its most frequent description, with ties
    going to the alphabetically first so the result is deterministic.

    Args:
        events: Output of ``build_events``.

    Returns:
        Columns ``code`` and ``description``, one row per code, sorted by code.
    """
    counts = events.groupby(["code", "description"]).size().reset_index(name="n")
    counts = counts.sort_values(["code", "n", "description"], ascending=[True, False, True])
    n_reworded = int((counts["code"].value_counts() > 1).sum())
    dictionary = counts.drop_duplicates("code")[["code", "description"]].reset_index(drop=True)
    logger.info(
        "Code dictionary: %d codes, %d of them with more than one description",
        len(dictionary),
        n_reworded,
    )
    return dictionary


def save_code_dictionary(dictionary: pd.DataFrame, path: Path) -> None:
    """Write the code dictionary to ``path`` as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(path, index=False)
    logger.info("Wrote %d codes to %s", len(dictionary), path)


def save_events(events: pd.DataFrame, path: Path) -> None:
    """Write the events table to ``path`` as Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(path, engine="pyarrow", index=False)
    logger.info("Wrote %d events to %s", len(events), path)


def validate(
    events: pd.DataFrame, patients: pd.DataFrame, reference_date: pd.Timestamp | None = None
) -> dict[str, int]:
    """Check the events table against the patients table and log every count.

    Hard checks (``HARD_CHECKS``) raise: duplicate patient IDs, events missing a
    patient_id, date or code, and events for patients not in ``patients``.
    Soft checks (``SOFT_CHECKS``) are logged and their rows kept: events dated
    after the patient's death date (compared by calendar day, so events on the
    day of death are fine) and events dated after ``reference_date``.

    Args:
        events: Output of ``build_events``.
        patients: ``RawTables.patients``.
        reference_date: UTC time the data runs to, for the future-date check. Defaults
            to the latest event date in ``events``, not today: Synthea can date events
            a few days past the run date, and those are not errors.

    Returns:
        The number of patients and events, plus the count for every check.

    Raises:
        ValueError: If any hard check is non-zero.
    """
    reference = events["date"].max() if reference_date is None else reference_date
    logger.info("Reference date for the future-date check: %s", reference)
    death_dates = patients.drop_duplicates("patient_id").set_index("patient_id")["death_date"]
    event_death_date = events["patient_id"].map(death_dates)
    checks = {
        "patients": len(patients),
        "events": len(events),
        "duplicate_patient_ids": int(patients["patient_id"].duplicated().sum()),
        "events_missing_fields": int(
            events[["patient_id", "date", "code"]].isna().any(axis=1).sum()
        ),
        "events_unknown_patient": int((~events["patient_id"].isin(patients["patient_id"])).sum()),
        "events_after_death": int((events["date"].dt.normalize() > event_death_date).sum()),
        "events_in_future": int((events["date"] > reference).sum()),
    }

    logger.info("Validating %d events for %d patients", checks["events"], checks["patients"])
    for name in HARD_CHECKS + SOFT_CHECKS:
        n = checks[name]
        if n == 0:
            level = logging.INFO
        elif name in HARD_CHECKS:
            level = logging.ERROR
        else:
            level = logging.WARNING
        logger.log(level, "check %s: %d", name, n)

    failed = {name: checks[name] for name in HARD_CHECKS if checks[name]}
    if failed:
        raise ValueError(f"data validation failed: {failed}")
    return checks


def merge_codes(events: pd.DataFrame, code_merge: Mapping[str, str]) -> pd.DataFrame:
    """Rename codes that duplicate a concept to their canonical code.

    Synthea versions sometimes record one concept under two codes (Prediabetes is
    15777000 in older exports and 714628002 in newer ones). Every event whose code
    is a key of ``code_merge`` takes the mapped code and keeps its own description.
    A renamed event that repeats an event already under the canonical code (same
    patient, encounter, date and stop) is dropped, so the concept counts once.
    Codes in ``code_merge`` that match no event are logged as warnings, never
    errors: a rule for another Synthea version's code simply renames nothing.

    Args:
        events: Output of ``build_events``.
        code_merge: ``cfg.data.code_merge``, old code to canonical code.

    Returns:
        ``events`` with merged codes, sorted by patient_id, date and code, with a
        fresh index.
    """
    unmatched = sorted({*code_merge, *code_merge.values()} - set(events["code"].dropna()))
    if unmatched:
        logger.warning("Code merge codes matching no event: %s", unmatched)
    for old, new in code_merge.items():
        rows = events["code"] == old
        logger.info(
            "Merging code %s into %s: %d events of %d patients",
            old,
            new,
            int(rows.sum()),
            events.loc[rows, "patient_id"].nunique(),
        )

    renamed = events["code"].isin(list(code_merge))
    merged = events.assign(code=events["code"].replace(dict(code_merge)))
    # Events already under the canonical code sort first, so a repeat drops the renamed copy.
    key = [name for name in EVENT_COLUMNS if name != "description"]
    duplicate = (
        merged.assign(_renamed=renamed)
        .sort_values("_renamed", kind="stable")
        .duplicated(key)
        .reindex(merged.index)
    )
    repeats = duplicate & renamed
    merged = (
        merged[~repeats].sort_values(["patient_id", "date", "code"]).reset_index(drop=True)
    )
    logger.info(
        "Code merge renamed %d events and dropped %d that repeated a canonical event",
        int(renamed.sum()),
        int(repeats.sum()),
    )
    return merged


def classify_codes(events: pd.DataFrame, code_filter: CodeFilterConfig) -> pd.DataFrame:
    """Decide for every code whether it is a clinical condition.

    A code is judged on all of its descriptions, because Synthea versions word
    some codes differently ("Stroke" vs "Cerebrovascular accident (disorder)"). It
    is kept if any description ends with one of ``keep_suffixes`` or equals one of
    ``keep_descriptions``, both compared case-insensitively. Whitelist entries
    that match no description are logged as warnings, since they are likely typos.

    Args:
        events: Events table (needs code and description).
        code_filter: ``cfg.data.code_filter``.

    Returns:
        One row per code, sorted by code: code, kept (bool) and rule, which is
        "suffix", "whitelist" or "dropped" (suffix wins when both apply).
    """
    wordings = events[["code", "description"]].drop_duplicates()
    text = wordings["description"].fillna("").str.strip().str.lower()
    suffixes = tuple(s.strip().lower() for s in code_filter.keep_suffixes)
    whitelist = {d.strip().lower() for d in code_filter.keep_descriptions}
    flags = pd.DataFrame(
        {
            "code": wordings["code"],
            "by_suffix": text.map(lambda s: s.endswith(suffixes)).astype(bool),
            "by_whitelist": text.isin(whitelist),
        }
    )
    codes = flags.groupby("code").agg(
        by_suffix=("by_suffix", "any"), by_whitelist=("by_whitelist", "any")
    )
    codes["kept"] = codes["by_suffix"] | codes["by_whitelist"]
    codes["rule"] = np.select(
        [codes["by_suffix"], codes["by_whitelist"]], ["suffix", "whitelist"], default="dropped"
    )

    matched = set(text[flags["by_whitelist"]])
    unmatched = [d for d in code_filter.keep_descriptions if d.strip().lower() not in matched]
    if unmatched:
        logger.warning("Code filter whitelist entries matching no description: %s", unmatched)
    logger.info(
        "Code filter: kept %d codes (%d by suffix, %d by whitelist), dropped %d",
        int(codes["kept"].sum()),
        int((codes["rule"] == "suffix").sum()),
        int((codes["rule"] == "whitelist").sum()),
        int((~codes["kept"]).sum()),
    )
    return codes.reset_index()[["code", "kept", "rule"]]


def filter_clinical_events(events: pd.DataFrame, code_filter: CodeFilterConfig) -> pd.DataFrame:
    """Keep only the events whose code is clinical (see ``classify_codes``).

    Args:
        events: Output of ``build_events``.
        code_filter: ``cfg.data.code_filter``.

    Returns:
        The kept rows of ``events``, in the same order, with a fresh index.
    """
    codes = classify_codes(events, code_filter)
    clinical = events[events["code"].isin(codes.loc[codes["kept"], "code"])]
    clinical = clinical.reset_index(drop=True)
    logger.info(
        "Clinical filter kept %d of %d events (dropped %d); %d patients have clinical events",
        len(clinical),
        len(events),
        len(events) - len(clinical),
        clinical["patient_id"].nunique(),
    )
    return clinical


def active_until(events: pd.DataFrame, patients: pd.DataFrame) -> pd.Series:
    """Return when each condition stops being active.

    That is its STOP date or the patient's death date, whichever comes first,
    because Synthea leaves conditions open when a patient dies. Missing (NaT) means
    the condition has neither: it is still active at the end of the data.

    Args:
        events: Events table (needs patient_id and stop).
        patients: ``RawTables.patients`` (needs unique patient_id and death_date).

    Returns:
        UTC timestamps aligned with ``events``.
    """
    death_date = events["patient_id"].map(patients.set_index("patient_id")["death_date"])
    stop = events["stop"]
    return stop.mask(stop.isna() | (death_date < stop), death_date)


def active_in_year(events: pd.DataFrame, patients: pd.DataFrame, year: int) -> pd.Series:
    """Flag the conditions that are active at some point during ``year``.

    A condition is active from its event date through ``active_until`` (its STOP
    date, or the patient's death if earlier), compared by calendar day. With
    neither it stays active, so a chronic condition that started before ``year``
    still counts in ``year``.

    Args:
        events: Events table (needs patient_id, date and stop).
        patients: ``RawTables.patients`` (needs unique patient_id and death_date).
        year: Calendar year of the window.

    Returns:
        Boolean mask aligned with ``events``.
    """
    window_start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    window_end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
    until = active_until(events, patients)
    return (events["date"] < window_end) & (until.isna() | (until >= window_start))


def active_patients_per_year(
    events: pd.DataFrame, patients: pd.DataFrame, start_year: int, end_year: int
) -> pd.Series:
    """Count the patients with at least one active condition in each year.

    Args:
        events: Events table (see ``active_in_year``).
        patients: ``RawTables.patients``.
        start_year: First year, inclusive.
        end_year: Last year, inclusive.

    Returns:
        Patient counts indexed by year.
    """
    counts = {
        year: events.loc[active_in_year(events, patients, year), "patient_id"].nunique()
        for year in range(start_year, end_year + 1)
    }
    return pd.Series(counts, name="patients").rename_axis("year")


def alive_patients_per_year(patients: pd.DataFrame, start_year: int, end_year: int) -> pd.Series:
    """Count the patients alive at some point during each year.

    Args:
        patients: ``RawTables.patients``.
        start_year: First year, inclusive.
        end_year: Last year, inclusive.

    Returns:
        Patient counts indexed by year.
    """
    counts = {}
    for year in range(start_year, end_year + 1):
        window_start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        window_end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
        born = patients["birth_date"] < window_end
        alive = patients["death_date"].isna() | (patients["death_date"] >= window_start)
        counts[year] = int((born & alive).sum())
    return pd.Series(counts, name="patients").rename_axis("year")


def patient_ages(patients: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Return each patient's age in completed years.

    Deceased patients are aged at their death date, everyone else at ``as_of``.
    A year is ``DAYS_PER_YEAR`` days, so ages can be off by one near a birthday.

    Args:
        patients: ``RawTables.patients``.
        as_of: UTC reference date for living patients.

    Returns:
        Integer ages indexed like ``patients``.
    """
    end = patients["death_date"].fillna(as_of)
    days = (end - patients["birth_date"]).dt.days
    return np.floor(days / DAYS_PER_YEAR).astype(int)


def top_conditions(events: pd.DataFrame, code_dictionary: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return the ``n`` codes recorded for the most patients.

    Ties are broken by event count, then by code.

    Args:
        events: Output of ``build_events``.
        code_dictionary: Output of ``build_code_dictionary``.
        n: Number of codes to return.

    Returns:
        Columns code, description, patients (distinct patients with the code) and events.
    """
    stats = (
        events.groupby("code")
        .agg(patients=("patient_id", "nunique"), events=("patient_id", "size"))
        .reset_index()
        .sort_values(["patients", "events", "code"], ascending=[False, False, True])
        .head(n)
    )
    top = stats.merge(code_dictionary, on="code", how="left")
    return top[["code", "description", "patients", "events"]].reset_index(drop=True)


def plot_events_per_patient(per_patient: pd.Series, path: Path, cfg: Config) -> None:
    """Save a histogram of condition events per patient to ``path``.

    Args:
        per_patient: Event count per patient, including patients with zero events.
        path: Output PNG.
        cfg: Project configuration (``summary`` section).
    """
    dpi = cfg.summary.figure_dpi
    px = _POINTS_PER_INCH / dpi
    # Integer-width bins, so every bar covers the same number of possible counts.
    max_count = int(per_patient.max())
    width = max(1, math.ceil((max_count + 1) / cfg.summary.events_hist_bins))
    edges = np.arange(0, max_count + width + 1, width)
    median = float(per_patient.median())

    with matplotlib.rc_context(chart_style(dpi)):
        fig = Figure(figsize=(_FIG_WIDTH_IN, _HIST_HEIGHT_IN), layout="constrained")
        ax = fig.subplots()
        ax.hist(
            per_patient.to_numpy(),
            bins=edges,
            color=_SERIES_COLOR,
            edgecolor=_SURFACE,
            linewidth=_SURFACE_GAP_PX * px,
        )
        ax.axvline(median, color=_TEXT_SECONDARY, linewidth=_LINE_PX * px)
        ax.annotate(
            f"median {median:g}",
            xy=(median, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            color=_TEXT_SECONDARY,
        )
        ax.set_title(f"Condition events per patient ({len(per_patient):,} patients)")
        ax.set_xlabel("Condition events")
        ax.set_ylabel("Patients")
        ax.set_xlim(0, edges[-1])
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        save_figure(fig, path)


def plot_top_conditions(top: pd.DataFrame, path: Path, cfg: Config) -> None:
    """Save a horizontal bar chart of the most common conditions to ``path``.

    Args:
        top: Output of ``top_conditions``.
        path: Output PNG.
        cfg: Project configuration (``summary`` section).
    """
    dpi = cfg.summary.figure_dpi
    row_in = _MAX_BAR_PX / (_BAR_FILL * dpi)  # keeps each bar at most _MAX_BAR_PX thick
    y = np.arange(len(top))

    with matplotlib.rc_context(chart_style(dpi)):
        fig = Figure(
            figsize=(_FIG_WIDTH_IN, len(top) * row_in + _BAR_CHROME_IN), layout="constrained"
        )
        ax = fig.subplots()
        bars = ax.barh(y, top["patients"], height=_BAR_FILL, color=_SERIES_COLOR)
        ax.bar_label(
            bars,
            labels=[f"{v:,}" for v in top["patients"]],
            padding=3,
            color=_TEXT_SECONDARY,
        )
        ax.set_yticks(y, top["description"])
        ax.invert_yaxis()  # rank 1 at the top
        ax.margins(x=_VALUE_LABEL_MARGIN)
        ax.set_title(f"Top {len(top)} conditions by number of patients")
        # Every bar carries its value, so the x axis would only repeat them.
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelcolor=_TEXT_SECONDARY)
        save_figure(fig, path)


def chart_style(dpi: int) -> dict[str, object]:
    """Matplotlib rc settings for the reference palette at ``dpi``."""
    hairline = _POINTS_PER_INCH / dpi  # 1 px in points
    return {
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "figure.facecolor": _SURFACE,
        "savefig.facecolor": _SURFACE,
        "axes.facecolor": _SURFACE,
        "axes.edgecolor": _BASELINE,
        "axes.linewidth": hairline,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titlecolor": _TEXT_PRIMARY,
        "axes.labelcolor": _TEXT_SECONDARY,
        "text.color": _TEXT_PRIMARY,
        "xtick.color": _BASELINE,
        "ytick.color": _BASELINE,
        "xtick.labelcolor": _TEXT_MUTED,
        "ytick.labelcolor": _TEXT_MUTED,
        "grid.color": _GRIDLINE,
        "grid.linewidth": hairline,
    }


def save_figure(fig: Figure, path: Path) -> None:
    """Save ``fig`` to ``path``, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    logger.info("Wrote figure %s", path)


def write_data_summary(
    events: pd.DataFrame,
    patients: pd.DataFrame,
    code_dictionary: pd.DataFrame,
    checks: dict[str, int],
    cfg: Config,
) -> str:
    """Write the data summary markdown and its two figures.

    Args:
        events: Output of ``build_events``.
        patients: ``RawTables.patients``.
        code_dictionary: Output of ``build_code_dictionary``.
        checks: Output of ``validate``.
        cfg: Project configuration (output paths and the ``summary`` section).

    Returns:
        The markdown written to ``cfg.paths.data_summary_md``.
    """
    paths, opts = cfg.paths, cfg.summary
    n_patients = len(patients)
    n_dead = int(patients["death_date"].notna().sum())
    first, last = events["date"].min(), events["date"].max()
    as_of = last.normalize()
    ages = patient_ages(patients, as_of)
    age_groups = pd.cut(
        ages, bins=list(opts.age_bins), right=False, labels=_age_group_labels(opts.age_bins)
    ).value_counts(sort=False)
    gender = patients["gender"].value_counts().sort_index()
    per_patient = (
        events.groupby("patient_id").size().reindex(patients["patient_id"], fill_value=0)
    )
    top = top_conditions(events, code_dictionary, opts.top_n_conditions)

    plot_events_per_patient(per_patient, paths.events_per_patient_fig, cfg)
    plot_top_conditions(top, paths.top_conditions_fig, cfg)

    def share(part: int, whole: int = n_patients) -> str:
        return f"{part / whole:.1%}"

    def link(path: Path) -> str:
        return Path(os.path.relpath(path, paths.data_summary_md.parent)).as_posix()

    lines = [
        "# Data summary",
        "",
        "Synthea synthetic EHR, prepared by `scripts/01_prepare_data.py`. A condition event is "
        "one row of conditions.csv, dated by its encounter's START (UTC).",
        "",
        "## Patients",
        "",
        markdown_table(
            ["Status", "Patients", "Share"],
            [
                ["Alive", f"{n_patients - n_dead:,}", share(n_patients - n_dead)],
                ["Deceased", f"{n_dead:,}", share(n_dead)],
                ["Total", f"{n_patients:,}", share(n_patients)],
            ],
            "lrr",
        ),
        "",
        "### Gender",
        "",
        markdown_table(
            ["Gender", "Patients", "Share"],
            [[g, f"{n:,}", share(n)] for g, n in gender.items()],
            "lrr",
        ),
        "",
        "### Age",
        "",
        f"Completed years at death for deceased patients, otherwise on {as_of:%Y-%m-%d} "
        f"(the last event date in the data). Mean {ages.mean():.1f}, "
        f"median {ages.median():.0f}, range {ages.min()} to {ages.max()}.",
        "",
        markdown_table(
            ["Age group", "Patients", "Share"],
            [[str(g), f"{n:,}", share(n)] for g, n in age_groups.items()],
            "lrr",
        ),
        "",
        "## Condition events",
        "",
        markdown_table(
            ["Measure", "Value"],
            [
                ["Events", f"{len(events):,}"],
                ["Unique condition codes (SNOMED-CT)", f"{events['code'].nunique():,}"],
                ["Date range", f"{first:%Y-%m-%d} to {last:%Y-%m-%d}"],
                ["Events per patient, mean", f"{per_patient.mean():.1f}"],
                ["Events per patient, median", f"{per_patient.median():g}"],
                ["Events per patient, max", f"{per_patient.max():,}"],
                ["Patients with no events", f"{int((per_patient == 0).sum()):,}"],
            ],
            "lr",
        ),
        "",
        f"![Condition events per patient]({link(paths.events_per_patient_fig)})",
        "",
        f"## Top {len(top)} conditions",
        "",
        "Ranked by the number of patients with at least one event of the code (ties by event "
        "count). Each description is the code's most frequent wording in the data.",
        "",
        markdown_table(
            ["Rank", "Code", "Description", "Patients", "Share of patients", "Events"],
            [
                [str(rank), row.code, row.description, f"{row.patients:,}",
                 share(row.patients), f"{row.events:,}"]
                for rank, row in enumerate(top.itertuples(index=False), start=1)
            ],
            "rllrrr",
        ),
        "",
        f"![Top {len(top)} conditions]({link(paths.top_conditions_fig)})",
        "",
        "## Validation",
        "",
        "Hard checks make `validate()` raise; soft checks are logged and their rows kept. "
        "events_in_future counts events after the reference date, which by default is the "
        "latest event date in the data rather than today.",
        "",
        markdown_table(
            ["Check", "Kind", "Count"],
            [[name, "hard", f"{checks[name]:,}"] for name in HARD_CHECKS]
            + [[name, "soft", f"{checks[name]:,}"] for name in SOFT_CHECKS],
            "llr",
        ),
        "",
    ]
    text = "\n".join(lines)
    paths.data_summary_md.parent.mkdir(parents=True, exist_ok=True)
    paths.data_summary_md.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", paths.data_summary_md)
    return text


def write_clinical_summary(
    events: pd.DataFrame,
    clinical: pd.DataFrame,
    patients: pd.DataFrame,
    code_dictionary: pd.DataFrame,
    cfg: Config,
) -> str:
    """Write the clinical code filter report to ``cfg.paths.clinical_summary_md``.

    Covers the code merges, the filter rule and whitelist; codes, events and
    patients kept versus dropped; the top clinical conditions; patients with an
    active clinical condition in each trajectory year; and every dropped code,
    for review. Everything after the merges section counts merged codes.

    Args:
        events: Output of ``build_events`` (unfiltered, codes not yet merged).
        clinical: Output of ``filter_clinical_events`` on the merged events.
        patients: ``RawTables.patients``.
        code_dictionary: Output of ``build_code_dictionary``.
        cfg: Project configuration.

    Returns:
        The markdown written to ``cfg.paths.clinical_summary_md``.
    """
    code_filter, code_merge = cfg.data.code_filter, cfg.data.code_merge
    opts, traj = cfg.summary, cfg.trajectories
    min_codes = cfg.data.min_conditions_per_patient
    n_patients = len(patients)
    descriptions = code_dictionary.set_index("code")["description"]

    merge_rows = []
    for old, new in code_merge.items():
        rows = events["code"] == old
        merge_rows.append(
            [old, descriptions.get(old, "not in data"), new, descriptions.get(new, "not in data"),
             f"{events.loc[rows, 'patient_id'].nunique():,}", f"{int(rows.sum()):,}"]
        )
    merged = merge_codes(events, code_merge)

    stats = (
        merged.groupby("code")
        .agg(patients=("patient_id", "nunique"), events=("patient_id", "size"))
        .reset_index()
    )
    # Left join: a canonical code absent from the raw data still counts.
    codes = (
        classify_codes(merged, code_filter)
        .merge(code_dictionary, on="code", how="left")
        .merge(stats, on="code")
    )
    n_kept_codes = int(codes["kept"].sum())
    dropped = codes[~codes["kept"]].sort_values(
        ["patients", "events", "code"], ascending=[False, False, True]
    )

    wordings = merged[["code", "description"]].drop_duplicates()
    wording_text = wordings["description"].str.strip().str.lower()
    whitelist_rows = []
    for entry in code_filter.keep_descriptions:
        matched = sorted(set(wordings.loc[wording_text == entry.strip().lower(), "code"]))
        n = merged.loc[merged["code"].isin(matched), "patient_id"].nunique()
        whitelist_rows.append([entry, ", ".join(matched) or "no match", f"{n:,}"])

    codes_per_patient = clinical.groupby("patient_id")["code"].nunique()
    n_multi = int((codes_per_patient >= min_codes).sum())
    top = top_conditions(clinical, code_dictionary, opts.top_n_conditions)
    active = active_patients_per_year(clinical, patients, traj.start_year, traj.end_year)
    alive = alive_patients_per_year(patients, traj.start_year, traj.end_year)
    threshold = opts.min_active_patients_per_year
    low_years = [int(y) for y in active.index[active < threshold]]

    def share(part: int, whole: int) -> str:
        return f"{part / whole:.1%}"

    lines = [
        "# Clinical code filter",
        "",
        "Merges codes that duplicate a concept, then restricts the condition events to "
        "clinical conditions, dropping social, administrative and symptom codes. Unfiltered "
        "events with their original codes stay in events.parquet; the merged, filtered ones "
        "are in events_clinical.parquet.",
        "",
        f"## Code merges ({len(code_merge)}, `data.code_merge`)",
        "",
        "Each old code is renamed to its canonical code before the filter, so the concept "
        "counts once; the events keep their own description. Patients and events are those "
        "recorded under the old code. Every count below is after merging.",
        "",
        markdown_table(
            ["Old code", "Old description", "Canonical code", "Canonical description",
             "Patients", "Events"],
            merge_rows,
            "llllrr",
        )
        if merge_rows
        else "No merges configured.",
        "",
        "## Rule",
        "",
        f"A code is kept if any of its descriptions ends with "
        f"{', '.join(f'`{s}`' for s in code_filter.keep_suffixes)} or equals one of the "
        "whitelist entries below (both case-insensitive). Every other code is dropped. "
        "Any wording counts because Synthea versions word some codes differently.",
        "",
        f"### Whitelist ({len(code_filter.keep_descriptions)} entries, `data.code_filter`)",
        "",
        markdown_table(["Description", "Codes matched", "Patients"], whitelist_rows, "llr"),
        "",
        "## Result",
        "",
        markdown_table(
            ["", "Kept", "Dropped", "Total", "Kept share"],
            [
                ["Codes", f"{n_kept_codes:,}", f"{len(codes) - n_kept_codes:,}",
                 f"{len(codes):,}", share(n_kept_codes, len(codes))],
                ["Events", f"{len(clinical):,}", f"{len(merged) - len(clinical):,}",
                 f"{len(merged):,}", share(len(clinical), len(merged))],
            ],
            "lrrrr",
        ),
        "",
        f"Kept codes by rule: {int((codes['rule'] == 'suffix').sum())} by suffix, "
        f"{int((codes['rule'] == 'whitelist').sum())} by whitelist only. Before merging "
        f"there were {events['code'].nunique():,} codes and {len(events):,} events.",
        "",
        markdown_table(
            ["Patients", "Count", "Share of all patients"],
            [
                ["With any condition event", f"{merged['patient_id'].nunique():,}",
                 share(merged["patient_id"].nunique(), n_patients)],
                ["With at least 1 clinical code", f"{len(codes_per_patient):,}",
                 share(len(codes_per_patient), n_patients)],
                [f"With at least {min_codes} distinct clinical codes", f"{n_multi:,}",
                 share(n_multi, n_patients)],
            ],
            "lrr",
        ),
        "",
        f"## Top {len(top)} clinical conditions",
        "",
        "Ranked by the number of patients with at least one event of the code.",
        "",
        markdown_table(
            ["Rank", "Code", "Description", "Patients", "Share of patients", "Events"],
            [
                [str(rank), row.code, row.description, f"{row.patients:,}",
                 share(row.patients, n_patients), f"{row.events:,}"]
                for rank, row in enumerate(top.itertuples(index=False), start=1)
            ],
            "rllrrr",
        ),
        "",
        f"## Patients with an active clinical condition, {traj.start_year}-{traj.end_year}",
        "",
        "A condition is active in a year if it started before the year ended, had not "
        "stopped before the year began (no STOP date means still active), and the "
        "patient had not died before the year began. Alive counts patients born before "
        "the year ended and not dead before it began.",
        "",
        markdown_table(
            ["Year", "Alive", "With active clinical condition", "Share of alive", "Flag"],
            [
                [str(year), f"{alive[year]:,}", f"{n:,}", share(n, alive[year]),
                 f"below {threshold:,}" if n < threshold else ""]
                for year, n in active.items()
            ],
            "rrrrl",
        ),
        "",
        f"Years below {threshold:,}: {', '.join(map(str, low_years)) if low_years else 'none'}.",
        "",
        f"## Dropped codes ({len(dropped)})",
        "",
        markdown_table(
            ["Code", "Description", "Patients", "Events"],
            [[r.code, r.description, f"{r.patients:,}", f"{r.events:,}"]
             for r in dropped.itertuples(index=False)],
            "llrr",
        ),
        "",
    ]
    text = "\n".join(lines)
    path = cfg.paths.clinical_summary_md
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)
    return text


def _age_group_labels(edges: tuple[float, ...]) -> list[str]:
    """Label left-closed integer age bins, e.g. [0, 18) -> "0-17" and [80, inf) -> "80+"."""
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"{lo:g}+" if math.isinf(hi) else f"{lo:g}-{hi - 1:g}")
    return labels


def markdown_table(header: list[str], rows: list[list[str]], align: str) -> str:
    """Render a GitHub markdown table; ``align`` holds one 'l' or 'r' per column."""
    rule = ["---:" if a == "r" else "---" for a in align]
    return "\n".join(
        "| " + " | ".join(str(cell).replace("|", r"\|") for cell in line) + " |"
        for line in [header, rule, *rows]
    )
