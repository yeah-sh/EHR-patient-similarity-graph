"""Comorbidity trajectories: ordered condition events, yearly windows and transitions.

Pipeline (run end to end by ``scripts/02_build_trajectories.py``)::

    trajectories = build_trajectories(events, patients, cfg)  # cohort, ordered events
    windows = window_trajectories(trajectories, cfg)  # windows[w][patient, code], CSR
    transitions = extract_transitions(trajectories)  # A -> next new condition B
    save_trajectories(trajectories, windows, transitions, cfg)  # + code x code counts
    write_trajectory_summary(events, patients, trajectories, windows, transitions, cfg)

A condition is active from its event date through ``active_until``: its STOP date
or the patient's death, whichever comes first, or the end of the data when it has
neither (see ``data.active_until``). Conditions active in the same window are
comorbid. The cohort is the patients with enough distinct codes and a condition
active in at least one window, so no patient's row is empty in every window.
Trajectories, windows and transitions all cover the full data range, past the
prediction cutoff; features must use only windows that close by the cutoff
(``config.feature_window_years``) and transitions that end before it
(``pre_cutoff_transitions``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ehrgraph.config import Config
from ehrgraph.data import DATE_DTYPE, active_until, build_code_dictionary, markdown_table

logger = logging.getLogger(__name__)

# chronic is True when the condition has no STOP date (it never resolved).
TRAJECTORY_COLUMNS: list[str] = ["patient_id", "date", "stop", "active_until", "chronic", "code"]
TRANSITION_COLUMNS: list[str] = ["patient_id", "from_code", "from_date", "to_code", "to_date"]
# Window entries are 0/1 but stored as floats: products such as X.T @ X would
# overflow a small integer type.
WINDOW_DTYPE = np.float32
COUNT_DTYPE = np.int64


@dataclass(frozen=True)
class Trajectories:
    """The cohort's condition events in time order, with each condition's active span.

    Attributes:
        events: ``TRAJECTORY_COLUMNS``, sorted by patient_id, date and code, over the
            full data range. ``active_until`` is never missing: conditions with no
            STOP and no death run to ``data_end``.
        patient_ids: Cohort patients, sorted; position i is row i of every window.
        codes: Code vocabulary, sorted; position j is column j of every window.
        descriptions: Each code's most frequent wording, aligned with ``codes``.
        data_end: The latest event or STOP date in the input events.
    """

    events: pd.DataFrame
    patient_ids: tuple[str, ...]
    codes: tuple[str, ...]
    descriptions: tuple[str, ...]
    data_end: pd.Timestamp

    def sequences(self) -> dict[str, list[tuple[pd.Timestamp, str]]]:
        """Return each patient's (date, code) events in time order."""
        return {
            str(patient_id): list(zip(group["date"], group["code"]))
            for patient_id, group in self.events.groupby("patient_id", sort=True)
        }


@dataclass(frozen=True)
class TrajectoryWindows:
    """Conditions active in each yearly window, indexable as ``windows[w][patient, code]``.

    Attributes:
        years: Calendar year of each window, ascending.
        matrices: One CSR matrix of shape (len(patient_ids), len(codes)) per window,
            1.0 where the code is active for the patient at some point in that year.
        patient_ids: Row labels.
        codes: Column labels.
    """

    years: tuple[int, ...]
    matrices: list[sp.csr_matrix]
    patient_ids: tuple[str, ...]
    codes: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.matrices)

    def __getitem__(self, window: int) -> sp.csr_matrix:
        return self.matrices[window]


def build_trajectories(events: pd.DataFrame, patients: pd.DataFrame, cfg: Config) -> Trajectories:
    """Order each patient's condition events in time and attach their active spans.

    Patients with fewer than ``cfg.data.min_conditions_per_patient`` distinct codes
    are dropped, and so are patients with no condition active in any window from
    ``start_year`` to ``end_year`` (all their conditions resolved before the first
    window or began after the last). Every event of the remaining patients is kept,
    whatever its date: the prediction cutoff is applied later.

    Args:
        events: Clinical events (``data.EVENT_COLUMNS``), e.g. events_clinical.parquet.
        patients: ``RawTables.patients``, for death dates.
        cfg: Project configuration.

    Returns:
        The cohort's trajectories.

    Raises:
        ValueError: If ``trajectories.window`` is not ``"yearly"``.
    """
    years = _window_years(cfg)
    min_codes = cfg.data.min_conditions_per_patient
    codes_per_patient = events.groupby("patient_id")["code"].nunique()
    multi_code = events[
        events["patient_id"].isin(codes_per_patient.index[codes_per_patient >= min_codes])
    ]
    data_end = pd.Series([events["date"].max(), events["stop"].max()]).max()

    until = active_until(multi_code, patients)
    until = until.where(until <= data_end, data_end).astype(DATE_DTYPE)  # NaT -> data_end
    ordered = (
        multi_code.assign(active_until=until, chronic=multi_code["stop"].isna())[
            TRAJECTORY_COLUMNS
        ]
        .sort_values(["patient_id", "date", "code"], kind="stable")
        .reset_index(drop=True)
    )
    in_some_window = np.logical_or.reduce([active_in_window(ordered, year) for year in years])
    patient_ids = tuple(sorted(map(str, ordered.loc[in_some_window, "patient_id"].unique())))
    ordered = ordered[ordered["patient_id"].isin(patient_ids)].reset_index(drop=True)
    cohort = multi_code[multi_code["patient_id"].isin(patient_ids)]

    codes = tuple(sorted(map(str, ordered["code"].unique())))
    wording = build_code_dictionary(cohort).set_index("code")["description"]
    descriptions = tuple(str(wording[code]) for code in codes)
    n_multi_code = multi_code["patient_id"].nunique()
    logger.info(
        "Trajectories: %d of %d patients have >= %d distinct codes; dropped %d with no "
        "condition active in %d-%d, kept %d; %d events, %d codes, %.1f%% without STOP; "
        "data end %s",
        n_multi_code,
        len(codes_per_patient),
        min_codes,
        n_multi_code - len(patient_ids),
        years[0],
        years[-1],
        len(patient_ids),
        len(ordered),
        len(codes),
        100 * ordered["chronic"].mean(),
        data_end,
    )
    return Trajectories(ordered, patient_ids, codes, descriptions, data_end)


def window_trajectories(trajectories: Trajectories, cfg: Config) -> TrajectoryWindows:
    """Record the conditions active for each patient in each yearly window.

    The window for year Y is [Y-01-01, (Y+1)-01-01). A condition is active in it if
    it started before the window ends and is active until a day on or after the
    window starts, as in ``data.active_in_year``. A code with several episodes in
    one window counts once. Every cohort patient has a row in every window, empty
    when nothing is active (including before birth and after death), but no row is
    empty in every window (see ``build_trajectories``).

    Args:
        trajectories: Output of ``build_trajectories``.
        cfg: Project configuration (``trajectories`` section).

    Returns:
        One CSR matrix per year from ``start_year`` to ``end_year``, inclusive.

    Raises:
        ValueError: If ``trajectories.window`` is not ``"yearly"``.
    """
    years = _window_years(cfg)
    events = trajectories.events
    rows = pd.Index(trajectories.patient_ids).get_indexer(events["patient_id"])
    cols = pd.Index(trajectories.codes).get_indexer(events["code"])
    shape = (len(trajectories.patient_ids), len(trajectories.codes))

    matrices = []
    for year in years:
        active = active_in_window(events, year)
        matrix = sp.csr_matrix(
            (np.ones(int(active.sum()), dtype=WINDOW_DTYPE), (rows[active], cols[active])),
            shape=shape,
        )
        matrix.data[:] = 1  # duplicates were summed; a set holds each code once
        matrices.append(matrix)
        logger.info(
            "Window %d: %d active patients, %d active (patient, code) pairs",
            year,
            int((np.diff(matrix.indptr) > 0).sum()),
            matrix.nnz,
        )
    return TrajectoryWindows(years, matrices, trajectories.patient_ids, trajectories.codes)


def extract_transitions(trajectories: Trajectories) -> pd.DataFrame:
    """List each patient's transitions from a condition to the next new condition.

    Each code enters a patient's trajectory once, at its first event (its onset).
    The patient's onsets in time order form steps; codes with the same onset time
    (usually diagnosed at one encounter) share a step, so no order is invented
    between them. Every code of one step leads to every code of the next step:
    A -> B means B was the first new condition after A. Later episodes of a code
    add no transitions.

    Transitions span the full data range. Features may use only those that end
    before the prediction cutoff (``pre_cutoff_transitions``); since a
    code's onset is its first event, those are exactly the transitions that events
    before the cutoff alone would give.

    Args:
        trajectories: Output of ``build_trajectories``.

    Returns:
        ``TRANSITION_COLUMNS``, with the onset dates of both codes, sorted by
        patient_id, from_date, from_code and to_code.
    """
    onsets = trajectories.events.groupby(["patient_id", "code"], as_index=False)["date"].min()
    # Group numbers follow (patient_id, date) order, so they count steps within a patient.
    group = onsets.groupby(["patient_id", "date"], sort=True).ngroup()
    onsets["step"] = group - group.groupby(onsets["patient_id"]).transform("min")
    following = onsets.assign(step=onsets["step"] - 1)
    transitions = (
        onsets.merge(following, on=["patient_id", "step"], suffixes=("_from", "_to"))
        .rename(
            columns={
                "code_from": "from_code",
                "date_from": "from_date",
                "code_to": "to_code",
                "date_to": "to_date",
            }
        )[TRANSITION_COLUMNS]
        .sort_values(["patient_id", "from_date", "from_code", "to_code"])
        .reset_index(drop=True)
    )
    logger.info(
        "Transitions: %d from %d patients, %d distinct code pairs",
        len(transitions),
        transitions["patient_id"].nunique(),
        len(transitions[["from_code", "to_code"]].drop_duplicates()),
    )
    return transitions


def pre_cutoff_transitions(transitions: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Keep the transitions that end strictly before the prediction cutoff.

    A transition ends at its ``to_date``, the onset of the later code, and
    ``from_date <= to_date``, so both onsets are before the cutoff. These are the
    only transitions features may use.

    Args:
        transitions: Output of ``extract_transitions`` (or ``load_transitions``).
        cutoff: ``labels.cutoff_date``, read as 00:00 UTC.

    Returns:
        The matching rows, in their original order.
    """
    boundary = pd.Timestamp(cutoff.isoformat(), tz="UTC")
    return transitions[transitions["to_date"] < boundary].reset_index(drop=True)


def transition_matrix(transitions: pd.DataFrame, codes: Sequence[str]) -> sp.csr_matrix:
    """Count transitions in a code x code matrix.

    Entry [i, j] counts transitions codes[i] -> codes[j]. A patient has each
    transition at most once (a code has one onset), so it is a patient count.

    Args:
        transitions: Output of ``extract_transitions`` (or a subset of its rows).
        codes: The code vocabulary, ``Trajectories.codes``.

    Returns:
        A (len(codes), len(codes)) CSR matrix of integer counts.

    Raises:
        ValueError: If a transition uses a code outside ``codes``.
    """
    index = pd.Index(codes)
    rows = index.get_indexer(transitions["from_code"])
    cols = index.get_indexer(transitions["to_code"])
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError("transitions use codes that are not in the vocabulary")
    return sp.csr_matrix(
        (np.ones(len(transitions), dtype=COUNT_DTYPE), (rows, cols)),
        shape=(len(codes), len(codes)),
    )


def save_trajectories(
    trajectories: Trajectories,
    windows: TrajectoryWindows,
    transitions: pd.DataFrame,
    cfg: Config,
) -> None:
    """Write the trajectories, windows and transitions to ``data/processed``.

    Files (paths from ``cfg.paths``): trajectories.parquet (the ordered events);
    patient_index.json (patient IDs in row order); code_vocabulary.json (code and
    description in column order); windows.npz (the window matrices stacked
    vertically in year order, so window w is rows w*N to (w+1)*N for N patients);
    windows.json (years and shape); transitions.parquet (every transition, per
    patient) and transitions.npz (their code x code counts). Both transition files
    span the full data range and are for analysis only.

    Args:
        trajectories: Output of ``build_trajectories``.
        windows: Output of ``window_trajectories`` on ``trajectories``.
        transitions: Output of ``extract_transitions`` on ``trajectories``.
        cfg: Project configuration.

    Raises:
        ValueError: If ``windows`` is labelled differently from ``trajectories``.
    """
    if (windows.patient_ids, windows.codes) != (trajectories.patient_ids, trajectories.codes):
        raise ValueError("windows were not built from these trajectories")
    paths = cfg.paths
    paths.trajectories_parquet.parent.mkdir(parents=True, exist_ok=True)
    trajectories.events.to_parquet(paths.trajectories_parquet, engine="pyarrow", index=False)
    _write_json(paths.patient_index_json, list(trajectories.patient_ids))
    _write_json(
        paths.code_vocabulary_json,
        [{"code": c, "description": d} for c, d in zip(trajectories.codes, trajectories.descriptions)],
    )
    _write_npz(paths.windows_npz, sp.vstack(windows.matrices, format="csr"))
    _write_json(
        paths.windows_json,
        {
            "window": cfg.trajectories.window,
            "years": list(windows.years),
            "n_patients": len(windows.patient_ids),
            "n_codes": len(windows.codes),
            "data_end": trajectories.data_end.isoformat(),
        },
    )
    transitions.to_parquet(paths.transitions_parquet, engine="pyarrow", index=False)
    logger.info("Wrote %d transitions to %s", len(transitions), paths.transitions_parquet)
    _write_npz(paths.transitions_npz, transition_matrix(transitions, trajectories.codes))
    logger.info("Wrote %d trajectory events to %s", len(trajectories.events), paths.trajectories_parquet)


def load_transitions(cfg: Config) -> pd.DataFrame:
    """Read the per-patient transitions written by ``save_trajectories``.

    They span the full data range; filter them with
    ``pre_cutoff_transitions`` before building features.

    Args:
        cfg: Project configuration.

    Returns:
        ``TRANSITION_COLUMNS``, as returned by ``extract_transitions``.
    """
    return pd.read_parquet(cfg.paths.transitions_parquet)


def load_trajectories(cfg: Config) -> Trajectories:
    """Read the trajectories written by ``save_trajectories``.

    Args:
        cfg: Project configuration.

    Returns:
        The trajectories, labelled by the saved patient index and code vocabulary.
    """
    paths = cfg.paths
    vocabulary = load_code_vocabulary(cfg)
    meta = json.loads(paths.windows_json.read_text(encoding="utf-8"))
    return Trajectories(
        events=pd.read_parquet(paths.trajectories_parquet),
        patient_ids=tuple(json.loads(paths.patient_index_json.read_text(encoding="utf-8"))),
        codes=tuple(vocabulary["code"]),
        descriptions=tuple(vocabulary["description"]),
        data_end=pd.Timestamp(meta["data_end"]),
    )


def load_code_vocabulary(cfg: Config) -> pd.DataFrame:
    """Read the code vocabulary written by ``save_trajectories``.

    Args:
        cfg: Project configuration.

    Returns:
        Columns code and description, in window column order.
    """
    entries = json.loads(cfg.paths.code_vocabulary_json.read_text(encoding="utf-8"))
    return pd.DataFrame(entries, columns=["code", "description"])


def load_windows(cfg: Config) -> TrajectoryWindows:
    """Read the windows written by ``save_trajectories``.

    Args:
        cfg: Project configuration.

    Returns:
        The windows, labelled by the saved patient index and code vocabulary.

    Raises:
        ValueError: If the saved matrix does not match the saved index and years.
    """
    paths = cfg.paths
    meta = json.loads(paths.windows_json.read_text(encoding="utf-8"))
    patient_ids = tuple(json.loads(paths.patient_index_json.read_text(encoding="utf-8")))
    codes = tuple(load_code_vocabulary(cfg)["code"])
    years = tuple(meta["years"])
    stacked = sp.load_npz(paths.windows_npz).tocsr()
    n = len(patient_ids)
    if stacked.shape != (n * len(years), len(codes)):
        raise ValueError(
            f"{paths.windows_npz.name} has shape {stacked.shape}, expected "
            f"({n} patients x {len(years)} windows, {len(codes)} codes)"
        )
    matrices = [stacked[w * n : (w + 1) * n] for w in range(len(years))]
    return TrajectoryWindows(years, matrices, patient_ids, codes)


def write_trajectory_summary(
    events: pd.DataFrame,
    patients: pd.DataFrame,
    trajectories: Trajectories,
    windows: TrajectoryWindows,
    transitions: pd.DataFrame,
    cfg: Config,
) -> str:
    """Write the trajectory report to ``cfg.paths.trajectories_summary_md``.

    Covers the cohort, active patients and mean active conditions per window, the
    most common transitions and a few example trajectories, drawn at random with
    ``cfg.seed``.

    Args:
        events: The clinical events ``trajectories`` were built from.
        patients: ``RawTables.patients``.
        trajectories: Output of ``build_trajectories``.
        windows: Output of ``window_trajectories``.
        transitions: Output of ``extract_transitions``.
        cfg: Project configuration.

    Returns:
        The markdown written.
    """
    opts, traj_cfg = cfg.summary, cfg.trajectories
    traj = trajectories.events
    n_cohort = len(trajectories.patient_ids)
    min_codes = cfg.data.min_conditions_per_patient
    n_multi_code = int((events.groupby("patient_id")["code"].nunique() >= min_codes).sum())
    cutoff = cfg.labels.cutoff_date
    n_pre_cutoff = len(pre_cutoff_transitions(transitions, cutoff))
    description = dict(zip(trajectories.codes, trajectories.descriptions))
    window_start, _ = _year_bounds(traj_cfg.start_year)
    early = traj["date"] < window_start

    def share(part: float, whole: float) -> str:
        return f"{part / whole:.1%}" if whole else "n/a"

    window_rows = []
    for year, matrix in zip(windows.years, windows.matrices):
        start, end = _year_bounds(year)
        per_patient = np.diff(matrix.indptr)
        active = per_patient[per_patient > 0]
        n_dated = int(((traj["date"] >= start) & (traj["date"] < end)).sum())
        stats = (
            [f"{active.mean():.2f}", f"{np.median(active):g}", f"{active.max():,}"]
            if active.size
            else ["n/a"] * 3
        )
        window_rows.append(
            [str(year), f"{active.size:,}", share(active.size, n_cohort), *stats, f"{n_dated:,}"]
        )

    pairs = (
        transitions.groupby(["from_code", "to_code"]).size().reset_index(name="patients")
        .sort_values(["patients", "from_code", "to_code"], ascending=[False, True, True])
    )
    top_pairs = pairs.head(opts.top_n_transitions)

    lines = [
        "# Comorbidity trajectories",
        "",
        "Built by `scripts/02_build_trajectories.py` from events_clinical.parquet. A condition "
        "is active from its event date through its STOP date or the patient's death, whichever "
        "comes first; with neither, it stays active to the end of the data "
        f"({trajectories.data_end:%Y-%m-%d}). Conditions active in the same window are comorbid.",
        "",
        "## Cohort",
        "",
        markdown_table(
            ["Measure", "Count"],
            [
                ["Patients with a clinical event", f"{events['patient_id'].nunique():,}"],
                [f"Patients with at least {min_codes} distinct codes", f"{n_multi_code:,}"],
                [f"Of those, with no condition active in any window {traj_cfg.start_year}-"
                 f"{traj_cfg.end_year} (dropped)", f"{n_multi_code - n_cohort:,}"],
                ["Patients in the cohort", f"{n_cohort:,}"],
                ["Events of cohort patients, all dates", f"{len(traj):,}"],
                ["Events without a STOP date (chronic)",
                 f"{int(traj['chronic'].sum()):,} ({share(traj['chronic'].sum(), len(traj))})"],
                ["Codes in the vocabulary", f"{len(trajectories.codes):,}"],
            ],
            "lr",
        ),
        "",
        f"## Yearly windows, {traj_cfg.start_year}-{traj_cfg.end_year}",
        "",
        f"Events dated before {traj_cfg.start_year}: {int(early.sum()):,}, of which "
        f"{share(traj.loc[early, 'chronic'].sum(), early.sum())} have no STOP date. From "
        f"{traj_cfg.start_year}: {int((~early).sum()):,}, of which "
        f"{share(traj.loc[~early, 'chronic'].sum(), (~early).sum())} have no STOP date. "
        "Earlier years hold mostly conditions still open when Synthea's full export history "
        "begins, so windows start at the first full year after it (see config.yaml). Earlier "
        "conditions that are still active count in the windows.",
        "",
        "Active patients have at least one active condition in the window. The mean, median "
        "and max count active conditions per active patient. Events dated counts the "
        "events (new conditions or episodes) dated within the year.",
        "",
        markdown_table(
            ["Year", "Active patients", "Share of cohort", "Mean active conditions", "Median",
             "Max", "Events dated"],
            window_rows,
            "rrrrrrr",
        ),
        "",
        "## Transitions",
        "",
        "Each code enters a patient's trajectory once, at its onset. A -> B means B was the "
        "first new condition after A's onset; codes with the same onset time share a step, "
        "with no transition between them. Transitions use every event, including those "
        "before the first window and after the prediction cutoff.",
        "",
        f"{len(transitions):,} transitions from {transitions['patient_id'].nunique():,} "
        f"patients, {len(pairs):,} distinct code pairs. {n_pre_cutoff:,} of them end before "
        f"the prediction cutoff ({cutoff:%Y-%m-%d}); only those may be used as features. "
        "transitions.parquet and transitions.npz hold all of them, for analysis only.",
        "",
        markdown_table(
            ["From", "To", "Patients"],
            [[description[r.from_code], description[r.to_code], f"{r.patients:,}"]
             for r in top_pairs.itertuples(index=False)],
            "llr",
        ),
        "",
        f"## Example trajectories ({opts.example_patients}, random with seed {cfg.seed})",
        "",
        "Every clinical event, in time order. Windows lists the years from "
        f"{traj_cfg.start_year}-{traj_cfg.end_year} in which the condition is active.",
        "",
    ]
    for patient_id in _example_patients(trajectories.patient_ids, opts.example_patients, cfg.seed):
        lines += _example_section(patient_id, patients, trajectories, cfg)

    text = "\n".join(lines)
    path = cfg.paths.trajectories_summary_md
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)
    return text


def _example_patients(patient_ids: Sequence[str], n: int, seed: int) -> list[str]:
    """Draw ``n`` patients at random with ``seed``, returned sorted."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(patient_ids), size=min(n, len(patient_ids)), replace=False)
    return sorted(patient_ids[i] for i in chosen)


def _example_section(
    patient_id: str, patients: pd.DataFrame, trajectories: Trajectories, cfg: Config
) -> list[str]:
    """Markdown lines for one example patient: a header and the event table."""
    traj_cfg = cfg.trajectories
    person = patients.set_index("patient_id").loc[patient_id]
    events = trajectories.events[trajectories.events["patient_id"] == patient_id]
    description = dict(zip(trajectories.codes, trajectories.descriptions))
    died = (
        f", died {person['death_date']:%Y-%m-%d}" if pd.notna(person["death_date"]) else ""
    )

    rows = []
    for e in events.itertuples(index=False):
        if pd.notna(e.stop) and e.active_until == e.stop:
            until = f"{e.stop:%Y-%m-%d}"
        elif e.active_until < trajectories.data_end:
            until = f"{e.active_until:%Y-%m-%d} (death)"
        else:
            until = "ongoing"
        first = max(e.date.year, traj_cfg.start_year)
        last = min(e.active_until.year, traj_cfg.end_year)
        span = "-" if first > last else str(first) if first == last else f"{first}-{last}"
        rows.append([f"{e.date:%Y-%m-%d}", description[e.code], e.code, until, span])

    return [
        f"### Patient {patient_id}",
        "",
        f"{person['gender']}, born {person['birth_date']:%Y}{died}. {len(events)} events, "
        f"{events['code'].nunique()} distinct conditions.",
        "",
        markdown_table(["Date", "Condition", "Code", "Active until", "Windows"], rows, "lllll"),
        "",
    ]


def _window_years(cfg: Config) -> tuple[int, ...]:
    """Return the window years, ``start_year`` to ``end_year``; only yearly windows exist."""
    traj_cfg = cfg.trajectories
    if traj_cfg.window != "yearly":
        raise ValueError(f"trajectories.window {traj_cfg.window!r} is not supported; use 'yearly'")
    return tuple(range(traj_cfg.start_year, traj_cfg.end_year + 1))


def active_in_window(events: pd.DataFrame, year: int) -> np.ndarray:
    """Flag the trajectory events active at some point during the window for ``year``.

    An event is active if it started before the window ends and is active until a
    day on or after the window starts; this is what ``window_trajectories`` records.

    Args:
        events: ``Trajectories.events`` (or a subset), with date and active_until.
        year: Calendar year of the window.

    Returns:
        Boolean array aligned with ``events``.
    """
    start, end = _year_bounds(year)
    return ((events["date"] < end) & (events["active_until"] >= start)).to_numpy()


def _year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the UTC start of ``year`` and of the year after."""
    return (
        pd.Timestamp(year=year, month=1, day=1, tz="UTC"),
        pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC"),
    )


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
