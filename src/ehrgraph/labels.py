"""Prediction labels for the model cohort: new onset of chronic condition groups.

Pipeline (run end to end by ``scripts/05_build_labels.py``)::

    patient_ids, _ = load_model_cohort(cfg)
    labels = build_labels(trajectories.events, patients, encounters, patient_ids, cfg)
    labels = attach_phenotypes(labels, load_phenotype_clusters(cfg))  # analysis only
    split = stratified_split(split_strata(labels), split_fractions(cfg), cfg.seed)
    save_labels(labels, split, cfg)
    write_labels_summary(labels.assign(split=...), phenotype_table, descriptions, cfg)

Timeline (project config): inputs use windows 2017-2022, all before the cutoff
2023-01-01; labels use the horizon [2023-01-01, 2026-01-01). Nothing on or after
the horizon end is read: later events, encounters and deaths are dropped first.

Targets: the primary target is ``ANY_ONSET``, a new onset in any onset group; the
per-group onset labels are secondary targets. Mortality is kept for descriptive
statistics only (too few deaths in the horizon to train or evaluate on).

Onset label of group g (``labels.onset_groups``): the onset is g's first event, of
any of its codes. In order of precedence the label is

* masked if the patient died before the cutoff;
* masked if the patient is selected by post-cutoff codes: fewer than
  ``data.min_conditions_per_patient`` distinct codes before the cutoff, so in the
  cohort only thanks to later codes (and so certain to have a later condition);
* masked if the onset is before the cutoff (the patient already had it);
* masked if the patient is lost to follow-up: alive through the horizon with a
  last encounter before ``labels.min_last_encounter_date`` (rule a, which masks
  every onset label of the patient; an onset seen in the horizon before the loss
  gets the status ``LOST_AFTER_ONSET``);
* 1 if the onset is in the horizon;
* masked if the patient died in the horizon without that onset (rule b);
* 0 otherwise.

``ANY_ONSET`` is 1 if any group label is 1, 0 if at least one group label is
observed and all observed ones are 0, and masked if every group label is masked
(see ``pooled_onset_status``). Mortality is masked if the patient died before the
cutoff or is selected by post-cutoff codes, 1 if they died in the horizon, else 0.

Labels are 1.0, 0.0 or NaN (masked): metrics must use only ``observed`` entries,
and a NaN that slips through propagates instead of counting as a negative. Every
cohort patient keeps a row (and a graph node) whatever their labels. Phase 4
phenotype clusters are attached for analysis and split stratification only; they
are never a target.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ehrgraph.config import Config, horizon_end
from ehrgraph.data import markdown_table

logger = logging.getLogger(__name__)

ANY_ONSET = "any_onset"
PRIMARY_TARGET = ANY_ONSET
MORTALITY = "mortality"  # descriptive statistics only, never a model target
ONSET_PREFIX = "onset_"
STATUS_PREFIX = "status_"
PHENOTYPE = "phenotype_cluster"
SPLIT = "split"
SPLITS: tuple[str, ...] = ("train", "val", "test")
# Label statuses, in order of precedence for onset labels.
DIED_BEFORE_CUTOFF = "died before cutoff"
SELECTED_BY_FUTURE = "selected by post-cutoff codes"
PREVALENT = "had it before cutoff"
LOST_AFTER_ONSET = "lost to follow-up after onset"  # rule (a) masks a seen onset
LOST_TO_FOLLOW_UP = "lost to follow-up"
POSITIVE = "positive"
DIED_WITHOUT_ONSET = "died without onset"
NEGATIVE = "negative"
STATUSES: tuple[str, ...] = (
    DIED_BEFORE_CUTOFF, SELECTED_BY_FUTURE, PREVALENT, LOST_AFTER_ONSET, LOST_TO_FOLLOW_UP,
    POSITIVE, DIED_WITHOUT_ONSET, NEGATIVE,
)
SUMMARY_COLUMNS: list[str] = ["label", "observed", "positives", "negatives", "masked", "rate"]


def onset_columns(cfg: Config) -> list[str]:
    """Return the per-group onset label columns (secondary targets), in config order."""
    return [ONSET_PREFIX + group for group in cfg.labels.onset_groups]


def target_columns(cfg: Config) -> list[str]:
    """Return the model targets: ``PRIMARY_TARGET`` first, then the per-group onsets."""
    return [PRIMARY_TARGET, *onset_columns(cfg)]


def label_columns(cfg: Config) -> list[str]:
    """Return every label column: the targets, then mortality (descriptive only)."""
    return [*target_columns(cfg), MORTALITY]


def pooled_onset_status(statuses: np.ndarray) -> np.ndarray:
    """Return each patient's ``ANY_ONSET`` status from their per-group onset statuses.

    ``POSITIVE`` if any group is positive; else ``NEGATIVE`` if any group is observed
    (every observed group is then negative); else masked, every group being masked,
    with the patient-level reason in this order: died before the cutoff, selected by
    post-cutoff codes, lost to follow-up (after an onset, then without), died in the
    horizon without an onset, and last ``PREVALENT``: had every group before the cutoff.

    Args:
        statuses: Onset statuses, patients x groups (at least one group).

    Returns:
        One status per patient.
    """
    order = [
        DIED_BEFORE_CUTOFF, SELECTED_BY_FUTURE, POSITIVE, NEGATIVE, LOST_AFTER_ONSET,
        LOST_TO_FOLLOW_UP, DIED_WITHOUT_ONSET,
    ]
    return np.select([(statuses == s).any(axis=1) for s in order], order, default=PREVALENT)


def last_encounters(
    encounters: pd.DataFrame, patient_ids: Sequence[str], cfg: Config
) -> pd.Series:
    """Return each patient's last encounter of any type before the horizon end.

    Args:
        encounters: ``data.load_encounter_dates`` output.
        patient_ids: Patients to report, in order.
        cfg: Project configuration.

    Returns:
        UTC timestamps indexed by patient_id; NaT for patients with none.
    """
    end = _utc(horizon_end(cfg))
    seen = encounters[encounters["encounter_start"] < end]
    return seen.groupby("patient_id")["encounter_start"].max().reindex(pd.Index(patient_ids))


def build_labels(
    events: pd.DataFrame,
    patients: pd.DataFrame,
    encounters: pd.DataFrame,
    patient_ids: Sequence[str],
    cfg: Config,
) -> pd.DataFrame:
    """Build the labels of ``patient_ids`` (see the module docstring).

    Args:
        events: Condition events with patient_id, date and code, e.g.
            ``Trajectories.events``. Any dates; those on or after the horizon end
            are dropped.
        patients: ``data.load_patients`` output (patient_id, death_date).
        encounters: ``data.load_encounter_dates`` output.
        patient_ids: The cohort, e.g. ``features.load_model_cohort``; row order.
        cfg: Project configuration.

    Returns:
        One row per patient, every patient kept: patient_id, node (position in
        ``patient_ids``), one float label column per ``label_columns`` (NaN =
        masked), its status column (``STATUS_PREFIX`` + label), and the flags
        died_before_cutoff, died_in_horizon, lost_to_follow_up and
        selected_by_future (fewer than ``data.min_conditions_per_patient`` distinct
        codes before the cutoff: in the cohort only thanks to later codes; every
        label masked).
    """
    labels_cfg = cfg.labels
    ids = pd.Index(patient_ids, name="patient_id")
    cutoff, end = _utc(labels_cfg.cutoff_date), _utc(horizon_end(cfg))
    events = events[(events["date"] < end) & events["patient_id"].isin(ids)]

    death = patients.set_index("patient_id")["death_date"].reindex(ids)
    death = death.where(death < end)  # a death on or after the horizon end is not seen
    died_before = (death < cutoff).to_numpy(dtype=bool)
    died_in = ((death >= cutoff) & (death < end)).to_numpy(dtype=bool)
    last = last_encounters(encounters, ids, cfg)
    followed = (last >= _utc(labels_cfg.min_last_encounter_date)).to_numpy(dtype=bool)
    lost = ~died_before & ~died_in & ~followed
    codes_before = (
        events[events["date"] < cutoff].groupby("patient_id")["code"].nunique()
        .reindex(ids, fill_value=0)
    )
    future = (codes_before < cfg.data.min_conditions_per_patient).to_numpy(dtype=bool)

    frame = pd.DataFrame({"patient_id": ids.astype("string"), "node": np.arange(len(ids))})
    statuses = []
    for group, codes in labels_cfg.onset_groups.items():
        onset = events[events["code"].isin(codes)].groupby("patient_id")["date"].min().reindex(ids)
        in_horizon = (onset >= cutoff).to_numpy(dtype=bool)  # and before the end: filtered
        status = np.select(
            [
                died_before,
                future,
                (onset < cutoff).to_numpy(dtype=bool),
                lost & in_horizon,
                lost,
                in_horizon,
                died_in,
            ],
            [DIED_BEFORE_CUTOFF, SELECTED_BY_FUTURE, PREVALENT, LOST_AFTER_ONSET,
             LOST_TO_FOLLOW_UP, POSITIVE, DIED_WITHOUT_ONSET],
            default=NEGATIVE,
        )
        _set_label(frame, ONSET_PREFIX + group, status)
        statuses.append(status)
    _set_label(frame, ANY_ONSET, pooled_onset_status(np.column_stack(statuses)))
    _set_label(
        frame,
        MORTALITY,
        np.select(
            [died_before, future, died_in],
            [DIED_BEFORE_CUTOFF, SELECTED_BY_FUTURE, POSITIVE],
            default=NEGATIVE,
        ),
    )
    frame["died_before_cutoff"] = died_before
    frame["died_in_horizon"] = died_in
    frame["lost_to_follow_up"] = lost
    frame["selected_by_future"] = future
    logger.info(
        "Labels for %d patients: %d died before the cutoff, %d died in the horizon, %d lost to "
        "follow-up, %d selected by post-cutoff codes (all labels masked); %s: %d positive, "
        "%d observed",
        len(frame),
        int(died_before.sum()),
        int(died_in.sum()),
        int(lost.sum()),
        int(future.sum()),
        ANY_ONSET,
        int((frame[ANY_ONSET] == 1).sum()),
        int(frame[ANY_ONSET].notna().sum()),
    )
    return frame


def observed(labels: pd.DataFrame, column: str) -> np.ndarray:
    """Flag the entries of a label column that are not masked; metrics use only these."""
    return labels[column].notna().to_numpy()


def label_summary(
    labels: pd.DataFrame, columns: Sequence[str], by: str | None = None
) -> pd.DataFrame:
    """Count observed, positive and masked entries per label, optionally per group.

    Masked (NaN) entries are counted only as masked: never as observed, positive
    or negative, and never in the rate.

    Args:
        labels: ``build_labels`` output (optionally with a split or phenotype column).
        columns: Label columns to summarize.
        by: Optional column to group by, e.g. ``SPLIT``.

    Returns:
        ``SUMMARY_COLUMNS`` (with ``by`` first when given); rate is positives /
        observed, NaN when nothing is observed.
    """
    groups = [(None, labels)] if by is None else list(labels.groupby(by, sort=False))
    rows = []
    for key, part in groups:
        for column in columns:
            values = part[column].to_numpy(dtype=float)
            seen = values[~np.isnan(values)]
            row = {
                "label": column,
                "observed": len(seen),
                "positives": int((seen == 1).sum()),
                "negatives": int((seen == 0).sum()),
                "masked": int(np.isnan(values).sum()),
                "rate": seen.mean() if len(seen) else np.nan,
            }
            rows.append(row if by is None else {by: key, **row})
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS if by is None else [by, *SUMMARY_COLUMNS])


def attach_phenotypes(labels: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    """Add the Phase 4 phenotype cluster of each patient, for analysis only.

    Args:
        labels: ``build_labels`` output.
        clusters: ``graph.load_phenotype_clusters`` output (cluster by patient_id).

    Returns:
        A copy with ``PHENOTYPE`` (nullable integer; missing if unclustered).
    """
    cluster = clusters.reindex(pd.Index(labels["patient_id"])).to_numpy()
    return labels.assign(**{PHENOTYPE: pd.array(cluster, dtype="Int64")})


def split_strata(labels: pd.DataFrame) -> pd.Series:
    """Return each patient's stratum: phenotype cluster x ``PRIMARY_TARGET`` label.

    Unclustered patients get cluster "none"; a masked label is its own value,
    "masked".

    Args:
        labels: ``attach_phenotypes`` output.

    Returns:
        Stratum strings indexed by patient_id.
    """
    cluster = labels[PHENOTYPE].astype("string").fillna("none")
    target = labels[PRIMARY_TARGET].map({0.0: "0", 1.0: "1"}).fillna("masked")
    return pd.Series(
        (cluster + "|" + target).to_numpy(), index=pd.Index(labels["patient_id"]), name="stratum"
    )


def stratified_split(strata: pd.Series, fractions: Mapping[str, float], seed: int) -> pd.Series:
    """Split patients so each stratum is divided in the given fractions.

    Within a stratum, patients are shuffled with ``seed`` and each split gets the
    floor of its quota; the patients left over go to the splits with the largest
    remainders (ties in the order of ``fractions``). Strata are taken in sorted order.

    Args:
        strata: Stratum per patient, indexed by patient_id.
        fractions: Split name -> fraction, summing to 1.
        seed: Random seed.

    Returns:
        Split name per patient, indexed like ``strata``.
    """
    rng = np.random.default_rng(seed)
    names = list(fractions)
    shares = np.array([fractions[name] for name in names])
    split = pd.Series(pd.NA, index=strata.index, dtype="string", name=SPLIT)
    for stratum in sorted(strata.unique()):
        members = strata.index[(strata == stratum).to_numpy()]
        members = members[rng.permutation(len(members))]
        quota = len(members) * shares
        counts = np.floor(quota).astype(int)
        leftover = len(members) - int(counts.sum())
        counts[np.lexsort((np.arange(len(names)), -(quota - counts)))[:leftover]] += 1
        split[members] = np.repeat(names, counts)
    return split


def split_fractions(cfg: Config) -> dict[str, float]:
    """Return the configured split fractions by split name."""
    return {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test}


def save_labels(labels: pd.DataFrame, split: pd.Series, cfg: Config) -> None:
    """Write labels.parquet (with the split) and splits.json.

    Args:
        labels: ``attach_phenotypes`` output.
        split: ``stratified_split`` output for the same patients.
        cfg: Project configuration.

    Raises:
        ValueError: If a patient is in no split or the labels and split disagree.
    """
    ids = pd.Index(labels["patient_id"])
    if not split.index.equals(ids) or split.isna().any():
        raise ValueError("every labelled patient needs exactly one split")
    paths = cfg.paths
    paths.labels_parquet.parent.mkdir(parents=True, exist_ok=True)
    labels.assign(**{SPLIT: split.to_numpy()}).to_parquet(
        paths.labels_parquet, engine="pyarrow", index=False
    )
    logger.info("Wrote %d patients' labels to %s", len(labels), paths.labels_parquet)
    payload: dict[str, Any] = {
        "seed": cfg.seed,
        "fractions": split_fractions(cfg),
        "stratified_by": f"Phase 4 phenotype cluster x {PRIMARY_TARGET} label",
        "sizes": {name: int((split == name).sum()) for name in SPLITS},
    }
    payload.update({name: [str(p) for p in ids[(split == name).to_numpy()]] for name in SPLITS})
    paths.splits_json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    logger.info("Wrote %s: %s", paths.splits_json, payload["sizes"])


def load_labels(cfg: Config) -> pd.DataFrame:
    """Read labels.parquet written by ``save_labels``."""
    return pd.read_parquet(cfg.paths.labels_parquet)


def load_splits(cfg: Config) -> dict[str, tuple[str, ...]]:
    """Read splits.json: patient IDs per split.

    Raises:
        ValueError: If the splits overlap.
    """
    payload = json.loads(cfg.paths.splits_json.read_text(encoding="utf-8"))
    splits = {name: tuple(payload[name]) for name in SPLITS}
    members = [p for name in SPLITS for p in splits[name]]
    if len(members) != len(set(members)):
        raise ValueError(f"{cfg.paths.splits_json.name}: a patient is in more than one split")
    return splits


def write_labels_summary(
    labels: pd.DataFrame,
    phenotype_table: pd.DataFrame,
    descriptions: Mapping[str, str],
    cfg: Config,
) -> str:
    """Write the label report to ``cfg.paths.labels_summary_md``.

    Args:
        labels: ``build_labels`` output with ``PHENOTYPE`` and ``SPLIT``.
        phenotype_table: ``graph.cluster_phenotypes`` of the Phase 4 clusters.
        descriptions: Code -> description.
        cfg: Project configuration.

    Returns:
        The markdown written.
    """
    labels_cfg = cfg.labels
    onsets = onset_columns(cfg)
    columns = label_columns(cfg)
    cutoff, end = labels_cfg.cutoff_date, horizon_end(cfg)
    years = (
        f"{cutoff.year}-{end.year - 1}" if cutoff == date(cutoff.year, 1, 1)
        else f"{cutoff:%Y-%m-%d} to {end:%Y-%m-%d}"
    )
    n = len(labels)

    def status_count(column: str, status: str, rows: pd.DataFrame = labels) -> int:
        return int((rows[STATUS_PREFIX + column] == status).sum())

    def rate(positives: int, observed_: int) -> str:
        return f"{positives / observed_:.1%}" if observed_ else "n/a"

    membership_rows = [
        [group, "; ".join(f"{descriptions[c]} ({c})" for c in codes)]
        for group, codes in labels_cfg.onset_groups.items()
    ]
    group_rows = []
    for group, column in zip(labels_cfg.onset_groups, onsets):
        positives, negatives = status_count(column, POSITIVE), status_count(column, NEGATIVE)
        lost_after = status_count(column, LOST_AFTER_ONSET)
        lost_total = status_count(column, LOST_TO_FOLLOW_UP) + lost_after
        review = "keep, **borderline**" if group in labels_cfg.borderline_groups else "keep"
        if positives < labels_cfg.min_onsets:
            review += f", **< {labels_cfg.min_onsets} onsets**"
        group_rows.append(
            [
                group,
                f"{status_count(column, PREVALENT):,}",
                f"{positives:,}",
                f"{negatives:,}",
                rate(positives, positives + negatives),
                f"{lost_total:,} ({lost_after:,})",
                f"{status_count(column, DIED_WITHOUT_ONSET):,}",
                f"{status_count(column, DIED_BEFORE_CUTOFF):,}",
                f"{status_count(column, SELECTED_BY_FUTURE):,}",
                review,
            ]
        )

    pooled = label_summary(labels, [ANY_ONSET]).iloc[0]
    positive = (labels[ANY_ONSET] == 1).to_numpy()
    multi = int(((labels[onsets].to_numpy(dtype=float) == 1).sum(axis=1)[positive] > 1).sum())
    pooled_names = {PREVALENT: "had every group before cutoff"}
    pooled_rows = [
        [pooled_names.get(status, status), {POSITIVE: "1", NEGATIVE: "0"}.get(status, "masked"),
         f"{status_count(ANY_ONSET, status):,}"]
        for status in (POSITIVE, NEGATIVE, *(s for s in STATUSES if s not in (POSITIVE, NEGATIVE)))
        if status_count(ANY_ONSET, status)
    ]
    all_masked = int(labels[columns].isna().all(axis=1).sum())

    lost = labels["lost_to_follow_up"]
    died_in = labels["died_in_horizon"]
    future = labels["selected_by_future"]
    lost_positive = sum(status_count(c, LOST_AFTER_ONSET) for c in onsets)
    lost_entries = sum(status_count(c, LOST_TO_FOLLOW_UP) for c in onsets) + lost_positive
    died_entries = sum(status_count(c, DIED_WITHOUT_ONSET) for c in onsets)
    died_kept = sum(status_count(c, POSITIVE, labels[died_in]) for c in onsets)
    censor_rows = [
        ["Died before the cutoff: every label masked",
         f"{int(labels['died_before_cutoff'].sum()):,}", "all"],
        [f"Selected by post-cutoff codes (fewer than {cfg.data.min_conditions_per_patient} "
         "distinct codes before the cutoff): every label masked",
         f"{int(future.sum()):,}", "all"],
        [f"(a) Alive through the horizon, last encounter before "
         f"{labels_cfg.min_last_encounter_date:%Y-%m-%d}: every onset label masked",
         f"{int(lost.sum()):,}", f"{lost_entries:,}"],
        ["(b) Died in the horizon: onset labels without that onset masked",
         f"{int(died_in.sum()):,}", f"{died_entries:,}"],
    ]

    mortality = label_summary(labels, [MORTALITY]).iloc[0]

    by_split = label_summary(labels, columns, by=SPLIT)
    roles = {PRIMARY_TARGET: " (primary)", MORTALITY: " (descriptive only)"}
    split_rows = []
    for column in columns:
        cells = []
        for name in SPLITS:
            row = by_split[(by_split[SPLIT] == name) & (by_split["label"] == column)].iloc[0]
            cells.append(f"{row.positives:,} / {row.observed:,} ({rate(row.positives, row.observed)})")
        split_rows.append([column + roles.get(column, ""), *cells])
    size_row = ["Patients", *[f"{int((labels[SPLIT] == name).sum()):,}" for name in SPLITS]]

    phenotype_rows = []
    for cluster, part in phenotype_table.groupby("cluster", sort=True):
        top = ", ".join(_short(descriptions[c]) for c in part.sort_values("rank")["code"])
        phenotype_rows.append(
            [str(cluster), f"{int((labels[PHENOTYPE] == cluster).sum()):,}", top]
        )
    for cluster in sorted(set(labels[PHENOTYPE].dropna()) - set(phenotype_table["cluster"])):
        phenotype_rows.append(
            [str(cluster), f"{int((labels[PHENOTYPE] == cluster).sum()):,}",
             "(no chronic condition common enough to list)"]
        )
    phenotype_rows.sort(key=lambda r: int(r[0]))

    lines = [
        "# Labels",
        "",
        "Built by `scripts/05_build_labels.py` for the model cohort "
        f"({n:,} patients, every one a graph node whatever its labels). Inputs use windows "
        f"before the cutoff {cutoff:%Y-%m-%d}; labels use the horizon {cutoff:%Y-%m-%d} to "
        f"{end:%Y-%m-%d} (exclusive). Events, encounters and deaths on or after the horizon end "
        "are never read. Labels are 1, 0 or masked (NaN); every count and rate below uses "
        "unmasked entries only.",
        "",
        "## Targets",
        "",
        f"- **Primary:** `{ANY_ONSET}`: 1 if the patient has a new onset in any onset group in "
        f"{years}; 0 if at least one group label is observed and every observed one is 0; "
        "masked if every group label is masked.",
        f"- **Secondary:** the per-group labels `{ONSET_PREFIX}<group>`.",
        f"- **Not a target:** `{MORTALITY}`, kept for descriptive statistics only (see below).",
        "",
        "## Onset groups",
        "",
        "A group's onset is its first event, of any of its codes. Precedence: died before "
        "the cutoff (masked) > selected by post-cutoff codes (masked) > had it before the "
        "cutoff (masked) > (a) lost to follow-up (masked) > onset in the horizon (1) > (b) "
        "died in the horizon without it (masked) > 0.",
        "",
        markdown_table(["Group", "Codes"], membership_rows, "ll"),
        "",
        markdown_table(
            ["Group", "Had it before cutoff (masked)", f"New onsets {years}", "Negatives",
             "Positive rate", "Masked: (a) lost (after onset)", "Masked: (b) died",
             "Masked: died before", "Masked: selected by future", "Review"],
            group_rows,
            "lrrrrrrrrl",
        ),
        "",
        "Groups in `labels.borderline_groups` are kept but flagged **borderline**: they have "
        f"few new onsets for a review threshold of {labels_cfg.min_onsets}, so their metrics "
        "will be noisy.",
        "",
        f"## Primary target: {ANY_ONSET}",
        "",
        f"Positives / observed: {int(pooled.positives):,} / {int(pooled.observed):,} "
        f"({rate(int(pooled.positives), int(pooled.observed))}); masked: {int(pooled.masked):,}. "
        f"{multi:,} of the positives have new onsets in more than one group.",
        "",
        markdown_table([f"{ANY_ONSET} status", "Value", "Patients"], pooled_rows, "llr"),
        "",
        f"Patients with every target label masked ({ANY_ONSET} and every group): "
        f"{int(pooled.masked):,}. With {MORTALITY} masked too, so every label: {all_masked:,} "
        "(died before the cutoff or selected by post-cutoff codes).",
        "",
        "## Follow-up and censoring",
        "",
        markdown_table(["Rule", "Patients", "Onset label entries masked"], censor_rows, "lrr"),
        "",
        f"Rule (a) also masks {lost_positive:,} onsets that fall in the horizon (the patient "
        "was lost later); rule (b) keeps the "
        f"{died_kept:,} onsets seen before death as positives. Patients who died in the "
        "horizon are judged by rule (b) only, so (a) does not mask their onsets before death.",
        "",
        "## Mortality (descriptive only)",
        "",
        f"Observed: {int(mortality.observed):,} patients alive on {cutoff:%Y-%m-%d} and not "
        f"selected by post-cutoff codes. Died {cutoff:%Y-%m-%d} to {end:%Y-%m-%d} (exclusive): "
        f"{int(mortality.positives):,} ({rate(int(mortality.positives), int(mortality.observed))}). "
        f"Masked: {int(mortality.masked):,}.",
        "",
        f"Mortality is not a model target: {int(mortality.positives):,} deaths are too few to "
        "train on or to evaluate (per-split counts in the split table). The column stays in "
        "`labels.parquet` for descriptive statistics only.",
        "",
        "## Cohort selection check",
        "",
        f"{int(future.sum()):,} model-cohort patients have fewer than "
        f"{cfg.data.min_conditions_per_patient} distinct codes before the cutoff: they are in "
        "the cohort (Phase 2 rule, applied to all dates) only thanks to later codes, so they "
        "are certain to have a new condition after the cutoff and to be alive at it. They "
        "stay in the graph as nodes, with their pre-cutoff features, but every label of "
        f"theirs is masked: each onset group, `{ANY_ONSET}` and `{MORTALITY}`.",
        "",
        "## Split",
        "",
        f"Patient-level {cfg.split.train:.0%}/{cfg.split.val:.0%}/{cfg.split.test:.0%}, "
        f"stratified by Phase 4 phenotype cluster x `{PRIMARY_TARGET}` label (masked is its own "
        f"value), seed {cfg.seed}; saved to `{cfg.paths.splits_json.name}`. Positives / "
        "observed (rate):",
        "",
        markdown_table(["Label", *SPLITS], [size_row, *split_rows], "lrrr"),
        "",
        "## Phenotypes (Phase 4, attached for analysis only)",
        "",
        "The k = 7 clusters of the 2022 graph, reused as computed in Phase 4, with its top "
        "over-represented chronic conditions. They stratify the split and support analysis; "
        "they are never a training target.",
        "",
        markdown_table(["Cluster", "Patients", "Top over-represented conditions"],
                       phenotype_rows, "rrl"),
        "",
    ]
    text = "\n".join(lines)
    path = cfg.paths.labels_summary_md
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)
    return text


def _set_label(frame: pd.DataFrame, column: str, status: np.ndarray) -> None:
    """Add a label column (1.0, 0.0 or NaN) and its status column to ``frame``."""
    frame[column] = np.select([status == POSITIVE, status == NEGATIVE], [1.0, 0.0], default=np.nan)
    frame[STATUS_PREFIX + column] = pd.Categorical(status, categories=STATUSES)


def _short(description: str) -> str:
    """Drop a trailing semantic tag such as " (disorder)"."""
    return description.rsplit(" (", 1)[0] if description.endswith(")") else description


def _utc(day: date) -> pd.Timestamp:
    """Return 00:00 UTC of ``day``."""
    return pd.Timestamp(day.isoformat(), tz="UTC")
