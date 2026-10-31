"""Phase 8 demo: place the example patients among the reference patients and show the results.

Loads the predictor once (inference.Predictor), predicts the inference.demo_patients of
config/config.yaml (a healthy young adult, a metabolic-syndrome pattern, a multimorbid
elderly patient, a chronic-pain and musculoskeletal patient: realistic profiles, not chosen
to land in different phenotypes) and logs each result: the nearest phenotype, the risk set
(the onset groups the patient does not already have), the most similar reference patients
and their shared chronic conditions, the neighbours' onsets in 2023-2025, the supervised
model's calibrated any_onset probability, the explanation (lift) and the trajectory. The
same text, with a summary table, goes to reports/predict_demo.md::

    conda run -n ehrgraph python scripts/08_predict_demo.py [--config PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ehrgraph.config import DEFAULT_CONFIG_PATH, Config, DemoPatientConfig
from ehrgraph.data import DAYS_PER_YEAR, markdown_table
from ehrgraph.inference import (
    RISK_SET_EXPLANATION,
    NewPatient,
    Prediction,
    Predictor,
    format_share,
)
from ehrgraph.labels import ANY_ONSET
from ehrgraph.utils import get_logger, set_seed

logger = get_logger("predict_demo")

SEX_NAMES = {"F": "female", "M": "male"}


def patient_section(
    number: int, demo: DemoPatientConfig, prediction: Prediction, explanation: pd.DataFrame,
    predictor: Predictor, seconds: float,
) -> str:
    """Render one example patient's prediction as markdown."""
    cfg = predictor.cfg
    cutoff = cfg.labels.cutoff_date
    age = (cutoff - demo.birth_date).days / DAYS_PER_YEAR
    lines = [f"## {number}. {demo.name}", "",
             f"{SEX_NAMES[demo.sex].capitalize()}, born {demo.birth_date} (age {age:.1f} at the "
             f"cutoff {cutoff}). {len(demo.history)} history entries; prediction in {seconds:.2f} s.",
             ""]
    lines += ["**Conditions used** (onset year; chronic = active from onset through the last "
              "window):", ""]
    lines.append(markdown_table(
        ["Onset", "Condition", "Code", "Chronic"],
        [[str(row.onset.year), row.condition, row.code, "yes" if row.chronic else "no"]
         for row in prediction.conditions.itertuples()], "lllc"))
    lines.append("")
    if prediction.notes:
        lines += ["**Input notes.**", "", *[f"- {note}" for note in prediction.notes], ""]
    if not prediction.placed:
        return "\n".join(lines)

    phenotype = prediction.phenotype
    lines += [f"**Nearest phenotype: cluster {phenotype.cluster}** ({phenotype.patients} reference "
              f"patients). Most over-represented chronic conditions: {phenotype.description}.", ""]
    lines += [f"**{prediction.risk_set}** ({group_list(prediction.at_risk_groups)}). Already "
              f"has: {group_list(prediction.prior_groups)}.", ""]

    probability = prediction.any_onset_probability
    any_row = prediction.outcomes.set_index("target").loc[ANY_ONSET]
    base = any_row["positives_all"] / any_row["observed_all"]
    if probability is not None:
        lines += [f"**Supervised `any_onset` probability: {probability:.1%}** (TemporalGCN, "
                  f"Platt-calibrated; {predictor.model_label}, an interval that includes 0.5, so "
                  f"near chance; reference base rate {base:.1%}).", ""]

    sizes = cfg.inference.outcome_neighbours
    lines += [f"**Neighbour outcomes**: new onsets in {cutoff.year}-{cutoff.year + cfg.labels.horizon_years - 1} "
              "among the nearest reference patients in the spectral embedding, over those whose "
              "label is observed (masked labels never count).", ""]
    rows = []
    for row in prediction.outcomes.itertuples():
        name = "any of the groups" if pd.isna(row.group) else row.group.replace("_", " ")
        had = "" if pd.isna(row.patient_had_it) else ("yes" if row.patient_had_it else "no")
        rows.append([name, had,
                     *[format_share(getattr(row, f"positives_{n}"), getattr(row, f"observed_{n}"))
                       for n in sizes],
                     format_share(row.positives_all, row.observed_all)])
    lines.append(markdown_table(["Onset", "Patient already had it",
                                 *[f"{n} nearest" for n in sizes], "All reference patients"],
                                rows, "ll" + "r" * (len(sizes) + 1)))
    lines.append("")

    lines += [f"**{len(prediction.similar)} most similar reference patients** (cosine similarity "
              f"in the {cutoff.year - 1} spectral embedding) and the chronic conditions they share "
              "with this patient:", ""]
    lines.append(markdown_table(
        ["#", "Patient", "Similarity", "Age", "Sex", "Phenotype", "Shared / their chronic",
         "Shared chronic conditions"],
        [[str(r.rank), r.patient_id[:8], f"{r.similarity:.3f}", f"{r.age:.0f}", r.sex,
          str(r.phenotype), f"{r.shared_conditions} / {r.chronic_conditions}", r.shared or "-"]
         for r in prediction.similar.itertuples()], "rlrrlrrl"))
    lines.append("")

    lines += [f"**Explanation**: the patient's chronic conditions most over-represented among the "
              f"{cfg.inference.explain_neighbours} nearest reference patients (lift = share there / "
              "share among all reference patients):", ""]
    if explanation.empty:
        lines += ["No chronic condition: the placement rests on acute conditions and transitions.",
                  ""]
    else:
        lines.append(markdown_table(
            ["Condition", "Neighbours with it", "All reference patients", "Lift"],
            [[r.condition, f"{r.neighbours_with}/{r.neighbours} ({r.neighbourhood_share:.0%})",
              f"{r.population_share:.1%}", f"{r.lift:.1f}"] for r in explanation.itertuples()],
            "lrrr"))
        lines.append("")

    states = prediction.window_states
    steps = np.linalg.norm(np.diff(states, axis=0), axis=1)
    lines += ["**Trajectory** (TemporalGraphAE GRU state after each window, inserted into each "
              "year's graph; the same space as the reference patients' trajectories):", ""]
    lines.append(markdown_table(
        ["Window", "Graph neighbours", "Mean similarity", "Step from previous year"],
        [[str(ins.year), str(len(ins.neighbours)),
          f"{ins.weights.mean():.3f}" if len(ins.weights) else "- (isolated)",
          "" if t == 0 else f"{steps[t - 1]:.3f}"]
         for t, ins in enumerate(prediction.insertions)], "lrrr"))
    lines.append("")
    return "\n".join(lines)


def group_list(groups: tuple[str, ...]) -> str:
    """Onset groups in words, e.g. "ckd, ischemic heart disease", or "none"."""
    return ", ".join(group.replace("_", " ") for group in groups) or "none"


def elderly_multimorbid_line(predictor: Predictor) -> str:
    """Where the elderly multimorbid reference patients fall (``inference.elderly_min_*``)."""
    inf = predictor.cfg.inference
    ages = predictor.demographics["age"].to_numpy(dtype=float)
    chronic = np.diff(predictor.chronic.indptr)
    elderly = (ages >= inf.elderly_min_age) & (chronic >= inf.elderly_min_chronic)
    counts = pd.Series(predictor.clusters[elderly]).value_counts()
    return (f"Phenotype {int(counts.index[0])} holds {int(counts.iloc[0])} of the "
            f"{int(elderly.sum())} reference patients aged {inf.elderly_min_age}+ with "
            f"{inf.elderly_min_chronic}+ chronic conditions (a cardio-metabolic continuum).")


def summary_table(demos: list[DemoPatientConfig], predictions: list[Prediction],
                  cfg: Config) -> str:
    """One row per example patient: phenotype, risk set and neighbour any_onset rates."""
    sizes = cfg.inference.outcome_neighbours
    rows = []
    for number, (demo, prediction) in enumerate(zip(demos, predictions), start=1):
        any_row = prediction.outcomes.set_index("target").loc[ANY_ONSET]
        rows.append([
            str(number), demo.name,
            str(prediction.phenotype.cluster) if prediction.placed else "-",
            prediction.risk_set.removeprefix("At risk for "),
            *[format_share(any_row[f"positives_{n}"], any_row[f"observed_{n}"]) for n in sizes],
            format_share(any_row["positives_all"], any_row["observed_all"]),
        ])
    return markdown_table(["#", "Patient", "Phenotype", "At risk for",
                           *[f"Neighbour any_onset, {n} nearest" for n in sizes],
                           "All reference patients"], rows, "llrl" + "r" * (len(sizes) + 1))


def report(sections: list[str], summary: str, predictor: Predictor, load_seconds: float,
           cfg: Config) -> str:
    """The whole demo report."""
    header = [
        f"# Phase 8: predictions for {len(sections)} example patients", "",
        "Built by `scripts/08_predict_demo.py` with `inference.Predictor`. Each patient is placed "
        f"among the {len(predictor.patient_ids):,} reference (model-cohort) patients without "
        "refitting anything: features from the saved transforms; in each year's k-NN graph "
        f"the patient is linked to their {cfg.graph.k} most similar reference patients "
        "(normalization updated for the affected nodes only); the Nystrom extension places "
        f"them in the {cfg.labels.cutoff_date.year - 1} spectral embedding, where the phenotype, "
        "the similar patients and the neighbour outcomes are read. "
        f"Predictor loaded in {load_seconds:.2f} s. Synthetic data (Synthea).", "",
        "The examples are realistic profiles, not chosen to land in different phenotypes: the "
        "multimorbid elderly patient lands in the same phenotype as the metabolic-syndrome "
        f"pattern. {elderly_multimorbid_line(predictor)}", "",
        summary, "",
        f"**Risk set.** `{ANY_ONSET}` is a new onset in {cfg.labels.cutoff_date.year}-"
        f"{cfg.labels.cutoff_date.year + cfg.labels.horizon_years - 1} in any of the "
        f"{len(cfg.labels.onset_groups)} onset groups; a group a patient already has is masked, so "
        "each neighbour counts only for the groups they were still at risk of. "
        f"{RISK_SET_EXPLANATION}", "",
    ]
    return "\n".join(header + sections)


def main(argv: list[str] | None = None) -> None:
    """Run the Phase 8 demo."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="config YAML (default: config/config.yaml)")
    args = parser.parse_args(argv)

    started = perf_counter()
    predictor = Predictor.load(config_path=args.config)
    load_seconds = perf_counter() - started
    cfg = predictor.cfg
    set_seed(cfg.seed)

    sections, predictions = [], []
    demos = list(cfg.inference.demo_patients)
    for number, demo in enumerate(demos, start=1):
        patient = NewPatient(history=list(demo.history), birth_date=demo.birth_date, sex=demo.sex)
        started = perf_counter()
        prediction = predictor.predict(patient)
        explanation = predictor.explain(prediction)
        seconds = perf_counter() - started
        section = patient_section(number, demo, prediction, explanation, predictor, seconds)
        logger.info("Example patient %d:\n%s", number, section)
        sections.append(section)
        predictions.append(prediction)
    summary = summary_table(demos, predictions, cfg)
    logger.info("Summary:\n%s", summary)

    path = cfg.paths.predict_demo_md
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report(sections, summary, predictor, load_seconds, cfg), encoding="utf-8")
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
