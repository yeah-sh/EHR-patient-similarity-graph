"""Build the onset labels for the model cohort and the train/val/test split.

Reads the model cohort (03_build_features.py), the trajectories (02), the raw
patients and encounters tables and the Phase 4 phenotype clusters (04), then writes
data/processed/labels.parquet, data/processed/splits.json and
reports/labels_summary.md (all paths from config/config.yaml). The primary target is
any_onset, the per-group onsets are secondary, and mortality is descriptive only.
Labels use only the horizon [labels.cutoff_date, cutoff + labels.horizon_years); the
phenotype clusters are reused as computed in Phase 4, for analysis and stratification
only::

    conda run -n ehrgraph python scripts/05_build_labels.py [--config PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ehrgraph.config import load_config
from ehrgraph.data import load_encounter_dates, load_patients
from ehrgraph.features import chronic_presence, load_model_cohort
from ehrgraph.graph import cluster_phenotypes, load_phenotype_clusters
from ehrgraph.labels import (
    SPLIT,
    attach_phenotypes,
    build_labels,
    save_labels,
    split_fractions,
    split_strata,
    stratified_split,
    write_labels_summary,
)
from ehrgraph.trajectories import load_code_vocabulary, load_trajectories
from ehrgraph.utils import get_logger, set_seed

logger = get_logger("build_labels")


def main(argv: list[str] | None = None) -> None:
    """Run the label pipeline end to end."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=None, help="config YAML (default: config/config.yaml)"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    patient_ids, _ = load_model_cohort(cfg)
    trajectories = load_trajectories(cfg)
    labels = build_labels(
        trajectories.events, load_patients(cfg), load_encounter_dates(cfg), patient_ids, cfg
    )
    clusters = load_phenotype_clusters(cfg)
    labels = attach_phenotypes(labels, clusters)
    split = stratified_split(split_strata(labels), split_fractions(cfg), cfg.seed)
    save_labels(labels, split, cfg)

    # The Phase 4 description of each cluster, from its saved members (not re-clustered).
    presence, codes = chronic_presence(trajectories, cfg)
    members = pd.Index(trajectories.patient_ids).get_indexer(clusters.index)
    phenotypes = cluster_phenotypes(
        presence[members], clusters.to_numpy(), codes, cfg.graph.phenotype_top_conditions,
        cfg.graph.phenotype_min_patients,
    )
    vocabulary = load_code_vocabulary(cfg)
    descriptions = dict(zip(vocabulary["code"], vocabulary["description"]))
    write_labels_summary(labels.assign(**{SPLIT: split.to_numpy()}), phenotypes, descriptions, cfg)
    logger.info("Label building finished")


if __name__ == "__main__":
    main()
