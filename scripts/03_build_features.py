"""Build per-window patient features, the model cohort and the comorbidity matrices.

Reads the windows, trajectories and code vocabulary from data/processed/ (from
02_build_trajectories.py) and the clinical events (from 01_prepare_data.py), then
writes features_<year>.npz per window and features.json to data/processed/features/,
model_cohort.json and the comorbidity counts and lift (all codes and chronic codes)
to data/processed/, the fitted IDF and SVD arrays to artifacts/, and
reports/features_summary.md with the chronic comorbidity heatmap (all paths from
config/config.yaml). Redundant general codes (features.redundant_codes) are left out
of the features, and only transitions that end before labels.cutoff_date are used::

    conda run -n ehrgraph python scripts/03_build_features.py [--config PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ehrgraph.config import load_config
from ehrgraph.features import (
    build_chronic_comorbidity,
    build_comorbidity,
    build_features,
    co_recorded_codes,
    feature_transitions,
    plot_comorbidity_heatmap,
    save_features,
    write_features_summary,
)
from ehrgraph.trajectories import load_code_vocabulary, load_trajectories, load_windows
from ehrgraph.utils import get_logger, set_seed

logger = get_logger("build_features")


def main(argv: list[str] | None = None) -> None:
    """Run the feature pipeline end to end."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=None, help="config YAML (default: config/config.yaml)"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    windows = load_windows(cfg)
    trajectories = load_trajectories(cfg)
    vocabulary = load_code_vocabulary(cfg)
    descriptions = dict(zip(vocabulary["code"], vocabulary["description"]))
    events = pd.read_parquet(cfg.paths.events_clinical_parquet)
    co_recorded = co_recorded_codes(
        events[events["patient_id"].isin(windows.patient_ids)],
        cfg.features.redundant_min_share,
        cfg.features.redundant_codes,
    )

    features = build_features(windows, feature_transitions(trajectories, cfg), cfg)
    comorbidity = build_comorbidity(windows, cfg)
    chronic = build_chronic_comorbidity(trajectories, cfg)
    save_features(features, comorbidity, chronic, cfg)
    plot_comorbidity_heatmap(
        chronic, descriptions, cfg.paths.comorbidity_heatmap_fig, cfg,
        min_frequency=cfg.summary.chronic_min_patients,
    )
    write_features_summary(features, comorbidity, chronic, co_recorded, descriptions, cfg)
    logger.info("Feature building finished")


if __name__ == "__main__":
    main()
