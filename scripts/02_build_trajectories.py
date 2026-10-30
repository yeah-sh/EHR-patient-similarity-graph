"""Build comorbidity trajectories, yearly active-condition windows and transitions.

Reads data/processed/events_clinical.parquet (from 01_prepare_data.py) and the raw
patients table, then writes the trajectories, the window matrices, the patient
index, the code vocabulary and the transition counts to data/processed/, and
reports/trajectories_summary.md (all paths from config/config.yaml). The cohort is
the patients with enough distinct codes and a condition active in some window::

    conda run -n ehrgraph python scripts/02_build_trajectories.py [--config PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ehrgraph.config import load_config
from ehrgraph.data import load_patients
from ehrgraph.trajectories import (
    build_trajectories,
    extract_transitions,
    save_trajectories,
    window_trajectories,
    write_trajectory_summary,
)
from ehrgraph.utils import get_logger, set_seed

logger = get_logger("build_trajectories")


def main(argv: list[str] | None = None) -> None:
    """Run the trajectory pipeline end to end."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=None, help="config YAML (default: config/config.yaml)"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    events = pd.read_parquet(cfg.paths.events_clinical_parquet)
    patients = load_patients(cfg)
    trajectories = build_trajectories(events, patients, cfg)
    windows = window_trajectories(trajectories, cfg)
    transitions = extract_transitions(trajectories)
    save_trajectories(trajectories, windows, transitions, cfg)
    write_trajectory_summary(events, patients, trajectories, windows, transitions, cfg)
    logger.info("Trajectory building finished")


if __name__ == "__main__":
    main()
