"""Score validation-only baselines for any_onset against the Phase 6 TemporalGCN.

Logistic regression on the last training window's features or on every window's
features concatenated, each without and with age at the cutoff and sex, plus the
Phase 6 TemporalGCN setting (baselines.reference_model) retrained with seed 42,
without and with demographics. Same train/validation split and masked labels as
the GCN; test labels are never loaded. Writes reports/baseline_results.csv,
reports/baselines_summary.md and artifacts/baseline_val_scores.csv (the validation
scores 06_train.py compares the final model with). Run after 05_build_labels.py::

    conda run -n ehrgraph python scripts/06a_baselines.py [--config PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ehrgraph.config import DEFAULT_CONFIG_PATH, load_config
from ehrgraph.evaluate import run_baselines, save_baselines, write_baselines_summary
from ehrgraph.train import load_training_data
from ehrgraph.utils import get_logger, set_seed

logger = get_logger("baselines")


def main(argv: list[str] | None = None) -> None:
    """Run the baselines end to end."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="config YAML (default: config/config.yaml)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    data = load_training_data(cfg)
    results, val_scores = run_baselines(data, cfg)
    save_baselines(results, val_scores, cfg)
    write_baselines_summary(results, data, cfg)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        logger.info("Validation baselines:\n%s", results.to_string(index=False))
    logger.info("Baselines finished")


if __name__ == "__main__":
    main()
