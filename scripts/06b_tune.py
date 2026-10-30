"""Tune the TemporalGCN on the validation split (any_onset AUROC), several seeds per setting.

Trains every setting of the tuning grid in config/config.yaml (or a random subset
of tuning.max_settings) once per seed in tuning.seeds, with model.lr, early stopping
on the validation AUROC of any_onset (model.patience) and every other model value
from the config. Writes every run to reports/tuning_results.csv (after each setting,
so a stopped run keeps its results) and the ranking, marginal means and chosen
setting to reports/tuning_summary.md. Test labels are never loaded. Copy the chosen
setting into the model section of config/config.yaml, then run 06_train.py::

    conda run -n ehrgraph python scripts/06b_tune.py [--config PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ehrgraph.config import DEFAULT_CONFIG_PATH, load_config
from ehrgraph.train import (
    TUNED_FIELDS,
    load_training_data,
    select_setting,
    summarize_tuning,
    tune,
    tuning_settings,
    write_tuning_summary,
)
from ehrgraph.utils import get_logger, set_seed

logger = get_logger("tune")


def main(argv: list[str] | None = None) -> None:
    """Run the tuning search end to end."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="config YAML (default: config/config.yaml)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    tuning = cfg.tuning
    data = load_training_data(cfg)
    n_grid = (len(tuning.weight_decay) * len(tuning.dropout) * len(tuning.dims)
              * len(tuning.smoothness_weight) * len(tuning.drop_edge))
    settings = tuning_settings(tuning, cfg.seed)
    logger.info("Tuning %d settings x %d seeds", len(settings), len(tuning.seeds))
    path = cfg.paths.tuning_results_csv
    path.parent.mkdir(parents=True, exist_ok=True)

    def save(runs: pd.DataFrame) -> None:
        runs.to_csv(path, index=False)

    results = tune(data, cfg.model, settings, tuning.seeds, on_progress=save)
    save(results)
    summary = summarize_tuning(results)
    chosen = select_setting(summary, tuning.tie_tolerance)
    write_tuning_summary(results, summary, chosen, cfg, n_grid)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        logger.info("Top settings:\n%s", summary.head(tuning.top_n_report).to_string(index=False))
    logger.info(
        "Chosen setting %d; set these in the model section of config.yaml: %s",
        int(chosen.setting), {name: chosen[name] for name in TUNED_FIELDS},
    )


if __name__ == "__main__":
    main()
