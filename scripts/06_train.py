"""Train the temporal GCN on the onset targets, calibrate it, save the model and embeddings.

Reads the model inputs (03_build_features.py), the per-window graphs (04) and the
labels and split (05), trains full batch with early stopping on the validation
AUROC of any_onset, and writes artifacts/model.pt (state_dict),
artifacts/config_used.yaml, artifacts/train_log.csv, artifacts/train_metrics.json,
the embeddings artifacts/embeddings_per_window.npy [N, T, d] and
artifacts/trajectory_embeddings.npy [N, d], reports/figures/training_curves.png and
reports/training_summary.md (all paths from config/config.yaml). Training is
transductive: every patient passes messages, only training labels enter the loss,
and test labels are never read.

Then calibrates the validation logits of any_onset with calibration.method (Platt
scaling by default; artifacts/calibration.json, reports/figures/reliability_any_onset.png)
and, if 06a_baselines.py has run, compares the model with the baselines on the same
validation bootstrap resamples. --quick runs model.quick_epochs epochs on
model.quick_patients patients, skips the comparison and writes to paths.quick_dir::

    conda run -n ehrgraph python scripts/06_train.py [--quick] [--config PATH]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ehrgraph.config import DEFAULT_CONFIG_PATH, Config, load_config
from ehrgraph.evaluate import (
    Calibration,
    calibrate,
    calibration_section,
    compare_models,
    comparison_section,
    load_baseline_scores,
    plot_reliability,
    save_calibration,
    sigmoid,
    validation_logits,
)
from ehrgraph.train import (
    TrainOutputs,
    fit,
    load_training_data,
    plot_training_curves,
    save_training,
    validation_report,
    write_training_summary,
)
from ehrgraph.utils import get_logger, set_seed

logger = get_logger("train")

FINAL = "final"


def main(argv: list[str] | None = None) -> None:
    """Run the training pipeline end to end."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="smoke test: model.quick_epochs epochs on model.quick_patients patients",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    model_cfg = cfg.model
    data = load_training_data(cfg, model_cfg.quick_patients if args.quick else None)
    result = fit(data, model_cfg, cfg.seed, model_cfg.quick_epochs if args.quick else None)
    report = validation_report(result.model, data, result.pos_weights)
    outputs = TrainOutputs.from_config(cfg, args.quick)
    save_training(result, report, data, cfg, outputs, args.config, args.quick)
    plot_training_curves(result.history, result.best_epoch, outputs.training_curves_fig, cfg)

    logits, targets, patient_ids = validation_logits(result.model, data)
    calibration = calibrate(logits, targets, cfg.calibration.method, cfg.evaluation, cfg.seed)
    save_calibration(calibration, outputs.calibration_json)
    plot_reliability(calibration, outputs.reliability_fig, cfg)
    sections = _comparison(cfg, logits, targets, patient_ids, data.years, args.quick)
    sections += _calibration(calibration, outputs, cfg)
    write_training_summary(result, report, data, cfg, outputs, args.quick, sections)
    logger.info("Validation metrics at the best epoch:\n%s", report.to_string(index=False))
    logger.info("Training finished")


def _comparison(
    cfg: Config,
    logits: np.ndarray,
    targets: np.ndarray,
    patient_ids: np.ndarray,
    years: tuple[int, ...],
    quick: bool,
) -> list[str]:
    """Markdown comparing the model with the saved baselines (the model alone if none)."""
    inputs = f"{len(years)} windows + graphs"
    if cfg.model.use_demographics:
        inputs += ", age and sex in the heads"
    labels = {FINAL: ("TemporalGCN, final", inputs)}
    scores = {FINAL: sigmoid(logits)}
    baseline = None if quick else load_baseline_scores(cfg)
    results_path = cfg.paths.baseline_results_csv
    if baseline is not None and results_path.exists():
        if baseline["patient_id"].tolist() != list(patient_ids) or not np.array_equal(
            baseline["any_onset"].to_numpy(), targets
        ):
            logger.warning("Baseline scores are for other validation patients or labels; "
                           "re-run scripts/06a_baselines.py. Skipping the comparison.")
        else:
            results = pd.read_csv(results_path)
            for row in results.itertuples():
                labels[row.key] = (row.model, row.inputs)
                scores[row.key] = baseline[row.key].to_numpy()
    elif not quick:
        logger.warning("No baseline scores; run scripts/06a_baselines.py to compare")
    table = compare_models(targets, scores, FINAL, cfg.evaluation, cfg.seed)
    final = table.iloc[0]
    logger.info(
        "Final model: validation AUROC %.4f (%.4f to %.4f), AUPRC %.4f (%.4f to %.4f)",
        final.val_auroc, final.val_auroc_low, final.val_auroc_high, final.val_auprc,
        final.val_auprc_low, final.val_auprc_high,
    )
    return comparison_section(table, labels, FINAL, cfg.evaluation)


def _calibration(calibration: Calibration, outputs: TrainOutputs, cfg: Config) -> list[str]:
    """Markdown for the calibration, with the figure linked relative to the report."""
    figure = Path(os.path.relpath(outputs.reliability_fig, outputs.training_summary_md.parent))
    return calibration_section(calibration, figure.as_posix(), outputs.calibration_json.name, cfg)


if __name__ == "__main__":
    main()
