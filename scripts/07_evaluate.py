"""Evaluate the TemporalGCN: cross-validated comparison, single-shot test, embedding analysis.

Every hyperparameter stays as in config/config.yaml; nothing is tuned here.

1. Cross-validation (the main comparison): every patient with an observed any_onset,
   cross_validation.folds folds stratified by phenotype cluster x any_onset. In each
   fold an inner validation set picks the GCNs' epochs and fits every model's
   calibration; the GCNs are trained once per seed in cross_validation.seeds. Eight
   models: age and sex only, logistic regressions on the final window and on every
   window, spectral embedding + logistic regression, a static GCN, the temporal model
   without a graph (A_hat = I), the TemporalGCN on rewired graphs, and the full
   TemporalGCN. Writes artifacts/cv_oof_predictions.csv and artifacts/cv_fits.csv.
2. Test set, scored once: the final model of 06_train.py (artifacts/model.pt) with its
   validation calibration (artifacts/calibration.json). Writes
   artifacts/test_predictions.csv.
3. Embedding analysis of the final model: UMAP figures, cluster separation, nearest
   neighbours, trajectories.
4. reports/results.md, reports/results.csv, reports/evaluation_summary.md and the
   figures in reports/figures/.

Run after 06_train.py. --reuse-cv reads the saved out-of-fold predictions instead of
re-running step 1::

    conda run -n ehrgraph python scripts/07_evaluate.py [--reuse-cv] [--config PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ehrgraph.config import DEFAULT_CONFIG_PATH, load_config
from ehrgraph.evaluate import (
    ABLATIONS,
    FULL_MODEL,
    analyse_embeddings,
    calibrated_label,
    cv_by_split,
    cv_fold_metrics,
    cv_group_results,
    cv_results,
    evaluate_test,
    f1_threshold,
    held_out_targets,
    load_cross_validation,
    load_patient_context,
    load_spectral_embedding,
    plot_cv_auroc,
    plot_embeddings,
    plot_reliability_bins,
    require_trained_model,
    run_cross_validation,
    save_cross_validation,
    test_predictions_frame,
    validation_logits,
    write_evaluation_summary,
    write_results,
)
from ehrgraph.labels import ANY_ONSET
from ehrgraph.train import load_model_data, load_trained_model, make_training_data, predict
from ehrgraph.utils import get_logger, set_seed

logger = get_logger("evaluate")


def main(argv: list[str] | None = None) -> None:
    """Run the Phase 7 evaluation end to end."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--reuse-cv", action="store_true",
        help="read the saved out-of-fold predictions instead of re-running the cross-validation",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    calibrator = require_trained_model(args.config, cfg)
    model_data = load_model_data(cfg)
    spectral = load_spectral_embedding(cfg, len(model_data.patient_ids))

    # 1. Cross-validation: the main comparison.
    if args.reuse_cv:
        cv = load_cross_validation(cfg, model_data, spectral.shape[1])
        logger.info("Read the out-of-fold predictions of %d patients", len(cv.predictions))
    else:
        cv = run_cross_validation(model_data, spectral, cfg)
        save_cross_validation(cv, cfg)
    table = cv_results(cv, cfg)
    fold_metrics = cv_fold_metrics(cv)
    group_table = cv_group_results(cv, cfg)
    by_split = cv_by_split(cv, model_data.labels)
    plot_cv_auroc(table, cfg.paths.cv_auroc_fig, cfg)

    # 2. The held-out test set: the final model, scored once.
    data = make_training_data(
        model_data.patient_ids, model_data.years, model_data.matrices, model_data.weights,
        model_data.adjacency, model_data.labels, model_data.groups, model_data.demographics,
    )
    model = load_trained_model(cfg.model, cfg.paths.model_pt, data.features[0].shape[1],
                               len(data.groups))
    output = predict(model, data)
    val_logits, val_targets, _ = validation_logits(model, data)
    val_auroc = float(roc_auc_score(val_targets, val_logits))
    threshold = f1_threshold(val_targets, calibrator.apply(val_logits))
    logits = output.any_onset_logits.numpy().astype(np.float64)
    onset_logits = output.onset_logits.numpy().astype(np.float64)
    targets, onset_targets = held_out_targets(model_data)
    test = evaluate_test(logits, onset_logits, targets, onset_targets, data.groups, calibrator,
                         threshold, cfg)
    predictions = test_predictions_frame(data.patient_ids, logits, onset_logits, targets,
                                         onset_targets, data.groups, calibrator)
    cfg.paths.test_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(cfg.paths.test_predictions_csv, index=False)
    logger.info("Wrote %s", cfg.paths.test_predictions_csv)
    plot_reliability_bins(
        test.bins_raw, test.bins_calibrated, calibrated_label(calibrator),
        f"Reliability of {ANY_ONSET}, test (n = {test.observed}, {len(test.bins_raw)} "
        "equal-count bins)", cfg.paths.test_reliability_fig, cfg,
    )

    # 3. Embedding analysis of the final model.
    context = load_patient_context(cfg, model_data)
    analysis = analyse_embeddings(
        output.window_embeddings.numpy(), output.trajectory_embeddings.numpy(), spectral,
        model_data.matrices[-1], context, cfg,
    )
    plot_embeddings(analysis, context, cfg)

    # 4. Reports.
    write_results(table, fold_metrics, group_table, cv, by_split, test, analysis, context,
                  val_auroc, cfg)
    write_evaluation_summary(table, group_table, cv, by_split, test, analysis, val_auroc, cfg)
    columns = ["number", "model", "oof_auroc", "oof_auroc_low", "oof_auroc_high", "oof_auprc",
               "delta_auroc", "delta_auroc_low", "delta_auroc_high", "folds_full_better"]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        logger.info("Cross-validation, sorted by pooled AUROC:\n%s",
                    table[columns].to_string(index=False))
    by_key = table.set_index("key")
    for question, key in ABLATIONS:
        row = by_key.loc[key]
        logger.info("%s full minus %s: %+.4f (%+.4f to %+.4f)", question, key, row.delta_auroc,
                    row.delta_auroc_low, row.delta_auroc_high)
    logger.info("Full model (%s) pooled AUROC %.4f; test AUROC %.4f %s", FULL_MODEL,
                by_key.loc[FULL_MODEL, "oof_auroc"], test.calibrated["auroc"],
                test.intervals["auroc"])
    logger.info("Evaluation finished")


if __name__ == "__main__":
    main()
