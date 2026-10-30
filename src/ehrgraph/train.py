"""Full-batch training of the temporal GCN on the onset targets.

Pipeline (run end to end by ``scripts/06_train.py``)::

    data = load_training_data(cfg)  # --quick: n_patients=cfg.model.quick_patients
    result = fit(data, cfg.model, cfg.seed)  # --quick: epochs=cfg.model.quick_epochs
    report = validation_report(result.model, data, result.pos_weights)
    outputs = TrainOutputs.from_config(cfg, quick)
    save_training(result, report, data, cfg, outputs, config_path, quick)
    plot_training_curves(result.history, result.best_epoch, outputs.training_curves_fig, cfg)
    write_training_summary(result, report, data, cfg, outputs, quick)

Training is transductive. Every model-cohort patient is a graph node and passes
messages with its pre-cutoff features (the windows of ``features.load_model_inputs``):
training, validation and test patients alike, and patients whose labels are
masked. Only the observed labels of training patients enter the loss. Validation
labels pick the epoch (early stopping on the validation AUROC of ``any_onset``)
and give the reported metrics. Test labels are set to NaN on loading, so this
phase cannot read them.

Targets: ``any_onset`` (primary) and the per-group onsets (secondary); see
``labels``. The loss is ``BCE(any_onset) + model.secondary_weight * masked
BCE(groups)``, each with ``pos_weight = negatives / positives`` of the training
split, plus ``model.smoothness_weight * models.graph_smoothness`` when that weight
is positive. With ``model.drop_edge`` > 0 each training step sees graphs with that
fraction of edges dropped (``models.EdgeDropper``); evaluation uses the full graphs.
With ``model.use_demographics`` the heads also see age at the cutoff (standardized
with training-split statistics) and sex.

Tuning (``scripts/06b_tune.py``, validation only)::

    settings = tuning_settings(cfg.tuning, cfg.seed)
    results = tune(data, cfg.model, settings, cfg.tuning.seeds)  # one row per run
    summary = summarize_tuning(results)
    best = select_setting(summary, cfg.tuning.tie_tolerance)
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor

from ehrgraph.config import Config, ModelConfig, TuningConfig
from ehrgraph.data import chart_style, load_patients, markdown_table, save_figure
from ehrgraph.features import DEMOGRAPHICS, load_model_inputs, patient_demographics
from ehrgraph.graph import load_graphs, normalize_adjacency
from ehrgraph.labels import ANY_ONSET, ONSET_PREFIX, SPLIT, load_labels
from ehrgraph.models import (
    EdgeDropper,
    ModelOutput,
    OnsetModel,
    build_model,
    graph_smoothness,
    onset_loss,
    positive_weights,
    smoothness_laplacian,
    sparse_tensor,
)
from ehrgraph.utils import set_seed

logger = logging.getLogger(__name__)

TRAIN, VAL, TEST = "train", "val", "test"
REPORT_COLUMNS: list[str] = [
    "target", "val_positives", "val_observed", "val_auroc", "val_auprc", "train_positives",
    "train_observed", "train_auroc", "pos_weight",
]
# The model values the tuning grid varies (tuning_settings), in report order.
TUNED_FIELDS: tuple[str, ...] = (
    "weight_decay", "dropout", "hidden_dim", "embed_dim", "smoothness_weight", "drop_edge",
)
TUNING_RUN_COLUMNS: list[str] = [
    "setting", *TUNED_FIELDS, "seed", "best_epoch", "epochs_run", "val_auroc", "val_auprc",
    "train_auroc", "seconds",
]

# Chart tokens from the dataviz reference palette (as in data.py and graph.py):
# categorical slots 1 and 2 for train and validation, secondary ink for annotations.
_TRAIN_COLOR = "#2a78d6"
_VAL_COLOR = "#eb6834"
_TEXT_SECONDARY = "#52514e"
_BASELINE = "#c3c2b7"
_POINTS_PER_INCH = 72
_FIG_SIZE_IN = (10.0, 4.2)
_LINE_PX = 2
_LABEL_OFFSET_PT = 4
_CHANCE_AUROC = 0.5


@dataclass(frozen=True)
class TrainingData:
    """Model inputs and targets of N nodes, T windows and G onset groups.

    Attributes:
        patient_ids: Node order: the model cohort (or, in quick mode, a subset of it).
        years: The windows, oldest first; all close by the prediction cutoff.
        groups: Onset group names, in config order: the columns of ``onsets``.
        features: Dense float32 [N, F] per window.
        weights: ``W`` per window, CSR (N, N), for the smoothness Laplacian.
        adjacency: ``A_hat`` per window, torch sparse [N, N]; isolated nodes have
            only their self-loop.
        any_onset: [N] targets: 1.0, 0.0 or NaN (masked); every test patient NaN.
        onsets: [N, G] targets, same convention.
        split: Split name per node.
        demographics: [N, len(DEMOGRAPHICS)] float32 head inputs: age at the cutoff
            standardized with ``age_mean`` and ``age_sd``, and sex (0/1).
        age_mean: Mean age of the training-split nodes, in years.
        age_sd: Their standard deviation (ddof 0), in years.
    """

    patient_ids: tuple[str, ...]
    years: tuple[int, ...]
    groups: tuple[str, ...]
    features: list[Tensor]
    weights: list[sp.csr_matrix]
    adjacency: list[Tensor]
    any_onset: Tensor
    onsets: Tensor
    split: np.ndarray
    demographics: Tensor
    age_mean: float
    age_sd: float


class PosWeights(NamedTuple):
    """``pos_weight`` of each target, from the training split's observed labels."""

    any_onset: Tensor
    onsets: Tensor


@dataclass
class FitResult:
    """Output of ``fit``.

    Attributes:
        model: The model with the weights of the best epoch, in eval mode.
        history: One row per epoch run (see ``fit``).
        best_epoch: Epoch with the highest validation AUROC of ``any_onset``.
        epochs_run: Epochs actually run.
        max_epochs: Epochs allowed.
        stopped_early: Whether early stopping ended training.
        seconds: Wall-clock training time.
        pos_weights: The ``pos_weight`` used for each target.
    """

    model: OnsetModel
    history: pd.DataFrame
    best_epoch: int
    epochs_run: int
    max_epochs: int
    stopped_early: bool
    seconds: float
    pos_weights: PosWeights


@dataclass(frozen=True)
class TrainOutputs:
    """Where ``save_training`` and the plots write; see ``from_config``."""

    model_pt: Path
    config_used_yaml: Path
    train_log_csv: Path
    train_metrics_json: Path
    embeddings_per_window_npy: Path
    trajectory_embeddings_npy: Path
    training_curves_fig: Path
    training_summary_md: Path
    temperature_json: Path
    reliability_fig: Path

    @classmethod
    def from_config(cls, cfg: Config, quick: bool = False) -> TrainOutputs:
        """Return the configured paths; in quick mode the same names in ``paths.quick_dir``."""
        paths = {f.name: getattr(cfg.paths, f.name) for f in fields(cls)}
        if quick:
            paths = {name: cfg.paths.quick_dir / path.name for name, path in paths.items()}
        return cls(**paths)


def make_training_data(
    patient_ids: Sequence[str],
    years: Sequence[int],
    matrices: Sequence[sp.spmatrix],
    weights: Sequence[sp.spmatrix],
    adjacency: Sequence[sp.spmatrix],
    labels: pd.DataFrame,
    groups: Sequence[str],
    demographics: pd.DataFrame,
) -> TrainingData:
    """Convert features, graphs, labels and demographics to tensors; drop test labels.

    Age is standardized with the mean and standard deviation (ddof 0) of the
    training-split nodes only; sex is kept as 0/1.

    Args:
        patient_ids: Node order.
        years: The windows, oldest first.
        matrices: Sparse (N, F) features per window.
        weights: ``W`` per window, (N, N).
        adjacency: ``A_hat`` per window, (N, N).
        labels: One row per node, in node order, with ``ANY_ONSET``, the onset
            columns of ``groups`` and ``SPLIT``.
        groups: Onset group names.
        demographics: ``features.patient_demographics`` of the nodes, in node order.

    Returns:
        The training data; every test patient's targets are NaN.

    Raises:
        ValueError: If the labels or demographics are not in node order, or the
            training ages do not vary.
    """
    if labels["patient_id"].tolist() != list(patient_ids):
        raise ValueError("labels are not in node order")
    if demographics.index.tolist() != list(patient_ids):
        raise ValueError("demographics are not in node order")
    split = labels[SPLIT].to_numpy(dtype=str)
    held_out = split == TEST
    any_onset = labels[ANY_ONSET].to_numpy(dtype=np.float32, copy=True)
    onsets = labels[[ONSET_PREFIX + g for g in groups]].to_numpy(dtype=np.float32, copy=True)
    any_onset[held_out] = np.nan
    onsets[held_out] = np.nan
    inputs = demographics[list(DEMOGRAPHICS)].to_numpy(dtype=np.float64, copy=True)
    age = DEMOGRAPHICS.index("age")
    train_ages = inputs[split == TRAIN, age]
    age_mean, age_sd = float(train_ages.mean()), float(train_ages.std())
    if not age_sd > 0:
        raise ValueError("the training-split ages do not vary; cannot standardize age")
    inputs[:, age] = (inputs[:, age] - age_mean) / age_sd
    return TrainingData(
        patient_ids=tuple(patient_ids),
        years=tuple(years),
        groups=tuple(groups),
        features=[torch.from_numpy(sp.csr_matrix(X).toarray().astype(np.float32)) for X in matrices],
        weights=[sp.csr_matrix(W) for W in weights],
        adjacency=[sparse_tensor(A) for A in adjacency],
        any_onset=torch.from_numpy(any_onset),
        onsets=torch.from_numpy(onsets),
        split=split,
        demographics=torch.from_numpy(inputs.astype(np.float32)),
        age_mean=age_mean,
        age_sd=age_sd,
    )


def load_training_data(cfg: Config, n_patients: int | None = None) -> TrainingData:
    """Load the Phase 3-5 outputs (model inputs, graphs, labels) and the demographics.

    Demographics are age on ``labels.cutoff_date`` and sex
    (``features.patient_demographics``).

    Args:
        cfg: Project configuration.
        n_patients: If given, keep this many patients drawn with ``cfg.seed`` (in
            model-cohort order) and use the subgraph they induce, renormalized with
            ``graph.normalize_adjacency``: a node whose neighbours were all dropped
            keeps only its self-loop.

    Returns:
        The training data, test labels dropped.

    Raises:
        ValueError: If the inputs, graphs and labels disagree on the nodes, or
            ``n_patients`` exceeds the cohort.
    """
    inputs = load_model_inputs(cfg)
    graphs = load_graphs(cfg)
    labels = load_labels(cfg)
    n = len(inputs.patient_ids)
    if graphs.patient_ids != inputs.patient_ids or graphs.years != inputs.years:
        raise ValueError("the graphs do not match the model inputs; re-run scripts/04_build_graphs.py")
    if not (labels["node"].to_numpy() == np.arange(n)).all():
        raise ValueError("labels.parquet is not in node order; re-run scripts/05_build_labels.py")
    weights, adjacency = graphs.weights, graphs.adjacency
    nodes = np.arange(n)
    if n_patients is not None:
        if not 0 < n_patients <= n:
            raise ValueError(f"n_patients must be in [1, {n}], got {n_patients}")
        nodes = np.sort(np.random.default_rng(cfg.seed).choice(n, n_patients, replace=False))
        weights = [W[nodes][:, nodes] for W in weights]
        adjacency = [normalize_adjacency(W) for W in weights]
    patient_ids = [inputs.patient_ids[i] for i in nodes]
    data = make_training_data(
        patient_ids,
        inputs.years,
        [X[nodes] for X in inputs.matrices],
        weights,
        adjacency,
        labels.iloc[nodes].reset_index(drop=True),
        list(cfg.labels.onset_groups),
        patient_demographics(load_patients(cfg), patient_ids, cfg.labels.cutoff_date),
    )
    logger.info(
        "Training data: %d nodes, windows %d-%d, %d features, %d onset groups; split sizes %s; "
        "training age %.1f +/- %.1f years on %s",
        len(nodes), data.years[0], data.years[-1], data.features[0].shape[1], len(data.groups),
        {name: int((data.split == name).sum()) for name in (TRAIN, VAL, TEST)},
        data.age_mean, data.age_sd, cfg.labels.cutoff_date,
    )
    return data


def split_targets(data: TrainingData, name: str) -> tuple[Tensor, Tensor]:
    """Return the targets of one split, every other node masked (NaN).

    Args:
        data: Training data.
        name: "train" or "val".

    Returns:
        ``any_onset`` [N] and ``onsets`` [N, G].

    Raises:
        ValueError: For the test split, whose labels are not loaded in this phase.
    """
    if name not in (TRAIN, VAL):
        raise ValueError(f"only the train and val labels are loaded, not {name!r}")
    inside = torch.from_numpy(data.split == name)
    return (
        torch.where(inside, data.any_onset, torch.nan),
        torch.where(inside[:, None], data.onsets, torch.nan),
    )


def binary_metrics(targets: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """AUROC and AUPRC over the observed (non-NaN) targets.

    Args:
        targets: [N] 1.0, 0.0 or NaN.
        scores: [N] predicted scores.

    Returns:
        observed, positives, auroc and auprc; the metrics are NaN unless both
        classes are observed.
    """
    seen = ~np.isnan(targets)
    y, s = targets[seen], scores[seen]
    positives = int((y == 1).sum())
    both = 0 < positives < len(y)
    return {
        "observed": int(len(y)),
        "positives": positives,
        "auroc": float(roc_auc_score(y, s)) if both else float("nan"),
        "auprc": float(average_precision_score(y, s)) if both else float("nan"),
    }


def predict(model: OnsetModel, data: TrainingData) -> ModelOutput:
    """Run the model on every node in eval mode, on the full graphs, without gradients."""
    model.eval()
    with torch.no_grad():
        return model(data.features, data.adjacency, data.demographics)


def train_step(
    model: OnsetModel,
    optimizer: torch.optim.Optimizer,
    data: TrainingData,
    targets: tuple[Tensor, Tensor],
    pos_weights: PosWeights,
    model_cfg: ModelConfig,
    regularizer: Callable[[ModelOutput], Tensor] | None = None,
    adjacency: Sequence[Tensor] | None = None,
) -> float:
    """One full-batch optimisation step (dropout on); returns the objective minimized.

    Args:
        model: The model.
        optimizer: Its optimizer.
        data: Training data; every node passes messages.
        targets: ``any_onset`` and ``onsets`` with every non-training node NaN.
        pos_weights: ``pos_weight`` per target.
        model_cfg: ``secondary_weight`` and ``grad_clip_norm``.
        regularizer: Optional extra loss term on the model output.
        adjacency: Graphs for this step (e.g. ``EdgeDropper.sample``); defaults to
            ``data.adjacency``.

    Returns:
        The objective before the step.
    """
    model.train()
    optimizer.zero_grad()
    graphs = data.adjacency if adjacency is None else adjacency
    output = model(data.features, graphs, data.demographics)
    loss = onset_loss(
        output, *targets, pos_weights.any_onset, pos_weights.onsets, model_cfg.secondary_weight
    ).total
    if regularizer is not None:
        loss = loss + regularizer(output)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), model_cfg.grad_clip_norm)
    optimizer.step()
    return float(loss.detach())


def fit(
    data: TrainingData,
    model_cfg: ModelConfig,
    seed: int,
    epochs: int | None = None,
    verbose: bool = True,
) -> FitResult:
    """Train an ``OnsetModel`` full-batch with Adam and early stopping.

    Each epoch takes one step on the training patients' observed labels (on graphs
    with edges dropped when ``model.drop_edge`` > 0), then scores every node in
    eval mode on the full graphs. Training stops after ``model.patience`` epochs
    without a higher validation AUROC of ``any_onset``, and the best epoch's
    weights are restored. History columns: epoch, objective (the dropout-on loss
    minimized, with the smoothness term if on), train_loss, train_loss_any,
    train_loss_groups, val_loss, val_loss_any, val_loss_groups (all in eval mode),
    train_auroc, val_auroc, val_auprc (of ``any_onset``), and smoothness (eval
    mode, only if ``model.smoothness_weight`` > 0).

    Args:
        data: Training data.
        model_cfg: The ``model`` section of the configuration.
        seed: Random seed for the initial weights, dropout and DropEdge.
        epochs: Maximum epochs; defaults to ``model.epochs``.
        verbose: Log the positive weights and per-epoch progress; the summary line
            at the end is always logged.

    Returns:
        The trained model and its history.
    """
    set_seed(seed)
    model = build_model(model_cfg, data.features[0].shape[1], len(data.groups))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.weight_decay
    )
    train_targets, val_targets = split_targets(data, TRAIN), split_targets(data, VAL)
    pos_weights = PosWeights(*(positive_weights(t) for t in train_targets))
    if verbose:
        _log_pos_weights(data, train_targets, pos_weights)

    smoothness = _smoothness_term(data) if model_cfg.smoothness_weight > 0 else None
    regularizer = None
    if smoothness is not None:
        regularizer = lambda output: model_cfg.smoothness_weight * smoothness(output)  # noqa: E731
    dropper = None
    if model_cfg.drop_edge > 0:
        dropper = EdgeDropper(data.weights, model_cfg.drop_edge, seed)

    max_epochs = model_cfg.epochs if epochs is None else epochs
    rows: list[dict[str, float]] = []
    best_auroc, best_epoch, best_state, since_best = -np.inf, 0, None, 0
    stopped_early = False
    start = perf_counter()
    for epoch in range(1, max_epochs + 1):
        objective = train_step(
            model, optimizer, data, train_targets, pos_weights, model_cfg, regularizer,
            None if dropper is None else dropper.sample(),
        )
        output = predict(model, data)
        scores = torch.sigmoid(output.any_onset_logits).numpy()
        row: dict[str, float] = {"epoch": epoch, "objective": objective}
        for name, targets in ((TRAIN, train_targets), (VAL, val_targets)):
            loss = onset_loss(
                output, *targets, pos_weights.any_onset, pos_weights.onsets,
                model_cfg.secondary_weight,
            )
            row |= {
                f"{name}_loss": float(loss.total),
                f"{name}_loss_any": float(loss.primary),
                f"{name}_loss_groups": float(loss.secondary),
            }
        train_metrics = binary_metrics(train_targets[0].numpy(), scores)
        val_metrics = binary_metrics(val_targets[0].numpy(), scores)
        row |= {
            "train_auroc": train_metrics["auroc"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
        }
        if smoothness is not None:
            row["smoothness"] = float(smoothness(output))
        rows.append(row)

        improved = row["val_auroc"] > best_auroc  # False while the AUROC is undefined (NaN)
        if improved:
            best_auroc, best_epoch, since_best = row["val_auroc"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1
        if verbose and (improved or epoch % model_cfg.log_every_epochs == 0 or epoch == 1):
            logger.info(
                "Epoch %3d: objective %.4f, train loss %.4f, val loss %.4f, train AUROC %.4f, "
                "val AUROC %.4f, val AUPRC %.4f%s",
                epoch, objective, row["train_loss"], row["val_loss"], row["train_auroc"],
                row["val_auroc"], row["val_auprc"], " (best)" if improved else "",
            )
        if since_best >= model_cfg.patience:
            stopped_early = True
            if verbose:
                logger.info(
                    "Early stopping at epoch %d: no higher validation AUROC for %d epochs",
                    epoch, model_cfg.patience,
                )
            break
    seconds = perf_counter() - start

    if best_state is None:
        logger.warning("The validation AUROC was never defined; keeping the last epoch's weights")
        best_epoch = len(rows)
    else:
        model.load_state_dict(best_state)
    model.eval()
    logger.info(
        "Trained %d epochs in %.1f s; best epoch %d, validation AUROC %.4f",
        len(rows), seconds, best_epoch, rows[best_epoch - 1]["val_auroc"],
    )
    return FitResult(
        model=model,
        history=pd.DataFrame(rows),
        best_epoch=best_epoch,
        epochs_run=len(rows),
        max_epochs=max_epochs,
        stopped_early=stopped_early,
        seconds=seconds,
        pos_weights=pos_weights,
    )


def validation_report(
    model: OnsetModel, data: TrainingData, pos_weights: PosWeights
) -> pd.DataFrame:
    """Score every target on the validation (and, for reference, training) patients.

    Args:
        model: Trained model.
        data: Training data.
        pos_weights: The ``pos_weight`` used for each target.

    Returns:
        ``REPORT_COLUMNS``; one row for ``any_onset``, then one per onset group.
    """
    output = predict(model, data)
    train_any, train_onsets = split_targets(data, TRAIN)
    val_any, val_onsets = split_targets(data, VAL)
    any_scores = torch.sigmoid(output.any_onset_logits).numpy()
    onset_scores = torch.sigmoid(output.onset_logits).numpy()
    columns = [
        (ANY_ONSET, train_any.numpy(), val_any.numpy(), any_scores,
         float(pos_weights.any_onset)),
        *(
            (ONSET_PREFIX + group, train_onsets[:, g].numpy(), val_onsets[:, g].numpy(),
             onset_scores[:, g], float(pos_weights.onsets[g]))
            for g, group in enumerate(data.groups)
        ),
    ]
    rows = []
    for target, train_y, val_y, scores, weight in columns:
        val_m, train_m = binary_metrics(val_y, scores), binary_metrics(train_y, scores)
        rows.append(
            {
                "target": target,
                "val_positives": val_m["positives"],
                "val_observed": val_m["observed"],
                "val_auroc": val_m["auroc"],
                "val_auprc": val_m["auprc"],
                "train_positives": train_m["positives"],
                "train_observed": train_m["observed"],
                "train_auroc": train_m["auroc"],
                "pos_weight": weight,
            }
        )
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def save_training(
    result: FitResult,
    report: pd.DataFrame,
    data: TrainingData,
    cfg: Config,
    outputs: TrainOutputs,
    config_path: Path,
    quick: bool,
) -> None:
    """Write the weights, the config used, the log, the metrics and the embeddings.

    The weights are saved with ``state_dict()``. Embeddings are computed in eval mode
    with the best epoch's weights, for every node in ``data.patient_ids`` order:
    per window [N, T, embed_dim] and trajectory [N, embed_dim].

    Args:
        result: Output of ``fit``.
        report: Output of ``validation_report``.
        data: Training data.
        cfg: Project configuration.
        outputs: Output paths.
        config_path: The YAML file ``cfg`` was loaded from; copied verbatim.
        quick: Whether this is a quick (smoke-test) run.
    """
    for path in (outputs.model_pt, outputs.train_metrics_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), outputs.model_pt)
    shutil.copyfile(config_path, outputs.config_used_yaml)
    result.history.to_csv(outputs.train_log_csv, index=False)
    output = predict(result.model, data)
    np.save(outputs.embeddings_per_window_npy, output.window_embeddings.numpy())
    np.save(outputs.trajectory_embeddings_npy, output.trajectory_embeddings.numpy())
    metrics: dict[str, Any] = {
        "quick": quick,
        "seed": cfg.seed,
        "n_nodes": len(data.patient_ids),
        "years": list(data.years),
        "in_dim": int(data.features[0].shape[1]),
        "groups": list(data.groups),
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "max_epochs": result.max_epochs,
        "stopped_early": result.stopped_early,
        "training_seconds": round(result.seconds, 2),
        "demographics": {
            "used": cfg.model.use_demographics,
            "columns": list(DEMOGRAPHICS),
            "age_as_of": cfg.labels.cutoff_date.isoformat(),
            "age_mean_train": data.age_mean,
            "age_sd_train": data.age_sd,
        },
        "validation": json.loads(report.to_json(orient="records")),
        "patient_ids": list(data.patient_ids),
    }
    outputs.train_metrics_json.write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8")
    logger.info(
        "Wrote %s, %s, %s, %s, %s and %s",
        outputs.model_pt, outputs.config_used_yaml, outputs.train_log_csv,
        outputs.train_metrics_json, outputs.embeddings_per_window_npy,
        outputs.trajectory_embeddings_npy,
    )


def load_trained_model(
    model_cfg: ModelConfig, path: Path, in_dim: int, n_groups: int
) -> OnsetModel:
    """Rebuild an ``OnsetModel`` and load a ``state_dict`` saved by ``save_training``.

    Args:
        model_cfg: The ``model`` section the weights were trained with.
        path: The saved ``state_dict``.
        in_dim: Number of input features per node.
        n_groups: Number of onset groups.

    Returns:
        The model, in eval mode.
    """
    model = build_model(model_cfg, in_dim, n_groups)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def tuning_settings(tuning: TuningConfig, seed: int) -> list[dict[str, float | int]]:
    """Expand the tuning grid into model settings (``TUNED_FIELDS`` each).

    The grid is the product of the lists in ``tuning``, in config order. When it
    has more than ``tuning.max_settings`` settings, a random subset of that size is
    drawn with ``seed`` (kept in grid order).

    Args:
        tuning: The ``tuning`` section of the configuration.
        seed: Random seed for the subset.

    Returns:
        One mapping of ``TUNED_FIELDS`` per setting.
    """
    grid = [
        dict(zip(TUNED_FIELDS, (wd, dropout, hidden, embed, smooth, drop)))
        for wd, dropout, (hidden, embed), smooth, drop in itertools.product(
            tuning.weight_decay, tuning.dropout, tuning.dims, tuning.smoothness_weight,
            tuning.drop_edge,
        )
    ]
    if len(grid) > tuning.max_settings:
        keep = np.sort(np.random.default_rng(seed).choice(len(grid), tuning.max_settings,
                                                          replace=False))
        logger.info("Tuning a random %d of the %d grid settings", len(keep), len(grid))
        grid = [grid[i] for i in keep]
    return grid


def tune(
    data: TrainingData,
    model_cfg: ModelConfig,
    settings: Sequence[Mapping[str, float | int]],
    seeds: Sequence[int],
    on_progress: Callable[[pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    """Train every setting once per seed; validation labels only pick the epoch and score.

    Each run is ``fit(data, model_cfg with the setting, seed)``: early stopping on
    the validation AUROC of ``any_onset``, which is also the score. Test labels are
    not in ``data``.

    Args:
        data: Training data.
        model_cfg: The base ``model`` section (lr, patience, epochs, ...).
        settings: ``tuning_settings`` output.
        seeds: Seeds per setting.
        on_progress: Called with every run so far after each setting (e.g. to save).

    Returns:
        ``TUNING_RUN_COLUMNS``, one row per run; ``setting`` numbers settings from 1.
    """
    rows: list[dict[str, Any]] = []
    start = perf_counter()
    for number, setting in enumerate(settings, start=1):
        run_cfg = replace(model_cfg, **setting)
        for seed in seeds:
            result = fit(data, run_cfg, seed, verbose=False)
            best = result.history.iloc[result.best_epoch - 1]
            rows.append(
                {"setting": number, **setting, "seed": seed, "best_epoch": result.best_epoch,
                 "epochs_run": result.epochs_run, "val_auroc": float(best["val_auroc"]),
                 "val_auprc": float(best["val_auprc"]), "train_auroc": float(best["train_auroc"]),
                 "seconds": round(result.seconds, 2)}
            )
        aurocs = [row["val_auroc"] for row in rows[-len(seeds):]]
        sd = float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else float("nan")
        logger.info(
            "Setting %d/%d %s: validation AUROC %.4f +/- %.4f (%.0f s elapsed)",
            number, len(settings), dict(setting), np.mean(aurocs), sd, perf_counter() - start,
        )
        if on_progress is not None:
            on_progress(pd.DataFrame(rows, columns=TUNING_RUN_COLUMNS))
    return pd.DataFrame(rows, columns=TUNING_RUN_COLUMNS)


def summarize_tuning(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ``tune`` runs per setting, best mean validation AUROC first.

    Args:
        results: ``tune`` output.

    Returns:
        setting, ``TUNED_FIELDS``, seeds, val_auroc_mean, val_auroc_sd (ddof 1; NaN
        for one seed), val_auprc_mean, val_auprc_sd, best_epoch_mean, seconds_mean.
    """
    keys = ["setting", *TUNED_FIELDS]
    summary = (
        results.groupby(keys, sort=False)
        .agg(
            seeds=("seed", "size"),
            val_auroc_mean=("val_auroc", "mean"),
            val_auroc_sd=("val_auroc", "std"),
            val_auprc_mean=("val_auprc", "mean"),
            val_auprc_sd=("val_auprc", "std"),
            best_epoch_mean=("best_epoch", "mean"),
            seconds_mean=("seconds", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(
        ["val_auroc_mean", "setting"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def select_setting(summary: pd.DataFrame, tolerance: float) -> pd.Series:
    """Pick the simplest setting whose mean AUROC is within ``tolerance`` of the best.

    Simplicity, in order: fewest optional parts switched on (smoothness loss,
    DropEdge), then the smaller model (hidden_dim, then embed_dim); remaining ties
    go to the higher mean AUROC, then the lower setting number.

    Args:
        summary: ``summarize_tuning`` output.
        tolerance: ``tuning.tie_tolerance``.

    Returns:
        The chosen row of ``summary``.
    """
    best = summary["val_auroc_mean"].max()
    tied = summary[summary["val_auroc_mean"] >= best - tolerance]
    tied = tied.assign(
        _parts=(tied["smoothness_weight"] > 0).astype(int) + (tied["drop_edge"] > 0).astype(int)
    )
    chosen = tied.sort_values(
        ["_parts", "hidden_dim", "embed_dim", "val_auroc_mean", "setting"],
        ascending=[True, True, True, False, True], kind="stable",
    ).iloc[0]
    return chosen.drop("_parts")


def write_tuning_summary(
    results: pd.DataFrame, summary: pd.DataFrame, chosen: pd.Series, cfg: Config,
    n_grid: int,
) -> str:
    """Write the tuning report to ``cfg.paths.tuning_summary_md``.

    Args:
        results: ``tune`` output (one row per run).
        summary: ``summarize_tuning`` output.
        chosen: ``select_setting`` output.
        cfg: Project configuration.
        n_grid: Size of the full grid (more than ``len(summary)`` if subsampled).

    Returns:
        The markdown written.
    """
    tuning, model_cfg = cfg.tuning, cfg.model
    best = summary.iloc[0]

    def setting_cells(row: Any) -> list[str]:
        return [str(int(row.setting)), f"{row.weight_decay:g}", f"{row.dropout:g}",
                f"{int(row.hidden_dim)}/{int(row.embed_dim)}", f"{row.smoothness_weight:g}",
                f"{row.drop_edge:g}", f"{row.val_auroc_mean:.4f} +/- {row.val_auroc_sd:.4f}",
                f"{row.val_auprc_mean:.4f} +/- {row.val_auprc_sd:.4f}",
                f"{row.best_epoch_mean:.0f}"]

    header = ["Setting", "Weight decay", "Dropout", "Hidden/embed", "Smoothness", "DropEdge",
              "Val AUROC mean +/- sd", "Val AUPRC mean +/- sd", "Best epoch (mean)"]
    align = "rrrrrrrrr"
    marginal_rows = []
    for keys, label in ((["weight_decay"], "Weight decay"), (["dropout"], "Dropout"),
                        (["hidden_dim", "embed_dim"], "Hidden/embed"),
                        (["smoothness_weight"], "Smoothness"), (["drop_edge"], "DropEdge")):
        for value, part in summary.groupby(keys, sort=True):
            shown = "/".join(f"{v:g}" for v in value)
            marginal_rows.append([label, shown, f"{len(part)}",
                                  f"{part['val_auroc_mean'].mean():.4f}",
                                  f"{part['val_auroc_mean'].max():.4f}"])
    subset = (
        f"a random {len(summary)} of the {n_grid} grid settings (seed {cfg.seed})"
        if len(summary) < n_grid else f"all {n_grid} grid settings"
    )
    tied = summary[summary["val_auroc_mean"] >= best.val_auroc_mean - tuning.tie_tolerance]
    lines = [
        "# Tuning (validation only)",
        "",
        f"Built by `scripts/06b_tune.py`: {subset}, each trained with seeds "
        f"{', '.join(map(str, tuning.seeds))} ({len(results)} runs, "
        f"{results['seconds'].sum() / 60:.1f} min of training). Every run uses lr "
        f"{model_cfg.lr}, at most {model_cfg.epochs} epochs, early stopping with patience "
        f"{model_cfg.patience} on the validation AUROC of `{ANY_ONSET}` and "
        f"{'age and sex in the heads' if model_cfg.use_demographics else 'no demographics'}; "
        "the score is the validation AUROC at the best epoch. Test labels are never loaded. "
        "Every run is in `tuning_results.csv`.",
        "",
        "Because the same validation patients pick each run's epoch and rank the settings, "
        "the best mean AUROC is optimistic; the test set (Phase 7) gives the unbiased number.",
        "",
        "## Chosen setting",
        "",
        f"Highest mean validation AUROC: setting {int(best.setting)} "
        f"({best.val_auroc_mean:.4f}). {len(tied)} setting(s) lie within "
        f"{tuning.tie_tolerance} of it; of those the simplest (fewest of smoothness loss and "
        "DropEdge switched on, then the smaller model, then the higher mean) is **setting "
        f"{int(chosen.setting)}**: weight decay {chosen.weight_decay:g}, dropout "
        f"{chosen.dropout:g}, hidden/embed {int(chosen.hidden_dim)}/{int(chosen.embed_dim)}, "
        f"smoothness {chosen.smoothness_weight:g}, DropEdge {chosen.drop_edge:g}; validation "
        f"AUROC {chosen.val_auroc_mean:.4f} +/- {chosen.val_auroc_sd:.4f} (sd over seeds).",
        "",
        f"## Top {tuning.top_n_report} settings",
        "",
        markdown_table(header, [setting_cells(r) for r in
                                summary.head(tuning.top_n_report).itertuples()], align),
        "",
        "## Marginal means",
        "",
        "Mean (and best) of the settings' mean validation AUROC at each value of one "
        "hyperparameter, over every setting run with that value.",
        "",
        markdown_table(["Hyperparameter", "Value", "Settings", "Mean AUROC", "Best AUROC"],
                       marginal_rows, "lrrrr"),
        "",
        "## Every setting",
        "",
        markdown_table(header, [setting_cells(r) for r in summary.itertuples()], align),
        "",
    ]
    text = "\n".join(lines)
    path = cfg.paths.tuning_summary_md
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)
    return text


def plot_training_curves(history: pd.DataFrame, best_epoch: int, path: Path, cfg: Config) -> None:
    """Save the per-epoch loss and ``any_onset`` AUROC, train vs validation.

    Args:
        history: ``FitResult.history``.
        best_epoch: Marked with a dashed line.
        path: Output PNG.
        cfg: Project configuration (``summary`` section).
    """
    dpi = cfg.summary.figure_dpi
    px = _POINTS_PER_INCH / dpi
    panels = [
        ("Loss (eval mode)", "train_loss", "val_loss", "Loss"),
        (f"AUROC of {ANY_ONSET}", "train_auroc", "val_auroc", "AUROC"),
    ]
    with matplotlib.rc_context(chart_style(dpi)):
        fig = Figure(figsize=_FIG_SIZE_IN, layout="constrained")
        axes = fig.subplots(1, 2, sharex=True)
        for ax, (title, train_column, val_column, ylabel) in zip(axes, panels):
            series = [(train_column, "train", _TRAIN_COLOR), (val_column, "validation", _VAL_COLOR)]
            ends = [history[column].iloc[-1] for column, _, _ in series]
            for (column, name, color), end, other in zip(series, ends, ends[::-1]):
                ax.plot(history["epoch"], history[column], color=color, linewidth=_LINE_PX * px,
                        label=name)
                above = end >= other  # the higher end label sits above its line, the other below
                ax.annotate(
                    name, xy=(history["epoch"].iloc[-1], end),
                    xytext=(_LABEL_OFFSET_PT, _LABEL_OFFSET_PT / 2 * (1 if above else -1)),
                    textcoords="offset points", ha="left", va="bottom" if above else "top",
                    color=_TEXT_SECONDARY,
                )
            if val_column == "val_auroc":
                ax.axhline(_CHANCE_AUROC, color=_BASELINE, linewidth=px, linestyle=":")
                ax.annotate("chance", xy=(0, _CHANCE_AUROC), xycoords=("axes fraction", "data"),
                            xytext=(_LABEL_OFFSET_PT, _LABEL_OFFSET_PT / 2),
                            textcoords="offset points", va="bottom", color=_TEXT_SECONDARY)
            # Left of the line, so it never meets the end labels on the right.
            ax.axvline(best_epoch, color=_TEXT_SECONDARY, linewidth=px, linestyle="--")
            ax.annotate(f"best epoch {best_epoch}", xy=(best_epoch, 1),
                        xycoords=("data", "axes fraction"), xytext=(-_LABEL_OFFSET_PT, -2),
                        textcoords="offset points", ha="right", va="top", color=_TEXT_SECONDARY)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.margins(x=0.12)  # room for the end labels
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        axes[0].legend(frameon=False, loc="lower left")
        save_figure(fig, path)


def write_training_summary(
    result: FitResult,
    report: pd.DataFrame,
    data: TrainingData,
    cfg: Config,
    outputs: TrainOutputs,
    quick: bool,
    extra_sections: Sequence[str] = (),
) -> str:
    """Write the training report (markdown) to ``outputs.training_summary_md``.

    Args:
        result: Output of ``fit``.
        report: Output of ``validation_report``.
        data: Training data.
        cfg: Project configuration.
        outputs: Output paths (the figure is linked relative to the report).
        quick: Whether this is a quick (smoke-test) run.
        extra_sections: Markdown lines placed after the result, e.g. the baseline
            comparison and calibration of ``evaluate``.

    Returns:
        The markdown written.
    """
    model_cfg = cfg.model
    n = len(data.patient_ids)
    sizes = {name: int((data.split == name).sum()) for name in (TRAIN, VAL, TEST)}
    best = result.history.iloc[result.best_epoch - 1]
    primary = report.iloc[0]

    def metric(value: float) -> str:
        return "n/a" if np.isnan(value) else f"{value:.3f}"

    def share(positives: int, observed: int) -> str:
        return f"{positives / observed:.1%}" if observed else "n/a"

    group_rows = [
        [
            row.target.removeprefix(ONSET_PREFIX),
            f"{row.val_positives} / {row.val_observed}",
            metric(row.val_auroc),
            metric(row.val_auprc),
            share(row.val_positives, row.val_observed),
            f"{row.train_positives} / {row.train_observed}",
            metric(row.train_auroc),
            f"{row.pos_weight:.1f}",
        ]
        for row in report.iloc[1:].itertuples()
    ]
    borderline = set(cfg.labels.borderline_groups) | set(cfg.labels.near_threshold_groups)
    for row in group_rows:
        if row[0] in borderline:
            row[0] += " *"
    stop = (
        f"early stopping after {model_cfg.patience} epochs without improvement"
        if result.stopped_early else f"the epoch limit ({result.max_epochs})"
    )
    smoothing = (
        f"plus {model_cfg.smoothness_weight} x graph smoothness (normalized trace(Z^T L Z))"
        if model_cfg.smoothness_weight > 0 else "no auxiliary smoothness loss (off)"
    )
    drop_edge = (
        f"DropEdge {model_cfg.drop_edge} (each epoch, that fraction of every window's edges "
        "dropped at random and A_hat renormalized; evaluation on the full graphs)"
        if model_cfg.drop_edge > 0 else "no DropEdge"
    )
    head_input = (
        f"its final state concatenated with age on {cfg.labels.cutoff_date:%Y-%m-%d} "
        f"(standardized with the training split's mean {data.age_mean:.1f} and sd "
        f"{data.age_sd:.1f} years) and sex (1 = male); demographics enter the heads only, "
        "never the graphs or node features"
        if model_cfg.use_demographics else "its final state (no demographics)"
    )
    figure = Path(os.path.relpath(outputs.training_curves_fig, outputs.training_summary_md.parent))
    title = "# Training" + (" (quick smoke test)" if quick else "")
    mode = (
        f"Quick mode: {n} of the model-cohort patients drawn with seed {cfg.seed} (induced "
        f"subgraph, renormalized), at most {result.max_epochs} epochs. A smoke test only: "
        "these numbers mean nothing."
        if quick else f"All {n:,} model-cohort patients."
    )
    lines = [
        title,
        "",
        f"Built by `scripts/06_train.py{' --quick' if quick else ''}`. {mode}",
        "",
        "## Setup",
        "",
        f"- **Inputs:** windows {data.years[0]}-{data.years[-1]} (all before the cutoff "
        f"{cfg.labels.cutoff_date:%Y-%m-%d}), {data.features[0].shape[1]} features per window, "
        "and each window's Phase 4 graph `A_hat = D^(-1/2) (W + I) D^(-1/2)`; isolated nodes "
        "keep only their self-loop.",
        "- **Transductive training:** every patient is a node and passes messages with its "
        "pre-cutoff features: train, validation and test patients, and those whose labels are "
        "masked. Only the observed labels of training patients enter the loss; validation "
        "labels choose the epoch; test labels are dropped on loading and never read here.",
        f"- **Nodes:** {sizes[TRAIN]:,} train, {sizes[VAL]:,} validation, {sizes[TEST]:,} test.",
        f"- **Model:** a shared 2-layer GCN per window ({data.features[0].shape[1]} -> "
        f"{model_cfg.hidden_dim} -> {model_cfg.embed_dim}, ReLU and dropout {model_cfg.dropout} "
        f"between the layers), a GRU ({model_cfg.embed_dim} units) over the windows, and two "
        f"linear heads on {head_input}: `{ANY_ONSET}` (primary) and the "
        f"{len(data.groups)} onset groups (secondary). No mortality head.",
        f"- **Loss:** BCE(`{ANY_ONSET}`) + {model_cfg.secondary_weight} x masked BCE(groups), "
        f"each with `pos_weight` = negatives / positives of the training split; {smoothing}. "
        "Only observed (unmasked) labels count.",
        f"- **Optimisation:** full batch, Adam (lr {model_cfg.lr}, weight decay "
        f"{model_cfg.weight_decay}), gradient norm clipped at {model_cfg.grad_clip_norm}, "
        f"{drop_edge}, early stopping on the validation AUROC of `{ANY_ONSET}` (patience "
        f"{model_cfg.patience}); seed {cfg.seed}.",
        *(
            ["- **Hyperparameters:** chosen on the validation split by `scripts/06b_tune.py` "
             f"(see `{cfg.paths.tuning_summary_md.name}`)."]
            if cfg.paths.tuning_summary_md.exists() and not quick else []
        ),
        "",
        "## Result",
        "",
        f"Best epoch **{result.best_epoch}** of {result.epochs_run} run (stopped by {stop}); "
        f"training time {result.seconds:.1f} s.",
        "",
        markdown_table(
            ["Validation", "Positives / observed", "AUROC", "AUPRC", "AUPRC of a random score"],
            [[f"`{ANY_ONSET}`", f"{primary.val_positives} / {primary.val_observed}",
              metric(primary.val_auroc), metric(primary.val_auprc),
              share(primary.val_positives, primary.val_observed)]],
            "lrrrr",
        ),
        "",
        f"Training AUROC of `{ANY_ONSET}` at the best epoch: {metric(primary.train_auroc)}. "
        f"Losses at the best epoch (eval mode): train {best.train_loss:.4f}, validation "
        f"{best.val_loss:.4f}.",
        "",
        *extra_sections,
        "## Onset groups (secondary)",
        "",
        "Validation AUROC per group, with its positives. With so few validation positives "
        "per group, these AUROCs are noisy: a single positive moves them by up to "
        "1 / positives.",
        "",
        markdown_table(
            ["Group", "Val positives / observed", "Val AUROC", "Val AUPRC",
             "AUPRC of a random score", "Train positives / observed", "Train AUROC",
             "pos_weight"],
            group_rows,
            "lrrrrrrr",
        ),
        "",
        "\\* borderline or near the review threshold of "
        f"{cfg.labels.min_onsets} onsets (see `labels_summary.md`).",
        "",
        "## Training curves",
        "",
        f"![Training curves]({figure.as_posix()})",
        "",
        "## Outputs",
        "",
        f"- `{outputs.model_pt.name}`: the best epoch's `state_dict`; "
        f"`{outputs.config_used_yaml.name}`: the config used; `{outputs.train_log_csv.name}`: "
        f"per-epoch losses and AUROC; `{outputs.train_metrics_json.name}`: the numbers above "
        "and the node order.",
        f"- `{outputs.embeddings_per_window_npy.name}`: the time-series embedding Z, "
        f"[{n}, {len(data.years)}, {model_cfg.embed_dim}], windows {data.years[0]}-"
        f"{data.years[-1]} in order; `{outputs.trajectory_embeddings_npy.name}`: the trajectory "
        f"embedding h, [{n}, {model_cfg.embed_dim}]. Rows follow "
        + ("`train_metrics.json` (patient_ids)." if quick else "`model_cohort.json`."),
        "",
    ]
    text = "\n".join(lines)
    path = outputs.training_summary_md
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)
    return text


def _smoothness_term(data: TrainingData) -> Callable[[ModelOutput], Tensor]:
    """Return output -> ``models.graph_smoothness`` of its window embeddings on data's graphs."""
    laplacians = [sparse_tensor(smoothness_laplacian(W)) for W in data.weights]
    connected = [torch.from_numpy(np.diff(W.indptr) > 0) for W in data.weights]

    def smoothness(output: ModelOutput) -> Tensor:
        return graph_smoothness(output.window_embeddings, laplacians, connected)

    return smoothness


def _log_pos_weights(
    data: TrainingData, train_targets: tuple[Tensor, Tensor], pos_weights: PosWeights
) -> None:
    """Log each target's training positives and ``pos_weight``; warn when it defaults to 1."""
    any_onset, onsets = train_targets
    entries = [(ANY_ONSET, any_onset, pos_weights.any_onset)] + [
        (group, onsets[:, g], pos_weights.onsets[g]) for g, group in enumerate(data.groups)
    ]
    for name, targets, weight in entries:
        positives, observed = int((targets == 1).sum()), int((~torch.isnan(targets)).sum())
        if positives in (0, observed):
            logger.warning("%s: %d / %d training positives; pos_weight set to 1", name,
                           positives, observed)
        else:
            logger.info("%s: %d / %d training positives, pos_weight %.2f", name, positives,
                        observed, float(weight))
