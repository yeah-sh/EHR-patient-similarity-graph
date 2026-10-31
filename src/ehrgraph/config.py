"""Load ``config/config.yaml`` into typed, immutable dataclasses.

Every section of the YAML maps to one frozen dataclass. Loading is strict:
missing or unknown keys and values of the wrong type raise ``ValueError``, so
a typo in the config fails loudly instead of silently falling back to a default.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, get_args, get_origin, get_type_hints

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations, resolved to absolute paths under the project root."""

    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    figures_dir: Path
    patients_csv: Path
    conditions_csv: Path
    encounters_csv: Path
    events_parquet: Path
    events_clinical_parquet: Path
    code_dictionary_csv: Path
    data_summary_md: Path
    clinical_summary_md: Path
    events_per_patient_fig: Path
    top_conditions_fig: Path
    trajectories_parquet: Path
    patient_index_json: Path
    code_vocabulary_json: Path
    windows_npz: Path
    windows_json: Path
    transitions_npz: Path
    transitions_parquet: Path
    trajectories_summary_md: Path
    features_dir: Path
    comorbidity_counts_npz: Path
    comorbidity_lift_npz: Path
    chronic_comorbidity_counts_npz: Path
    chronic_comorbidity_lift_npz: Path
    model_cohort_json: Path
    tfidf_idf_npy: Path
    transition_svd_npz: Path
    features_summary_md: Path
    comorbidity_heatmap_fig: Path
    graphs_dir: Path
    graphs_summary_md: Path
    eigenvalues_fig: Path
    graph_drift_fig: Path
    labels_parquet: Path
    splits_json: Path
    labels_summary_md: Path
    model_pt: Path
    config_used_yaml: Path
    train_log_csv: Path
    train_metrics_json: Path
    embeddings_per_window_npy: Path
    trajectory_embeddings_npy: Path
    training_curves_fig: Path
    training_summary_md: Path
    calibration_json: Path
    reliability_fig: Path
    quick_dir: Path
    baseline_results_csv: Path
    baselines_summary_md: Path
    baseline_val_scores_csv: Path
    tuning_results_csv: Path
    tuning_summary_md: Path
    results_md: Path
    results_csv: Path
    evaluation_summary_md: Path
    cv_predictions_csv: Path
    cv_fits_csv: Path
    test_predictions_csv: Path
    cv_auroc_fig: Path
    test_reliability_fig: Path
    umap_phenotype_fig: Path
    umap_age_fig: Path
    umap_chronic_fig: Path
    trajectory_umap_fig: Path
    graph_ae_dir: Path
    embeddings_npz: Path
    embedding_results_md: Path
    embedding_results_csv: Path
    embedding_summary_md: Path
    ae_umap_phenotype_fig: Path
    ae_umap_age_fig: Path
    ae_umap_chronic_fig: Path
    ae_trajectory_fig: Path
    next_year_auroc_fig: Path


@dataclass(frozen=True)
class CodeFilterConfig:
    """Which condition codes count as clinical; see ``data.classify_codes``."""

    keep_suffixes: tuple[str, ...]
    keep_descriptions: tuple[str, ...]


@dataclass(frozen=True)
class DataConfig:
    """Cohort selection settings.

    ``code_merge`` maps a duplicate code to its canonical code (read-only); see
    ``data.merge_codes``.
    """

    min_conditions_per_patient: int
    code_merge: Mapping[str, str]
    code_filter: CodeFilterConfig


@dataclass(frozen=True)
class TrajectoriesConfig:
    """Time windowing of each patient's condition history."""

    window: str
    start_year: int
    end_year: int


@dataclass(frozen=True)
class FeaturesConfig:
    """Per-window patient feature settings; see ``features.build_features``."""

    redundant_codes: tuple[str, ...]
    redundant_min_share: float
    tfidf_norm: str
    time_decay_lambda: float
    transition_svd_components: int
    transition_svd_n_iter: int
    transition_weight: float
    chronic_min_duration_days: int


@dataclass(frozen=True)
class GraphConfig:
    """Patient-similarity graph, spectral analysis and drift settings; see ``graph``."""

    k: int
    metric: str
    knn_working_memory_mb: int
    spectral_dim: int
    eigengap_n_eigenvalues: int
    eigengap_min_clusters: int
    kmeans_n_init: int
    phenotype_top_conditions: int
    phenotype_min_patients: int
    drift_jump_threshold: float


@dataclass(frozen=True)
class LabelsConfig:
    """Prediction targets and the leakage cutoff; see ``labels``.

    ``onset_groups`` maps a group name to its codes (read-only, in config order);
    ``borderline_groups`` names onset groups the report flags as borderline, and
    ``near_threshold_groups`` those it notes as also close to ``min_onsets``.
    """

    cutoff_date: date
    horizon_years: int
    min_last_encounter_date: date
    min_onsets: int
    borderline_groups: tuple[str, ...]
    near_threshold_groups: tuple[str, ...]
    onset_groups: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class ModelConfig:
    """Temporal GCN architecture, optimiser and training settings; see ``train.fit``."""

    hidden_dim: int
    embed_dim: int
    dropout: float
    lr: float
    weight_decay: float
    epochs: int
    patience: int
    grad_clip_norm: float
    secondary_weight: float
    smoothness_weight: float
    drop_edge: float
    use_demographics: bool
    log_every_epochs: int
    quick_epochs: int
    quick_patients: int


@dataclass(frozen=True)
class ReferenceModelConfig:
    """The Phase 6 model setting, retrained as the baselines' reference row.

    Its fields replace the same fields of ``ModelConfig`` (``reference_model_config``).
    """

    hidden_dim: int
    embed_dim: int
    dropout: float
    weight_decay: float
    smoothness_weight: float
    drop_edge: float
    use_demographics: bool


@dataclass(frozen=True)
class BaselinesConfig:
    """Logistic-regression baselines; see ``evaluate.run_baselines``."""

    C: float
    max_iter: int
    balanced_class_weight: bool
    reference_model: ReferenceModelConfig


@dataclass(frozen=True)
class TuningConfig:
    """Validation-only hyperparameter grid; see ``train.tuning_settings``.

    ``dims`` holds (hidden_dim, embed_dim) pairs.
    """

    seeds: tuple[int, ...]
    weight_decay: tuple[float, ...]
    dropout: tuple[float, ...]
    dims: tuple[tuple[int, int], ...]
    smoothness_weight: tuple[float, ...]
    drop_edge: tuple[float, ...]
    max_settings: int
    tie_tolerance: float
    top_n_report: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Bootstrap confidence intervals and calibration; see ``evaluate``."""

    bootstrap_resamples: int
    ci_level: float
    calibration_bins: int
    calibration_cv_folds: int
    temperature_bounds: tuple[float, float]


# Calibration methods of the any_onset logits (evaluate.fit_calibrator).
CALIBRATION_METHODS: tuple[str, ...] = ("platt", "temperature")


@dataclass(frozen=True)
class CalibrationConfig:
    """How ``any_onset`` logits become probabilities; one of ``CALIBRATION_METHODS``."""

    method: str


@dataclass(frozen=True)
class CrossValidationConfig:
    """Phase 7 cross-validation; see ``evaluate.run_cross_validation``."""

    folds: int
    inner_val: float
    seeds: tuple[int, ...]
    rewire_swaps_per_edge: int
    demographic_rank_correlation: float


@dataclass(frozen=True)
class EmbeddingAnalysisConfig:
    """Phase 7 embedding analysis; see ``evaluate.analyse_embeddings``."""

    umap_n_neighbors: int
    umap_min_dist: float
    chronic_count_bins: tuple[float, ...]
    trajectory_patients: int
    trajectory_min_onsets: int
    max_onset_labels: int
    onset_label_chars: int
    neighbour_patients: int
    n_neighbours: int
    neighbour_min_conditions: int


@dataclass(frozen=True)
class GraphAEConfig:
    """Phase 7b self-supervised temporal graph autoencoder; see ``train.fit_graph_ae``."""

    hidden_dim: int
    embed_dim: int
    dropout: float
    lr: float
    weight_decay: float
    epochs: int
    patience: int
    grad_clip_norm: float
    next_window_weight: float
    log_every_epochs: int
    tuning_next_window_weights: tuple[float, ...]
    tuning_seeds: tuple[int, ...]


@dataclass(frozen=True)
class EmbeddingEvalConfig:
    """Phase 7b embedding comparison; see ``embeddings``."""

    dim: int
    min_train_onsets: int
    C: float
    max_iter: int
    balanced_class_weight: bool
    n_neighbours: int


@dataclass(frozen=True)
class SplitConfig:
    """Train/validation/test fractions; must sum to 1."""

    train: float
    val: float
    test: float


@dataclass(frozen=True)
class SummaryConfig:
    """Settings for the summary reports and their figures."""

    top_n_conditions: int
    age_bins: tuple[float, ...]
    events_hist_bins: int
    figure_dpi: int
    min_active_patients_per_year: int
    example_patients: int
    top_n_transitions: int
    comorbidity_top_pairs: int
    comorbidity_min_count: int
    chronic_min_patients: int
    chronic_pair_min_patients: int
    heatmap_codes: int
    heatmap_log2_lift_limit: float


@dataclass(frozen=True)
class Config:
    """Top-level project configuration."""

    paths: PathsConfig
    seed: int
    data: DataConfig
    trajectories: TrajectoriesConfig
    features: FeaturesConfig
    graph: GraphConfig
    labels: LabelsConfig
    model: ModelConfig
    baselines: BaselinesConfig
    tuning: TuningConfig
    evaluation: EvaluationConfig
    calibration: CalibrationConfig
    cross_validation: CrossValidationConfig
    embedding_analysis: EmbeddingAnalysisConfig
    graph_ae: GraphAEConfig
    embedding_eval: EmbeddingEvalConfig
    split: SplitConfig
    summary: SummaryConfig


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the project configuration.

    Args:
        path: YAML file to load. Defaults to ``<project root>/config/config.yaml``.

    Returns:
        The parsed configuration. Relative paths in the ``paths`` section are
        resolved against the project root, not the current working directory.

    Raises:
        ValueError: If keys are missing or unknown, a value has the wrong type,
            or the values are inconsistent (see ``_validate``).
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config = _coerce(raw, Config, "config")
    _validate(config)
    return config


def _coerce(value: Any, hint: Any, where: str) -> Any:
    """Convert a raw YAML value to the annotated type ``hint``."""
    if is_dataclass(hint):
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected a mapping, got {type(value).__name__}")
        names = [f.name for f in fields(hint)]
        missing = sorted(set(names) - value.keys())
        unknown = sorted(value.keys() - set(names))
        if missing or unknown:
            raise ValueError(f"{where}: missing keys {missing}, unknown keys {unknown}")
        hints = get_type_hints(hint)
        return hint(**{n: _coerce(value[n], hints[n], f"{where}.{n}") for n in names})

    if get_origin(hint) is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where}: expected a list, got {type(value).__name__}")
        args = get_args(hint)
        item_types = [args[0]] * len(value) if args[-1] is Ellipsis else list(args)
        if len(item_types) != len(value):
            raise ValueError(f"{where}: expected {len(item_types)} items, got {len(value)}")
        return tuple(
            _coerce(v, t, f"{where}[{i}]") for i, (v, t) in enumerate(zip(value, item_types))
        )

    if get_origin(hint) is Mapping:
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected a mapping, got {type(value).__name__}")
        key_type, value_type = get_args(hint)
        items = {
            _coerce(k, key_type, f"{where} key {k!r}"): _coerce(v, value_type, f"{where}[{k!r}]")
            for k, v in value.items()
        }
        return MappingProxyType(items)

    if hint is Path:
        if not isinstance(value, str):
            raise ValueError(f"{where}: expected a path string, got {value!r}")
        p = Path(value)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    if hint is date:
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value)
        raise ValueError(f"{where}: expected an ISO date (YYYY-MM-DD), got {value!r}")

    if hint is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{where}: expected true or false, got {value!r}")
        return value

    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{where}: expected a number, got {value!r} "
                "(PyYAML needs a decimal point in exponents, e.g. 5.0e-4 not 5e-4)"
            )
        return float(value)

    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: expected an integer, got {value!r}")
        return value

    if hint is str:
        # Numeric-looking values such as SNOMED-CT codes are parsed by YAML as ints.
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f"{where}: expected a string, got {value!r}")
        return str(value)

    raise TypeError(f"{where}: unsupported config field type {hint!r}")


def _validate(config: Config) -> None:
    """Check cross-field consistency that per-field typing cannot express."""
    split = config.split
    fractions = (split.train, split.val, split.test)
    if any(not 0.0 <= f <= 1.0 for f in fractions) or not math.isclose(sum(fractions), 1.0):
        raise ValueError(f"split fractions must lie in [0, 1] and sum to 1, got {fractions}")

    traj = config.trajectories
    if traj.start_year > traj.end_year:
        raise ValueError(
            f"trajectories.start_year ({traj.start_year}) is after end_year ({traj.end_year})"
        )

    labels = config.labels
    if labels.horizon_years < 1:
        raise ValueError(f"labels.horizon_years must be >= 1, got {labels.horizon_years}")
    if not labels.cutoff_date <= labels.min_last_encounter_date < horizon_end(config):
        raise ValueError(
            f"labels.min_last_encounter_date ({labels.min_last_encounter_date}) must fall in the "
            f"label horizon [{labels.cutoff_date}, {horizon_end(config)})"
        )
    group_of: dict[str, str] = {}
    shared = sorted(
        code
        for group, codes in labels.onset_groups.items()
        for code in codes
        if group_of.setdefault(code, group) != group
    )
    if shared:
        raise ValueError(f"labels.onset_groups: codes in more than one group: {shared}")
    if not labels.onset_groups:
        raise ValueError("labels.onset_groups must name at least one group")
    for name in ("borderline_groups", "near_threshold_groups"):
        unknown = sorted(set(getattr(labels, name)) - set(labels.onset_groups))
        if unknown:
            raise ValueError(f"labels.{name}: not onset groups: {unknown}")

    model = config.model
    if not 0.0 <= model.dropout < 1.0:
        raise ValueError(f"model.dropout must lie in [0, 1), got {model.dropout}")
    positive = ("hidden_dim", "embed_dim", "epochs", "patience", "log_every_epochs",
                "quick_epochs", "quick_patients", "lr", "grad_clip_norm")
    bad = [name for name in positive if getattr(model, name) <= 0]
    if bad:
        raise ValueError(f"model: these values must be positive: {bad}")
    negative = [n for n in ("weight_decay", "secondary_weight", "smoothness_weight")
                if getattr(model, n) < 0]
    if negative:
        raise ValueError(f"model: these values must not be negative: {negative}")
    if not 0.0 <= model.drop_edge < 1.0:
        raise ValueError(f"model.drop_edge must lie in [0, 1), got {model.drop_edge}")

    baselines = config.baselines
    if baselines.C <= 0 or baselines.max_iter <= 0:
        raise ValueError("baselines.C and baselines.max_iter must be positive")
    _validate_model_values(asdict(baselines.reference_model), "baselines.reference_model")

    tuning = config.tuning
    lists = ("seeds", "weight_decay", "dropout", "dims", "smoothness_weight", "drop_edge")
    empty = [name for name in lists if not getattr(tuning, name)]
    if empty:
        raise ValueError(f"tuning: these lists must not be empty: {empty}")
    for i, (hidden_dim, embed_dim) in enumerate(tuning.dims):
        _validate_model_values({"hidden_dim": hidden_dim, "embed_dim": embed_dim},
                               f"tuning.dims[{i}]")
    for name in ("weight_decay", "dropout", "smoothness_weight", "drop_edge"):
        for value in getattr(tuning, name):
            _validate_model_values({name: value}, f"tuning.{name}")
    if tuning.max_settings < 1 or tuning.top_n_report < 1 or tuning.tie_tolerance < 0:
        raise ValueError(
            "tuning: max_settings and top_n_report must be positive and tie_tolerance >= 0"
        )

    evaluation = config.evaluation
    if evaluation.bootstrap_resamples < 1 or not 0.0 < evaluation.ci_level < 1.0:
        raise ValueError("evaluation: bootstrap_resamples must be >= 1 and ci_level in (0, 1)")
    if evaluation.calibration_bins < 2 or evaluation.calibration_cv_folds < 2:
        raise ValueError("evaluation: calibration_bins and calibration_cv_folds must be >= 2")
    low, high = evaluation.temperature_bounds
    if not 0 < low < 1 < high:
        raise ValueError(
            f"evaluation.temperature_bounds must satisfy 0 < low < 1 < high, got {(low, high)}"
        )
    if config.calibration.method not in CALIBRATION_METHODS:
        raise ValueError(
            f"calibration.method must be one of {list(CALIBRATION_METHODS)}, "
            f"got {config.calibration.method!r}"
        )

    cv = config.cross_validation
    if cv.folds < 2 or not 0.0 < cv.inner_val < 1.0:
        raise ValueError("cross_validation: folds must be >= 2 and inner_val in (0, 1)")
    if not cv.seeds or cv.rewire_swaps_per_edge < 1:
        raise ValueError(
            "cross_validation: seeds must not be empty and rewire_swaps_per_edge must be >= 1"
        )
    if not 0.0 < cv.demographic_rank_correlation <= 1.0:
        raise ValueError("cross_validation.demographic_rank_correlation must lie in (0, 1]")

    analysis = config.embedding_analysis
    counts = ("umap_n_neighbors", "trajectory_patients", "max_onset_labels", "onset_label_chars",
              "neighbour_patients", "n_neighbours")
    bad = [name for name in counts if getattr(analysis, name) < 1]
    if bad or analysis.umap_n_neighbors < 2 or analysis.umap_min_dist < 0:
        raise ValueError(
            f"embedding_analysis: these values must be positive: {bad}; umap_n_neighbors >= 2 "
            "and umap_min_dist >= 0"
        )
    edges = analysis.chronic_count_bins
    if len(edges) < 3 or edges[0] != 0 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(
            "embedding_analysis.chronic_count_bins must start at 0, increase strictly and "
            f"give at least 2 bins, got {list(edges)}"
        )

    graph_ae = config.graph_ae
    positive = ("hidden_dim", "embed_dim", "epochs", "patience", "log_every_epochs", "lr",
                "grad_clip_norm")
    bad = [name for name in positive if getattr(graph_ae, name) <= 0]
    if bad:
        raise ValueError(f"graph_ae: these values must be positive: {bad}")
    if not 0.0 <= graph_ae.dropout < 1.0 or graph_ae.weight_decay < 0:
        raise ValueError("graph_ae: dropout must lie in [0, 1) and weight_decay be >= 0")
    weights = (graph_ae.next_window_weight, *graph_ae.tuning_next_window_weights)
    if not graph_ae.tuning_next_window_weights or not graph_ae.tuning_seeds or min(weights) <= 0:
        raise ValueError("graph_ae: the next-window weights must be positive and the tuning "
                         "lists must not be empty")

    embedding_eval = config.embedding_eval
    counts = ("dim", "min_train_onsets", "max_iter", "n_neighbours")
    bad = [name for name in counts if getattr(embedding_eval, name) < 1]
    if embedding_eval.C <= 0:
        bad.append("C")
    if bad:
        raise ValueError(f"embedding_eval: these values must be positive: {bad}")

    # A canonical code that is itself merged would make the result depend on order.
    code_merge = config.data.code_merge
    chained = sorted(old for old, new in code_merge.items() if new in code_merge)
    if chained:
        raise ValueError(
            f"data.code_merge: these codes map to a code that is itself merged (or to "
            f"themselves): {chained}"
        )


def _validate_model_values(values: Mapping[str, Any], where: str) -> None:
    """Check model settings given outside the ``model`` section (reference, tuning grid)."""
    if any(not 0.0 <= values.get(name, 0.0) < 1.0 for name in ("dropout", "drop_edge")):
        raise ValueError(f"{where}: dropout and drop_edge must lie in [0, 1)")
    if any(values.get(name, 1) <= 0 for name in ("hidden_dim", "embed_dim")):
        raise ValueError(f"{where}: hidden_dim and embed_dim must be positive")
    if any(values.get(name, 0.0) < 0 for name in ("weight_decay", "smoothness_weight")):
        raise ValueError(f"{where}: weight_decay and smoothness_weight must not be negative")


def reference_model_config(config: Config) -> ModelConfig:
    """Return ``config.model`` with the fields of ``baselines.reference_model`` replaced."""
    return replace(config.model, **asdict(config.baselines.reference_model))


def horizon_end(config: Config) -> date:
    """Return the first day after the label horizon: the cutoff plus ``horizon_years``.

    Labels use events in ``[labels.cutoff_date, horizon_end)``; nothing later.

    Args:
        config: The project configuration.

    Returns:
        The exclusive end date of the horizon (2026-01-01 for the project config).
    """
    cutoff = config.labels.cutoff_date
    return cutoff.replace(year=cutoff.year + config.labels.horizon_years)


def feature_window_years(config: Config) -> tuple[int, ...]:
    """Return the trajectory years whose windows may be used as prediction features.

    Trajectories cover the full data range, so they may run past
    ``labels.cutoff_date``. The yearly window for year ``Y`` spans
    ``[Y-01-01, (Y+1)-01-01)``; only windows that close on or before the cutoff
    are returned, so features never see post-cutoff events. Later windows are
    reserved for labels.

    Args:
        config: The project configuration.

    Returns:
        The feature-window years in ascending order (empty if none close by the cutoff).

    Raises:
        ValueError: If ``trajectories.window`` is not ``"yearly"``.
    """
    traj = config.trajectories
    if traj.window != "yearly":
        raise ValueError(f"trajectories.window {traj.window!r} is not supported; use 'yearly'")
    cutoff = config.labels.cutoff_date
    return tuple(
        year for year in range(traj.start_year, traj.end_year + 1) if date(year + 1, 1, 1) <= cutoff
    )
