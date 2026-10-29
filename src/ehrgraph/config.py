"""Load ``config/config.yaml`` into typed, immutable dataclasses.

Every section of the YAML maps to one frozen dataclass. Loading is strict:
missing or unknown keys and values of the wrong type raise ``ValueError``, so
a typo in the config fails loudly instead of silently falling back to a default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass
from datetime import date
from pathlib import Path
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


@dataclass(frozen=True)
class DataConfig:
    """Cohort selection settings."""

    min_conditions_per_patient: int


@dataclass(frozen=True)
class TrajectoriesConfig:
    """Time windowing of each patient's condition history."""

    window: str
    start_year: int
    end_year: int


@dataclass(frozen=True)
class FeaturesConfig:
    """Text and temporal feature settings."""

    tfidf_ngram_range: tuple[int, int]
    time_decay_lambda: float


@dataclass(frozen=True)
class GraphConfig:
    """Patient-similarity k-NN graph settings."""

    k: int
    metric: str


@dataclass(frozen=True)
class LabelsConfig:
    """Prediction target definition and leakage cutoff."""

    cutoff_date: date
    horizon_years: int
    target_codes: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    """Temporal GCN architecture and optimiser settings."""

    hidden_dim: int
    embed_dim: int
    dropout: float
    lr: float
    weight_decay: float
    epochs: int


@dataclass(frozen=True)
class SplitConfig:
    """Train/validation/test fractions; must sum to 1."""

    train: float
    val: float
    test: float


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
    split: SplitConfig


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

    # The last trajectory window must close before the prediction cutoff, otherwise
    # trajectory features would include post-cutoff events (label leakage).
    last_window_end = date(traj.end_year + 1, 1, 1)
    if last_window_end > config.labels.cutoff_date:
        raise ValueError(
            f"trajectories.end_year ({traj.end_year}) must end before "
            f"labels.cutoff_date ({config.labels.cutoff_date}) to avoid leakage"
        )
