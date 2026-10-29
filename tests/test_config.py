"""Tests for loading config/config.yaml into typed dataclasses."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ehrgraph.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, Config, load_config


def _write_variant(tmp_path: Path, section: str, key: str, value: object) -> Path:
    """Copy the real config with one value replaced and return the new file's path."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw[section][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_config_loads() -> None:
    config = load_config()
    assert isinstance(config, Config)


def test_seed_is_42() -> None:
    assert load_config().seed == 42


def test_values_are_typed() -> None:
    config = load_config()
    assert isinstance(config.model.weight_decay, float)
    assert config.model.weight_decay == pytest.approx(5e-4)
    assert config.features.tfidf_ngram_range == (1, 2)
    assert config.graph.k == 10
    assert config.paths.raw_dir == (PROJECT_ROOT / "data" / "raw").resolve()


def test_split_must_sum_to_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        load_config(_write_variant(tmp_path, "split", "train", 0.9))


def test_unknown_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(_write_variant(tmp_path, "graph", "kk", 5))


def test_trajectories_must_end_before_cutoff(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="leakage"):
        load_config(_write_variant(tmp_path, "trajectories", "end_year", 2021))
