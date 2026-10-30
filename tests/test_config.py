"""Tests for loading config/config.yaml into typed dataclasses."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
import yaml

from ehrgraph.config import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    Config,
    feature_window_years,
    load_config,
)


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
    assert config.graph.k == 10
    assert config.paths.raw_dir == (PROJECT_ROOT / "data" / "raw").resolve()


def test_code_filter_is_typed() -> None:
    code_filter = load_config().data.code_filter
    assert code_filter.keep_suffixes == ("(disorder)",)
    assert isinstance(code_filter.keep_descriptions, tuple)
    assert "Prediabetes (finding)" in code_filter.keep_descriptions


def test_tfidf_ngram_range_is_removed() -> None:
    assert not hasattr(load_config().features, "tfidf_ngram_range")
    assert "tfidf_ngram_range" not in DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")


def test_redundant_codes_are_strings() -> None:
    redundant = load_config().features.redundant_codes
    assert redundant and all(isinstance(code, str) for code in redundant)
    assert "312608009" in redundant  # Laceration - injury


def test_code_merge_is_typed_and_read_only() -> None:
    code_merge = load_config().data.code_merge
    assert dict(code_merge) == {"15777000": "714628002"}  # Prediabetes, approved
    with pytest.raises(TypeError):
        code_merge["1"] = "2"  # type: ignore[index]


def test_code_merge_numeric_codes_become_strings(tmp_path: Path) -> None:
    config = load_config(_write_variant(tmp_path, "data", "code_merge", {15777000: 714628002}))
    assert dict(config.data.code_merge) == {"15777000": "714628002"}


@pytest.mark.parametrize(
    "code_merge",
    [
        {"1": "2", "2": "3"},  # chain: 1 -> 2 -> 3
        {"1": "1"},  # self-map
    ],
)
def test_code_merge_rejects_chains(tmp_path: Path, code_merge: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="code_merge"):
        load_config(_write_variant(tmp_path, "data", "code_merge", code_merge))


def test_split_must_sum_to_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        load_config(_write_variant(tmp_path, "split", "train", 0.9))


def test_unknown_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(_write_variant(tmp_path, "graph", "kk", 5))


def test_trajectories_may_extend_past_cutoff(tmp_path: Path) -> None:
    cutoff_year = load_config().labels.cutoff_date.year
    config = load_config(_write_variant(tmp_path, "trajectories", "end_year", cutoff_year + 2))
    assert config.trajectories.end_year > cutoff_year


@pytest.mark.parametrize(
    ("cutoff", "last_feature_year"),
    [
        (date(2021, 1, 1), 2020),  # window 2020 closes exactly at the cutoff
        (date(2021, 7, 1), 2020),  # window 2021 straddles the cutoff, so it is excluded
        (date(2021, 12, 31), 2020),
        (date(2022, 1, 1), 2021),
        (date(2022, 12, 31), 2021),  # the previous project cutoff lost window 2022
        (date(2023, 1, 1), 2022),  # the project cutoff keeps it
    ],
)
def test_feature_windows_do_not_extend_past_cutoff(cutoff: date, last_feature_year: int) -> None:
    base = load_config()
    config = replace(
        base,
        trajectories=replace(base.trajectories, start_year=2010, end_year=2025),
        labels=replace(base.labels, cutoff_date=cutoff),
    )
    years = feature_window_years(config)
    assert years == tuple(range(2010, last_feature_year + 1))
    assert all(date(year + 1, 1, 1) <= cutoff for year in years)


def test_project_cutoff_splits_feature_and_label_windows() -> None:
    config = load_config()
    labels, traj = config.labels, config.trajectories
    assert labels.cutoff_date == date(2023, 1, 1)
    features = feature_window_years(config)
    assert features == tuple(range(2017, 2023))
    # Every later window is for labels only, and the horizon covers exactly those windows.
    later = tuple(y for y in range(traj.start_year, traj.end_year + 1) if y not in features)
    assert later == (2023, 2024, 2025)
    assert labels.cutoff_date.year + labels.horizon_years - 1 == later[-1]
