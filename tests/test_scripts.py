"""Tests for the Phase 10 scripts: run_all.py (step list, safe cleaning, timing report) and
10_final_figures.py (numbered export, dpi check, captions), on temporary directories."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import ModuleType

import pytest
from matplotlib.figure import Figure

from ehrgraph.config import PROJECT_ROOT, Config, FinalFigureConfig, FinalFiguresConfig, load_config

SCRIPTS = PROJECT_ROOT / "scripts"


def _script(name: str) -> ModuleType:
    """Import scripts/<name>.py as a module (file names may start with a digit)."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while building
    spec.loader.exec_module(module)
    return module


run_all = _script("run_all")
final_figures = _script("10_final_figures")


# --- run_all.py -----------------------------------------------------------------


def test_steps_run_every_phase_script_in_order() -> None:
    scripts = [step.script for step in run_all.STEPS]
    assert all((SCRIPTS / s).exists() for s in scripts)
    assert scripts[-1] == "08_predict_demo.py"  # the demo runs last, after the app assets
    prefixes = [s[:2] for s in scripts[:-1]]
    assert prefixes == sorted(prefixes)  # 01 ... 09 is the phase order
    assert scripts.index("06a_baselines.py") < scripts.index("06_train.py")  # 06 reads 06a
    assert [step.phase for step in run_all.STEPS] == [
        "1", "2", "3", "4", "5", "6a", "6", "7", "7b", "9", "8"]
    assert "06b_tune.py" not in scripts  # the tuned setting is already in the config


def test_output_dirs_are_processed_data_and_artifacts() -> None:
    cfg = load_config()
    assert run_all.output_dirs(cfg) == (cfg.paths.processed_dir.resolve(),
                                        cfg.paths.artifacts_dir.resolve())


@pytest.mark.parametrize("field, value", [
    ("artifacts_dir", PROJECT_ROOT),  # the project itself
    ("artifacts_dir", PROJECT_ROOT.parent / "elsewhere"),  # outside the project
    ("processed_dir", PROJECT_ROOT / "data"),  # holds data/raw
    ("processed_dir", PROJECT_ROOT / "data" / "raw"),
    ("artifacts_dir", PROJECT_ROOT / "reports" / "artifacts"),  # inside reports
])
def test_cleaning_refuses_dangerous_dirs(field: str, value: Path) -> None:
    cfg = load_config()
    cfg = replace(cfg, paths=replace(cfg.paths, **{field: value}))
    with pytest.raises(ValueError, match="refusing to delete"):
        run_all.output_dirs(cfg)


def test_timing_report_lists_steps_total_and_failure() -> None:
    steps = run_all.STEPS
    started = datetime(2026, 10, 31, 12, 0, 0)
    text = run_all.timing_report([(steps[0], 30.0), (steps[1], 90.0)], started)
    assert "| 1 | `01_prepare_data.py` |" in text
    assert "| 30.0 | 0:30 | 25.0% |" in text
    assert "| **Total** |  | 120.0 | 2:00 | 100.0% |" in text
    assert "All 2 steps finished." in text
    failed = run_all.timing_report([(steps[0], 5.0)], started, failed=steps[0])
    assert "**Failed at step 1 (`01_prepare_data.py`)**" in failed


@pytest.mark.parametrize("seconds, text", [(0.4, "0:00"), (59.6, "1:00"), (3725.0, "62:05")])
def test_minutes(seconds: float, text: str) -> None:
    assert run_all.minutes(seconds) == text


# --- 10_final_figures.py --------------------------------------------------------


@pytest.fixture
def figure_cfg(tmp_path: Path) -> Config:
    """Two final figures drawn into ``tmp_path``, exported to ``tmp_path/final``."""
    base = load_config()
    figures_dir = tmp_path / "figures"
    paths = replace(base.paths, figures_dir=figures_dir,
                    eigenvalues_fig=figures_dir / "eigen.png",
                    graph_drift_fig=figures_dir / "drift.png",
                    final_figures_dir=figures_dir / "final",
                    final_captions_md=figures_dir / "final" / "captions.md")
    figures = FinalFiguresConfig((
        FinalFigureConfig("eigenvalues_fig", "eigenvalues", "The spectrum."),
        FinalFigureConfig("graph_drift_fig", "drift", "The drift."),
    ))
    cfg = replace(base, paths=paths, final_figures=figures)
    figures_dir.mkdir()
    for path in (paths.eigenvalues_fig, paths.graph_drift_fig):
        _draw(path, cfg.summary.figure_save_dpi)
    return cfg


def _draw(path: Path, dpi: int) -> None:
    fig = Figure(figsize=(2, 1), dpi=dpi)
    fig.subplots().plot([0, 1], [0, 1])
    fig.savefig(path, dpi=dpi)


def test_export_numbers_figures_in_order_and_writes_captions(figure_cfg: Config) -> None:
    out = figure_cfg.paths.final_figures_dir
    out.mkdir()
    (out / "07_stale.png").write_bytes(b"old")  # from an earlier export
    exported = final_figures.export_figures(figure_cfg)
    assert sorted(p.name for p in out.glob("*.png")) == ["01_eigenvalues.png", "02_drift.png"]
    assert [e.size for e in exported] == [(600, 300)] * 2  # 2 x 1 inches at 300 dpi
    assert (out / "01_eigenvalues.png").read_bytes() == figure_cfg.paths.eigenvalues_fig.read_bytes()
    text = final_figures.captions_markdown(exported, figure_cfg)
    assert "**Figure 1.** The spectrum." in text
    assert "![Figure 2](02_drift.png)" in text
    assert "| 1 | `01_eigenvalues.png` | `../eigen.png` | 600 x 300 |" in text


def test_export_rejects_figures_at_another_dpi(figure_cfg: Config) -> None:
    _draw(figure_cfg.paths.graph_drift_fig, 150)
    with pytest.raises(ValueError, match="drift.png"):
        final_figures.export_figures(figure_cfg)


def test_export_rejects_missing_figures(figure_cfg: Config) -> None:
    figure_cfg.paths.eigenvalues_fig.unlink()
    with pytest.raises(FileNotFoundError, match="eigen.png"):
        final_figures.export_figures(figure_cfg)
