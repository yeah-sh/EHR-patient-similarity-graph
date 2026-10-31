"""Run the whole pipeline from scratch, in order, with the time each step takes.

Deletes paths.processed_dir and paths.artifacts_dir (data/processed/ and artifacts/), then
runs each step's script in a fresh process of this Python interpreter, stopping at the
first failure:

    1   01_prepare_data.py        events, code dictionary, data reports
    2   02_build_trajectories.py  yearly windows 2017-2025
    3   03_build_features.py      TF-IDF + transition features, comorbidity, model cohort
    4   04_build_graphs.py        k-NN graphs, A_hat, spectrum, phenotypes
    5   05_build_labels.py        onset labels and the split
    6a  06a_baselines.py          validation baselines, which 06_train.py compares with
    6   06_train.py               final TemporalGCN and its calibration
    7   07_evaluate.py            cross-validation, test set, embedding analysis
    7b  07b_embeddings.py         TemporalGraphAE and the embedding comparison
    9   09_app_assets.py          assets of the Streamlit app
    8   08_predict_demo.py        example-patient predictions with the final models

Not run: 06b_tune.py (its chosen setting is already in the model section of the config)
and 10_final_figures.py (run it afterwards). data/raw/ and reports/
are never deleted; each step rewrites its own reports and figures. The time per step is
logged and written to paths.run_all_timing_md::

    conda run -n ehrgraph python scripts/run_all.py [--config PATH]
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ehrgraph.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, Config, load_config
from ehrgraph.data import markdown_table
from ehrgraph.utils import get_logger, set_seed

logger = get_logger("run_all")

SCRIPTS_DIR = Path(__file__).resolve().parent
SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class Step:
    """One pipeline step: its phase label, script (in scripts/) and what it produces."""

    phase: str
    script: str
    what: str


STEPS: tuple[Step, ...] = (
    Step("1", "01_prepare_data.py", "events, code dictionary, data reports"),
    Step("2", "02_build_trajectories.py", "yearly windows and transitions"),
    Step("3", "03_build_features.py", "TF-IDF + transition features, comorbidity, cohort"),
    Step("4", "04_build_graphs.py", "k-NN graphs, A_hat, spectrum, phenotypes"),
    Step("5", "05_build_labels.py", "onset labels and the train/validation/test split"),
    Step("6a", "06a_baselines.py", "validation baselines that 06_train.py compares with"),
    Step("6", "06_train.py", "final TemporalGCN and its Platt calibration"),
    Step("7", "07_evaluate.py", "cross-validation, test set, embedding analysis"),
    Step("7b", "07b_embeddings.py", "TemporalGraphAE and the embedding comparison"),
    Step("9", "09_app_assets.py", "assets of the Streamlit app"),
    Step("8", "08_predict_demo.py", "example-patient predictions with the final models"),
)


def output_dirs(cfg: Config) -> tuple[Path, ...]:
    """The directories a run from scratch deletes: processed data and artifacts.

    Raises:
        ValueError: If one is not strictly inside the project root, or holds (or is
            inside) the raw data or the reports.
    """
    paths = cfg.paths
    protected = (paths.raw_dir.resolve(), paths.reports_dir.resolve())
    dirs = (paths.processed_dir.resolve(), paths.artifacts_dir.resolve())
    for directory in dirs:
        if PROJECT_ROOT not in directory.parents:
            raise ValueError(f"refusing to delete {directory}: not inside {PROJECT_ROOT}")
        clash = [p for p in protected
                 if p == directory or directory in p.parents or p in directory.parents]
        if clash:
            raise ValueError(f"refusing to delete {directory}: it overlaps {clash[0]}")
    return dirs


def clean_outputs(cfg: Config) -> None:
    """Delete and recreate ``output_dirs``."""
    for directory in output_dirs(cfg):
        if directory.exists():
            shutil.rmtree(directory)
            logger.info("Deleted %s", directory)
        directory.mkdir(parents=True)


def run_step(step: Step, config_path: Path) -> float:
    """Run one step's script with ``config_path``; return its wall-clock seconds.

    Raises:
        subprocess.CalledProcessError: If the script exits with an error.
    """
    start = time.perf_counter()
    subprocess.run([sys.executable, str(SCRIPTS_DIR / step.script), "--config", str(config_path)],
                   check=True, cwd=PROJECT_ROOT)
    return time.perf_counter() - start


def minutes(seconds: float) -> str:
    """``seconds`` as m:ss."""
    whole = round(seconds)
    return f"{whole // SECONDS_PER_MINUTE}:{whole % SECONDS_PER_MINUTE:02d}"


def timing_report(timings: list[tuple[Step, float]], started: datetime,
                  failed: Step | None = None) -> str:
    """Markdown of the time per step, the total, and the step that failed if any."""
    total = sum(seconds for _, seconds in timings)
    rows = [[step.phase, f"`{step.script}`", step.what, f"{seconds:,.1f}", minutes(seconds),
             f"{seconds / total:.1%}" if total else "n/a"]
            for step, seconds in timings]
    rows.append(["", "**Total**", "", f"{total:,.1f}", minutes(total), "100.0%" if total else ""])
    status = (f"**Failed at step {failed.phase} (`{failed.script}`)**; later steps did not run."
              if failed else f"All {len(timings)} steps finished.")
    return "\n".join([
        "# Pipeline run from scratch",
        "",
        f"`scripts/run_all.py`, started {started:%Y-%m-%d %H:%M:%S}, deleted `data/processed/` "
        "and `artifacts/` first. Python "
        f"{platform.python_version()} on {platform.system()} {platform.release()} "
        f"({platform.machine()}). Wall-clock time per step, each in a fresh process.",
        "",
        markdown_table(["Phase", "Script", "Produces", "Seconds", "m:ss", "Share"], rows,
                       "lllrrr"),
        "",
        status,
        "",
    ])


def main(argv: list[str] | None = None) -> None:
    """Delete the outputs and run every step in order."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="config YAML (default: config/config.yaml)")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    set_seed(cfg.seed)  # also exports PYTHONHASHSEED to every step
    started = datetime.now()
    clean_outputs(cfg)

    timings: list[tuple[Step, float]] = []
    report = cfg.paths.run_all_timing_md
    report.parent.mkdir(parents=True, exist_ok=True)
    for i, step in enumerate(STEPS, start=1):
        logger.info("Step %d/%d, phase %s: %s (%s)", i, len(STEPS), step.phase, step.script,
                    step.what)
        start = time.perf_counter()
        try:
            seconds = run_step(step, config_path)
        except subprocess.CalledProcessError as error:
            timings.append((step, time.perf_counter() - start))
            report.write_text(timing_report(timings, started, failed=step), encoding="utf-8")
            logger.error("Phase %s (%s) failed with exit code %d; wrote %s", step.phase,
                         step.script, error.returncode, report)
            raise SystemExit(error.returncode) from error
        timings.append((step, seconds))
        logger.info("Phase %s finished in %s (%.1f s)", step.phase, minutes(seconds), seconds)

    report.write_text(timing_report(timings, started), encoding="utf-8")
    for step, seconds in timings:
        logger.info("  %-3s %-26s %8.1f s  %6s", step.phase, step.script, seconds,
                    minutes(seconds))
    logger.info("All steps finished in %s; wrote %s",
                minutes(sum(s for _, s in timings)), report)


if __name__ == "__main__":
    main()
