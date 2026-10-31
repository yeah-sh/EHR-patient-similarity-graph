"""Copy the report figures, numbered in report order, to reports/figures/final/ with captions.

Reads the final_figures section of the config: each entry names a figure path of the paths
section and is copied to paths.final_figures_dir as <number>_<name>.png (01, 02, ...). The
figures are the pipeline's own, not redrawn, so they share its styling; each must have been
saved at summary.figure_save_dpi (300). Numbered PNGs already in the folder are removed first.
Writes paths.final_captions_md: the numbered captions, each under its figure. Run after
run_all.py (or the phase scripts)::

    conda run -n ehrgraph python scripts/10_final_figures.py [--config PATH]
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ehrgraph.config import DEFAULT_CONFIG_PATH, Config, FinalFigureConfig, load_config
from ehrgraph.data import markdown_table
from ehrgraph.utils import get_logger

logger = get_logger("final_figures")

MIN_NUMBER_WIDTH = 2  # 01, 02, ...


@dataclass(frozen=True)
class ExportedFigure:
    """A copied figure: its number, config entry, source and copy, and size in pixels."""

    number: int
    figure: FinalFigureConfig
    source: Path
    path: Path
    size: tuple[int, int]


def saved_dpi(path: Path) -> float | None:
    """The resolution recorded in a PNG (None if it records none)."""
    with Image.open(path) as image:
        dpi = image.info.get("dpi")
    return float(dpi[0]) if dpi else None


def export_figures(cfg: Config) -> list[ExportedFigure]:
    """Copy every ``final_figures`` figure, numbered, to ``paths.final_figures_dir``.

    Raises:
        FileNotFoundError: If a source figure is missing (run the pipeline first).
        ValueError: If a source figure was not saved at ``summary.figure_save_dpi``.
    """
    figures = cfg.final_figures.figures
    sources = [getattr(cfg.paths, figure.source) for figure in figures]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing figures (run scripts/run_all.py): {missing}")
    target = cfg.summary.figure_save_dpi
    wrong = {path.name: saved_dpi(path) for path in sources}
    wrong = {name: dpi for name, dpi in wrong.items() if dpi is None or round(dpi) != target}
    if wrong:
        raise ValueError(f"figures not saved at {target} dpi (summary.figure_save_dpi); re-run "
                         f"the phase that draws them: {wrong}")

    out = cfg.paths.final_figures_dir
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("[0-9]*_*.png"):
        old.unlink()
    width = max(MIN_NUMBER_WIDTH, len(str(len(figures))))
    exported = []
    for number, (figure, source) in enumerate(zip(figures, sources), start=1):
        path = out / f"{number:0{width}d}_{figure.name}.png"
        shutil.copyfile(source, path)
        with Image.open(path) as image:
            size = image.size
        exported.append(ExportedFigure(number, figure, source, path, size))
        logger.info("Figure %d: %s <- %s (%d x %d px)", number, path.name, source.name, *size)
    return exported


def captions_markdown(exported: list[ExportedFigure], cfg: Config) -> str:
    """The captions file: an index table, then each figure with its numbered caption."""
    out = cfg.paths.final_captions_md.parent

    def link(path: Path) -> str:
        return Path(os.path.relpath(path, out)).as_posix()

    lines = [
        "# Report figures",
        "",
        f"{len(exported)} figures, numbered in report order (`final_figures` in "
        f"`config/config.yaml`), saved at {cfg.summary.figure_save_dpi} dpi. They are the "
        "pipeline's own figures, copied by `scripts/10_final_figures.py`; the numbers in the "
        "captions are from `reports/findings.md`, `results.md` and `embedding_results.md`.",
        "",
        markdown_table(
            ["Figure", "File", "Drawn by", "Pixels"],
            [[str(e.number), f"`{e.path.name}`", f"`{link(e.source)}`",
              f"{e.size[0]:,} x {e.size[1]:,}"] for e in exported],
            "rllr",
        ),
        "",
    ]
    for e in exported:
        lines += [f"![Figure {e.number}]({link(e.path)})", "",
                  f"**Figure {e.number}.** {e.figure.caption}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Export the report figures and write their captions."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="config YAML (default: config/config.yaml)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    exported = export_figures(cfg)
    path = cfg.paths.final_captions_md
    path.write_text(captions_markdown(exported, cfg), encoding="utf-8")
    logger.info("Wrote %d figures and %s", len(exported), path)


if __name__ == "__main__":
    main()
