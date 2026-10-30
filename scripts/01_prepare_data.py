"""Prepare the Synthea data: tidy events, code dictionary, validation and summaries.

Reads the raw CSVs and writes data/processed/events.parquet, the code-merged and
clinically filtered data/processed/events_clinical.parquet, data/processed/code_dictionary.csv,
reports/data_summary.md with its figures and reports/clinical_summary.md (all
paths from config/config.yaml)::

    conda run -n ehrgraph python scripts/01_prepare_data.py [--config PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ehrgraph.config import load_config
from ehrgraph.data import (
    build_code_dictionary,
    build_events,
    filter_clinical_events,
    load_raw,
    merge_codes,
    save_code_dictionary,
    save_events,
    validate,
    write_clinical_summary,
    write_data_summary,
)
from ehrgraph.utils import get_logger, set_seed

logger = get_logger("prepare_data")


def main(argv: list[str] | None = None) -> None:
    """Run the data-preparation pipeline end to end."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=None, help="config YAML (default: config/config.yaml)"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    raw = load_raw(cfg)
    events = build_events(raw)
    checks = validate(events, raw.patients)
    code_dictionary = build_code_dictionary(events)
    save_code_dictionary(code_dictionary, cfg.paths.code_dictionary_csv)
    save_events(events, cfg.paths.events_parquet)
    merged = merge_codes(events, cfg.data.code_merge)
    clinical = filter_clinical_events(merged, cfg.data.code_filter)
    save_events(clinical, cfg.paths.events_clinical_parquet)
    write_data_summary(events, raw.patients, code_dictionary, checks, cfg)
    write_clinical_summary(events, clinical, raw.patients, code_dictionary, cfg)
    logger.info("Data preparation finished")


if __name__ == "__main__":
    main()
