"""Build patient-similarity graphs per training-period window and analyse them.

Reads the model inputs (features of the model cohort for the windows that close by
labels.cutoff_date, via features.load_model_inputs) and the trajectories, then
writes W_<year>.npz, A_hat_<year>.npz and graphs.json to data/processed/graphs/,
with the final window's spectral embedding and phenotype clusters, and
reports/graphs_summary.md with the eigenvalue and drift figures (all paths from
config/config.yaml). Windows after the cutoff never get a graph::

    conda run -n ehrgraph python scripts/04_build_graphs.py [--config PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ehrgraph.config import load_config
from ehrgraph.features import chronic_presence, load_model_inputs
from ehrgraph.graph import (
    build_graphs,
    cluster_phenotypes,
    final_window_spectrum,
    graph_statistics,
    neighbourhood_drift,
    plot_drift,
    plot_eigenvalues,
    save_graphs,
    write_graph_summary,
)
from ehrgraph.trajectories import load_code_vocabulary, load_trajectories
from ehrgraph.utils import get_logger, set_seed

logger = get_logger("build_graphs")


def main(argv: list[str] | None = None) -> None:
    """Run the graph pipeline end to end."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=None, help="config YAML (default: config/config.yaml)"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    graphs = build_graphs(load_model_inputs(cfg), cfg)
    stats = [graph_statistics(W) for W in graphs.weights]
    spectrum = final_window_spectrum(graphs, cfg)

    # Chronic conditions of the clustered patients: graph node -> patient_index row.
    presence, codes = chronic_presence(load_trajectories(cfg), cfg)
    clustered = presence[graphs.rows[spectrum.nodes]]
    phenotypes = cluster_phenotypes(
        clustered, spectrum.labels, codes, cfg.graph.phenotype_top_conditions,
        cfg.graph.phenotype_min_patients,
    )
    drift = neighbourhood_drift(graphs, cfg)

    save_graphs(graphs, spectrum, cfg)
    plot_eigenvalues(spectrum, cfg.paths.eigenvalues_fig, cfg)
    plot_drift(drift, cfg.paths.graph_drift_fig, cfg)
    vocabulary = load_code_vocabulary(cfg)
    descriptions = dict(zip(vocabulary["code"], vocabulary["description"]))
    write_graph_summary(graphs, stats, spectrum, phenotypes, clustered, drift, descriptions, cfg)
    logger.info("Graph building finished")


if __name__ == "__main__":
    main()
