"""Phase 9: precompute what the Streamlit app shows beyond inference.Predictor.

Nothing is trained or refitted: every asset is read or derived from earlier phases' saved
outputs, so the app itself only loads files. Writes (``paths.app_*``):

- ``state_umap.npz``: one UMAP (``embedding_analysis`` settings, global seed) of every
  non-empty patient-window's TemporalGraphAE state, as in Phase 7b
  (``evaluate.window_umap_coordinates``): ``coords`` [N, T, 2] (NaN for empty windows),
  ``patient_ids`` and ``years``. The app draws trajectories on it and places a new patient's
  states with ``inference.map_positions``.
- ``comorbidity_nodes.csv`` and ``comorbidity_pairs.csv``: the Phase 3 chronic comorbidity
  matrices as a network. Nodes: chronic codes in at least ``summary.chronic_min_patients``
  patients, clustered as the Phase 3 heatmap (``features.lift_linkage`` of their clipped log2
  lift profiles), cut into ``app.network_clusters`` branches; branches of at least
  ``app.network_min_cluster_size`` codes are clusters 1, 2, ... in the dendrogram's leaf
  order, the rest 0 (unclustered); ``position`` is the leaf order. Pairs: every pair of those codes active together in a patient,
  with lift and Jaccard (``features.top_comorbidity_pairs``).
- ``graph_drift.csv``: neighbourhood Jaccard between consecutive windows' graphs
  (``graph.neighbourhood_drift``, as in Phase 4).
- ``app_stats.json``: dataset and graph statistics, the next-year base-rate AUROC that
  Phase 7b compares the embeddings with (``embeddings.base_rate_scores``, same bootstrap),
  and Phase 7b's neighbour-concordance references (random patients, input features).

    conda run -n ehrgraph python scripts/09_app_assets.py [--config PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, leaves_list

from ehrgraph.config import DEFAULT_CONFIG_PATH, Config, load_config
from ehrgraph.data import load_patients
from ehrgraph.embeddings import (
    BASE_RATE,
    GRAPH_AE,
    base_rate_scores,
    concordance_results,
    neighbour_jaccard,
    next_year_results,
    next_year_task,
    random_jaccard,
)
from ehrgraph.evaluate import window_umap_coordinates
from ehrgraph.features import (
    Comorbidity,
    chronic_presence,
    feature_window_years,
    lift_linkage,
    lift_profiles,
    load_feature_codes,
    load_model_cohort,
    load_model_inputs,
    most_common_codes,
    top_comorbidity_pairs,
)
from ehrgraph.graph import load_graphs, load_phenotype_clusters, neighbourhood_drift
from ehrgraph.labels import ANY_ONSET, load_labels
from ehrgraph.train import TEST, GraphAEData, load_graph_ae_data
from ehrgraph.trajectories import load_code_vocabulary, load_trajectories
from ehrgraph.utils import get_logger, set_seed

logger = get_logger("app_assets")


def state_umap(cfg: Config) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]:
    """The UMAP of every non-empty patient-window's GraphAE state (Phase 7b's trajectory map)."""
    inputs = load_model_inputs(cfg)
    stored = np.load(cfg.paths.embeddings_npz)
    if tuple(stored["patient_ids"]) != inputs.patient_ids:
        raise ValueError(f"{cfg.paths.embeddings_npz.name} is for other patients than the inputs")
    nonempty = np.column_stack([np.diff(sp.csr_matrix(X).indptr) > 0 for X in inputs.matrices])
    coords = window_umap_coordinates(stored[f"{GRAPH_AE}.states"], nonempty, cfg)
    logger.info("UMAP of %d non-empty patient-windows (%d empty left out)", int(nonempty.sum()),
                int((~nonempty).sum()))
    return coords, inputs.patient_ids, inputs.years


def chronic_comorbidity(cfg: Config) -> Comorbidity:
    """The saved Phase 3 chronic comorbidity matrices (counted in patients)."""
    counts = sp.load_npz(cfg.paths.chronic_comorbidity_counts_npz).tocsr()
    lift = sp.load_npz(cfg.paths.chronic_comorbidity_lift_npz).tocsr()
    frequency = counts.diagonal()
    present = np.flatnonzero(frequency > 0)
    # The diagonal of the lift is population / frequency (features.comorbidity_lift).
    populations = np.round(lift.diagonal()[present] * frequency[present]).astype(np.int64)
    if len(set(populations.tolist())) != 1:
        raise ValueError("the saved chronic lift and counts disagree on the population")
    return Comorbidity(counts, lift, int(populations[0]), "patients", "chronic conditions",
                       feature_window_years(cfg), load_feature_codes(cfg))


def comorbidity_network(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nodes (clustered and ordered as the Phase 3 heatmap) and pairs of the chronic network."""
    chronic = chronic_comorbidity(cfg)
    vocabulary = load_code_vocabulary(cfg)
    descriptions = dict(zip(vocabulary["code"], vocabulary["description"]))
    min_patients = cfg.summary.chronic_min_patients
    nodes = most_common_codes(chronic, len(chronic.codes), min_patients)
    linkage = lift_linkage(lift_profiles(chronic, nodes, cfg.summary.heatmap_log2_lift_limit))
    order = leaves_list(linkage)
    raw = fcluster(linkage, t=cfg.app.network_clusters, criterion="maxclust")
    # Number the multi-code clusters in leaf order (cluster 1 comes first around the
    # circle); codes on smaller branches are unclustered (0).
    sizes = np.bincount(raw)
    kept = [c for c in dict.fromkeys(raw[order]) if sizes[c] >= cfg.app.network_min_cluster_size]
    renumber = {**{c: 0 for c in raw}, **{c: i for i, c in enumerate(kept, start=1)}}
    frequency = chronic.counts.diagonal()
    node_table = pd.DataFrame({
        "code": [chronic.codes[i] for i in nodes[order]],
        "description": [descriptions[chronic.codes[i]] for i in nodes[order]],
        "patients": frequency[nodes[order]].astype(np.int64),
        "cluster": [renumber[c] for c in raw[order]],
        "position": np.arange(len(nodes)),
    })
    pairs = top_comorbidity_pairs(chronic, descriptions, sp.triu(chronic.counts, k=1).nnz, 1,
                                  min_patients)
    logger.info("Comorbidity network: %d codes chronic in >= %d of %d patients, %d clusters "
                "(%d codes unclustered), %d pairs active together", len(node_table), min_patients,
                chronic.population, len(kept), int((node_table["cluster"] == 0).sum()), len(pairs))
    for cluster, members in node_table.groupby("cluster")["description"]:
        logger.info("  cluster %d: %s", cluster, "; ".join(members))
    return node_table, pairs


def concordance_references(cfg: Config, data: GraphAEData) -> dict[str, dict[str, float]]:
    """Phase 7b's neighbour-concordance references with their intervals.

    Test-split patients with a chronic condition, as in Phase 7b: random other patients
    (chance), the final input features (the graph's own neighbours) and, as a check that
    this reproduces Phase 7b, the TemporalGraphAE.
    """
    ev = cfg.embedding_eval
    presence, _ = chronic_presence(load_trajectories(cfg), cfg)
    _, rows = load_model_cohort(cfg)
    chronic = sp.csr_matrix(presence[rows])
    queries = np.flatnonzero((data.split == TEST) & (np.diff(chronic.indptr) > 0))
    memory = cfg.graph.knn_working_memory_mb
    final = np.load(cfg.paths.embeddings_npz)[f"{GRAPH_AE}.final"]
    features = load_model_inputs(cfg).matrices[-1].toarray()
    table = concordance_results(
        {GRAPH_AE: neighbour_jaccard(final, chronic, queries, ev.n_neighbours, memory)},
        {"random": random_jaccard(chronic, queries, ev.n_neighbours, cfg.seed),
         "features": neighbour_jaccard(features, chronic, queries, ev.n_neighbours, memory)},
        GRAPH_AE, cfg.evaluation, cfg.seed,
    ).set_index("key")
    return {key: {"concordance": float(table.loc[key, "concordance"]),
                  "low": float(table.loc[key, "low"]), "high": float(table.loc[key, "high"])}
            for key in (GRAPH_AE, "random", "features")}


def next_year_base_rate(cfg: Config, data: GraphAEData) -> dict[str, float]:
    """Phase 7b's next-year micro AUROC of the codes' base rates alone, with its interval."""
    task = next_year_task(load_trajectories(cfg).events, load_patients(cfg), data.patient_ids,
                          data.codes, data.split, data.years[-1], cfg.labels.cutoff_date,
                          cfg.embedding_eval.min_train_onsets)
    results = next_year_results(task, {BASE_RATE: base_rate_scores(task, data.split)}, data.split,
                                BASE_RATE, cfg.evaluation, cfg.seed)
    row = results.table.set_index("key").loc[BASE_RATE]
    return {"auroc": float(row["micro_auroc"]), "low": float(row["micro_low"]),
            "high": float(row["micro_high"]), "year": int(task.year), "codes": len(task.codes)}


def dataset_stats(cfg: Config) -> dict[str, object]:
    """Counts shown on the app's overview page."""
    events = pd.read_parquet(cfg.paths.events_parquet, columns=["patient_id", "code"])
    clinical = pd.read_parquet(cfg.paths.events_clinical_parquet, columns=["patient_id", "code"])
    inputs = load_model_inputs(cfg)
    graphs = load_graphs(cfg)
    clusters = load_phenotype_clusters(cfg)
    labels = load_labels(cfg)
    test = pd.read_csv(cfg.paths.test_predictions_csv)
    data = load_graph_ae_data(cfg)
    cutoff = cfg.labels.cutoff_date
    per_year = []
    for year, W in zip(graphs.years, graphs.weights):
        degree = np.diff(sp.csr_matrix(W).indptr)
        per_year.append({"year": year, "nodes_with_edges": int((degree > 0).sum()),
                         "isolated": int((degree == 0).sum()), "edges": int(W.nnz // 2),
                         "mean_weight": float(W.data.mean())})
    return {
        "patients": len(load_patients(cfg)),
        "condition_events": len(events),
        "condition_codes": int(events["code"].nunique()),
        "clinical_events": len(clinical),
        "clinical_codes": int(clinical["code"].nunique()),
        "reference_patients": len(inputs.patient_ids),
        "feature_codes": len(load_feature_codes(cfg)),
        "feature_dim": int(inputs.matrices[0].shape[1]),
        "years": list(inputs.years),
        "cutoff_date": cutoff.isoformat(),
        "label_years": [cutoff.year, cutoff.year + cfg.labels.horizon_years - 1],
        "onset_groups": len(cfg.labels.onset_groups),
        "k": cfg.graph.k,
        "graphs": per_year,
        "phenotypes": {str(c): int(n) for c, n in clusters.value_counts().sort_index().items()},
        "any_onset_observed": int(labels[ANY_ONSET].notna().sum()),
        "any_onset_positives": int((labels[ANY_ONSET] == 1).sum()),
        "test_observed": len(test),
        "test_positives": int((test[ANY_ONSET] == 1).sum()),
        "next_year_base_rate": next_year_base_rate(cfg, data),
        "concordance_references": concordance_references(cfg, data),
        "seed": cfg.seed,
    }


def main(argv: list[str] | None = None) -> None:
    """Write the Phase 9 app assets."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="config YAML (default: config/config.yaml)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    paths = cfg.paths
    paths.app_dir.mkdir(parents=True, exist_ok=True)

    nodes, pairs = comorbidity_network(cfg)
    nodes.to_csv(paths.app_network_nodes_csv, index=False)
    pairs.to_csv(paths.app_network_pairs_csv, index=False)
    drift = neighbourhood_drift(load_graphs(cfg), cfg)
    drift.to_csv(paths.app_drift_csv, index=False)
    stats = dataset_stats(cfg)
    paths.app_stats_json.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    logger.info("Next-year base-rate micro AUROC %.4f (%.4f-%.4f)",
                *(stats["next_year_base_rate"][k] for k in ("auroc", "low", "high")))
    for key, value in stats["concordance_references"].items():
        logger.info("Concordance %s: %.4f (%.4f-%.4f)", key, value["concordance"], value["low"],
                    value["high"])
    coords, patient_ids, years = state_umap(cfg)
    np.savez(paths.app_state_umap_npz, coords=coords, patient_ids=np.array(patient_ids),
             years=np.array(years))
    logger.info("Wrote %s", ", ".join(p.name for p in (
        paths.app_network_nodes_csv, paths.app_network_pairs_csv, paths.app_drift_csv,
        paths.app_stats_json, paths.app_state_umap_npz)))


if __name__ == "__main__":
    main()
