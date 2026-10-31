"""Phase 7b: train the self-supervised TemporalGraphAE and compare its embeddings.

1. Tuning, validation only: every graph_ae.tuning_next_window_weights value trained
   with every graph_ae.tuning_seeds seed; the weight with the lowest mean validation
   next-window loss is chosen. No onset label is read.
2. The final TemporalGraphAE (chosen weight, global seed) and three variants: no
   graph (A_hat = I), degree-preserving rewired graphs (Phase 7 model 7's), and link
   only. Their state_dicts and logs go to artifacts/graph_ae/.
3. The other embeddings: PCA of the 2022 features and of every window concatenated,
   the spectral embedding of the 2022 graph, and the supervised TemporalGCN's h
   (artifacts/model.pt). Every embedding goes to artifacts/embeddings_7b.npz.
4. Metrics on test-split patients: next-year new conditions 2021 -> 2022, neighbour
   concordance, phenotype separation; the PC1 share; the any_onset linear probe on
   Phase 7's folds.
5. reports/embedding_results.md and .csv, reports/embedding_summary.md and the figures.

Run after 07_evaluate.py. --reuse-embeddings skips steps 1-3 and reads
artifacts/embeddings_7b.npz and the training summaries in artifacts/graph_ae/::

    conda run -n ehrgraph python scripts/07b_embeddings.py [--reuse-embeddings] [--config PATH]
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from ehrgraph.config import DEFAULT_CONFIG_PATH, Config, load_config
from ehrgraph.data import load_patients
from ehrgraph.embeddings import (
    BASE_RATE,
    GRAPH_AE,
    LINK_ONLY,
    NO_GRAPH,
    QUESTIONS,
    REWIRED,
    SPECTRAL,
    SUPERVISED,
    Embedding,
    VariantFit,
    base_rate_scores,
    check_model_2,
    concordance_results,
    embedding_specs,
    embedding_table,
    linear_probe,
    micro_auroc,
    neighbour_jaccard,
    next_year_results,
    next_year_scores,
    next_year_task,
    plot_ae_embeddings,
    plot_next_year_auroc,
    probe_results,
    random_jaccard,
    spectral_window_embedding,
    states_embedding,
    step_lengths,
    supervised_states,
    variant_fits_frame,
    window_pca_embeddings,
    write_embedding_results,
    write_embedding_summary,
)
from ehrgraph.evaluate import (
    cluster_scores,
    gcn_inputs,
    load_patient_context,
    require_trained_model,
    trajectory_example_nodes,
    umap_coordinates,
    window_umap_coordinates,
)
from ehrgraph.models import pc1_share
from ehrgraph.train import (
    TEST,
    GraphAEData,
    ModelData,
    fit_graph_ae,
    graph_ae_states,
    load_graph_ae_data,
    load_model_data,
    save_graph_ae,
    select_next_window_weight,
    tune_graph_ae,
    with_graphs,
)
from ehrgraph.trajectories import load_trajectories
from ehrgraph.utils import get_logger, set_seed

logger = get_logger("embeddings")

TUNING_RUNS_CSV = "tuning_runs.csv"
VARIANTS_CSV = "variants.csv"
CONFIG_USED = "config_used.yaml"
STATES = f"{GRAPH_AE}.states"  # the final GraphAE's GRU state after every window


def train_embeddings(
    data: GraphAEData, model_data: ModelData, cfg: Config, config_path: Path
) -> tuple[dict[str, Embedding], np.ndarray, pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    """Steps 1-3: tune, train the four GraphAE variants, build the other embeddings, save.

    Returns:
        The embeddings by key, the final GraphAE's states [N, T, d], the tuning runs,
        the variants' training summaries, and the previous (2021) embedding of the
        chosen setting per tuning seed.
    """
    ae_dir = cfg.paths.graph_ae_dir
    runs, fits = tune_graph_ae(data, cfg.graph_ae)
    weight, summary = select_next_window_weight(runs)
    logger.info("Tuning (validation next-window loss, lower is better):\n%s\nChosen weight: %g",
                summary.to_string(index=False), weight)
    chosen = replace(cfg.graph_ae, next_window_weight=weight)
    n = len(data.patient_ids)
    rewired = gcn_inputs(model_data, cfg.cross_validation.rewire_swaps_per_edge,
                         cfg.seed)["tgcn_rewired"].weights
    variant_data = {
        GRAPH_AE: data,
        NO_GRAPH: with_graphs(data, [sp.csr_matrix((n, n)) for _ in data.years]),
        REWIRED: with_graphs(data, rewired),
        LINK_ONLY: data,
    }
    embeddings, variants, states = {}, [], {}
    for key, inputs in variant_data.items():
        result = fits.get((weight, cfg.seed)) if key == GRAPH_AE else None
        if result is None:
            result = fit_graph_ae(inputs, chosen, cfg.seed, link_only=key == LINK_ONLY)
        save_graph_ae(result, ae_dir, key)
        states[key] = graph_ae_states(result.model, inputs)
        embeddings[key] = states_embedding(states[key])
        best = result.history.iloc[result.best_epoch - 1]
        variants.append(VariantFit(key, result.best_epoch, result.epochs_run, result.stopped_early,
                                   float(best["val_next_loss"]), result.link_auroc,
                                   result.pc1_share, round(result.seconds, 1)))
    seed_previous = {seed: graph_ae_states(fits[(weight, seed)].model, data)[:, -2]
                     for seed in cfg.graph_ae.tuning_seeds}

    embeddings |= window_pca_embeddings(model_data.matrices, cfg.embedding_eval.dim, cfg.seed)
    embeddings[SPECTRAL] = Embedding(
        spectral_window_embedding(model_data.weights[-1], cfg.embedding_eval.dim, cfg.seed),
        spectral_window_embedding(model_data.weights[-2], cfg.embedding_eval.dim, cfg.seed),
    )
    embeddings[SUPERVISED] = states_embedding(supervised_states(cfg, data))

    variants_frame = variant_fits_frame(variants)
    runs.to_csv(ae_dir / TUNING_RUNS_CSV, index=False)
    variants_frame.to_csv(ae_dir / VARIANTS_CSV, index=False)
    shutil.copyfile(config_path, ae_dir / CONFIG_USED)
    arrays = {f"{key}.{part}": getattr(emb, part) for key, emb in embeddings.items()
              for part in ("final", "previous")}
    arrays |= {STATES: states[GRAPH_AE], "patient_ids": np.array(data.patient_ids)}
    arrays |= {f"seed{seed}.previous": prev for seed, prev in seed_previous.items()}
    cfg.paths.embeddings_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.paths.embeddings_npz, **arrays)
    logger.info("Wrote %s, %s, %s and %s", cfg.paths.embeddings_npz, ae_dir / TUNING_RUNS_CSV,
                ae_dir / VARIANTS_CSV, ae_dir / CONFIG_USED)
    return embeddings, states[GRAPH_AE], runs, variants_frame, seed_previous


def load_embeddings(
    data: GraphAEData, cfg: Config, config_path: Path, keys: list[str]
) -> tuple[dict[str, Embedding], np.ndarray, pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    """Read what ``train_embeddings`` saved; same return values.

    Raises:
        ValueError: If the saved files were made with another graph_ae or
            embedding_eval section, or for other patients.
    """
    ae_dir = cfg.paths.graph_ae_dir
    used = yaml.safe_load((ae_dir / CONFIG_USED).read_text(encoding="utf-8"))
    current = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    for section in ("graph_ae", "embedding_eval", "model", "seed"):
        if used.get(section) != current.get(section):
            raise ValueError(f"the saved embeddings were made with another {section!r}; "
                             "run without --reuse-embeddings")
    stored = np.load(cfg.paths.embeddings_npz)
    if tuple(stored["patient_ids"]) != data.patient_ids:
        raise ValueError("the saved embeddings are for other patients")
    embeddings = {key: Embedding(stored[f"{key}.final"], stored[f"{key}.previous"]) for key in keys}
    seed_previous = {seed: stored[f"seed{seed}.previous"] for seed in cfg.graph_ae.tuning_seeds}
    return (embeddings, stored[STATES], pd.read_csv(ae_dir / TUNING_RUNS_CSV),
            pd.read_csv(ae_dir / VARIANTS_CSV), seed_previous)


def main(argv: list[str] | None = None) -> None:
    """Run Phase 7b end to end."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="config YAML (default: config/config.yaml)")
    parser.add_argument("--reuse-embeddings", action="store_true",
                        help="read the saved embeddings instead of training (steps 1-3)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    ev = cfg.embedding_eval
    if cfg.graph_ae.embed_dim != ev.dim:
        raise ValueError(f"graph_ae.embed_dim ({cfg.graph_ae.embed_dim}) must equal "
                         f"embedding_eval.dim ({ev.dim})")
    require_trained_model(args.config, cfg)  # the supervised h (v) must match the config
    data = load_graph_ae_data(cfg)
    model_data = load_model_data(cfg)
    if data.patient_ids != model_data.patient_ids or data.years != model_data.years:
        raise ValueError("the GraphAE data and the model data disagree on the nodes or windows")
    specs = embedding_specs(data.years, ev.dim)
    keys = [spec.key for spec in specs]

    # 1-3. The embeddings.
    if args.reuse_embeddings:
        embeddings, ae_states, runs, variants, seed_previous = load_embeddings(
            data, cfg, args.config, keys)
        logger.info("Read the saved embeddings from %s", cfg.paths.embeddings_npz)
    else:
        embeddings, ae_states, runs, variants, seed_previous = train_embeddings(
            data, model_data, cfg, args.config)
    weight, tuning_summary = select_next_window_weight(runs)
    embeddings = {key: embeddings[key] for key in keys}
    history = pd.read_csv(cfg.paths.graph_ae_dir / f"{GRAPH_AE}_log.csv")
    pc1_history = history.loc[history["pc1_share"].notna(),
                              ["epoch", "pc1_share", "link_auroc", "val_next_loss"]]

    # 4. Metrics.
    context = load_patient_context(cfg, model_data)
    trajectories = load_trajectories(cfg)
    task = next_year_task(trajectories.events, load_patients(cfg), data.patient_ids, data.codes,
                          data.split, data.years[-1], cfg.labels.cutoff_date, ev.min_train_onsets)
    scores = {key: next_year_scores(emb.previous, task, data.split, ev)
              for key, emb in embeddings.items()}
    scores[BASE_RATE] = base_rate_scores(task, data.split)  # reference: no patient information
    next_year = next_year_results(task, scores, data.split, GRAPH_AE, cfg.evaluation, cfg.seed)
    seed_sensitivity = pd.DataFrame([
        {"seed": seed,
         "micro_auroc": micro_auroc(task, next_year_scores(previous, task, data.split, ev),
                                    data.split),
         "pc1_share": float(runs.set_index(["next_window_weight", "seed"])
                            .loc[(weight, seed), "pc1_share"])}
        for seed, previous in seed_previous.items()
    ])

    memory = cfg.graph.knn_working_memory_mb
    test = data.split == TEST
    queries = np.flatnonzero(test & (np.diff(context.chronic.indptr) > 0))
    year = data.years[-1]
    concordance = concordance_results(
        {key: neighbour_jaccard(emb.final, context.chronic, queries, ev.n_neighbours, memory)
         for key, emb in embeddings.items()},
        {"Random other patients (chance)": random_jaccard(context.chronic, queries,
                                                          ev.n_neighbours, cfg.seed),
         f"{year} input features (the graph's neighbours)": neighbour_jaccard(
             model_data.matrices[-1].toarray(), context.chronic, queries, ev.n_neighbours,
             memory)},
        GRAPH_AE, cfg.evaluation, cfg.seed,
    )
    clusters = cluster_scores({key: emb.final[test] for key, emb in embeddings.items()},
                              context.phenotype[test], cfg.seed, memory)
    probe_predictions, probe_fits = linear_probe(
        {key: emb.final for key, emb in embeddings.items()}, model_data, cfg)
    model_2_check = check_model_2(probe_predictions, cfg)
    probe = probe_results(probe_predictions, probe_fits, keys, cfg)
    table = embedding_table(specs, embeddings, next_year, concordance, clusters, probe)

    # 5. Figures and reports.
    final_umap = umap_coordinates(embeddings[GRAPH_AE].final, cfg)
    window_umap = window_umap_coordinates(ae_states, context.nonempty, cfg)
    plot_ae_embeddings(final_umap, window_umap, trajectory_example_nodes(context, cfg), context,
                       cfg)
    base_micro = float(next_year.table.set_index("key").loc[BASE_RATE, "micro_auroc"])
    plot_next_year_auroc(table, specs, task.year, base_micro, cfg.paths.next_year_auroc_fig, cfg)
    write_embedding_results(table, specs, runs, tuning_summary, weight, variants, pc1_history,
                            task, next_year, concordance, clusters, probe, seed_sensitivity,
                            context.names, len(data.codes), model_2_check,
                            step_lengths(ae_states, context.nonempty), cfg)
    write_embedding_summary(table, specs, weight, task, next_year, concordance, seed_sensitivity,
                            context.names, cfg)

    columns = ["number", "name", "next_micro_auroc", "next_micro_low", "next_micro_high",
               "next_macro_auroc", "next_macro_low", "next_macro_high", "concordance",
               "silhouette", "davies_bouldin", "pc1_share", "probe_auroc", "probe_vs_model2",
               "probe_folds_flipped", "probe_fold_auroc_raw"]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        logger.info("Embedding results:\n%s", table[columns].to_string(index=False))
    by_key = table.set_index("key")
    for question, key in QUESTIONS:
        r = by_key.loc[key]
        logger.info("%s i minus %s: next-year micro AUROC %+.4f (%+.4f to %+.4f); macro "
                    "%+.4f (%+.4f to %+.4f); concordance %+.4f (%+.4f to %+.4f); probe %+.4f "
                    "(%+.4f to %+.4f)", question, r.number,
                    r.next_micro_diff, r.next_micro_diff_low, r.next_micro_diff_high,
                    r.next_macro_diff, r.next_macro_diff_low, r.next_macro_diff_high,
                    r.concordance_diff, r.concordance_diff_low, r.concordance_diff_high,
                    r.probe_diff, r.probe_diff_low, r.probe_diff_high)
    logger.info("Next-year micro AUROC of the codes' base rates alone: %.4f", base_micro)
    logger.info("PC1 share of the GraphAE's final embedding: %.4f (all patients); chosen "
                "next-window weight %g; model 2 check %s", pc1_share(embeddings[GRAPH_AE].final),
                weight, model_2_check)
    logger.info("Phase 7b finished")


if __name__ == "__main__":
    main()
