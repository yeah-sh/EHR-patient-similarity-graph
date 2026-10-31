"""Phase 9: Streamlit app for the EHR comorbidity-trajectory project.

The app is about patient similarity and comorbidity trajectories; prediction is secondary.
Pages: overview, new patient, patient explorer, comorbidity network, similarity graph over
time, embeddings and results.

Nothing is trained here. The app loads ``inference.Predictor`` once (``st.cache_resource``),
the assets of ``scripts/09_app_assets.py`` and the saved reports of earlier phases; every
path comes from ``config/config.yaml`` and is resolved against the project root. Run from
the project root::

    conda run -n ehrgraph streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sp
import streamlit as st
from sklearn.metrics import roc_curve

from ehrgraph.inference import (
    RISK_SET_EXPLANATION,
    RISK_SET_RULE,
    NewPatient,
    Prediction,
    Predictor,
    format_share,
    map_positions,
)

logger = logging.getLogger(__name__)

TITLE = ("Electronic health record networks: time-series embedding of comorbidity "
         "trajectories via linear-algebraic patient-similarity graphs")
DISCLAIMER = "Synthetic data (Synthea). Research prototype. Not for clinical use."

# --- Styling: the validated reference palette (dataviz skill), light and dark steps ------

CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
              "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9",
             "#e66767"],
}
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
TEXT = {"light": "#0b0b0b", "dark": "#f0efec"}
TEXT_SECONDARY = {"light": "#52514e", "dark": "#c3c2b7"}
CONTEXT = {"light": "#bdbcb5", "dark": "#55544f"}  # recessive grey for context marks
FILL = {"light": "#cde2fb", "dark": "#184f95"}  # diagram boxes (sequential blue 100 / 600)
CONTEXT_OPACITY = 0.35
LINE_WIDTH = 2
MARKER_SIZE = 9
CONTEXT_MARKER_SIZE = 3
LABEL_CHARS = 42  # condition names are cut to this many characters on axes
ARC_POINTS = 25  # points per arc of the comorbidity arc diagram
EDGE_WIDTHS = (1.0, 2.2, 3.8)  # edge tertiles, thin to thick
DOT_LINE_BREAK = "\\n"  # Graphviz's line-break escape inside a label
_SHARE_SUFFIX = re.compile(r"\s*\(\d+% vs \d+% overall\)")


def mode() -> str:
    """The active Streamlit theme, "light" or "dark"."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


def colour(slot: int) -> str:
    """Categorical slot ``slot`` (0-based), in fixed order, for the active theme."""
    return CATEGORICAL[mode()][slot % len(CATEGORICAL[mode()])]


def rgba(hex_colour: str, alpha: float) -> str:
    """``#rrggbb`` with an opacity, for plotly."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def cluster_colour(cluster: int) -> str:
    """Clusters 1, 2, ... keep the categorical slots in fixed order on every page (phenotype
    k is always slot k); 0 (unclustered) is grey."""
    return colour(cluster - 1) if cluster > 0 else CONTEXT[mode()]


def style(fig: go.Figure, height: int, legend: bool = True) -> go.Figure:
    """Shared layout: compact margins, horizontal legend above the plot, unified hover font."""
    fig.update_layout(
        height=height, margin={"l": 10, "r": 10, "t": 40, "b": 10}, showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
    )
    return fig


def hide_axes(fig: go.Figure) -> go.Figure:
    """For maps whose coordinates carry no units (UMAP): no ticks, grid or zero lines."""
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig


def ci(value: float, low: float, high: float, digits: int = 3, signed: bool = False) -> str:
    """"0.579 (0.549-0.608)"; signed values as "+0.023 (-0.010 to +0.057)"."""
    if signed:
        return f"{value:+.{digits}f} ({low:+.{digits}f} to {high:+.{digits}f})"
    return f"{value:.{digits}f} ({low:.{digits}f}-{high:.{digits}f})"


def group_name(group: str) -> str:
    """Onset group name for display."""
    return "CKD" if group == "ckd" else group.replace("_", " ")


def cut(text: str, chars: int = LABEL_CHARS) -> str:
    """Shorten a label to ``chars`` characters with an ellipsis."""
    return text if len(text) <= chars else text[:chars - 1] + "…"


# --- Loading (once per server process) -----------------------------------------------------


@dataclass(frozen=True)
class Assets:
    """Everything the pages show beyond the Predictor.

    Attributes:
        stats: ``app_stats.json``.
        coords: [N, T, 2] reference map (UMAP) position of each patient-window; NaN if empty.
        map_nodes: [M] node of each mapped patient-window.
        map_states: [M, d] their GraphAE states.
        map_xy: [M, 2] their map positions.
        weights: The similarity graph W of each window (from A_hat and the degrees).
        nodes, pairs: The chronic comorbidity network.
        drift: Neighbourhood Jaccard between consecutive windows.
        findings: ``reports/findings.md``.
        results: Phase 7 cross-validation table.
        embeddings: Phase 7b embedding comparison.
        oof: Phase 7 out-of-fold predictions.
        test: The final model's test predictions.
    """

    stats: dict[str, Any]
    coords: np.ndarray
    map_nodes: np.ndarray
    map_states: np.ndarray
    map_xy: np.ndarray
    weights: list[sp.csr_matrix]
    nodes: pd.DataFrame
    pairs: pd.DataFrame
    drift: pd.DataFrame
    findings: str
    results: pd.DataFrame
    embeddings: pd.DataFrame
    oof: pd.DataFrame
    test: pd.DataFrame


@st.cache_resource(show_spinner="Loading the reference patients, graphs and models (once)…")
def load_predictor() -> Predictor:
    """The Phase 8 predictor: saved transforms, graphs, weights and reference tables."""
    return Predictor.load()


@st.cache_resource(show_spinner="Loading the app assets (once)…")
def load_assets(_predictor: Predictor) -> Assets:
    """Read the saved assets; raise FileNotFoundError naming the script that writes them."""
    paths = _predictor.cfg.paths
    needed = [paths.app_stats_json, paths.app_state_umap_npz, paths.app_network_nodes_csv,
              paths.app_network_pairs_csv, paths.app_drift_csv, paths.findings_md,
              paths.results_csv, paths.embedding_results_csv, paths.cv_predictions_csv,
              paths.test_predictions_csv]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing {', '.join(missing)}: run scripts/09_app_assets.py "
                                "(and the earlier phases) first")
    umap = np.load(paths.app_state_umap_npz)
    if tuple(umap["patient_ids"]) != _predictor.patient_ids:
        raise ValueError(f"{paths.app_state_umap_npz.name} is for other reference patients")
    coords = umap["coords"]
    mapped = ~np.isnan(coords[..., 0])
    nodes, windows = np.nonzero(mapped)
    weights = []
    for A, degrees in zip(_predictor.adjacency, _predictor.degrees):
        root = sp.diags(np.sqrt(degrees))
        W = sp.csr_matrix(root @ A @ root)  # D^(1/2) A_hat D^(1/2) = W + I
        W.setdiag(0.0)
        W.eliminate_zeros()
        weights.append(W)
    return Assets(
        stats=json.loads(paths.app_stats_json.read_text(encoding="utf-8")),
        coords=coords,
        map_nodes=nodes,
        map_states=_predictor.reference_states[nodes, windows],
        map_xy=coords[nodes, windows],
        weights=weights,
        nodes=pd.read_csv(paths.app_network_nodes_csv, dtype={"code": str}),
        pairs=pd.read_csv(paths.app_network_pairs_csv, dtype={"code_a": str, "code_b": str}),
        drift=pd.read_csv(paths.app_drift_csv),
        findings=paths.findings_md.read_text(encoding="utf-8"),
        results=pd.read_csv(paths.results_csv),
        embeddings=pd.read_csv(paths.embedding_results_csv),
        oof=pd.read_csv(paths.cv_predictions_csv),
        test=pd.read_csv(paths.test_predictions_csv),
    )


def resources() -> tuple[Predictor, Assets]:
    """The cached predictor and assets, or an error message and a stopped page."""
    try:
        predictor = load_predictor()
        return predictor, load_assets(predictor)
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Cannot load the saved artifacts: {error}")
        st.stop()
        raise  # unreachable: st.stop() ends the run


@st.cache_data(max_entries=64, show_spinner="Placing the patient…")
def predict(_predictor: Predictor, history: tuple[tuple[Any, ...], ...], birth_date: str,
            sex: str) -> Prediction:
    """``Predictor.predict``, cached on the input."""
    return _predictor.predict(NewPatient(list(history), birth_date, sex))


# --- Shared pieces -------------------------------------------------------------------------


def phenotype_label(predictor: Predictor, cluster: int) -> str:
    """"2 · Neuropathy due to type 2 diabetes mellitus, Nonproliferative …"."""
    description = predictor.phenotypes[cluster].description
    top = [_SHARE_SUFFIX.sub("", part) for part in description.split("; ")][:2]
    return cut(f"{cluster} · {', '.join(top)}", 60)


def phenotype_card(predictor: Predictor, cluster: int) -> None:
    """The phenotype, its size and its most over-represented chronic conditions."""
    phenotype = predictor.phenotypes[cluster]
    st.markdown(f"**Phenotype {cluster}** · {phenotype.patients:,} reference patients")
    st.markdown("\n".join(f"- {part}" for part in phenotype.description.split("; ")))
    st.caption("Most over-represented chronic conditions: prevalence in the phenotype vs all "
               "reference patients (spectral clustering of the 2022 similarity graph, Phase 4).")


def similar_table(similar: pd.DataFrame) -> None:
    """The similar patients with the chronic conditions they share."""
    table = pd.DataFrame({
        "#": similar["rank"],
        "Patient": similar["patient_id"].str[:8],
        "Similarity": similar["similarity"],
        "Age": similar["age"].round().astype(int),
        "Sex": similar["sex"],
        "Phenotype": similar["phenotype"],
        "Shared / their chronic": [f"{s} / {c}" for s, c in
                                   zip(similar["shared_conditions"], similar["chronic_conditions"])],
        "Shared chronic conditions": similar["shared"].replace("", "–"),
    })
    st.dataframe(table, hide_index=True, column_config={
        "Similarity": st.column_config.NumberColumn(format="%.3f",
                                                    help="Cosine similarity in the 2022 "
                                                         "spectral embedding"),
        "Age": st.column_config.NumberColumn(help="Age on the prediction cutoff, 2023-01-01"),
    })


def map_figure(assets: Assets, predictor: Predictor, cluster: int | None, path: np.ndarray,
               years: list[int], hover: list[str], name: str,
               others: pd.DataFrame | None = None) -> go.Figure:
    """A trajectory over the reference map, with the phenotype's patient-years highlighted.

    Args:
        assets: The map.
        predictor: For the phenotypes.
        cluster: Phenotype whose patient-years are coloured (others grey), or None.
        path: [m, 2] the trajectory's map positions, in year order.
        years: Their years.
        hover: Their hover texts.
        name: Legend name of the trajectory.
        others: Optional x, y, text of extra points (e.g. similar patients).
    """
    m = mode()
    in_cluster = (predictor.clusters[assets.map_nodes] == cluster) if cluster else \
        np.zeros(len(assets.map_nodes), dtype=bool)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=assets.map_xy[~in_cluster, 0], y=assets.map_xy[~in_cluster, 1], mode="markers",
        marker={"size": CONTEXT_MARKER_SIZE, "color": CONTEXT[m], "opacity": CONTEXT_OPACITY},
        name="Reference patient-years", hoverinfo="skip"))
    if cluster:
        fig.add_trace(go.Scattergl(
            x=assets.map_xy[in_cluster, 0], y=assets.map_xy[in_cluster, 1], mode="markers",
            marker={"size": CONTEXT_MARKER_SIZE, "color": cluster_colour(cluster),
                    "opacity": CONTEXT_OPACITY},
            name=f"Phenotype {cluster} patient-years", hoverinfo="skip"))
    if others is not None and len(others):
        fig.add_trace(go.Scatter(
            x=others["x"], y=others["y"], mode="markers", text=others["text"],
            hovertemplate="%{text}<extra></extra>", name="Similar patients, 2022",
            marker={"size": MARKER_SIZE, "symbol": "diamond-open", "color": TEXT_SECONDARY[m],
                    "line": {"width": LINE_WIDTH}}))
    fig.add_trace(go.Scatter(
        x=path[:, 0], y=path[:, 1], mode="lines+markers+text", text=[str(y) for y in years],
        textposition="top center", textfont={"color": TEXT[m]}, hovertext=hover,
        hovertemplate="%{hovertext}<extra></extra>", name=name,
        line={"color": TEXT[m], "width": LINE_WIDTH},
        marker={"size": MARKER_SIZE, "color": TEXT[m],
                "line": {"color": SURFACE[m], "width": LINE_WIDTH}}))
    return hide_axes(style(fig, 560))


# --- Page 1: overview ------------------------------------------------------------------------


def pipeline_dot(stats: dict[str, Any]) -> str:
    """The pipeline as a Graphviz diagram, in the active theme's colours."""
    m = mode()
    years = stats["years"]
    graph_2022 = stats["graphs"][-1]
    phenotypes = len(stats["phenotypes"])
    boxes = {
        "data": ["Synthea EHR", f"{stats['patients']:,} patients",
                 f"{stats['clinical_events']:,} clinical condition events"],
        "traj": ["Comorbidity trajectories", f"yearly windows {years[0]}-{years[-1]}",
                 f"outcomes {stats['label_years'][0]}-{stats['label_years'][1]} only"],
        "feat": ["Features per window", f"time-decayed TF-IDF of {stats['feature_codes']} codes",
                 f"+ transition SVD = {stats['feature_dim']}"],
        "graph": ["Patient-similarity graphs", f"cosine k-NN, k = {stats['k']}, per year",
                  "Â = D^-½ (W + I) D^-½", f"{graph_2022['edges']:,} edges in {graph_2022['year']}"],
        "spec": ["Spectral embedding", f"normalized Laplacian, {years[-1]}",
                 f"→ {phenotypes} phenotypes"],
        "gcn": ["Temporal graph network", "GCN per year + GRU", "(GraphAE, self-supervised)"],
        "out": ["Embeddings and trajectories", "similar patients, phenotypes,",
                "neighbour outcomes"],
    }
    lines = [
        "digraph pipeline {",
        '  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.35;',
        f'  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, '
        f'fillcolor="{FILL[m]}", color="{FILL[m]}", fontcolor="{TEXT[m]}", margin="0.15,0.08"];',
        f'  edge [color="{TEXT_SECONDARY[m]}", arrowsize=0.7];',
        *[f'  {key} [label="{DOT_LINE_BREAK.join(parts)}"];' for key, parts in boxes.items()],
        "  data -> traj -> feat -> graph; graph -> spec; graph -> gcn; spec -> out; gcn -> out;",
        "}",
    ]
    return "\n".join(lines)


def findings_sections(text: str) -> tuple[str, list[tuple[str, str]], list[tuple[str, str]]]:
    """The intro, the numbered findings and the other sections of findings.md."""
    parts = re.split(r"^## ", text, flags=re.MULTILINE)
    intro = "\n".join(line for line in parts[0].splitlines() if not line.startswith("# "))
    numbered, other = [], []
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        (numbered if re.match(r"\d+\.", heading) else other).append((heading.strip(), body.strip()))
    return intro.strip(), numbered, other


def overview_page() -> None:
    """Title, pipeline, dataset statistics and the four findings."""
    predictor, assets = resources()
    stats = assets.stats
    st.title(TITLE)
    st.caption(DISCLAIMER)

    st.subheader("Pipeline")
    st.graphviz_chart(pipeline_dot(stats), width="stretch")
    years = stats["years"]
    st.markdown(
        f"- **Trajectories.** Each patient's clinical conditions, as yearly windows "
        f"{years[0]}-{years[-1]}; events from the cutoff {stats['cutoff_date']} on are used only "
        "as outcomes.\n"
        "- **Per-year similarity graphs.** Each year, every patient is linked to the "
        f"{stats['k']} patients whose condition profiles are most similar (cosine), giving one "
        "sparse graph per year.\n"
        "- **Linear algebra.** The normalized Laplacian of the last year's graph gives a spectral "
        "embedding; its eigenvectors cluster the patients into phenotypes.\n"
        "- **Temporal graph network.** A graph convolution per year passes information between "
        "similar patients; a recurrent unit (GRU) reads the years in order. Its state after each "
        "year is the patient's trajectory embedding.\n"
        "- **Use.** A new patient is inserted into each year's graph without retraining, placed "
        "in the spectral embedding, and compared with their most similar reference patients.")

    st.subheader("Dataset")
    graphs = pd.DataFrame(stats["graphs"])
    observed, positives = stats["any_onset_observed"], stats["any_onset_positives"]
    row = st.columns(4)
    row[0].metric("Patients (Synthea)", f"{stats['patients']:,}", border=True,
                  help="Synthetic patients in the raw data")
    row[1].metric("Reference patients", f"{stats['reference_patients']:,}", border=True,
                  help=f"With a condition or transition in {years[0]}-{years[-1]}: the graph "
                       "nodes every page compares with")
    row[2].metric("Clinical condition events", f"{stats['clinical_events']:,}", border=True,
                  help=f"{stats['clinical_codes']} clinical SNOMED-CT codes; "
                       f"{stats['feature_codes']} enter the features")
    row[3].metric("Features per patient-year", f"{stats['feature_dim']}", border=True,
                  help=f"{stats['feature_codes']} code weights + transition components")
    row = st.columns(4)
    row[0].metric("Yearly windows", f"{years[0]}-{years[-1]}", border=True,
                  help=f"Outcomes {stats['label_years'][0]}-{stats['label_years'][1]}")
    row[1].metric(f"Graph edges, {graphs['year'].iloc[-1]}", f"{graphs['edges'].iloc[-1]:,}",
                  border=True, help=f"k = {stats['k']} nearest neighbours, symmetrized")
    row[2].metric("Phenotypes", f"{len(stats['phenotypes'])}", border=True,
                  help="Spectral clusters of the 2022 graph (eigengap)")
    row[3].metric("New onsets 2023-2025", f"{positives / observed:.1%}", border=True,
                  help=f"{positives:,} of {observed:,} patients with an observed label had a new "
                       f"onset in one of the {stats['onset_groups']} onset groups")
    with st.expander("Graphs per year"):
        st.dataframe(graphs.rename(columns={
            "year": "Year", "nodes_with_edges": "Patients with neighbours",
            "isolated": "Isolated (no condition yet)", "edges": "Edges",
            "mean_weight": "Mean cosine weight"}), hide_index=True,
            column_config={"Year": st.column_config.NumberColumn(format="%d"),
                           "Mean cosine weight": st.column_config.NumberColumn(format="%.3f")})

    st.subheader("Findings")
    intro, numbered, other = findings_sections(assets.findings)
    st.caption(intro)
    for heading, body in numbered:
        st.markdown(f"#### {heading}")
        st.markdown(body)
    for heading, body in other:
        with st.expander(heading):
            st.markdown(body)


# --- Page 2: new patient ------------------------------------------------------------------


def condition_options(predictor: Predictor) -> dict[str, str]:
    """Feature-vocabulary code -> display name (name, plus the code if names repeat)."""
    names = {code: predictor.names[code] for code in predictor.feature_codes}
    counts = pd.Series(list(names.values())).value_counts()
    labels = {code: name if counts[name] == 1 else f"{name} ({code})" for code, name in names.items()}
    return dict(sorted(labels.items(), key=lambda item: item[1].lower()))


def load_example(predictor: Predictor, name: str, options: dict[str, str]) -> None:
    """Fill the inputs from a configured demo patient (codes outside the options are skipped)."""
    demo = next(d for d in predictor.cfg.inference.demo_patients if d.name == name)
    codes = [str(code) for _, code in demo.history if str(code) in options]
    st.session_state["np_codes"] = list(dict.fromkeys(codes))
    for onset, code in demo.history:
        st.session_state[f"np_onset_{code}"] = int(onset)
        st.session_state.pop(f"np_stop_{code}", None)  # the stop input starts blank
    st.session_state["np_birth"] = demo.birth_date
    st.session_state["np_sex"] = demo.sex
    skipped = len(demo.history) - len(codes)
    st.session_state["np_example_note"] = (
        f"{skipped} entr{'y' if skipped == 1 else 'ies'} of this example cannot be selected here "
        "(a code dropped from the features as redundant, or not in the vocabulary)." if skipped
        else "")


def history_inputs(predictor: Predictor, options: dict[str, str]) -> list[tuple[Any, ...]] | None:
    """One row per selected condition: onset year and optional stop year. None if invalid."""
    cutoff = predictor.cfg.labels.cutoff_date
    first_year = int(predictor.events["date"].min().year)
    last_year = int(predictor.data_end.year)
    codes = st.multiselect("Conditions", list(options), format_func=options.get, key="np_codes",
                           placeholder="Type to search, e.g. hypertension, prediabetes, asthma")
    if not codes:
        return []
    header = st.columns([4, 1.3, 1.3])
    header[0].caption("Condition (default without a stop year)")
    header[1].caption(f"Onset year (before {cutoff.year})")
    header[2].caption("Stop year (blank = default)")
    history, valid = [], True
    for code in codes:
        onset_key, stop_key = f"np_onset_{code}", f"np_stop_{code}"
        st.session_state.setdefault(onset_key, cutoff.year - 2)
        row = st.columns([4, 1.3, 1.3], vertical_alignment="center")
        default = ("chronic: active to the end of the data" if code in predictor.chronic_default
                   else "acute: active in its onset year only")
        row[0].markdown(f"{options[code]}  \n:gray[{default} · SNOMED {code}]")
        onset = row[1].number_input("Onset year", min_value=first_year, max_value=cutoff.year - 1,
                                    step=1, key=onset_key, label_visibility="collapsed")
        stop = row[2].number_input("Stop year", min_value=first_year, max_value=last_year,
                                   value=None, step=1, key=stop_key,
                                   label_visibility="collapsed", placeholder="–")
        if stop is not None and stop < onset:
            row[0].error(f"Stop year {stop} is before the onset {onset}.")
            valid = False
        # Onset: 1 January of the year; stop: the last day of the last active year.
        history.append((int(onset), code) if stop is None
                       else (int(onset), code, f"{int(stop)}-12-31"))
    return history if valid else None


def outcome_frame(predictor: Predictor, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Neighbour outcomes per group, counts over observed labels, next to the population."""
    sizes = predictor.cfg.inference.outcome_neighbours
    rows = []
    for row in outcomes.itertuples():
        is_any = pd.isna(row.group)
        rate = row.positives_all / row.observed_all if row.observed_all else float("nan")
        rows.append({
            "Onset 2023-2025": "Any group (any_onset)" if is_any else group_name(row.group),
            "Patient already has it": "" if is_any else ("yes" if row.patient_had_it else "no"),
            **{f"{n} most similar": format_share(getattr(row, f"positives_{n}"),
                                                 getattr(row, f"observed_{n}")) for n in sizes},
            "All reference patients": f"{row.positives_all:,}/{row.observed_all:,} ({rate:.1%})",
        })
    return pd.DataFrame(rows)


def lift_frame(explanation: pd.DataFrame) -> pd.DataFrame:
    """The explanation (lift) for display."""
    return pd.DataFrame({
        "Condition": explanation["condition"],
        "Neighbours with it": [f"{w}/{n} ({s:.0%})" for w, n, s in zip(
            explanation["neighbours_with"], explanation["neighbours"],
            explanation["neighbourhood_share"])],
        "All reference patients": [f"{s:.1%}" for s in explanation["population_share"]],
        "Lift": explanation["lift"],
    })


def new_patient_page() -> None:
    """Place a new patient: phenotype, similar patients, outcomes, trajectory, explanation."""
    predictor, assets = resources()
    cfg = predictor.cfg
    inf = cfg.inference
    cutoff = cfg.labels.cutoff_date
    st.title("New patient")
    st.warning(DISCLAIMER, icon=":material/warning:")
    st.caption(f"Enter a history of conditions before {cutoff}. The patient is placed among the "
               f"{len(predictor.patient_ids):,} reference patients with the saved transforms, "
               "graphs and weights; nothing is retrained.")

    options = condition_options(predictor)
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    example = left.selectbox("Start from an example profile",
                             [d.name for d in inf.demo_patients], index=None,
                             placeholder="Choose an example (optional)", key="np_example")
    if right.button("Load example", disabled=example is None, key="np_load", width="stretch"):
        load_example(predictor, example, options)
    if st.session_state.get("np_example_note"):
        st.caption(st.session_state["np_example_note"])

    st.session_state.setdefault("np_birth", date(1970, 1, 1))
    st.session_state.setdefault("np_sex", "F")
    left, right = st.columns([1, 1])
    birth = left.date_input("Birth date", min_value=date(1900, 1, 1),
                            max_value=cutoff - timedelta(days=1), key="np_birth")
    sex = right.radio("Sex", ["F", "M"], format_func={"F": "Female", "M": "Male"}.get,
                      horizontal=True, key="np_sex")
    history = history_inputs(predictor, options)
    if history is None:
        return
    if not history:
        st.info("Select at least one condition, or load an example profile.")
        return

    prediction = predict(predictor, tuple(history), birth.isoformat(), sex)
    for note in prediction.notes:
        st.info(note)
    if not prediction.placed:
        return

    st.header("Where this patient lands")
    left, right = st.columns([3, 2])
    with left:
        phenotype_card(predictor, prediction.phenotype.cluster)
    with right:
        st.markdown(f"**{prediction.risk_set}**")
        st.markdown(", ".join(group_name(g) for g in prediction.at_risk_groups) or "none")
        if prediction.prior_groups:
            st.caption("Already has: " + ", ".join(group_name(g) for g in prediction.prior_groups))
        st.caption(RISK_SET_RULE, help=RISK_SET_EXPLANATION)

    st.subheader(f"The {inf.n_similar} most similar reference patients")
    band = inf.age_band_years
    use_band = st.toggle(f"Only patients within ±{band} years of this patient's age "
                         f"({prediction.age:.0f} on {cutoff})", value=False, key="np_band")
    age_band = band if use_band else None
    similar = predictor.similar_patients(prediction, age_band)
    if len(similar) < inf.n_similar:
        st.caption(f"Only {len(similar)} reference patients are in the age band.")
    similar_table(similar)
    st.caption("Cosine similarity in the 2022 spectral embedding (Laplacian eigenvectors), where "
               "the patient is placed by the Nyström extension. The age band only filters the "
               "list, reading further down the same ranking; the graph and embeddings are "
               "unchanged.")

    st.subheader("What happened to similar patients in 2023-2025")
    st.dataframe(outcome_frame(predictor, predictor.neighbour_outcomes(prediction, age_band)),
                 hide_index=True)
    st.caption("New onsets among the most similar reference patients, over those whose label is "
               "observed: a neighbour who already had the group, died before 2023 or was lost to "
               "follow-up does not count for it, so the denominators vary (e.g. 3/9). With 10 "
               "neighbours these are counts, not risks; compare them with the population column.")

    st.subheader("Trajectory")
    nonempty = np.abs(prediction.features).sum(axis=1) > 0
    years = [y for y, keep in zip(prediction.years, nonempty) if keep]
    path = map_positions(prediction.window_states[nonempty], assets.map_states, assets.map_xy,
                         cfg.app.map_neighbours)
    insertions = [ins for ins, keep in zip(prediction.insertions, nonempty) if keep]
    hover = [f"{y}: GraphAE state after {y}<br>{len(ins.neighbours)} graph neighbours, mean "
             f"similarity {ins.weights.mean():.2f}" for y, ins in zip(years, insertions)]
    st.plotly_chart(map_figure(assets, predictor, prediction.phenotype.cluster, path, years,
                               hover, "This patient"), key="np_map")
    skipped = [str(y) for y, keep in zip(prediction.years, nonempty) if not keep]
    st.caption("The TemporalGraphAE's state after each year (the patient inserted into that "
               "year's graph), drawn over one UMAP of every reference patient-year state. Each "
               f"point sits at the mean map position of its {cfg.app.map_neighbours} nearest "
               "reference states; the map is not refitted. Early steps are long partly because the "
               "recurrent state settles as it reads more years (Phase 7b)."
               + (f" No condition active in {', '.join(skipped)}: no point." if skipped else ""))

    st.subheader("Why here: conditions over-represented among the neighbours")
    explanation = predictor.explain(prediction)
    if explanation.empty:
        st.caption("No chronic condition: the placement rests on acute conditions and transitions.")
    else:
        st.dataframe(lift_frame(explanation), hide_index=True, column_config={
            "Lift": st.column_config.NumberColumn(format="%.1f",
                                                  help="Share among the neighbours / share "
                                                       "among all reference patients")})
        st.caption(f"The patient's chronic conditions, by their share among the "
                   f"{inf.explain_neighbours} nearest reference patients over their share among "
                   "all reference patients.")

    auroc, low, high = predictor.any_onset_auroc
    with st.container(border=True):
        st.caption("**Supervised model (secondary)**")
        any_row = prediction.outcomes.set_index("target").loc["any_onset"]
        base = any_row["positives_all"] / any_row["observed_all"]
        if prediction.any_onset_probability is None:
            st.caption("The calibrated probability needs a birth date and sex.")
        else:
            st.caption(f"Calibrated any_onset probability: **{prediction.any_onset_probability:.1%}** "
                       f"(reference base rate {base:.1%}).")
        st.caption(f"Exploratory model, test AUROC {auroc:.2f} ({low:.2f}-{high:.2f}): near "
                   "chance, not for decisions.")


# --- Page 3: patient explorer -------------------------------------------------------------


def timeline_figure(predictor: Predictor, patient_id: str) -> go.Figure:
    """The patient's condition episodes, from onset to stop (or the end of the data)."""
    cfg = predictor.cfg
    m = mode()
    events = predictor.events[predictor.events["patient_id"] == patient_id]
    duration = events["active_until"] - events["date"]
    chronic = events["stop"].isna() | (duration > pd.Timedelta(days=cfg.features.chronic_min_duration_days))
    shortest = events["date"] + pd.Timedelta(days=cfg.app.timeline_min_days)
    frame = pd.DataFrame({
        "Condition": [cut(predictor.names.get(str(c), str(c))) for c in events["code"]],
        "Start": events["date"].dt.tz_convert(None),
        "End": events["active_until"].where(events["active_until"] > shortest, shortest)
                                     .dt.tz_convert(None),
        "Episode": np.where(chronic, "chronic (no stop, or > 1 year)", "acute"),
    })
    order = frame.groupby("Condition")["Start"].min().sort_values().index.tolist()
    fig = px.timeline(frame, x_start="Start", x_end="End", y="Condition", color="Episode",
                      color_discrete_map={"chronic (no stop, or > 1 year)": colour(0),
                                          "acute": colour(1)},
                      category_orders={"Condition": order, "Episode": [
                          "chronic (no stop, or > 1 year)", "acute"]})
    fig.update_traces(marker_line_width=0)
    cutoff = pd.Timestamp(cfg.labels.cutoff_date)
    horizon_end = cutoff + pd.DateOffset(years=cfg.labels.horizon_years)
    first = pd.Timestamp(year=predictor.years[0], month=1, day=1)
    fig.add_vrect(x0=cutoff, x1=horizon_end, fillcolor=CONTEXT[m], opacity=0.25, line_width=0,
                  annotation_text="outcomes 2023-2025", annotation_position="top left")
    fig.add_vline(x=first, line_dash="dot", line_color=TEXT_SECONDARY[m], line_width=1)
    fig.update_xaxes(range=[first - pd.DateOffset(years=3), predictor.data_end.tz_convert(None)],
                     title=None)
    fig.update_yaxes(title=None, autorange="reversed")  # earliest onset at the top
    return style(fig, 120 + 26 * len(order))


def explorer_page() -> None:
    """A reference patient: timeline, phenotype, trajectory and the most similar patients."""
    predictor, assets = resources()
    cfg = predictor.cfg
    inf = cfg.inference
    st.title("Patient explorer")
    st.caption(DISCLAIMER)
    demographics = predictor.demographics
    sexes = np.where(demographics["sex"].to_numpy() == 1.0, "M", "F")
    ages = demographics["age"].to_numpy()

    left, right = st.columns([1, 3])
    clusters = sorted(predictor.phenotypes)
    chosen = left.selectbox("Phenotype", [0, *clusters],
                            format_func=lambda c: "All phenotypes" if c == 0
                            else phenotype_label(predictor, c), key="ex_phenotype")
    nodes = (np.arange(len(predictor.patient_ids)) if chosen == 0
             else np.flatnonzero(predictor.clusters == chosen))
    node = right.selectbox(
        f"Reference patient ({len(nodes):,})", nodes.tolist(), key=f"ex_node_{chosen}",
        format_func=lambda i: (f"#{i} · {predictor.patient_ids[i][:8]} · age {ages[i]:.0f} "
                               f"{sexes[i]} · phenotype {predictor.clusters[i]} · "
                               f"{predictor.chronic[i].nnz} chronic conditions"))
    patient_id = predictor.patient_ids[node]
    cluster = int(predictor.clusters[node])
    labels = dict(zip(predictor.targets, predictor.labels[node]))
    onsets = [group_name(t.removeprefix("onset_")) for t, v in labels.items()
              if t != "any_onset" and v == 1.0]

    row = st.columns(4)
    row[0].metric("Age on 2023-01-01", f"{ages[node]:.0f}", border=True)
    row[1].metric("Sex", {"F": "Female", "M": "Male"}[sexes[node]], border=True)
    row[2].metric("Chronic conditions 2017-2022", f"{predictor.chronic[node].nnz}", border=True)
    any_label = labels["any_onset"]
    row[3].metric("New onsets 2023-2025",
                  "masked" if np.isnan(any_label) else (", ".join(onsets) or "none"),
                  border=True, help="Observed labels; masked if every group was already present, "
                                    "the patient died before 2023 or was lost to follow-up")

    st.subheader("Condition timeline")
    st.plotly_chart(timeline_figure(predictor, patient_id), key="ex_timeline")
    st.caption(f"Each bar is an episode, from onset to stop (open episodes run to the end of the "
               f"data). The dotted line marks the first feature window ({predictor.years[0]}); "
               "the shaded band is the outcome period, never seen by the features. Zoom out for "
               "earlier conditions.")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Phenotype")
        phenotype_card(predictor, cluster)
    with right:
        st.subheader("Trajectory")
        path = assets.coords[node]
        keep = ~np.isnan(path[:, 0])
        years = [y for y, k in zip(predictor.years, keep) if k]
        st.plotly_chart(map_figure(assets, predictor, cluster, path[keep], years,
                                   [f"{y}: GraphAE state after {y}" for y in years],
                                   "This patient"), key="ex_map")
        st.caption("The patient's TemporalGraphAE state after each year on the reference UMAP "
                   "(Phase 7b's map); years without an active condition have no point.")

    st.subheader(f"The {inf.n_similar} most similar reference patients")
    band = inf.age_band_years
    use_band = st.toggle(f"Only patients within ±{band} years of this patient's age",
                         value=False, key="ex_band")
    similar_table(predictor.similar_to_reference(patient_id, band if use_band else None))
    st.caption("Cosine similarity in the 2022 spectral embedding; the patient themselves is "
               "excluded. Shared conditions are chronic conditions both had in 2017-2022.")


# --- Page 4: comorbidity network ----------------------------------------------------------


def cluster_names(predictor: Predictor, nodes: pd.DataFrame) -> dict[int, str]:
    """Cluster -> legend name from its two most common codes."""
    names = {0: "Unclustered (own branch)"}
    for cluster, members in nodes[nodes["cluster"] > 0].groupby("cluster"):
        top = members.sort_values("patients", ascending=False)["code"].head(2)
        names[int(cluster)] = cut(f"Cluster {cluster}: " + ", ".join(predictor.names[c] for c in top)
                                  + f" (+{len(members) - 2})" * (len(members) > 2), 70)
    return names


def network_figure(predictor: Predictor, nodes: pd.DataFrame, edges: pd.DataFrame,
                   weight: str) -> go.Figure:
    """Arc diagram: codes in the Phase 3 dendrogram order, pairs as arcs, width by ``weight``."""
    m = mode()
    position = dict(zip(nodes["code"], nodes["position"]))
    cluster = dict(zip(nodes["code"], nodes["cluster"]))
    fig = go.Figure()
    t = np.linspace(0.0, np.pi, ARC_POINTS)
    if len(edges):
        tertile = pd.qcut(edges[weight].rank(method="first"), q=min(3, len(edges)), labels=False)
        groups: dict[tuple[int, int], list[np.ndarray]] = {}
        apex_x, apex_y, apex_text = [], [], []
        for (row, width) in zip(edges.itertuples(), tertile):
            ya, yb = -position[row.code_a], -position[row.code_b]
            radius, centre = abs(ya - yb) / 2, (ya + yb) / 2
            same = cluster[row.code_a] if cluster[row.code_a] == cluster[row.code_b] else 0
            groups.setdefault((same, int(width)), []).append(
                np.column_stack([radius * np.sin(t), centre + radius * np.cos(t)]))
            apex_x.append(radius)
            apex_y.append(centre)
            apex_text.append(
                f"{predictor.names[row.code_a]} + {predictor.names[row.code_b]}<br>"
                f"{row.count} patients together (expected {row.expected:.1f})<br>"
                f"lift {row.lift:.1f} · Jaccard {row.jaccard:.2f}")
        for (same, width), arcs in sorted(groups.items()):
            xy = np.vstack([np.vstack([arc, [np.nan, np.nan]]) for arc in arcs])
            line_colour = rgba(cluster_colour(same), 0.7) if same else \
                rgba(TEXT_SECONDARY[m], 0.45)
            fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="lines", hoverinfo="skip",
                                     showlegend=False,
                                     line={"color": line_colour, "width": EDGE_WIDTHS[width]}))
        fig.add_trace(go.Scatter(x=apex_x, y=apex_y, mode="markers", text=apex_text,
                                 hovertemplate="%{text}<extra></extra>", showlegend=False,
                                 marker={"size": MARKER_SIZE, "opacity": 0}))
    degree = pd.concat([edges["code_a"], edges["code_b"]]).value_counts()
    names = cluster_names(predictor, nodes)
    largest = nodes["patients"].max()
    for value, members in nodes.groupby("cluster"):
        fig.add_trace(go.Scatter(
            x=np.zeros(len(members)), y=-members["position"], mode="markers", name=names[value],
            text=[f"{predictor.names[c]}<br>{p:,} patients with it chronic<br>"
                  f"{int(degree.get(c, 0))} pairs shown" for c, p in
                  zip(members["code"], members["patients"])],
            hovertemplate="%{text}<extra></extra>",
            marker={"size": MARKER_SIZE + 10 * np.sqrt(members["patients"] / largest),
                    "color": cluster_colour(int(value)),
                    "line": {"color": SURFACE[m], "width": LINE_WIDTH}}))
    ordered = nodes.sort_values("position")
    fig.update_yaxes(tickvals=-ordered["position"],
                     ticktext=[cut(predictor.names[c]) for c in ordered["code"]],
                     showgrid=False, zeroline=False, range=[-len(nodes) + 0.3, 0.7])
    fig.update_xaxes(visible=False, range=[-0.6, len(nodes) / 2 + 0.6])
    return style(fig, 22 * len(nodes) + 90)


def network_page() -> None:
    """The Phase 3 chronic comorbidity pairs as an interactive network."""
    predictor, assets = resources()
    cfg = predictor.cfg
    st.title("Comorbidity network")
    st.caption(DISCLAIMER)
    nodes, pairs = assets.nodes, assets.pairs
    filters = st.columns([2, 2, 1.4])
    max_lift = float(np.ceil(pairs["lift"].max()))
    min_lift = filters[0].slider("Minimum lift", 1.0, max_lift, float(cfg.app.network_min_lift),
                                 step=0.5, key="net_lift",
                                 help="Patients with both / patients expected with both if "
                                      "independent")
    min_patients = filters[1].slider("Minimum patients with the pair", 1, int(pairs["count"].max()),
                                     int(cfg.app.network_min_patients), key="net_patients")
    weight = filters[2].radio("Arc width", ["lift", "jaccard"], horizontal=True, key="net_weight",
                              format_func={"lift": "Lift", "jaccard": "Jaccard"}.get)
    edges = pairs[(pairs["lift"] >= min_lift) & (pairs["count"] >= min_patients)]
    shown = pd.concat([edges["code_a"], edges["code_b"]]).nunique()
    st.markdown(f"**{len(edges):,} pairs** among **{shown} of {len(nodes)}** chronic conditions")
    st.plotly_chart(network_figure(predictor, nodes, edges, weight), key="net_figure")
    st.caption(
        f"Chronic conditions (no stop, or longer than {cfg.features.chronic_min_duration_days} "
        f"days) in at least {cfg.summary.chronic_min_patients} of the "
        f"{len(predictor.patient_ids):,} reference patients, counted in 2017-2022 (Phase 3). "
        "Lift = patients with both / patients expected with both if independent; Jaccard = "
        "patients with both / patients with either. Codes are listed in the order of the Phase 3 "
        "heatmap's clustering (average linkage of their clipped log2-lift profiles) and coloured "
        f"by its clusters (the dendrogram cut into {cfg.app.network_clusters} branches; "
        "single-code branches are grey). Arcs within a cluster take its colour; arc width shows "
        "the tertile of the chosen measure among the pairs shown. Hover an arc's apex for its "
        "numbers.")
    st.dataframe(pd.DataFrame({
        "Condition A": [predictor.names[c] for c in edges["code_a"]],
        "Condition B": [predictor.names[c] for c in edges["code_b"]],
        "Patients A": edges["frequency_a"], "Patients B": edges["frequency_b"],
        "Together": edges["count"], "Expected": edges["expected"],
        "Lift": edges["lift"], "Jaccard": edges["jaccard"],
    }), hide_index=True, column_config={
        "Expected": st.column_config.NumberColumn(format="%.1f"),
        "Lift": st.column_config.NumberColumn(format="%.1f"),
        "Jaccard": st.column_config.NumberColumn(format="%.2f")})


# --- Page 5: similarity graph over time ----------------------------------------------------


def graph_figure(predictor: Predictor, assets: Assets, sample: np.ndarray, t: int,
                 positions: str, edges_shown: str, highlight: int) -> tuple[go.Figure, dict[str, int]]:
    """The sampled subgraph of window ``t``, nodes coloured by phenotype."""
    m = mode()
    year = predictor.years[t]
    W = assets.weights[t]
    xy = assets.coords[sample, -1] if positions == "fixed" else assets.coords[sample, t]
    upper = sp.triu(W[sample][:, sample], k=1).tocsr()
    present = (upper != 0).astype(np.int8)
    if t > 0:
        previous = (sp.triu(assets.weights[t - 1][sample][:, sample], k=1) != 0).astype(np.int8)
        new = sp.csr_matrix(present - present.multiply(previous))
    else:
        new = present
    new.eliminate_zeros()
    drawn = (new if edges_shown == "new" else present).tocoo()
    keep = ~np.isnan(xy[drawn.row, 0]) & ~np.isnan(xy[drawn.col, 0])
    segments = np.full((3 * int(keep.sum()), 2), np.nan)
    segments[0::3] = xy[drawn.row[keep]]
    segments[1::3] = xy[drawn.col[keep]]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=segments[:, 0], y=segments[:, 1], mode="lines", hoverinfo="skip",
                               name="new since the previous year" if edges_shown == "new"
                               else f"edges in {year}",
                               line={"color": rgba(TEXT_SECONDARY[m], 0.35), "width": 1}))
    degree = np.diff(W.indptr)[sample]
    demographics = predictor.demographics
    for cluster in sorted(predictor.phenotypes):
        members = np.flatnonzero((predictor.clusters[sample] == cluster) & ~np.isnan(xy[:, 0]))
        nodes = sample[members]
        dim = highlight and cluster != highlight
        isolated = degree[members] == 0
        fig.add_trace(go.Scattergl(
            x=xy[members, 0], y=xy[members, 1], mode="markers", name=phenotype_label(predictor, cluster),
            text=[f"{predictor.patient_ids[n][:8]} · age {demographics['age'].iloc[n]:.0f} · "
                  f"phenotype {cluster}<br>{d} neighbours in {year}" for n, d in
                  zip(nodes, degree[members])],
            hovertemplate="%{text}<extra></extra>",
            marker={"size": MARKER_SIZE - 2,
                    "color": CONTEXT[m] if dim else cluster_colour(cluster),
                    "symbol": np.where(isolated, "circle-open", "circle"),
                    "line": {"color": SURFACE[m], "width": 1}}))
    counts = {"patients": len(sample), "placed": int((~np.isnan(xy[:, 0])).sum()),
              "edges": int(present.nnz), "new": int(new.nnz), "isolated": int((degree == 0).sum())}
    return hide_axes(style(fig, 620)), counts


def drift_figure(drift: pd.DataFrame, year: int) -> go.Figure:
    """Neighbourhood Jaccard between consecutive years: mean, median and interquartile range."""
    m = mode()
    labels = [f"{a}→{b}" for a, b in zip(drift["from_year"], drift["to_year"])]
    blue = colour(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels + labels[::-1],
                             y=list(drift["q75"]) + list(drift["q25"])[::-1], fill="toself",
                             fillcolor=rgba(blue, 0.18), line={"width": 0}, hoverinfo="skip",
                             name="Interquartile range"))
    fig.add_trace(go.Scatter(x=labels, y=drift["median"], mode="lines+markers", name="Median",
                             line={"color": blue, "width": LINE_WIDTH, "dash": "dot"},
                             marker={"size": MARKER_SIZE - 2, "color": blue},
                             hovertemplate="%{x}: median %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=drift["mean"], mode="lines+markers", name="Mean",
                             line={"color": blue, "width": LINE_WIDTH},
                             marker={"size": MARKER_SIZE, "color": blue,
                                     "line": {"color": SURFACE[m], "width": LINE_WIDTH}},
                             customdata=drift[["patients", "unchanged", "replaced"]],
                             hovertemplate="%{x}: mean %{y:.3f}<br>%{customdata[0]:,} patients; "
                                           "unchanged %{customdata[1]:.1%}, replaced "
                                           "%{customdata[2]:.1%}<extra></extra>"))
    selected = [label for label, to in zip(labels, drift["to_year"]) if to == year]
    if selected:
        fig.add_vline(x=labels.index(selected[0]), line_color=TEXT_SECONDARY[m], line_width=1,
                      line_dash="dash")
    fig.update_yaxes(range=[0, 1], title="Neighbourhood Jaccard")
    fig.update_xaxes(title=None)
    return style(fig, 360)


def graph_page() -> None:
    """A sampled subgraph per year, coloured by phenotype, and the drift between years."""
    predictor, assets = resources()
    cfg = predictor.cfg
    st.title("Similarity graph over time")
    st.caption(DISCLAIMER)
    years = list(predictor.years)
    controls = st.columns([3, 2, 2, 2])
    year = controls[0].select_slider("Year", years, value=years[-1], key="g_year")
    positions = controls[1].radio("Node positions", ["fixed", "year"], key="g_positions",
                                  format_func={"fixed": f"fixed ({years[-1]} state)",
                                               "year": "that year's state"}.get)
    edges_shown = controls[2].radio("Edges", ["all", "new"], key="g_edges",
                                    format_func={"all": "all of the year",
                                                 "new": "new since the year before"}.get)
    highlight = controls[3].selectbox("Highlight", [0, *sorted(predictor.phenotypes)],
                                      format_func=lambda c: "all phenotypes" if c == 0
                                      else f"phenotype {c}", key="g_highlight")
    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.app.graph_sample_patients, len(predictor.patient_ids))
    sample = np.sort(rng.choice(len(predictor.patient_ids), size=n, replace=False))
    t = years.index(year)
    fig, counts = graph_figure(predictor, assets, sample, t, positions, edges_shown, highlight)

    full = assets.stats["graphs"][t]
    row = st.columns(4)
    row[0].metric(f"Patients with neighbours, {year}", f"{full['nodes_with_edges']:,}",
                  border=True, help="Whole graph")
    row[1].metric("Isolated (no condition yet)", f"{full['isolated']:,}", border=True)
    row[2].metric("Edges (whole graph)", f"{full['edges']:,}", border=True)
    drift = assets.drift.set_index("to_year")
    row[3].metric("Neighbours kept from the year before",
                  f"{drift.loc[year, 'mean']:.2f}" if year in drift.index else "–", border=True,
                  help="Mean neighbourhood Jaccard with the previous year (1 = same neighbours)")
    st.plotly_chart(fig, key="g_figure")
    change = (f"{counts['new']:,} of them new since {year - 1}" if t
              else "the first year, so every edge is new")
    st.caption(
        f"{counts['patients']} reference patients drawn once (seed {cfg.seed}); "
        f"{counts['edges']:,} edges among them in {year}, {change}. Positions are the patients' "
        "TemporalGraphAE "
        "states on the reference UMAP" + (f" after {years[-1]}, so only the edges change." if
                                          positions == "fixed" else
                                          f" after {year}; patients with no condition that year "
                                          "have no position.")
        + " Open circles: no neighbour in that year. Colour: phenotype (2022 spectral clusters).")

    st.subheader("Graph drift")
    st.plotly_chart(drift_figure(assets.drift, year), key="g_drift")
    st.caption("For each patient with neighbours in both years: the share of neighbours kept, "
               "|N_t ∩ N_t+1| / |N_t ∪ N_t+1| (Phase 4). The graphs change most early, when "
               "patients still gain their first conditions, and stabilise year by year.")


# --- Page 6: embeddings and results --------------------------------------------------------


def forest_figure(labels: list[str], values: np.ndarray, lows: np.ndarray, highs: np.ndarray,
                  references: list[tuple[str, float, float, float]], title: str) -> go.Figure:
    """Point estimates with intervals, one per row, and reference bands."""
    m = mode()
    fig = go.Figure()
    for i, (name, value, low, high) in enumerate(references):
        fig.add_vrect(x0=low, x1=high, fillcolor=CONTEXT[m], opacity=0.3, line_width=0)
        fig.add_vline(x=value, line_color=TEXT_SECONDARY[m], line_width=1, line_dash="dash",
                      annotation_text=name, annotation_position="top right" if i % 2 else "top left",
                      annotation_font_color=TEXT_SECONDARY[m])
    fig.add_trace(go.Scatter(
        x=values, y=labels, mode="markers", showlegend=False,
        error_x={"type": "data", "symmetric": False, "array": highs - values,
                 "arrayminus": values - lows, "color": colour(0), "thickness": LINE_WIDTH},
        marker={"size": MARKER_SIZE, "color": colour(0),
                "line": {"color": SURFACE[m], "width": LINE_WIDTH}},
        customdata=np.column_stack([lows, highs]),
        hovertemplate="%{y}: %{x:.3f} (%{customdata[0]:.3f}-%{customdata[1]:.3f})<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title=title)
    return style(fig, 60 + 34 * len(labels), legend=False)


def embeddings_tab(assets: Assets) -> None:
    """Phase 7b: the embedding comparison table and its two forest plots."""
    table = assets.embeddings
    stats = assets.stats
    base = stats["next_year_base_rate"]
    refs = stats["concordance_references"]
    labels = [f"{n}. {name}" for n, name in zip(table["number"], table["name"])]
    st.dataframe(pd.DataFrame({
        "Embedding (32-d)": labels,
        "Graph": table["graph"],
        "Trained by": table["signal"],
        "Neighbour concordance": [ci(v, lo, hi) for v, lo, hi in
                                  zip(table["concordance"], table["concordance_low"],
                                      table["concordance_high"])],
        "PC1 share": table["pc1_share"],
        "Next-year micro AUROC": [ci(v, lo, hi) for v, lo, hi in
                                  zip(table["next_micro_auroc"], table["next_micro_low"],
                                      table["next_micro_high"])],
        "Minus base rate": table["next_micro_auroc"] - base["auroc"],
        "Next-year macro AUROC": [ci(v, lo, hi) for v, lo, hi in
                                  zip(table["next_macro_auroc"], table["next_macro_low"],
                                      table["next_macro_high"])],
    }), hide_index=True, column_config={
        "PC1 share": st.column_config.NumberColumn(format="percent",
                                                   help="Share of variance on the first principal "
                                                        "component (99.8% = collapsed)"),
        "Minus base rate": st.column_config.NumberColumn(
            format="%+.3f", help=f"Next-year micro AUROC minus the codes' base rates alone "
                                 f"({base['auroc']:.3f}), point estimates")})
    st.caption(
        f"Test-split patients, 95% bootstrap intervals. **Concordance**: mean Jaccard of chronic "
        f"conditions with the 5 nearest neighbours (cosine); random patients "
        f"{ci(refs['random']['concordance'], refs['random']['low'], refs['random']['high'])}, the "
        f"2022 input features {ci(refs['features']['concordance'], refs['features']['low'], refs['features']['high'])}. "
        f"**Next year**: new conditions in {base['year']} from the {base['year'] - 1} embedding, "
        f"{base['codes']} codes; the codes' base rates alone reach a micro AUROC of "
        f"{ci(base['auroc'], base['low'], base['high'])}, so micro AUROC mostly rewards knowing "
        "which codes are common; macro AUROC ranks patients within each code.")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(forest_figure(
            labels, table["concordance"].to_numpy(), table["concordance_low"].to_numpy(),
            table["concordance_high"].to_numpy(),
            [("random patients", refs["random"]["concordance"], refs["random"]["low"],
              refs["random"]["high"]),
             ("input features", refs["features"]["concordance"], refs["features"]["low"],
              refs["features"]["high"])],
            "Neighbour concordance (higher = similar patients share conditions)"), key="emb_conc")
    with right:
        st.plotly_chart(forest_figure(
            labels, table["next_micro_auroc"].to_numpy(), table["next_micro_low"].to_numpy(),
            table["next_micro_high"].to_numpy(),
            [("base rates alone", base["auroc"], base["low"], base["high"])],
            "Next-year micro AUROC"), key="emb_next")


def mark(low: float, high: float) -> str:
    """"shown" when an interval excludes 0, else "not shown"."""
    return "shown" if low > 0 or high < 0 else "not shown (CI includes 0)"


def ablations_tab(assets: Assets) -> None:
    """Phase 7 (forecasting) and Phase 7b (embedding) ablations, as paired differences."""
    results = assets.results.set_index("key")
    full = results.loc["tgcn"]
    st.markdown("**Forecasting `any_onset` (Phase 7, 5-fold cross-validation, pooled out-of-fold "
                "AUROC).** Full TemporalGCN minus the other model; paired bootstrap.")
    questions = [("Does the graph help?", "temporal_no_graph"), ("Does time help?", "gcn_static"),
                 ("Does the real similarity structure matter?", "tgcn_rewired"),
                 ("Better than the best non-GCN model?", "lr_last"),
                 ("Better than age and sex alone?", "age_sex")]
    st.dataframe(pd.DataFrame([{
        "Question": question, "Compared with": results.loc[key, "model"],
        "Full AUROC": full["oof_auroc"], "Other AUROC": results.loc[key, "oof_auroc"],
        "Full minus other": ci(results.loc[key, "delta_auroc"], results.loc[key, "delta_auroc_low"],
                               results.loc[key, "delta_auroc_high"], signed=True),
        "Reading": mark(results.loc[key, "delta_auroc_low"], results.loc[key, "delta_auroc_high"]),
    } for question, key in questions]), hide_index=True, column_config={
        "Full AUROC": st.column_config.NumberColumn(format="%.3f"),
        "Other AUROC": st.column_config.NumberColumn(format="%.3f")})

    table = assets.embeddings.set_index("key")
    st.markdown("**Embeddings (Phase 7b, test-split patients).** TemporalGraphAE (i) minus the "
                "other embedding; positive = the GraphAE is better.")
    questions = [("Does the graph help?", "graph_ae_no_graph"),
                 ("Is the real graph better than a random one?", "graph_ae_rewired"),
                 ("Does the temporal (next-window) objective help?", "graph_ae_link_only"),
                 ("Does it beat PCA of the final window?", "pca_final"),
                 ("Does it beat PCA of every window?", "pca_windows"),
                 ("Does it beat the spectral embedding?", "spectral")]
    rows = []
    for question, key in questions:
        r = table.loc[key]
        rows.append({
            "Question": question, "Compared with": f"{r['number']}. {r['name']}",
            "Neighbour concordance": ci(r["concordance_diff"], r["concordance_diff_low"],
                                       r["concordance_diff_high"], signed=True),
            "Concordance reading": mark(r["concordance_diff_low"], r["concordance_diff_high"]),
            "Next-year micro AUROC": ci(r["next_micro_diff"], r["next_micro_diff_low"],
                                       r["next_micro_diff_high"], signed=True),
            "AUROC reading": mark(r["next_micro_diff_low"], r["next_micro_diff_high"]),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("A difference is shown only if its 95% interval excludes 0. Rewiring the graphs "
               "(same degrees, random partners) collapses concordance to near chance: who is "
               "linked to whom carries the structure. No ablation changes the forecasting AUROC.")


def roc_figure(curves: list[tuple[str, np.ndarray, np.ndarray]]) -> go.Figure:
    """ROC curves with the chance diagonal."""
    m = mode()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                             line={"color": TEXT_SECONDARY[m], "width": 1, "dash": "dash"},
                             hoverinfo="skip"))
    for slot, (name, y, score) in enumerate(curves):
        fpr, tpr, _ = roc_curve(y, score)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name,
                                 line={"color": colour(slot), "width": LINE_WIDTH},
                                 hovertemplate="FPR %{x:.2f}, TPR %{y:.2f}<extra>" + name
                                               + "</extra>"))
    fig.update_xaxes(range=[0, 1], title="False positive rate", constrain="domain")
    fig.update_yaxes(range=[0, 1], title="True positive rate", scaleanchor="x", scaleratio=1)
    return style(fig, 460)


def performance_tab(predictor: Predictor, assets: Assets) -> None:
    """Phase 7: the cross-validation table, ROC curves and the held-out test result."""
    results = assets.results.sort_values("oof_auroc", ascending=False)
    st.markdown(f"**5-fold cross-validation**, {int(results['oof_observed'].iloc[0]):,} patients "
                f"with an observed label, {int(results['oof_positives'].iloc[0]):,} positives.")
    st.dataframe(pd.DataFrame({
        "#": results["number"], "Model": results["model"], "Inputs": results["inputs"],
        "Pooled OOF AUROC": [ci(v, lo, hi) for v, lo, hi in
                             zip(results["oof_auroc"], results["oof_auroc_low"],
                                 results["oof_auroc_high"])],
        "Pooled OOF AUPRC": [ci(v, lo, hi) for v, lo, hi in
                             zip(results["oof_auprc"], results["oof_auprc_low"],
                                 results["oof_auprc_high"])],
        "Full minus this": [("reference" if np.isnan(d) else ci(d, lo, hi, signed=True))
                            for d, lo, hi in zip(results["delta_auroc"],
                                                 results["delta_auroc_low"],
                                                 results["delta_auroc_high"])],
        "Brier": results["oof_brier"],
    }), hide_index=True, column_config={"Brier": st.column_config.NumberColumn(format="%.3f")})

    test = assets.test
    y, p = test["any_onset"].to_numpy(), test["probability"].to_numpy()
    auroc, low, high = predictor.any_onset_auroc
    rate = y.mean()
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Held-out test set (single shot)**")
        st.dataframe(pd.DataFrame({
            "Metric": ["Patients (positives)", "AUROC", "Brier score", "Brier, always the base rate"],
            "Value": [f"{len(y)} ({int(y.sum())})", ci(auroc, low, high),
                      f"{np.mean((p - y) ** 2):.3f}", f"{rate * (1 - rate):.3f}"],
        }), hide_index=True)
        st.caption("The final TemporalGCN (trained on the training split, Platt-calibrated on "
                   "validation), scored once. The AUROC interval includes 0.5.")
    with right:
        oof = assets.oof[assets.oof["any_onset"].notna()]
        names = assets.results.set_index("key")["model"]
        curves = [(f"{names[key]}, out of fold", oof["any_onset"].to_numpy(), oof[key].to_numpy())
                  for key in ("tgcn", "lr_last", "age_sex")]
        st.plotly_chart(roc_figure(curves), key="res_roc")
        st.caption("Out-of-fold ROC curves of the full TemporalGCN, the best logistic regression "
                   "and age and sex alone: they nearly coincide.")


def figures_tab(predictor: Predictor) -> None:
    """The key report figures."""
    paths = predictor.cfg.paths
    figures = [
        (paths.cv_auroc_fig, "Pooled out-of-fold AUROC of every model (Phase 7)"),
        (paths.next_year_auroc_fig, "Next-year AUROC of every embedding vs base rates (Phase 7b)"),
        (paths.ae_trajectory_fig, "GraphAE trajectories of five patients (Phase 7b)"),
        (paths.ae_umap_phenotype_fig, "GraphAE embeddings by phenotype (Phase 7b)"),
        (paths.eigenvalues_fig, "Smallest Laplacian eigenvalues of the 2022 graph (Phase 4)"),
        (paths.comorbidity_heatmap_fig, "Chronic comorbidity lift (Phase 3)"),
        (paths.test_reliability_fig, "Reliability of the calibrated test probabilities (Phase 7)"),
        (paths.training_curves_fig, "Training curves of the TemporalGCN (Phase 6)"),
    ]
    available = [(path, caption) for path, caption in figures if path.exists()]
    for start in range(0, len(available), 2):
        columns = st.columns(2)
        for column, (path, caption) in zip(columns, available[start:start + 2]):
            column.image(str(path), caption=caption, width="stretch")


def results_page() -> None:
    """Embedding comparison, ablations, cross-validation, test result and key figures."""
    predictor, assets = resources()
    st.title("Embeddings and results")
    st.caption(DISCLAIMER)
    results = assets.results.set_index("key")
    full, age_sex = results.loc["tgcn"], results.loc["age_sex"]
    auroc, low, high = predictor.any_onset_auroc
    base = assets.stats["next_year_base_rate"]
    ae = assets.embeddings.set_index("key").loc["graph_ae"]
    st.warning(
        "**Forecasting new onsets is near chance on this data.** Cross-validated AUROC of the "
        f"full TemporalGCN {ci(full['oof_auroc'], full['oof_auroc_low'], full['oof_auroc_high'])}; "
        f"all {len(results)} models lie between {results['oof_auroc'].min():.3f} and "
        f"{results['oof_auroc'].max():.3f}, and age and sex alone reach {age_sex['oof_auroc']:.3f}. "
        f"Held-out test AUROC {ci(auroc, low, high)}, an interval that includes 0.5. Next year, the "
        f"GraphAE's micro AUROC ({ae['next_micro_auroc']:.3f}) matches the codes' base rates alone "
        f"({base['auroc']:.3f}). What the graphs do carry is similarity: see the embedding table.",
        icon=":material/info:")
    tabs = st.tabs(["Embeddings", "Ablations", "Cross-validation and test", "Figures"])
    with tabs[0]:
        embeddings_tab(assets)
    with tabs[1]:
        ablations_tab(assets)
    with tabs[2]:
        performance_tab(predictor, assets)
    with tabs[3]:
        figures_tab(predictor)


# --- App -------------------------------------------------------------------------------------


def main() -> None:
    """Configure the page, the navigation and the permanent disclaimer."""
    st.set_page_config(page_title="EHR comorbidity trajectories", page_icon=":material/hub:",
                       layout="wide")
    pages = [
        st.Page(overview_page, title="Overview", icon=":material/home:", default=True),
        st.Page(new_patient_page, title="New patient", icon=":material/person_add:"),
        st.Page(explorer_page, title="Patient explorer", icon=":material/person_search:"),
        st.Page(network_page, title="Comorbidity network", icon=":material/hub:"),
        st.Page(graph_page, title="Similarity graph over time", icon=":material/timeline:"),
        st.Page(results_page, title="Embeddings and results", icon=":material/analytics:"),
    ]
    navigation = st.navigation(pages)
    with st.sidebar:
        st.warning(DISCLAIMER, icon=":material/warning:")
        st.caption("Comorbidity trajectories, patient-similarity graphs and their embeddings. "
                   "Every number comes from the saved artifacts; nothing is trained here.")
    navigation.run()


if __name__ == "__main__":  # streamlit run executes the script as __main__
    main()
