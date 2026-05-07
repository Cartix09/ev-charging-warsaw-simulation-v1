"""Aggregate scenario results, write CSVs, and render figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .simulation import SimulationResult


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_scenario_outputs(
    results: Iterable[SimulationResult],
    tables_dir: Path = config.TABLES_DIR,
) -> Dict[str, Path]:
    """Write the three required CSVs aggregated across scenarios."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict] = []
    agent_frames: List[pd.DataFrame] = []
    station_frames: List[pd.DataFrame] = []

    for r in results:
        summary_rows.append(r.summary)
        a = r.agents_df.copy()
        a["scenario"] = r.scenario
        agent_frames.append(a)
        s = r.stations_df.copy()
        s["scenario"] = r.scenario
        station_frames.append(s)

    summary_df = pd.DataFrame(summary_rows)
    agents_df = (
        pd.concat(agent_frames, ignore_index=True)
        if agent_frames
        else pd.DataFrame()
    )
    stations_df = (
        pd.concat(station_frames, ignore_index=True)
        if station_frames
        else pd.DataFrame()
    )

    paths = {
        "scenario_summary": tables_dir / "scenario_summary.csv",
        "agent_results": tables_dir / "agent_results.csv",
        "station_results": tables_dir / "station_results.csv",
    }
    summary_df.to_csv(paths["scenario_summary"], index=False)
    agents_df.to_csv(paths["agent_results"], index=False)
    stations_df.to_csv(paths["station_results"], index=False)
    return paths


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_queue_over_time(
    results: Iterable[SimulationResult],
    figure_path: Path = config.FIGURES_DIR / "queue_over_time.png",
) -> Path:
    """Total queue length across all stations, per minute, one line per scenario."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in results:
        if r.queue_timeseries.empty:
            continue
        total = r.queue_timeseries.sum(axis=1)
        ax.plot(total.index, total.values, label=r.scenario)
    ax.set_xlabel("simulation minute")
    ax.set_ylabel("total queued vehicles (sum over stations)")
    ax.set_title("Queue length over time, by scenario")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def plot_waiting_time_comparison(
    results: Iterable[SimulationResult],
    figure_path: Path = config.FIGURES_DIR / "scenario_comparison_waiting_time.png",
) -> Path:
    """Box plot of per-agent waiting time, one box per scenario."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    data, labels = [], []
    for r in results:
        if r.agents_df.empty:
            continue
        data.append(r.agents_df["waiting_time_min"].values)
        labels.append(r.scenario)

    fig, ax = plt.subplots(figsize=(8, 5))
    if data:
        # `labels=` works in all matplotlib versions; `tick_labels` only in 3.9+.
        ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("waiting time (minutes)")
    ax.set_title("Per-agent waiting time at chargers")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def plot_station_utilisation(
    results: Iterable[SimulationResult],
    figure_path: Path = config.FIGURES_DIR / "station_utilisation.png",
) -> Path:
    """Per-station utilisation distribution, one strip per scenario."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    results = list(results)
    labels = [r.scenario for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results):
        if r.stations_df.empty:
            continue
        ax.scatter(
            [i] * len(r.stations_df),
            r.stations_df["utilisation"].values,
            alpha=0.6,
            label=r.scenario,
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("scenario")
    ax.set_ylabel("port utilisation (fraction of horizon)")
    ax.set_title("Station port utilisation by scenario")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def plot_waited_only_boxplot(
    results: Iterable[SimulationResult],
    figure_path: Path = config.FIGURES_DIR / "waiting_time_waiters_only.png",
) -> Path:
    """Box plot of waiting time, restricted to agents who actually waited.

    The full-population boxplot is dominated by zeros and hides between-scenario
    differences; this view zooms in on agents whose waiting_time_min > 0.
    """
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    data, labels, counts = [], [], []
    for r in results:
        if r.agents_df.empty:
            continue
        waited = r.agents_df.loc[
            r.agents_df["waiting_time_min"] > 0, "waiting_time_min"
        ]
        data.append(waited.values)
        labels.append(f"{r.scenario}\n(n={len(waited)})")
        counts.append(len(waited))

    fig, ax = plt.subplots(figsize=(8, 5))
    if any(len(d) > 0 for d in data):
        ax.boxplot(data, labels=labels, showmeans=True)
    else:
        ax.text(
            0.5, 0.5, "no agents waited in any scenario",
            ha="center", va="center", transform=ax.transAxes,
        )
    ax.set_ylabel("waiting time (minutes)")
    ax.set_title("Waiting time — agents who actually waited")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def plot_detour_distribution(
    results: Iterable[SimulationResult],
    figure_path: Path = config.FIGURES_DIR / "detour_distribution.png",
) -> Path:
    """Histogram of per-agent detour distance (km), one series per scenario.

    Only agents that actually detoured (detour_distance_m > 0) are shown,
    so the histogram reflects realised detours, not the long zero-spike.
    """
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for r in results:
        if r.agents_df.empty:
            continue
        det_km = r.agents_df["detour_distance_m"] / 1000.0
        det_km = det_km[det_km > 0]
        if len(det_km) == 0:
            continue
        ax.hist(det_km, bins=30, alpha=0.5, label=f"{r.scenario} (n={len(det_km)})")
        plotted = True
    ax.set_xlabel("detour distance (km)")
    ax.set_ylabel("agents")
    ax.set_title("Detour distance distribution by scenario")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def plot_utilisation_histogram(
    results: Iterable[SimulationResult],
    figure_path: Path = config.FIGURES_DIR / "utilisation_histogram.png",
) -> Path:
    """Histogram of per-station utilisation, one series per scenario."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    bins = np.linspace(0, 1, 21)
    for r in results:
        if r.stations_df.empty:
            continue
        u = r.stations_df["utilisation"].values
        ax.hist(u, bins=bins, alpha=0.5, label=f"{r.scenario}")
        plotted = True
    ax.set_xlabel("station port utilisation (fraction of horizon)")
    ax.set_ylabel("stations")
    ax.set_title("Station utilisation distribution by scenario")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def plot_charging_events_bar(
    results: Iterable[SimulationResult],
    figure_path: Path = config.FIGURES_DIR / "charging_events_by_scenario.png",
) -> Path:
    """Side-by-side bars: started vs. completed charging events per scenario."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    results = list(results)
    labels = [r.scenario for r in results]
    starts = [int(r.summary.get("started_charging_events", 0)) for r in results]
    completes = [int(r.summary.get("completed_charging_events", 0)) for r in results]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, starts, width, label="started", color="#1f77b4")
    ax.bar(x + width / 2, completes, width, label="completed", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("events")
    ax.set_title("Charging events by scenario (started vs. completed)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    for i, (s, c) in enumerate(zip(starts, completes)):
        ax.text(i - width / 2, s, str(s), ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, c, str(c), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


# ---------------------------------------------------------------------------
# Objective score (decision-support summary)
# ---------------------------------------------------------------------------

OBJECTIVE_COMPONENTS = {
    # human label                  scenario_summary key
    "W (waiting pressure)":        "mean_waiting_time_among_waiters_min",
    "D (detour distance)":         "mean_detour_distance_m",
    "Q (queue pressure)":          "total_queue_minutes",
    "Uimb (util. imbalance)":      "utilisation_imbalance_sd",
}


def compute_objective_scores(
    results: Iterable[SimulationResult],
    weights: config.ObjectiveWeights = config.OBJECTIVE_WEIGHTS,
) -> pd.DataFrame:
    """Compute the normalised objective score per scenario.

        J_norm = α·W_norm + β·D_norm + γ·Q_norm + δ·Uimb_norm

    Each component is min-max normalised across the scenarios in `results`.
    Lower score = better. Returns a DataFrame with raw + normalised
    components, the score, and a rank (1 = best).
    """
    rows = []
    for r in results:
        s = r.summary or {}
        rows.append(
            {
                "scenario": r.scenario,
                "W_raw": float(s.get(OBJECTIVE_COMPONENTS["W (waiting pressure)"], 0.0)),
                "D_raw": float(s.get(OBJECTIVE_COMPONENTS["D (detour distance)"], 0.0)),
                "Q_raw": float(s.get(OBJECTIVE_COMPONENTS["Q (queue pressure)"], 0.0)),
                "Uimb_raw": float(s.get(OBJECTIVE_COMPONENTS["Uimb (util. imbalance)"], 0.0)),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for raw_col in ["W_raw", "D_raw", "Q_raw", "Uimb_raw"]:
        norm_col = raw_col.replace("_raw", "_norm")
        lo, hi = df[raw_col].min(), df[raw_col].max()
        df[norm_col] = (df[raw_col] - lo) / (hi - lo) if hi > lo else 0.0

    df["objective_score_equal_weights"] = (
        weights.alpha * df["W_norm"]
        + weights.beta * df["D_norm"]
        + weights.gamma * df["Q_norm"]
        + weights.delta * df["Uimb_norm"]
    )
    df["rank"] = (
        df["objective_score_equal_weights"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    return df


def write_professor_summary(
    results: List[SimulationResult],
    tables_dir: Path = config.TABLES_DIR,
    weights: config.ObjectiveWeights = config.OBJECTIVE_WEIGHTS,
) -> Path:
    """Write the compact, decision-support 'professor' summary table.

    Columns include the headline metrics plus the objective score and rank.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    obj = compute_objective_scores(results, weights=weights)

    rows = []
    for r in results:
        s = r.summary or {}
        rows.append(
            {
                "scenario": r.scenario,
                "completed_trips": s.get("completed_trips", 0),
                "started_charging_events": s.get("started_charging_events", 0),
                "completed_charging_events": s.get("completed_charging_events", 0),
                "pct_waited_at_least_once": s.get("pct_waited_at_least_once", 0.0),
                "mean_waiting_time_among_waiters_min": s.get(
                    "mean_waiting_time_among_waiters_min", 0.0
                ),
                "p95_waiting_time_min": s.get("p95_waiting_time_min", 0.0),
                "mean_detour_distance_m": s.get("mean_detour_distance_m", 0.0),
                "total_queue_minutes": s.get("total_queue_minutes", 0),
                "max_queue_length": s.get("max_queue_length", 0),
                "mean_station_utilisation": s.get("mean_station_utilisation", 0.0),
                "utilisation_imbalance_sd": s.get("utilisation_imbalance_sd", 0.0),
            }
        )
    df = pd.DataFrame(rows)
    df = df.merge(
        obj[["scenario", "objective_score_equal_weights", "rank"]],
        on="scenario",
        how="left",
    )
    df = df.sort_values("rank").reset_index(drop=True)
    path = tables_dir / "professor_summary.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Professor-facing figures
# ---------------------------------------------------------------------------

def plot_professor_queue_over_time(
    results: Iterable[SimulationResult],
    figure_path: Path = config.FIGURES_DIR / "professor_queue_over_time.png",
) -> Path:
    """Clean total-queue-length-over-time line chart, one line per scenario.

    Same source data as `plot_queue_over_time`, formatted for the thesis
    front-matter: thicker lines, named scenarios in the legend, time
    axis in hours.
    """
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = {"S1_real": "#1f77b4", "S2_clustered": "#d62728", "S3_distributed": "#2ca02c"}

    for r in results:
        if r.queue_timeseries.empty:
            continue
        total = r.queue_timeseries.sum(axis=1)
        hours = total.index / 60.0
        ax.plot(
            hours, total.values,
            label=r.scenario,
            linewidth=2.0,
            color=palette.get(r.scenario),
        )
    ax.set_xlabel("simulation time (hours)")
    ax.set_ylabel("total queued vehicles (sum over stations)")
    ax.set_title("Queue pressure over time, by station-layout scenario")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def plot_professor_objective_components(
    results: Iterable[SimulationResult],
    figure_path: Path = config.FIGURES_DIR / "professor_objective_components.png",
    weights: config.ObjectiveWeights = config.OBJECTIVE_WEIGHTS,
) -> Path:
    """Grouped bar chart of the four normalised objective components.

    For each component (W, D, Q, Uimb) we show one bar per scenario, in
    [0, 1] after min-max normalisation. Below the bars we annotate the
    composite J = αW + βD + γQ + δUimb (lower = better) and the rank.
    """
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    results = list(results)
    obj = compute_objective_scores(results, weights=weights)
    if obj.empty:
        # Empty placeholder
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "no scenarios", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
        return figure_path

    components = ["W_norm", "D_norm", "Q_norm", "Uimb_norm"]
    component_labels = ["W\n(waiting)", "D\n(detour)", "Q\n(queue)", "Uimb\n(util. imbalance)"]
    scenarios = obj["scenario"].tolist()
    palette = {"S1_real": "#1f77b4", "S2_clustered": "#d62728", "S3_distributed": "#2ca02c"}

    x = np.arange(len(components))
    width = 0.8 / max(len(scenarios), 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, sc in enumerate(scenarios):
        vals = obj.loc[obj["scenario"] == sc, components].values.flatten()
        offset = (i - (len(scenarios) - 1) / 2.0) * width
        score = float(obj.loc[obj["scenario"] == sc, "objective_score_equal_weights"].iloc[0])
        rank = int(obj.loc[obj["scenario"] == sc, "rank"].iloc[0])
        ax.bar(
            x + offset, vals, width,
            label=f"{sc}  (J = {score:.2f}, rank {rank})",
            color=palette.get(sc),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(component_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("normalised component value (0 = best, 1 = worst)")
    ax.set_title(
        "Objective components by scenario\n"
        f"J = {weights.alpha:.2f}·W + {weights.beta:.2f}·D + "
        f"{weights.gamma:.2f}·Q + {weights.delta:.2f}·Uimb   (lower = better)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def plot_professor_waiters_only(
    results: Iterable[SimulationResult],
    figure_path: Path = config.FIGURES_DIR / "professor_waiters_only.png",
) -> Path:
    """Boxplot of waiting time among agents who actually waited."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    data, labels = [], []
    for r in results:
        if r.agents_df.empty:
            continue
        waited = r.agents_df.loc[
            r.agents_df["waiting_time_min"] > 0, "waiting_time_min"
        ]
        data.append(waited.values)
        labels.append(f"{r.scenario}\n(n={len(waited)})")

    fig, ax = plt.subplots(figsize=(8, 5))
    if any(len(d) > 0 for d in data):
        ax.boxplot(data, labels=labels, showmeans=True)
    else:
        ax.text(0.5, 0.5, "no agents waited", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("waiting time (minutes)")
    ax.set_title("Waiting time among agents who actually waited")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def write_all_outputs(results: List[SimulationResult]) -> Dict[str, Path]:
    """Write all required tables and figures, returning a dict of paths."""
    paths = write_scenario_outputs(results)
    paths["queue_over_time"] = plot_queue_over_time(results)
    paths["scenario_comparison_waiting_time"] = plot_waiting_time_comparison(results)
    paths["waiting_time_waiters_only"] = plot_waited_only_boxplot(results)
    paths["station_utilisation"] = plot_station_utilisation(results)
    paths["utilisation_histogram"] = plot_utilisation_histogram(results)
    paths["detour_distribution"] = plot_detour_distribution(results)
    paths["charging_events_by_scenario"] = plot_charging_events_bar(results)

    # Decision-support / professor-facing outputs.
    paths["professor_summary"] = write_professor_summary(results)
    paths["professor_queue_over_time"] = plot_professor_queue_over_time(results)
    paths["professor_objective_components"] = plot_professor_objective_components(results)
    paths["professor_waiters_only"] = plot_professor_waiters_only(results)
    return paths
