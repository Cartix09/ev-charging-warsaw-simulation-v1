"""Aggregate scenario results, write CSVs, and render figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
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


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def write_all_outputs(results: List[SimulationResult]) -> Dict[str, Path]:
    """Write all required tables and figures, returning a dict of paths."""
    paths = write_scenario_outputs(results)
    paths["queue_over_time"] = plot_queue_over_time(results)
    paths["scenario_comparison_waiting_time"] = plot_waiting_time_comparison(results)
    paths["station_utilisation"] = plot_station_utilisation(results)
    return paths
