"""Scenario builders.

All three scenarios share:
- the same agent population (so demand is identical),
- the same RNG seed, simulation horizon, and tick size,
- the same total number of ports as the real layout,
so the only thing that varies is *where* the ports are placed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import networkx as nx
import numpy as np
import pandas as pd

from . import config
from .graph_utils import _haversine_m, snap_points_to_nodes
from .stations import ChargingStation, StationRegistry, stations_from_dataframe


@dataclass
class Scenario:
    name: str
    stations: StationRegistry


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def scenario_real(chargers_clean: pd.DataFrame) -> Scenario:
    """S1: real OCM/EIPA layout."""
    stations = stations_from_dataframe(chargers_clean)
    return Scenario(name="S1_real", stations=StationRegistry(stations))


def scenario_clustered(
    chargers_clean: pd.DataFrame,
    G: nx.MultiDiGraph,
    centre_lat: float = config.WARSAW_CENTER_LAT,
    centre_lon: float = config.WARSAW_CENTER_LON,
    radius_m: float = config.SCENARIO.clustered_radius_m,
) -> Scenario:
    """S2: same number of stations and total ports as `chargers_clean`,
    but all sites are placed on graph nodes within `radius_m` of the centre.
    """
    if chargers_clean.empty:
        return Scenario(name="S2_clustered", stations=StationRegistry([]))

    n_sites = len(chargers_clean)
    total_ports = int(chargers_clean["number_of_points"].sum())

    # Find graph nodes within radius_m of (centre_lat, centre_lon).
    node_ids = np.array(list(G.nodes))
    lats = np.array([G.nodes[n]["y"] for n in node_ids])
    lons = np.array([G.nodes[n]["x"] for n in node_ids])
    dists = _haversine_m(
        np.full_like(lats, centre_lat),
        np.full_like(lons, centre_lon),
        lats,
        lons,
    )
    candidates = node_ids[dists <= radius_m]
    if len(candidates) == 0:
        # Fall back: nearest N nodes to centre
        order = np.argsort(dists)
        candidates = node_ids[order[: max(n_sites, 10)]]

    rng = np.random.default_rng(config.SIM.seed)
    chosen = rng.choice(
        candidates, size=min(n_sites, len(candidates)), replace=False
    )

    return Scenario(
        name="S2_clustered",
        stations=_stations_from_nodes(
            chosen.tolist(),
            G,
            total_ports=total_ports,
            source="synthetic_clustered",
        ),
    )


def scenario_distributed(
    chargers_clean: pd.DataFrame,
    G: nx.MultiDiGraph,
    grid_size: int = config.SCENARIO.distributed_grid_size,
) -> Scenario:
    """S3: same number of stations and total ports, placed on a grid covering
    the bounding box of the graph nodes, snapped to the nearest road node.
    """
    if chargers_clean.empty:
        return Scenario(name="S3_distributed", stations=StationRegistry([]))

    n_sites = len(chargers_clean)
    total_ports = int(chargers_clean["number_of_points"].sum())

    lats = np.array([G.nodes[n]["y"] for n in G.nodes])
    lons = np.array([G.nodes[n]["x"] for n in G.nodes])
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    # Lay an oversized grid, then keep the first `n_sites` whose snap distance
    # is reasonable. Oversize so we have room after deduplication.
    side = max(grid_size, int(np.ceil(np.sqrt(n_sites)))) + 1
    grid_lats = np.linspace(lat_min, lat_max, side + 2)[1:-1]
    grid_lons = np.linspace(lon_min, lon_max, side + 2)[1:-1]
    glats, glons = np.meshgrid(grid_lats, grid_lons)
    pts = list(zip(glats.ravel(), glons.ravel()))

    nodes, dists = snap_points_to_nodes(G, [p[0] for p in pts], [p[1] for p in pts])
    # Sort by snap distance ascending and dedupe nodes.
    order = np.argsort(dists)
    chosen: List[int] = []
    seen = set()
    for idx in order:
        n = nodes[idx]
        if n in seen:
            continue
        seen.add(n)
        chosen.append(int(n))
        if len(chosen) >= n_sites:
            break

    return Scenario(
        name="S3_distributed",
        stations=_stations_from_nodes(
            chosen,
            G,
            total_ports=total_ports,
            source="synthetic_distributed",
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stations_from_nodes(
    nodes: List[int],
    G: nx.MultiDiGraph,
    total_ports: int,
    source: str,
) -> StationRegistry:
    """Construct `len(nodes)` stations whose port counts sum to `total_ports`,
    distributed as evenly as possible.
    """
    if not nodes:
        return StationRegistry([])
    n = len(nodes)
    base, extra = divmod(total_ports, n)
    stations: List[ChargingStation] = []
    for i, node in enumerate(nodes):
        ports = base + (1 if i < extra else 0)
        if ports < 1:
            ports = 1
        stations.append(
            ChargingStation(
                station_id=10_000 + i,
                node=node,
                name=f"{source}_{i:03d}",
                operator="(synthetic)",
                n_ports=ports,
                latitude=float(G.nodes[node]["y"]),
                longitude=float(G.nodes[node]["x"]),
                source=source,
                ports_imputed=False,
            )
        )
    return StationRegistry(stations)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def build_all_scenarios(
    chargers_clean: pd.DataFrame, G: nx.MultiDiGraph
) -> List[Scenario]:
    return [
        scenario_real(chargers_clean),
        scenario_clustered(chargers_clean, G),
        scenario_distributed(chargers_clean, G),
    ]
