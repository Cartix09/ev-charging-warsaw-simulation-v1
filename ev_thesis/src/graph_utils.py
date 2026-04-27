"""Road-network graph utilities (OSMnx).

Responsibilities:
- download the drivable network for the study area as a *directed* graph;
- preserve one-way streets;
- restrict to the largest strongly connected component;
- annotate edges with `length` (m) and `travel_time_min`;
- expose helpers for shortest-time routing and lat/lon → node snapping;
- pickle/unpickle the cleaned graph to avoid repeated downloads.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import osmnx as ox

from . import config


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

def download_graph(query=None) -> nx.MultiDiGraph:
    """Download the drivable network and return a cleaned MultiDiGraph.

    `query` may be:
    - None (default) → use `config.study_area_query()` based on STUDY_AREA_MODE;
    - a place string (e.g. "Warsaw, Poland");
    - a list of place strings (multi-district union);
    - a (north, south, east, west) bbox tuple, in which case
      `osmnx.graph_from_bbox` is used.

    Cleaning:
    - directed (one-way preserved by `network_type='drive'`);
    - largest strongly connected component;
    - edge `length` (m) from OSMnx default;
    - edge `travel_time_min` derived from `maxspeed` (OSMnx fills defaults).
    """
    if query is None:
        query = config.study_area_query()

    if isinstance(query, tuple) and len(query) == 4 and all(
        isinstance(v, (int, float)) for v in query
    ):
        north, south, east, west = query
        print(
            f"[graph_utils] downloading drivable graph for bbox "
            f"N={north} S={south} E={east} W={west}"
        )
        G = ox.graph_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            network_type=config.NETWORK_TYPE,
            simplify=True,
        )
    else:
        descr = query if isinstance(query, str) else f"{len(query)} places"
        print(f"[graph_utils] downloading drivable graph for: {descr}")
        G = ox.graph_from_place(
            query, network_type=config.NETWORK_TYPE, simplify=True
        )

    G = _post_process(G)
    print(
        f"[graph_utils] graph after post-processing: "
        f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
        f"(largest strongly connected component)"
    )
    return G


def _post_process(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    G = ox.add_edge_speeds(G)               # adds 'speed_kph' to every edge
    G = ox.add_edge_travel_times(G)         # adds 'travel_time' (seconds)

    # Convert seconds → minutes and ensure 'length' is set.
    for _, _, data in G.edges(data=True):
        if "travel_time" in data:
            data["travel_time_min"] = float(data["travel_time"]) / 60.0
        else:
            # Fallback: 30 km/h
            data["travel_time_min"] = float(data.get("length", 0.0)) / 1000.0 / 30.0 * 60.0
        data["length"] = float(data.get("length", 0.0))

    # Largest strongly connected component (directed).
    if not nx.is_strongly_connected(G):
        sccs = list(nx.strongly_connected_components(G))
        largest = max(sccs, key=len)
        G = G.subgraph(largest).copy()

    return G


def save_graph(G: nx.MultiDiGraph, path: Path = config.GRAPH_PICKLE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    return path


def load_graph(path: Path = config.GRAPH_PICKLE) -> nx.MultiDiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_or_build_graph(force_download: bool = False) -> nx.MultiDiGraph:
    """Load the pickled graph if available, else download and pickle it."""
    if (not force_download) and config.GRAPH_PICKLE.exists():
        return load_graph()
    G = download_graph()
    save_graph(G)
    return G


# ---------------------------------------------------------------------------
# Snapping & geometry
# ---------------------------------------------------------------------------

def snap_points_to_nodes(
    G: nx.MultiDiGraph,
    lats: Sequence[float],
    lons: Sequence[float],
) -> Tuple[List[int], List[float]]:
    """Snap (lat, lon) points to the nearest graph nodes.

    Returns (node_ids, snap_distances_m). Uses OSMnx haversine via
    `nearest_nodes`. Distances are computed manually because `nearest_nodes`
    can return them only as a side-effect on some versions.
    """
    if len(lats) == 0:
        return [], []
    nodes = ox.distance.nearest_nodes(G, X=list(lons), Y=list(lats))
    if not isinstance(nodes, (list, np.ndarray)):
        nodes = [nodes]

    node_lats = np.array([G.nodes[n]["y"] for n in nodes])
    node_lons = np.array([G.nodes[n]["x"] for n in nodes])
    dists = _haversine_m(np.asarray(lats), np.asarray(lons), node_lats, node_lons)
    return list(map(int, nodes)), list(map(float, dists))


def _haversine_m(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    R = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def shortest_path_by_time(
    G: nx.MultiDiGraph, source: int, target: int
) -> Optional[List[int]]:
    """Shortest path minimising travel_time_min. Returns None if unreachable."""
    if source == target:
        return [source]
    try:
        return nx.shortest_path(G, source, target, weight="travel_time_min")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def path_length_m(G: nx.MultiDiGraph, path: Sequence[int]) -> float:
    """Sum of edge lengths along a path (metres). Handles multi-edges."""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edges = G.get_edge_data(u, v)
        if not edges:
            continue
        total += min(d.get("length", 0.0) for d in edges.values())
    return float(total)


def path_time_min(G: nx.MultiDiGraph, path: Sequence[int]) -> float:
    """Sum of edge travel times along a path (minutes)."""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edges = G.get_edge_data(u, v)
        if not edges:
            continue
        total += min(d.get("travel_time_min", 0.0) for d in edges.values())
    return float(total)


def nearest_reachable_station(
    G: nx.MultiDiGraph,
    source: int,
    station_nodes: Iterable[int],
) -> Optional[Tuple[int, List[int], float]]:
    """Return (station_node, path, travel_time_min) for the nearest reachable
    station from `source`, or None if no station is reachable.

    Uses Dijkstra from `source` and picks the candidate station with the
    minimum cumulative travel time, then reconstructs the path.
    """
    station_set = set(station_nodes)
    if not station_set:
        return None
    # Single-source Dijkstra over travel time.
    distances, paths = nx.single_source_dijkstra(
        G, source, weight="travel_time_min"
    )
    best: Optional[Tuple[int, List[int], float]] = None
    for s in station_set:
        if s in distances:
            t = distances[s]
            if best is None or t < best[2]:
                best = (s, list(paths[s]), float(t))
    return best


def random_node_pairs(
    G: nx.MultiDiGraph, n: int, rng: np.random.Generator
) -> List[Tuple[int, int]]:
    """Sample `n` (origin, destination) pairs uniformly from graph nodes,
    rejecting self-loops. We do *not* check reachability here because the
    largest-SCC restriction already guarantees pairwise reachability.
    """
    nodes = np.array(list(G.nodes))
    pairs: List[Tuple[int, int]] = []
    while len(pairs) < n:
        o, d = rng.choice(nodes, 2, replace=False)
        if int(o) != int(d):
            pairs.append((int(o), int(d)))
    return pairs
