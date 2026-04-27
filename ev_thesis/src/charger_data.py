"""Charger-data acquisition and cleaning.

Primary source: Open Charge Map REST API (real, live).
Secondary fallback: a manually downloaded CSV (e.g. exported from EIPA/UDT
or an OCM web download) at `data/raw/chargers_manual.csv`.

The cleaned/snapped charger table is saved to `data/processed/chargers_clean.csv`
with the columns required by the simulation:
    station_id, name, latitude, longitude, operator,
    number_of_points, source, ports_imputed, node, snap_distance_m
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd
import requests

from . import config
from .graph_utils import snap_points_to_nodes


REQUIRED_COLUMNS = [
    "station_id",
    "name",
    "latitude",
    "longitude",
    "operator",
    "number_of_points",
    "source",
]


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def fetch_chargers_openchargemap(
    lat: float = config.WARSAW_CENTER_LAT,
    lon: float = config.WARSAW_CENTER_LON,
    radius_km: float = config.OCM_QUERY_RADIUS_KM,
    max_results: int = config.OCM_MAX_RESULTS,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Query Open Charge Map for chargers around (lat, lon).

    Returns a DataFrame with the REQUIRED_COLUMNS. If the API call fails,
    or if the API key is missing and no fallback is desired here, raises.
    Higher-level callers should use `load_chargers` which handles fallbacks.
    """
    api_key = api_key or config.get_ocm_api_key()
    if not api_key:
        raise RuntimeError(config.ocm_key_help_message())

    params = {
        "output": "json",
        "countrycode": config.OCM_COUNTRY_CODE,
        "latitude": lat,
        "longitude": lon,
        "distance": radius_km,
        "distanceunit": "KM",
        "maxresults": max_results,
        "compact": "true",
        "verbose": "false",
        "key": api_key,
    }
    headers = {"X-API-Key": api_key, "User-Agent": "ev-thesis-sgh/1.0"}
    print(
        f"[charger_data] querying Open Charge Map: "
        f"lat={lat}, lon={lon}, radius={radius_km} km, max={max_results}"
    )
    resp = requests.get(
        config.OCM_API_URL, params=params, headers=headers, timeout=timeout
    )
    resp.raise_for_status()
    payload = resp.json()
    df = _ocm_payload_to_df(payload)
    print(f"[charger_data] OCM returned {len(df)} raw POIs.")
    return df


def _ocm_payload_to_df(payload) -> pd.DataFrame:
    """Flatten an OCM JSON response to the required columns."""
    if not isinstance(payload, list) or len(payload) == 0:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    rows = []
    for poi in payload:
        addr = poi.get("AddressInfo") or {}
        op = poi.get("OperatorInfo") or {}
        connections = poi.get("Connections") or []
        # Number of points: prefer NumberOfPoints, else len(connections).
        n_points = poi.get("NumberOfPoints")
        if not n_points:
            n_points = len(connections) or None
        rows.append(
            {
                "station_id": poi.get("ID"),
                "name": addr.get("Title"),
                "latitude": addr.get("Latitude"),
                "longitude": addr.get("Longitude"),
                "operator": (op.get("Title") if isinstance(op, dict) else None),
                "number_of_points": n_points,
                "source": "openchargemap",
            }
        )
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    return df


def save_raw_chargers(
    df: pd.DataFrame, path: Path = config.RAW_CHARGER_CSV
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_manual_chargers(path: Path = config.MANUAL_CHARGER_CSV) -> pd.DataFrame:
    """Load a manually exported charger CSV (e.g. EIPA dump). Validates columns."""
    if not path.exists():
        raise FileNotFoundError(f"Manual charger CSV not found at {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Manual charger CSV {path} is missing required columns: {missing}"
        )
    if "source" not in df.columns or df["source"].isna().all():
        df["source"] = "manual"
    return df


def load_chargers(prefer_cached: bool = True) -> pd.DataFrame:
    """High-level loader.

    Order of preference:
    1. If `prefer_cached` and `RAW_CHARGER_CSV` exists, load it.
    2. Else, query Open Charge Map (requires API key in env).
    3. Else, fall back to `MANUAL_CHARGER_CSV`.
    Otherwise raise with a help message.
    """
    if prefer_cached and config.RAW_CHARGER_CSV.exists():
        return pd.read_csv(config.RAW_CHARGER_CSV)

    if config.get_ocm_api_key():
        try:
            df = fetch_chargers_openchargemap()
            save_raw_chargers(df)
            return df
        except Exception as e:  # network / API errors → try manual fallback
            print(f"[charger_data] OCM fetch failed ({e}); falling back to manual.")

    if config.MANUAL_CHARGER_CSV.exists():
        df = load_manual_chargers()
        save_raw_chargers(df)
        return df

    raise RuntimeError(config.ocm_key_help_message())


# ---------------------------------------------------------------------------
# Cleaning & snapping
# ---------------------------------------------------------------------------

def clean_chargers(
    df: pd.DataFrame,
    G: nx.MultiDiGraph,
    max_snap_distance_m: float = config.MAX_SNAP_DISTANCE_M,
    default_ports: int = config.DEFAULT_PORTS_WHEN_MISSING,
) -> pd.DataFrame:
    """Clean, impute, and snap chargers to the graph.

    Steps (each step prints a diagnostic so the thesis can show the funnel):
    1. drop rows without lat/lon;
    2. fill missing names / operators;
    3. impute missing port counts with `default_ports` and flag them;
    4. snap each charger to the nearest graph node;
    5. drop rows whose snap distance exceeds `max_snap_distance_m`
       (effectively outside the study area);
    6. aggregate chargers that snap to the same graph node.
    """
    n_raw = len(df)
    print(f"[charger_data] cleaning funnel — raw rows: {n_raw}")
    if df.empty:
        print("[charger_data] empty input; nothing to clean.")
        return _empty_clean()

    df = df.copy()

    # 1. drop rows without lat/lon
    n_before = len(df)
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    n_no_coords = n_before - len(df)
    if n_no_coords:
        print(
            f"[charger_data]   - dropped {n_no_coords} rows missing lat/lon"
        )
    if df.empty:
        print("[charger_data] no rows with valid coordinates; aborting.")
        return _empty_clean()

    # 2. fill text columns
    df["name"] = df["name"].fillna("(unnamed)")
    df["operator"] = df["operator"].fillna("(unknown)")

    # 3. impute missing/zero port counts
    df["ports_imputed"] = df["number_of_points"].isna() | (
        pd.to_numeric(df["number_of_points"], errors="coerce") <= 0
    )
    n_imputed = int(df["ports_imputed"].sum())
    df["number_of_points"] = pd.to_numeric(
        df["number_of_points"], errors="coerce"
    ).fillna(default_ports).astype(int)
    df.loc[df["number_of_points"] <= 0, "number_of_points"] = default_ports
    if n_imputed:
        print(
            f"[charger_data]   - imputed port count for {n_imputed} rows "
            f"(default = {default_ports}, flagged via ports_imputed=True)"
        )

    n_after_clean = len(df)
    print(f"[charger_data]   after cleaning: {n_after_clean} rows")

    # 4. snap to graph nodes
    nodes, dists = snap_points_to_nodes(
        G, df["latitude"].tolist(), df["longitude"].tolist()
    )
    df["node"] = nodes
    df["snap_distance_m"] = dists
    n_snapped = len(df)
    if n_snapped:
        median_snap = float(pd.Series(dists).median())
        max_snap = float(pd.Series(dists).max())
        print(
            f"[charger_data]   snapped: {n_snapped} rows  "
            f"(median snap distance = {median_snap:.1f} m, "
            f"max = {max_snap:.1f} m)"
        )

    # 5. boundary filter
    keep = df["snap_distance_m"] <= max_snap_distance_m
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(
            f"[charger_data]   - dropped {n_dropped} rows whose snap distance "
            f"> {max_snap_distance_m:.0f} m (outside graph boundary)"
        )
    df = df[keep].reset_index(drop=True)
    n_after_filter = len(df)
    print(
        f"[charger_data]   after graph-boundary filter: {n_after_filter} rows"
    )

    # 6. aggregate by node
    df = _aggregate_by_node(df)
    n_after_agg = len(df)
    if n_after_agg < n_after_filter:
        print(
            f"[charger_data]   - merged {n_after_filter - n_after_agg} "
            f"duplicate-node rows by summing ports"
        )

    total_ports = int(df["number_of_points"].sum()) if not df.empty else 0
    print(
        f"[charger_data] FINAL: {n_after_agg} stations, "
        f"{total_ports} total ports."
    )
    return df


def _empty_clean() -> pd.DataFrame:
    return pd.DataFrame(
        columns=REQUIRED_COLUMNS + ["ports_imputed", "node", "snap_distance_m"]
    )


def _aggregate_by_node(df: pd.DataFrame) -> pd.DataFrame:
    """Combine multiple chargers that snap to the same graph node."""
    if df.empty:
        return df
    grouped = df.groupby("node", as_index=False).agg(
        station_id=("station_id", "first"),
        name=("name", lambda s: " | ".join(sorted(set(map(str, s))))),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        operator=("operator", lambda s: " | ".join(sorted(set(map(str, s))))),
        number_of_points=("number_of_points", "sum"),
        source=("source", "first"),
        ports_imputed=("ports_imputed", "any"),
        snap_distance_m=("snap_distance_m", "mean"),
    )
    return grouped


def save_clean_chargers(
    df: pd.DataFrame, path: Path = config.PROCESSED_CHARGER_CSV
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_clean_chargers(
    path: Path = config.PROCESSED_CHARGER_CSV,
) -> pd.DataFrame:
    return pd.read_csv(path)
