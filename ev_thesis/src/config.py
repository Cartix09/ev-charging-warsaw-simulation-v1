"""Central configuration for the EV-charging agent-based simulation.

All paths are resolved relative to the `ev_thesis/` project root, so the code
works regardless of where Python is invoked from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
TABLES_DIR: Path = OUTPUTS_DIR / "tables"

for _d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Study area
# ---------------------------------------------------------------------------

STUDY_AREA_PLACE: str = "Śródmieście, Warszawa, Polska"
# Centroid of Śródmieście — used for OCM radius queries and clustered layout.
WARSAW_CENTER_LAT: float = 52.2330
WARSAW_CENTER_LON: float = 21.0173
OCM_QUERY_RADIUS_KM: float = 4.0


# ---------------------------------------------------------------------------
# Open Charge Map
# ---------------------------------------------------------------------------

OCM_API_URL: str = "https://api.openchargemap.io/v3/poi"
OCM_COUNTRY_CODE: str = "PL"
OCM_MAX_RESULTS: int = 500
OCM_API_KEY_ENV: str = "OPENCHARGEMAP_API_KEY"
MANUAL_CHARGER_CSV: Path = RAW_DIR / "chargers_manual.csv"
RAW_CHARGER_CSV: Path = RAW_DIR / "chargers_openchargemap_warsaw.csv"
PROCESSED_CHARGER_CSV: Path = PROCESSED_DIR / "chargers_clean.csv"

DEFAULT_PORTS_WHEN_MISSING: int = 2
MAX_SNAP_DISTANCE_M: float = 150.0


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

GRAPH_PICKLE: Path = PROCESSED_DIR / "graph_srodmiescie.gpickle"
NETWORK_TYPE: str = "drive"


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationParams:
    """All parameters that drive the simulation, in one place."""

    seed: int = 42
    # Discrete time step: 1 minute per tick.
    dt_minutes: int = 1
    # Simulated horizon. 6 hours covers a meaningful peak; expand for a full
    # day when running longer experiments.
    horizon_minutes: int = 6 * 60

    # Agent population (kept identical across scenarios).
    n_agents: int = 200
    # Number of trips each agent attempts during the horizon.
    trips_per_agent: Tuple[int, int] = (1, 3)

    # Battery model.
    battery_capacity_kwh: float = 50.0
    consumption_kwh_per_km: float = 0.18
    initial_soc_range: Tuple[float, float] = (0.30, 0.90)
    low_battery_threshold: float = 0.20  # SoC fraction at which agent seeks charger
    target_soc: float = 0.80              # SoC fraction at which agent leaves charger

    # Charging.
    charger_power_kw: float = 22.0

    # Departure-time distribution: bimodal (morning + afternoon peak), in minutes
    # since simulation start. Used by the synthetic demand generator.
    morning_peak_min: int = 60       # 1 h after sim start
    afternoon_peak_min: int = 240    # 4 h after sim start
    peak_std_min: int = 45

    # Decision model.
    max_seek_radius_nodes: int = 5000  # bound on reachable-station search
    # Hard cap on how long an agent will wait before giving up its charge attempt
    # and trying the next nearest station (no balking by default → very large).
    max_wait_minutes: int = 10_000


SIM = SimulationParams()


# ---------------------------------------------------------------------------
# Scenario knobs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioParams:
    clustered_radius_m: float = 600.0   # cluster sites within this radius of centre
    # In the distributed layout we lay sites on an N×N grid clipped to the polygon.
    distributed_grid_size: int = 5


SCENARIO = ScenarioParams()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_ocm_api_key() -> str | None:
    """Return the OCM API key from the environment, or None if unset."""
    return os.environ.get(OCM_API_KEY_ENV)


def ocm_key_help_message() -> str:
    return (
        "Open Charge Map API key not found.\n"
        f"  1. Register at https://openchargemap.org/site/develop/api\n"
        f"  2. Copy your API key.\n"
        f"  3. export {OCM_API_KEY_ENV}=...\n"
        f"Alternatively, place a manually downloaded CSV at\n"
        f"  {MANUAL_CHARGER_CSV}\n"
        f"with columns: station_id,name,latitude,longitude,operator,"
        f"number_of_points,source"
    )
