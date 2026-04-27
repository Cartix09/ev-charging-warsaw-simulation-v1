"""Charging-station model and registry.

A station owns a fixed number of ports, a FIFO queue of agent IDs, and a
small log of utilisation. The simulator calls `request_port`, `release_port`,
and `tick` on each station.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import pandas as pd

from . import config


@dataclass
class ChargingStation:
    station_id: int
    node: int
    name: str = "(unnamed)"
    operator: str = "(unknown)"
    n_ports: int = 1
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    source: str = "synthetic"
    ports_imputed: bool = False

    # Mutable state.
    occupied_ports: int = 0
    queue: Deque[int] = field(default_factory=deque)

    # Per-minute samples; index = simulation minute.
    queue_log: List[int] = field(default_factory=list)
    occupancy_log: List[int] = field(default_factory=list)

    # Counters.
    total_arrivals: int = 0
    total_charging_minutes: int = 0
    total_started: int = 0
    total_completed: int = 0

    @property
    def has_free_port(self) -> bool:
        return self.occupied_ports < self.n_ports

    def request_port(self, agent_id: int) -> bool:
        """Try to allocate a port. Returns True if granted, False if queued."""
        self.total_arrivals += 1
        if self.has_free_port:
            self.occupied_ports += 1
            self.total_started += 1
            return True
        self.queue.append(agent_id)
        return False

    def release_port(self) -> Optional[int]:
        """Free a port. If the queue is non-empty, dequeue the next agent and
        immediately reassign the port to them. Returns the agent_id that
        starts charging now, or None if the queue is empty.
        """
        if self.occupied_ports > 0:
            self.occupied_ports -= 1
            self.total_completed += 1
        if self.queue:
            self.occupied_ports += 1
            self.total_started += 1
            return self.queue.popleft()
        return None

    def tick(self) -> None:
        """Record one minute of state and accumulate utilisation."""
        self.queue_log.append(len(self.queue))
        self.occupancy_log.append(self.occupied_ports)
        self.total_charging_minutes += self.occupied_ports

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def utilisation(self, horizon_minutes: int) -> float:
        """Mean port-occupancy fraction over the horizon."""
        if self.n_ports == 0 or horizon_minutes == 0:
            return 0.0
        return self.total_charging_minutes / (self.n_ports * horizon_minutes)

    def summary(self, horizon_minutes: int) -> Dict:
        return {
            "station_id": self.station_id,
            "node": self.node,
            "name": self.name,
            "operator": self.operator,
            "n_ports": self.n_ports,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "source": self.source,
            "ports_imputed": self.ports_imputed,
            "total_arrivals": self.total_arrivals,
            "total_started": self.total_started,
            "total_completed": self.total_completed,
            "total_charging_minutes": self.total_charging_minutes,
            "total_queue_minutes": int(sum(self.queue_log)),
            "mean_queue": (
                sum(self.queue_log) / len(self.queue_log) if self.queue_log else 0.0
            ),
            "max_queue": max(self.queue_log) if self.queue_log else 0,
            "utilisation": self.utilisation(horizon_minutes),
        }


# ---------------------------------------------------------------------------
# Registry: maps node -> station, gives O(1) lookup
# ---------------------------------------------------------------------------

class StationRegistry:
    def __init__(self, stations: List[ChargingStation]):
        # Multiple chargers at the same node have already been merged in
        # cleaning, so node -> single station.
        self._by_node: Dict[int, ChargingStation] = {s.node: s for s in stations}

    def __len__(self) -> int:
        return len(self._by_node)

    def __iter__(self):
        return iter(self._by_node.values())

    @property
    def nodes(self) -> List[int]:
        return list(self._by_node.keys())

    @property
    def stations(self) -> List[ChargingStation]:
        return list(self._by_node.values())

    def get(self, node: int) -> Optional[ChargingStation]:
        return self._by_node.get(node)

    def total_ports(self) -> int:
        return sum(s.n_ports for s in self._by_node.values())

    def tick(self) -> None:
        for s in self._by_node.values():
            s.tick()

    def to_dataframe(self, horizon_minutes: int) -> pd.DataFrame:
        return pd.DataFrame([s.summary(horizon_minutes) for s in self._by_node.values()])


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def stations_from_dataframe(df: pd.DataFrame) -> List[ChargingStation]:
    """Build a list of ChargingStation from a cleaned charger DataFrame."""
    if df.empty:
        return []
    stations: List[ChargingStation] = []
    for i, row in df.reset_index(drop=True).iterrows():
        n_ports = int(row.get("number_of_points", config.DEFAULT_PORTS_WHEN_MISSING) or config.DEFAULT_PORTS_WHEN_MISSING)
        stations.append(
            ChargingStation(
                station_id=int(row.get("station_id", i) or i),
                node=int(row["node"]),
                name=str(row.get("name", "(unnamed)")),
                operator=str(row.get("operator", "(unknown)")),
                n_ports=n_ports,
                latitude=float(row["latitude"]) if "latitude" in row else None,
                longitude=float(row["longitude"]) if "longitude" in row else None,
                source=str(row.get("source", "synthetic")),
                ports_imputed=bool(row.get("ports_imputed", False)),
            )
        )
    return stations
