"""EV driver agent and synthetic demand generation.

An agent owns its own state machine and is stepped by the simulator each tick.
It does not know about stations directly — it queries the simulator's station
registry through dependency injection (see `simulation.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from . import config
from .graph_utils import (
    nearest_reachable_station,
    path_length_m,
    random_node_pairs,
    shortest_path_by_time,
)


class AgentState(Enum):
    IDLE = auto()           # waiting for next planned departure
    DRIVING = auto()        # following a route to a destination
    SEEKING_CHARGER = auto()  # routing to nearest reachable station
    QUEUEING = auto()       # waiting at a station for a free port
    CHARGING = auto()       # plugged in
    DONE = auto()           # all planned trips completed


@dataclass
class TripPlan:
    origin: int
    destination: int
    depart_minute: int


@dataclass
class EVAgent:
    """A single EV driver."""

    agent_id: int
    home_node: int
    trips: List[TripPlan]

    # Battery / vehicle parameters.
    battery_capacity_kwh: float = config.SIM.battery_capacity_kwh
    consumption_kwh_per_km: float = config.SIM.consumption_kwh_per_km
    low_threshold: float = config.SIM.low_battery_threshold
    target_soc: float = config.SIM.target_soc

    # Mutable state.
    state: AgentState = AgentState.IDLE
    current_node: int = field(init=False)
    soc: float = 0.5  # fraction
    route: List[int] = field(default_factory=list)
    route_idx: int = 0
    trip_ptr: int = 0

    # Used while a charging detour is active to remember the original goal.
    pending_destination: Optional[int] = None

    # Aggregates.
    waiting_time_min: float = 0.0
    detour_distance_m: float = 0.0
    completed_trips: int = 0
    # Charging events split: starts (port allocated to this agent) vs.
    # completions (charging session reached target SoC). They differ when
    # the simulation horizon ends mid-session.
    started_charging_events: int = 0
    completed_charging_events: int = 0
    times_queued: int = 0       # number of times this agent joined a queue
    failed_trips: int = 0       # destination unreachable, etc.
    stranded: bool = False      # ran out of battery with no reachable charger

    # Bookkeeping while at a station.
    current_station_node: Optional[int] = None
    queued_at_minute: Optional[int] = None
    charging_started_minute: Optional[int] = None

    def __post_init__(self) -> None:
        self.current_node = self.home_node

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def battery_kwh(self) -> float:
        return self.soc * self.battery_capacity_kwh

    def _consume(self, distance_m: float) -> None:
        used = distance_m / 1000.0 * self.consumption_kwh_per_km
        self.soc = max(0.0, self.soc - used / self.battery_capacity_kwh)

    def _next_trip(self) -> Optional[TripPlan]:
        if self.trip_ptr < len(self.trips):
            return self.trips[self.trip_ptr]
        return None

    # ------------------------------------------------------------------
    # The simulator drives the agent through these methods. Each returns
    # any 'event' the simulator needs to react to (e.g. a queue join).
    # ------------------------------------------------------------------

    def begin_trip(self, G: nx.MultiDiGraph, trip: TripPlan) -> bool:
        """Plan a route to `trip.destination`. Returns True on success."""
        path = shortest_path_by_time(G, self.current_node, trip.destination)
        if path is None:
            self.failed_trips += 1
            self.trip_ptr += 1
            return False
        self.route = path
        self.route_idx = 0
        self.pending_destination = trip.destination
        self.state = AgentState.DRIVING
        return True

    def step_drive(self, G: nx.MultiDiGraph, dt_min: float) -> str:
        """Advance along the current route by up to `dt_min` minutes.

        Returns one of: 'arrived', 'low_battery', 'continuing', 'stranded'.
        """
        time_budget = dt_min
        while time_budget > 0 and self.route_idx < len(self.route) - 1:
            u = self.route[self.route_idx]
            v = self.route[self.route_idx + 1]
            edges = G.get_edge_data(u, v)
            if not edges:
                # Should not happen for valid routes; bail out safely.
                return "stranded"
            # Pick the fastest parallel edge.
            best = min(edges.values(), key=lambda d: d.get("travel_time_min", float("inf")))
            t_edge = float(best.get("travel_time_min", 0.0))
            length = float(best.get("length", 0.0))

            if t_edge <= time_budget + 1e-9:
                self._consume(length)
                self.current_node = v
                self.route_idx += 1
                time_budget -= t_edge
                if self.soc <= 0.0:
                    return "stranded"
            else:
                # Partial traversal: consume proportional energy and stop here.
                # Position is approximated as still at u (we don't track
                # mid-edge geometry) but battery is debited correctly.
                fraction = time_budget / t_edge if t_edge > 0 else 0.0
                self._consume(length * fraction)
                # Bank the partial progress as time elapsed; node is unchanged
                # until the next tick completes the edge. We approximate by
                # leaving the agent at u; the remainder of the edge is paid
                # next tick (slight discretisation error of < 1 minute).
                time_budget = 0.0
                if self.soc <= 0.0:
                    return "stranded"

        if self.route_idx >= len(self.route) - 1:
            return "arrived"
        if self.soc <= self.low_threshold:
            return "low_battery"
        return "continuing"

    def divert_to_charger(
        self,
        G: nx.MultiDiGraph,
        station_nodes: List[int],
    ) -> bool:
        """Re-route to the nearest reachable station. Returns True on success."""
        result = nearest_reachable_station(G, self.current_node, station_nodes)
        if result is None:
            self.stranded = True
            self.state = AgentState.DONE
            return False
        station_node, path, _ = result
        # Detour distance = length of the path to the station (extra travel
        # the agent would not otherwise have done).
        self.detour_distance_m += path_length_m(G, path)
        self.route = path
        self.route_idx = 0
        self.current_station_node = station_node
        self.state = AgentState.SEEKING_CHARGER
        return True

    def join_queue(self, current_minute: int) -> None:
        self.state = AgentState.QUEUEING
        self.queued_at_minute = current_minute
        self.times_queued += 1

    def start_charging(self, current_minute: int) -> None:
        if self.queued_at_minute is not None:
            self.waiting_time_min += current_minute - self.queued_at_minute
            self.queued_at_minute = None
        self.charging_started_minute = current_minute
        self.started_charging_events += 1
        self.state = AgentState.CHARGING

    def step_charge(self, dt_min: float, charger_power_kw: float) -> bool:
        """Apply one tick of charging. Returns True when target_soc reached."""
        delta_kwh = charger_power_kw * (dt_min / 60.0)
        self.soc = min(1.0, self.soc + delta_kwh / self.battery_capacity_kwh)
        return self.soc >= self.target_soc

    def finish_charging(self, G: nx.MultiDiGraph) -> bool:
        """After charging, resume the original destination route."""
        self.completed_charging_events += 1
        self.charging_started_minute = None
        self.current_station_node = None
        if self.pending_destination is None:
            self.state = AgentState.IDLE
            return True
        path = shortest_path_by_time(G, self.current_node, self.pending_destination)
        if path is None:
            self.failed_trips += 1
            self.trip_ptr += 1
            self.pending_destination = None
            self.state = AgentState.IDLE
            return False
        self.route = path
        self.route_idx = 0
        self.state = AgentState.DRIVING
        return True

    def complete_current_trip(self) -> None:
        self.completed_trips += 1
        self.trip_ptr += 1
        self.pending_destination = None
        self.route = []
        self.route_idx = 0
        self.state = AgentState.IDLE


# ---------------------------------------------------------------------------
# Synthetic demand generator
# ---------------------------------------------------------------------------

def generate_agents(
    G: nx.MultiDiGraph,
    n_agents: int = config.SIM.n_agents,
    seed: int = config.SIM.seed,
) -> List[EVAgent]:
    """Produce a deterministic synthetic agent population.

    - Home nodes sampled uniformly from graph nodes.
    - Each agent gets between SIM.trips_per_agent[0] and [1] trips.
    - Departure times sampled from a bimodal Gaussian (morning + afternoon).
    - Initial SoC sampled uniformly from SIM.initial_soc_range.
    """
    rng = np.random.default_rng(seed)
    nodes = np.array(list(G.nodes))
    homes = rng.choice(nodes, size=n_agents, replace=True)

    agents: List[EVAgent] = []
    lo_t, hi_t = config.SIM.trips_per_agent
    lo_s, hi_s = config.SIM.initial_soc_range
    horizon = config.SIM.horizon_minutes

    for i in range(n_agents):
        home = int(homes[i])
        n_trips = int(rng.integers(lo_t, hi_t + 1))
        trips: List[TripPlan] = []
        for k in range(n_trips):
            o, d = random_node_pairs(G, 1, rng)[0]
            # Anchor first trip at the home node.
            if k == 0:
                o = home
            depart = int(_sample_bimodal_time(rng, horizon))
            trips.append(TripPlan(origin=o, destination=d, depart_minute=depart))
        # Sort trips chronologically so the agent attempts them in order.
        trips.sort(key=lambda t: t.depart_minute)

        agents.append(
            EVAgent(
                agent_id=i,
                home_node=home,
                trips=trips,
                soc=float(rng.uniform(lo_s, hi_s)),
            )
        )
    return agents


def _sample_bimodal_time(rng: np.random.Generator, horizon: int) -> int:
    if rng.random() < 0.5:
        mu = config.SIM.morning_peak_min
    else:
        mu = config.SIM.afternoon_peak_min
    t = int(rng.normal(mu, config.SIM.peak_std_min))
    return max(0, min(horizon - 1, t))
