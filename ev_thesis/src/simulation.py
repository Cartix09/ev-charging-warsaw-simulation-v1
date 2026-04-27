"""Discrete-time simulator that ties together the graph, stations, and agents.

One tick = SIM.dt_minutes minutes (default 1). Each tick:
1.  Activate idle agents whose departure minute has arrived.
2.  Drive every DRIVING agent for one tick.
3.  Handle 'arrived' / 'low_battery' / 'stranded' events.
4.  Move SEEKING_CHARGER agents toward their target station.
5.  At each station, charge currently plugged-in agents and dequeue waiters.
6.  Snapshot per-station queues for time-series metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd
from tqdm import tqdm

from . import config
from .agents import AgentState, EVAgent
from .stations import StationRegistry


@dataclass
class SimulationResult:
    scenario: str
    horizon_minutes: int
    n_agents: int
    agents_df: pd.DataFrame
    stations_df: pd.DataFrame
    queue_timeseries: pd.DataFrame  # rows = minute, cols = station nodes
    summary: Dict[str, float] = field(default_factory=dict)


class Simulator:
    def __init__(
        self,
        G: nx.MultiDiGraph,
        agents: List[EVAgent],
        stations: StationRegistry,
        params=config.SIM,
        scenario_name: str = "scenario",
    ):
        self.G = G
        self.agents = agents
        self.stations = stations
        self.p = params
        self.scenario_name = scenario_name
        self.minute = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, progress: bool = True) -> SimulationResult:
        horizon = self.p.horizon_minutes
        dt = self.p.dt_minutes
        iterator = range(0, horizon, dt)
        if progress:
            iterator = tqdm(iterator, desc=f"sim[{self.scenario_name}]")

        for t in iterator:
            self.minute = t
            self._activate_due_agents()
            self._step_driving_and_seeking()
            self._step_charging()
            self.stations.tick()

        return self._collect_results()

    # ------------------------------------------------------------------
    # Phase: activate
    # ------------------------------------------------------------------
    def _activate_due_agents(self) -> None:
        for a in self.agents:
            if a.state != AgentState.IDLE:
                continue
            trip = a._next_trip()
            if trip is None:
                a.state = AgentState.DONE
                continue
            if trip.depart_minute <= self.minute:
                # Skip impossibly-stale planned trips (shouldn't normally happen).
                if not a.begin_trip(self.G, trip):
                    # Failed to plan; leave IDLE so next planned trip can fire.
                    a.state = AgentState.IDLE

    # ------------------------------------------------------------------
    # Phase: drive
    # ------------------------------------------------------------------
    def _step_driving_and_seeking(self) -> None:
        dt = self.p.dt_minutes
        for a in self.agents:
            if a.state == AgentState.DRIVING:
                outcome = a.step_drive(self.G, dt)
                if outcome == "arrived":
                    a.complete_current_trip()
                elif outcome == "low_battery":
                    self._handle_low_battery(a)
                elif outcome == "stranded":
                    a.stranded = True
                    a.state = AgentState.DONE
            elif a.state == AgentState.SEEKING_CHARGER:
                outcome = a.step_drive(self.G, dt)
                if outcome == "arrived":
                    self._arrive_at_station(a)
                elif outcome == "stranded":
                    a.stranded = True
                    a.state = AgentState.DONE
            # QUEUEING agents accumulate waiting time implicitly via timestamps.

    def _handle_low_battery(self, a: EVAgent) -> None:
        if not self.stations.nodes:
            # No stations exist at all — nothing to be done; let the agent run
            # its battery to zero on the next ticks.
            return
        a.divert_to_charger(self.G, self.stations.nodes)

    def _arrive_at_station(self, a: EVAgent) -> None:
        node = a.current_station_node
        if node is None:
            a.state = AgentState.IDLE
            return
        st = self.stations.get(node)
        if st is None:
            a.state = AgentState.IDLE
            return
        if st.request_port(a.agent_id):
            a.start_charging(self.minute)
        else:
            a.join_queue(self.minute)

    # ------------------------------------------------------------------
    # Phase: charging
    # ------------------------------------------------------------------
    def _step_charging(self) -> None:
        dt = self.p.dt_minutes
        # Index agents by id so station release events can find them.
        agents_by_id = {a.agent_id: a for a in self.agents}

        for a in self.agents:
            if a.state == AgentState.CHARGING:
                done = a.step_charge(dt, self.p.charger_power_kw)
                if done:
                    st = self.stations.get(a.current_station_node)
                    if st is not None:
                        next_id = st.release_port()
                        if next_id is not None:
                            next_agent = agents_by_id.get(next_id)
                            if next_agent is not None:
                                next_agent.start_charging(self.minute)
                    a.finish_charging(self.G)

    # ------------------------------------------------------------------
    # Collect
    # ------------------------------------------------------------------
    def _collect_results(self) -> SimulationResult:
        horizon = self.p.horizon_minutes

        agents_df = pd.DataFrame(
            [
                {
                    "agent_id": a.agent_id,
                    "completed_trips": a.completed_trips,
                    "failed_trips": a.failed_trips,
                    "started_charging_events": a.started_charging_events,
                    "completed_charging_events": a.completed_charging_events,
                    "times_queued": a.times_queued,
                    "waiting_time_min": a.waiting_time_min,
                    "detour_distance_m": a.detour_distance_m,
                    "final_soc": a.soc,
                    "stranded": a.stranded,
                    "n_planned_trips": len(a.trips),
                }
                for a in self.agents
            ]
        )

        stations_df = self.stations.to_dataframe(horizon)

        # Queue timeseries: row per minute, column per station node.
        if self.stations.nodes:
            queue_ts = pd.DataFrame(
                {s.node: s.queue_log for s in self.stations.stations}
            )
            queue_ts.index.name = "minute"
        else:
            queue_ts = pd.DataFrame()

        summary = self._summarise(agents_df, stations_df)

        return SimulationResult(
            scenario=self.scenario_name,
            horizon_minutes=horizon,
            n_agents=len(self.agents),
            agents_df=agents_df,
            stations_df=stations_df,
            queue_timeseries=queue_ts,
            summary=summary,
        )

    def _summarise(self, agents_df: pd.DataFrame, stations_df: pd.DataFrame) -> Dict[str, float]:
        if agents_df.empty:
            return {}
        wait = agents_df["waiting_time_min"]
        det = agents_df["detour_distance_m"]
        n = len(agents_df)
        charged_at_least_once = agents_df["started_charging_events"] > 0
        waited_at_least_once = wait > 0
        wait_among_waiters = wait[waited_at_least_once]

        return {
            "scenario": self.scenario_name,
            "n_agents": n,
            "n_stations": len(stations_df),
            "total_ports": int(stations_df["n_ports"].sum()) if not stations_df.empty else 0,
            "completed_trips": int(agents_df["completed_trips"].sum()),
            "failed_trips": int(agents_df["failed_trips"].sum()),
            "stranded_agents": int(agents_df["stranded"].sum()),
            # Started vs completed are now distinct.
            "started_charging_events": int(agents_df["started_charging_events"].sum()),
            "completed_charging_events": int(agents_df["completed_charging_events"].sum()),
            "total_queued_agents": int(waited_at_least_once.sum()),
            "pct_charged_at_least_once": float(charged_at_least_once.mean() * 100),
            "pct_waited_at_least_once": float(waited_at_least_once.mean() * 100),
            "mean_waiting_time_min": float(wait.mean()),
            "max_waiting_time_min": float(wait.max()),
            "median_waiting_time_min": float(wait.median()),
            "p95_waiting_time_min": float(wait.quantile(0.95)),
            "mean_waiting_time_among_waiters_min": (
                float(wait_among_waiters.mean()) if len(wait_among_waiters) else 0.0
            ),
            "mean_detour_distance_m": float(det.mean()),
            "median_detour_distance_m": float(det.median()),
            "mean_station_utilisation": (
                float(stations_df["utilisation"].mean())
                if not stations_df.empty
                else 0.0
            ),
            "total_queue_minutes": (
                int(stations_df["total_queue_minutes"].sum())
                if not stations_df.empty
                else 0
            ),
            "max_queue_length": (
                int(stations_df["max_queue"].max())
                if not stations_df.empty
                else 0
            ),
        }
