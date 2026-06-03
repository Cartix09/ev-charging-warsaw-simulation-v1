# Agent-Based Simulation of Electric Vehicle Charging Systems on an Urban Road Network

Master's thesis project (SGH). Study area: **Śródmieście, Warsaw, Poland**.
This is a public decision-support simulation written from the perspective of a city government / municipal planner.
It is **not** a private company profit model and does **not** claim to predict real EV demand.

## What the model does

It uses a real road network (OpenStreetMap via OSMnx) and real public-charger
locations (Open Charge Map, optionally EIPA/UDT) and overlays synthetic EV
driver agents. Agents drive through the directed graph, consume battery,
re-route to chargers when their state-of-charge falls below a threshold,
queue if all ports are occupied, charge, and continue with their planned
trips. The model compares three station-layout scenarios under identical
demand: the existing real layout, a clustered counterfactual, and a
distributed counterfactual.

## Project structure

```
ev_thesis/
├── requirements.txt
├── README.md
├── data/
│   ├── raw/          # raw OSM extracts, raw charger CSV
│   └── processed/    # cleaned/snapped chargers, graph pickle
├── outputs/
│   ├── figures/
│   └── tables/
├── notebooks/
│   ├── 01_download_data.ipynb
│   ├── 02_build_graph_and_chargers.ipynb
│   ├── 03_run_scenarios.ipynb
│   └── 04_analyse_results.ipynb
└── src/
    ├── config.py
    ├── graph_utils.py
    ├── charger_data.py
    ├── agents.py
    ├── stations.py
    ├── simulation.py
    ├── scenarios.py
    └── metrics.py
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENCHARGEMAP_API_KEY=your_key_here   # see "Charger data" below
jupyter lab
```

Then run the notebooks in order: `01_download_data.ipynb` →
`02_build_graph_and_chargers.ipynb` → `03_run_scenarios.ipynb` →
`04_analyse_results.ipynb`. Each notebook is a thin wrapper around
`src/`; everything reproducible lives in `src/`.

## Data sources

| Source                  | Use                        | Real / simulated |
|-------------------------|----------------------------|------------------|
| OpenStreetMap (OSMnx)   | Road network               | Real             |
| Open Charge Map API     | Public charger inventory   | Real             |
| EIPA / UDT (optional)   | Polish official registry   | Real (manual)    |
| `src/agents.py`         | EV driver trips            | Simulated        |
| `src/scenarios.py`      | Counterfactual layouts     | Simulated        |

### Charger data

The default loader queries Open Charge Map. To get a key:

1. Register at https://openchargemap.org/profile/applications
2. Copy your API key
3. `export OPENCHARGEMAP_API_KEY=...`

If you cannot get a key, you can also export the **EIPA / UDT** public
registry of alternative-fuel infrastructure
(https://eipa.udt.gov.pl) as CSV and place it at
`data/raw/chargers_manual.csv` with columns:
`station_id,name,latitude,longitude,operator,number_of_points,source`.
The loader will pick that file up automatically when the API key is missing.

## Scenarios

| Scenario        | Layout                                            |
|-----------------|---------------------------------------------------|
| `S1_real`       | Real OCM/EIPA chargers snapped to graph           |
| `S2_clustered`  | Same total ports, sites near city centre          |
| `S3_distributed`| Same total ports, sites on a coarse grid          |

All scenarios share agent count, RNG seed, simulation duration, and total
station capacity, so differences in metrics are attributable to layout only.

## Outputs

- `outputs/tables/scenario_summary.csv`
- `outputs/tables/agent_results.csv`
- `outputs/tables/station_results.csv`
- `outputs/tables/professor_summary.csv`  (compact decision-support table with the objective score and rank)
- `outputs/figures/queue_over_time.png`
- `outputs/figures/scenario_comparison_waiting_time.png`
- `outputs/figures/station_utilisation.png`
- `outputs/figures/professor_queue_over_time.png`        (Figure 1 — preliminary results)
- `outputs/figures/professor_objective_components.png`   (Figure 2 — preliminary results)
- `outputs/figures/professor_waiters_only.png`           (Figure 3 — preliminary results, optional)

## Objective and preliminary outputs

From the perspective of a city government, the planning problem is **where to allocate scarce public charging capacity in central Warsaw**. The simulation does not solve a closed-form optimisation. Instead it estimates, for any candidate station layout, four signals that a planner would minimise:

```
J = α·W + β·D + γ·Q + δ·Uimb
```

| symbol | meaning                                                                  | source metric                              |
|--------|--------------------------------------------------------------------------|--------------------------------------------|
| W      | waiting pressure on affected drivers                                     | `mean_waiting_time_among_waiters_min`      |
| D      | induced detour distance                                                  | `mean_detour_distance_m`                   |
| Q      | cumulative queue pressure                                                | `total_queue_minutes`                      |
| Uimb   | utilisation imbalance across stations                                    | `utilisation_imbalance_sd` (std-dev of util)|

Each component is min-max normalised across the scenarios in a single run, so the composite J ∈ [0, 1] and **lower = better**. Default weights are equal (α = β = γ = δ = 0.25); they can be changed in `src/config.py → ObjectiveWeights`. The thesis reports the four raw components separately. **J is a decision-support summary, not a claim of true social welfare.**

Two key preliminary figures summarise the simulation output:

1. `outputs/figures/professor_queue_over_time.png` — total queue length over time, one line per scenario. Reads directly off the queue-pressure component (Q).
2. `outputs/figures/professor_objective_components.png` — all four normalised components grouped per scenario, with the composite J and the rank.

Preliminary finding (with current parameters): the **distributed** layout dominates on Q, W and D; the **clustered** layout is worst on Q and D; the **real** layout sits in between. Rerun `notebooks/03_run_scenarios.ipynb` to refresh.

## Two-objective optimisation layer

A separate, optional layer (`src/optimisation.py`, `notebooks/05_pareto_optimisation.ipynb`) restates the planning problem as a **two-objective minimisation** suitable for a Pareto-frontier figure:

- $G = (V(G), E(G))$ — the directed road graph (largest strongly connected component).
- $x \subset V(G) \times \mathbb{N}^+$ — a candidate layout = a finite set of `(node, ports)` pairs.
- $K(x) = |x|$ — number of stations.
- $P(x) = \sum_{(n,p) \in x} p$ — total ports.

$$\text{minimise } C(x) = c_{\text{station}}\,K(x) + c_{\text{port}}\,P(x)\quad\text{(infrastructure cost)}$$

$$\text{minimise } L(x) = w_1\,\hat{L}_{\text{wait}} + w_2\,\hat{L}_{\text{detour}} + w_3\,\hat{L}_{95}\quad\text{(service loss)}$$

with $L$ aggregated from the ABM summary (mean wait among waiters, mean detour, p95 wait) and each component min-max-normalised across the candidate set.

For each $\alpha \in \{0.0, 0.1, \ldots, 1.0\}$ the layer also solves

$$\arg\min_x\;J_\alpha(x) = \alpha\,\hat{C}(x) + (1-\alpha)\,\hat{L}(x)$$

with $\hat{C}, \hat{L}$ normalised across the candidate set.

**Evaluation function.** The agent-based simulation is the evaluation function: for every candidate $x$, the ABM runs once with the same agent population, seed, and horizon, and the resulting summary feeds $L(x)$.

**Search method.** Structured candidate search over $(K, P, \text{pattern})$ — *not* a genetic algorithm. The structured grid spans the cost / quality space coarsely; a GA could be substituted as the search driver without changing the Pareto or α-sweep machinery. This is documented honestly in `src/optimisation.py` to address the natural reviewer question.

**Outputs**:
- `outputs/tables/pareto_candidates.csv`
- `outputs/tables/pareto_frontier.csv`
- `outputs/tables/alpha_solutions.csv`
- `outputs/figures/pareto_frontier_cost_quality.png`
- `outputs/figures/alpha_tradeoff_solutions.png`

The three-scenario analysis (S1_real / S2_clustered / S3_distributed) in notebooks 03–04 remains unchanged — it is the preliminary scenario analysis. The optimisation layer is an *additional* search-and-trade-off layer on top of it.

## Modelling assumptions (short)

Free-flow edge travel times, linear battery consumption, constant-power
charging, FIFO queues, greedy nearest-station rule, static graph, no grid
constraints, OCM port count defaulted to 2 when missing (flagged). See
`src/config.py` for all parameters and the thesis Chapter 4 for the full
list and justification.
