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

1. Register at https://openchargemap.org/site/develop/api
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
- `outputs/figures/queue_over_time.png`
- `outputs/figures/scenario_comparison_waiting_time.png`
- `outputs/figures/station_utilisation.png`

## Modelling assumptions (short)

Free-flow edge travel times, linear battery consumption, constant-power
charging, FIFO queues, greedy nearest-station rule, static graph, no grid
constraints, OCM port count defaulted to 2 when missing (flagged). See
`src/config.py` for all parameters and the thesis Chapter 4 for the full
list and justification.
