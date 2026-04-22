---
title: Traffic Signal OpenEnv
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.main:app
pinned: false
---

# Traffic Signal OpenEnv: 11-Phase Hierarchical Traffic Benchmark

Traffic Signal OpenEnv is a high-fidelity, deterministic traffic-light control environment exposed as an HTTP API. It is designed for OpenEnv-style evaluation where a remote controller (Central Oversight) adjusts policy vectors for local intersection agents.

Live deployment: https://guuru-dev-traffic-signal-openenv.hf.space

## Overview: The 11-Phase Refactor
This project has undergone a comprehensive 11-phase refactor to transition from a basic grid simulation into a robust, hierarchical multi-agent orchestration benchmark. Key upgrades include:
- **Hierarchical Control**: A Central Controller managing grid-level policy vectors.
- **High-Fidelity Physics**: Turn-movement sub-queues, hard lane capacities, and spillback throttling.
- **Incident Modeling**: Deterministic disruptions (closures, surges, blockages).
- **Advanced Telemetry**: Complex metrics for fairness, stability, and recovery efficiency.
- **Adaptive Curriculum**: Automated task progression based on performance.

## Core Simulation Architecture

### 1. Traffic Physics & Realism
The simulation operates on a 2x2 grid (NW, NE, SW, SE) with the following physical layers:
- **Turn-Movement Queues**: Each lane tracks `left`, `straight`, and `right` sub-queues separately.
- **Hard Lane Capacity**: Lanes have physical limits. Reaching capacity triggers **Spillback Throttling**, reducing upstream outflow by 50-100% to simulate realistic congestion propagation.
- **3-Step Transit Buffers**: FIFO buffers on edges between intersections simulate travel time (3 steps delay).
- **Deterministic Noise**: Seeded stochasticity (10% arrival jitter, 5% service variance) ensures reproducibility while mimicking real-world randomness.

### 2. Hierarchical Orchestration
- **Local Agents**: Use a multi-term scoring system with **1-step lookahead**, **oscillation penalties**, and **fairness constraints** to select phases.
- **Central Oversight**: Detects multi-corridor flow patterns and adjusts global policy weights (e.g., `corridor_priority`, `emergency_boost`).
- **Safety Monitor**: A watchdog system that intervenes if emergency routing fails or if the grid approaches a deadlock state.
- **Rationale Logging**: Every central action includes a "Chain-of-Thought" rationale explaining the systemic triggers (e.g., "Addressing spillback; prioritizing horizontal corridor").

## Task Profiles & Curriculum
The environment supports 7 deterministic task profiles:
1. `easy_fixed`: Stable baseline with minimal jitter.
2. `medium_dynamic`: Periodic demand spikes.
3. `hard_multi`: Asymmetric surges, emergencies, and spillback traps.
4. `gridlock_risk`: **[NEW]** High demand (1.8x) and low capacity (15) to test pressure stability.
5. `corridor_flow`: **[NEW]** Optimized for testing horizontal "Green Wave" synchronization.
6. `incident_response`: **[NEW]** Sequential failure events (closures, surges, blockages).
7. `dynamic_demand`: **[NEW]** Rotating traffic patterns that shift every 25 steps.

**Adaptive Curriculum Runner**: Use `python inference.py --curriculum` to run an automated training loop that advances or demotes task difficulty based on performance (0.8 threshold to advance, 0.4 to step down).

## Advanced Metrics & Telemetry
The environment returns an extensive `info` dict and episode summary:
- **Efficiency**: `throughput_efficiency` (served/spawned ratio).
- **Service Quality**: `travel_time_mean/variance`, `fairness_score` (wait time distribution).
- **System Stability**: `stability_index`, `policy_stability`, `recovery_time` (steps to baseline after incident).
- **Reward Breakdown**: A strictly bounded `[-1.0, 1.0]` reward with a detailed dictionary containing components like `queue_reward`, `coordination_bonus`, and `stability_bonus`.

## Getting Started

### Installation
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

### Running the Environment
```bash
# Run the API server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Run the inference client with curriculum and metrics export
python inference.py --curriculum --export-metrics
```

### Testing
The project includes a comprehensive suite of **22 tests** (11 original regression + 11 extended feature tests).
```bash
pytest -v
```

## Repository Structure
- `env/traffic_env.py`: Core simulation engine and hierarchical logic.
- `env/metrics_exporter.py`: Telemetry persistence (JSON/CSV).
- `inference.py`: Agent logic, curriculum loop, and CLI runner.
- `tasks/`: Task profile definitions.
- `graders/`: Specialized grading logic for each task type.
- `tests/`: Original and extended test suites.

## Performance Validation
- **Hard Ablation Gap**: 36.2% improvement with Central Coordination enabled.
- **Medium Ablation Gap**: 23.0% improvement.
- **System Stability**: 100% deterministic and crash-free under extreme stress.
- **OpenEnv Compliance**: Fully compliant with HTTP API and grading contracts.
