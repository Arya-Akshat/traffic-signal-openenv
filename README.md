---
title: Traffic Signal OpenEnv
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.main:app
pinned: false
---

# Traffic Signal OpenEnv 🚦

Welcome to the **Traffic Signal OpenEnv**! This repository provides an OpenEnv-compliant reinforcement learning environment for traffic signal control. 

Instead of training a model natively inside the repository, this acts as an isolated, containerized environment that exposes deterministic simulation endpoints. External agents, LLM controllers, or rule-based clients (such as the provided `inference.py`) can interface with the environment over standard HTTP APIs.

**Live Deployment:** [guuru-dev-traffic-signal-openenv.hf.space](https://guuru-dev-traffic-signal-openenv.hf.space)

---

## 🚦 Features & Architecture

The primary goal of traffic signal optimization is to reduce queueing and idle waiting times without sacrificing overall throughput. The environment simulates a standard 4-way signalized intersection, providing dense, immediate feedback at every step.

**Flow:**
`Controller Client` ➔ `FastAPI Server` ➔ `TrafficEnv Simulation`

You pass an action to the server, it validates the schema, advances the deterministic simulation by one tick, and returns the new observation, reward, and step info.

### 📊 Observation & Action Space

**Observation Schema:**
```json
{
    "queue_lengths": [N, S, E, W],
    "waiting_times": [N, S, E, W],
    "current_phase": 0,
    "time_in_phase": 5
}
```

**Valid Actions:**
```json
["KEEP", "SWITCH", "PHASE_0", "PHASE_1", "PHASE_2", "PHASE_3"]
```
Invalid actions will immediately be rejected by the FastAPI validation layer.

---

## 🏆 Reward Design

We use a dense reward function per step to balance competing priorities (delay reduction vs. flow) and explicitly penalize signal oscillation:

```python
reward = -avg_wait_norm - 0.5 * queue_norm + throughput_norm * 2.0 - switching_penalty
```

This encourages the controller to maintain smooth traffic flow while discouraging erratic switching behavior.

---

## 📝 Tasks & Graders

The environment comes with three predefined tasks of increasing difficulty, managed by `openenv.yaml`. Each task is fully deterministic:

| Task ID | Description |
|---|---|
| `easy_fixed` | Fixed traffic demand with low variance. |
| `medium_dynamic` | Introduces dynamic demand spikes to test signal responsiveness. |
| `hard_multi` | Extensive pressure, representing a multi-intersection grid with emergency vehicles. |

Each task has an associated grader that yields a final composite score strictly clamped within `(0.0, 1.0)`. The score evaluates:
1. Average waiting time
2. Vehicle throughput
3. Maximum queue lengths

---

## 🚀 Quick Start (Local)

### 1. Installation

Set up your virtual environment and install the dependencies:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Run the API Server

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Verify Endpoints

```bash
curl http://127.0.0.1:8000/reset
curl "http://127.0.0.1:8000/reset?task_id=medium_dynamic"
curl -X POST http://127.0.0.1:8000/step -H "Content-Type: application/json" -d '{"action":"SWITCH"}'
```

### 4. Run the Client

You can run the baseline inference client to evaluate performance across all 3 tasks:
```bash
BASE_URL=http://127.0.0.1:8000 python inference.py
```

---

## 🐋 Docker Validation

To ensure OpenEnv compliance locally using Docker:

```bash
docker build -t traffic-env .
docker run --rm -p 7860:7860 traffic-env
```

And test it against the exposed port:
```bash
curl http://127.0.0.1:7860/reset
```

---

## 🛡️ Validation & Pre-Commit

Before submitting or pushing changes, please ensure code quality checks pass. Run all checks from the repository root:

```bash
source .venv/bin/activate
ruff check .
black --check .
pytest -q
```

You can optionally format the code automatically:
```bash
black . && ruff check . --fix
```

We also include a helpful submission validator script to verify your deployment matches the OpenEnv specs:
```bash
./validate-submission.sh https://guuru-dev-traffic-signal-openenv.hf.space
```

---

## ☁️ Hugging Face Deployment

This project uses the Docker SDK for Hugging Face Spaces.

1. Create a Space on HF (select `Docker` SDK).
2. Connect your GitHub repository or upload the files.
3. Configure your endpoint tokens in the Space settings (e.g., `HF_TOKEN`).

The application automatically exposes port `7860` as required by HF Spaces.
