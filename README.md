---
title: Traffic Signal OpenEnv
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.main:app
pinned: false
---
# Traffic Signal OpenEnv

This repository provides an OpenEnv-compliant traffic signal control environment.
An external controller client (for example, `inference.py`) interacts through HTTP APIs:

- GET /reset
- POST /step
- GET /state

The server does not train an RL policy. It exposes deterministic simulation APIs for controller-driven control.

Live deployment:

- https://guuru-dev-traffic-signal-openenv.hf.space

## Problem Description

Traffic signal optimization aims to reduce queueing and waiting while preserving throughput.
The environment simulates a 4-direction signalized intersection and returns dense feedback at each step.

## Architecture Overview

```text
Controller client (inference.py) -> FastAPI API -> TrafficEnv -> deterministic traffic simulation
```

The controller selects actions, the API exposes OpenEnv endpoints, and the environment applies the traffic dynamics.

## State and Action Space

Observation schema:

```json
{
	"queue_lengths": [N, S, E, W],
	"waiting_times": [N, S, E, W],
	"current_phase": 0,
	"time_in_phase": 5
}
```

Action space:

```json
["KEEP", "SWITCH", "PHASE_0", "PHASE_1", "PHASE_2", "PHASE_3"]
```

Invalid actions are rejected by schema validation.

## Reward Design

Dense reward per step:

```text
reward = (
		- total_waiting_time
		- 0.5 * total_queue_length
		+ throughput * 2
		- switching_penalty
)
```

This balances delay reduction, queue control, and traffic flow while discouraging excessive switching.

## Tasks

- easy_fixed: fixed traffic demand
- medium_dynamic: dynamic spikes
- hard_multi: multi-intersection/e-mergency pressure

All tasks are deterministic and defined in tasks/.

## Grader

Deterministic grader output in [0.0, 1.0] using a composite score:

```text
score = 0.5 * normalized_wait + 0.3 * throughput_score + 0.2 * queue_score
```

Where:

- normalized_wait rewards lower waiting time
- throughput_score rewards more vehicle movement per step
- queue_score rewards shorter queues

This makes the evaluation look closer to a research benchmark than a single-metric threshold.

## Baseline Comparison

A simple fixed-signal baseline is useful for sanity checking the environment.

| Approach | Behavior | Strength | Weakness |
| --- | --- | --- | --- |
| easy_fixed | Lowest demand, fixed traffic | Highest expected score | Least challenging |
| medium_dynamic | Random spikes and changing load | Middle expected score | More variance |
| hard_multi | Multi-intersection with emergency pressure | Lowest expected score | Most difficult |

Recommended comparison in experiments:

1. Run the fixed-signal baseline for the same number of steps as the controller.
2. Compare average waiting time, throughput, and final grader score.
3. Report whether the controller reduces queueing under medium and hard tasks.

Baseline score table:

| Task | Fixed-signal baseline | Rule-based controller | Notes |
| --- | --- | --- | --- |
| easy_fixed | High | Very high | Lowest demand, easiest to stabilize |
| medium_dynamic | Medium | High | Demand spikes reward responsive phase changes |
| hard_multi | Low | Medium | Multi-intersection pressure makes coordination harder |

## Quick Start (Local)

### 1. Setup

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Run API server

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Verify endpoints

```bash
curl http://127.0.0.1:8000/reset
curl "http://127.0.0.1:8000/reset?task_id=medium_dynamic"
curl -X POST http://127.0.0.1:8000/step -H "Content-Type: application/json" -d '{"action":"SWITCH"}'
curl http://127.0.0.1:8000/state
```

### 4. Run inference

```bash
BASE_URL=http://127.0.0.1:8000 python inference.py
```

## API Usage Notes

### Reset with task switch

You can switch task deterministically at reset time:

```bash
curl "http://127.0.0.1:8000/reset?task_id=easy_fixed"
curl "http://127.0.0.1:8000/reset?task_id=medium_dynamic"
curl "http://127.0.0.1:8000/reset?task_id=hard_multi"
```

Invalid task IDs return HTTP 400.

### Supported actions

```json
["KEEP", "SWITCH", "PHASE_0", "PHASE_1", "PHASE_2", "PHASE_3"]
```

## Full Test and Quality Checks

Run all checks from the repository root:

```bash
source .venv311/bin/activate
ruff check .
black --check .
pytest -q
```

Optional formatting fix:

```bash
black .
ruff check . --fix
```

Run deployment validator against a target URL:

```bash
./validate-submission.sh https://guuru-dev-traffic-signal-openenv.hf.space
```

## Inference Environment Variables

inference.py reads these variables:

- BASE_URL: required environment endpoint base URL
- API_BASE_URL: model API base URL
- MODEL_NAME: model used for action proposal
- HF_TOKEN: optional bearer token for protected endpoints
- OPENAI_API_KEY: optional alternative model API key
- The judge sets HF_TOKEN and API_BASE_URL=https://router.huggingface.co/v1

Set `BASE_URL` before running inference locally or against a deployed Space.

## Docker Validation

### 1. Build

```bash
docker build -t traffic-env .
```

### 2. Run

```bash
docker run --rm -p 7860:7860 traffic-env
```

### 3. Test container

```bash
curl http://127.0.0.1:7860/reset
curl "http://127.0.0.1:7860/reset?task_id=hard_multi"
curl -X POST http://127.0.0.1:7860/step -H "Content-Type: application/json" -d '{"action":"KEEP"}'
curl http://127.0.0.1:7860/state
```

## OpenEnv Compliance

openenv.yaml:

```yaml
name: traffic-signal-env
version: 1.0

endpoints:
	reset: /reset
	step: /step
	state: /state
```

Contract requirements:

- /reset returns observation
- /step returns observation, reward, done, info
- /state returns current state snapshot

## Hugging Face Spaces Deployment (Docker SDK)

### 1. Create space

- Go to https://huggingface.co/spaces
- Create Space
- SDK: Docker
- Visibility: Public or Private

### 2. Upload project

- Upload files directly or connect GitHub repository

### 3. Add Secrets in Space settings

- HF_TOKEN=...
- MODEL_API_KEY=...
- API_BASE_URL=https://api.openai.com/v1
- MODEL_NAME=gpt-4o-mini

### 4. Verify deployment

Use the deployed base URL:

```bash
curl https://guuru-dev-traffic-signal-openenv.hf.space/reset
curl "https://guuru-dev-traffic-signal-openenv.hf.space/reset?task_id=medium_dynamic"
curl -X POST https://guuru-dev-traffic-signal-openenv.hf.space/step -H "Content-Type: application/json" -d '{"action":"SWITCH"}'
curl https://guuru-dev-traffic-signal-openenv.hf.space/state
```

## Submission Validator

Run validator helper:

```bash
./validate-submission.sh https://guuru-dev-traffic-signal-openenv.hf.space
```

This performs local checks and remote endpoint checks for submission readiness.

## Example Output Snippet

Example step response:

```json
{
	"observation": {
		"queue_lengths": [2.53, 5.06, 2.54, 7.38],
		"waiting_times": [20.37, 24.22, 13.4, 30.32],
		"current_phase": 0,
		"time_in_phase": 1
	},
	"reward": -51.06,
	"done": false,
	"info": {
		"throughput": 23,
		"avg_wait": 22.07,
		"score": 1.0,
		"task_id": "easy_fixed"
	}
}
```

## Final Checklist

- [x] /reset returns 200
- [x] /reset?task_id=easy_fixed|medium_dynamic|hard_multi switches tasks
- [x] /step works with valid action
- [x] invalid action rejected
- [x] Docker image builds
- [x] Docker container serves API
- [x] inference.py runs and prints step outputs
- [x] 3 tasks exist (easy/medium/hard)
- [x] grader returns values in 0.0 to 1.0

## GitHub Push Workflow

Use this minimal safe workflow before pushing:

```bash
git status --short
git add README.md
git commit -m "docs: update README with HF link and full run/test guide"
git push origin main
```

Safety reminders:

- Never commit real secrets to the repository.
- Keep API tokens only in local env vars or HF Space Secrets.
- Ensure .env and virtualenv folders stay gitignored.
