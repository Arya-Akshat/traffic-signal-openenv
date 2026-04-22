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

Traffic Signal OpenEnv is a deterministic traffic-light control environment exposed as an HTTP API. It is designed for OpenEnv-style evaluation where a remote controller sends actions and receives observations, rewards, and task scores.

Live deployment: https://guuru-dev-traffic-signal-openenv.hf.space

## What This Project Does

- Simulates a 4-lane intersection abstraction with queue and waiting-time dynamics.
- Supports 3 deterministic task profiles: easy, medium, hard.
- Exposes OpenEnv-compatible endpoints: `/reset`, `/step`, `/state`.
- Computes both per-step reward (for online control) and bounded score (for validation/grading).
- Includes local tests, a rule-based/LLM-capable client, Docker packaging, and a submission validation script.

## End-to-End Architecture

Controller flow:

1. Client calls `POST /reset` (optionally with `task_id`) to initialize an episode.
2. Client calls `POST /step` with one action.
3. API validates action schema with Pydantic.
4. Environment advances one deterministic simulation tick.
5. Environment returns:
     - new observation
     - dense reward
     - done flag
     - info including throughput, avg wait, bounded score, episode aggregates

Main execution chain:

`inference.py` or external controller -> `app/main.py` -> `env/traffic_env.py` -> `tasks/task_*.py` + `graders/*.py`

## Core Simulation Logic

Simulation state tracks:

- `queue_lengths`: current queue pressure per lane
- `waiting_times`: lane-level waiting proxy
- `current_phase`: active green lane index 0..3
- `time_in_phase`: consecutive ticks in active phase
- `step_count`, `total_throughput`, `total_wait`, `done`, history records

Per-step algorithm in `env/traffic_env.py`:

1. Validate action (must be one of KEEP/SWITCH/PHASE_0..PHASE_3).
2. Update phase and `time_in_phase`.
3. Compute arrivals using base rates + sinusoidal jitter + optional spike multiplier + optional emergency multiplier.
4. Compute lane service rate from:
     - base capacity
     - whether lane is green
     - switching penalty
     - multi-intersection boost
     - hard-task pressure reduction
5. Update queue and waiting values lane-by-lane.
6. Compute throughput from served demand.
7. Build metrics (`avg_wait`, `total_waiting_time`, `total_queue_length`, `throughput`, `switching_penalty`).
8. Compute reward:

```python
reward = (
        -avg_wait_norm
        - 0.5 * queue_norm
        + throughput_norm * 2.0
        - switching_penalty * 0.1
)
reward = clip(reward, -20.0, 10.0)
```

9. Compute score via task grader, then sanitize score into strict open interval `(0, 1)`:
     - fallback if `None`
     - float conversion guard
     - NaN guard
     - clamp to `0.01`/`0.99` at edges
10. Return `(observation, reward, done, info)`.

## API Contract

### Endpoints

- `GET|POST /reset`
    - Optional task switch using query or JSON body `{"task_id": "..."}`.
    - Returns initial observation and active task id.
- `POST /step`
    - Body: `{"action": "KEEP|SWITCH|PHASE_0..PHASE_3"}`
    - Returns observation, reward, done, info.
- `GET /state`
    - Returns snapshot of current episode-level state and metrics.
- `GET /`
    - Service discovery payload with endpoint list.
- `GET /health`
    - Liveness status.
- `GET /metadata`
    - Basic project metadata.
- `GET /schema`
    - JSON schema for action/observation/state models.
- `POST /mcp`
    - Minimal JSON-RPC style OK response.

### Request/Response Models

Defined in `app/models.py`:

- `StepRequest.action` is a strict `Literal` action set.
- `Observation` includes queue/wait vectors and phase/timer.
- `ResetResponse`, `StepResponse`, and `StateResponse` define contract shape for endpoint outputs.

Note: because `StepRequest` is strongly typed, invalid API actions are rejected at request validation time. The environment itself still has internal action validation when used directly (for example, in direct Python calls outside FastAPI).

## Task Profiles and Determinism

Task factories return immutable `TrafficTask` configs (`env/types.py`):

- `easy_fixed`:
    - lower arrivals
    - minimal jitter
    - stable baseline behavior
- `medium_dynamic`:
    - higher jitter
    - periodic demand spikes at predefined steps
- `hard_multi`:
    - heavy arrivals and jitter
    - many spike steps
    - emergency event lane multiplier
    - multi-intersection tuning enabled

Determinism:

- Each task uses fixed seed (`7`, `21`, `99` respectively).
- `reset()` reseeds RNG, giving repeatable trajectories for same task/action sequence.

## Grading and Score Bounds

Grading pipeline:

- `graders/common.py` implements `compute_score()` and guarded `grade()`.
- Task graders (`graders/grader_easy.py`, `graders/grader_medium.py`, `graders/grader_hard.py`) delegate to common grading with extra safety guards.
- `graders/grader.py` is a generic wrapper used by tests/import convenience.

Score composition in `compute_score()`:

- wait component: lower wait -> higher score
- throughput component: higher throughput -> higher score
- queue component: lower queue -> higher score
- weighted mix:

```python
score = 0.5 * normalized_wait + 0.3 * throughput_score + 0.2 * queue_score
```

Safety behavior:

- catches conversion/compute failures
- handles NaN and infinities
- enforces strict open interval with `0.01 <=clamp<= 0.99` semantics
- includes runtime assertions

## Repository File-by-File Reference (Complete)

This section documents every file currently present in the repository tree.

### Root files

- `.env.example`
    - Sample environment values for task selection, max steps, SUMO toggles, base URL, HF token.
- `.gitignore`
    - Excludes env files, virtualenvs, caches, logs, secrets, editor folders, build artifacts.
- `Dockerfile`
    - Builds API container from Python 3.11 uv image.
    - Adds cache-busting `RUN echo` lines.
    - Copies project, installs runtime dependencies, runs uvicorn on port 7860.
- `inference.py`
    - Baseline controller client.
    - Supports optional OpenAI/HF-router action generation when token/client is available.
    - Falls back to local rule-based policy (`_rule_based_action`) selecting largest queue lane.
    - Iterates all tasks, logs START/STEP/END events.
- `openenv.yaml`
    - OpenEnv descriptor: endpoint paths and task->grader mapping.
- `pyproject.toml`
    - Project metadata, Python requirement, dependencies, setuptools build system.
    - Defines CLI script entrypoint `server = server.app:main`.
- `pytest.ini`
    - Test config (`pythonpath = .`).
- `README.md`
    - This documentation.
- `requirements-dev.txt`
    - Tooling deps: black, ruff, pytest.
- `requirements.txt`
    - Runtime deps: fastapi, uvicorn, pydantic, requests, openai.
- `session.md`
    - Local session notes/history document (not part of runtime execution path).
- `uv.lock`
    - Lockfile with fully resolved transitive package set and hashes.
- `validate-submission.sh`
    - Build/check helper:
        - verifies required files exist
        - builds Docker image
        - calls remote `/reset`, `/step`, `/state`
        - validates basic response contract fields

### App package

- `app/__init__.py`
    - Package marker docstring.
- `app/config.py`
    - `Settings` dataclass reading env vars:
        - `TASK_ID`
        - `MAX_STEPS`
        - `USE_REAL_SUMO`
        - `SUMO_HOME`
- `app/main.py`
    - FastAPI app and endpoint definitions.
    - Creates singleton env instance at import time.
    - Handles reset payload parsing and error translation to HTTP 400.
- `app/models.py`
    - Pydantic contracts for actions, observations, and endpoint responses.

### Env package

- `env/__init__.py`
    - Package marker docstring.
- `env/traffic_env.py`
    - Main deterministic simulator and reward/metric logic.
    - Action constants, task registry, state dataclass, and all transition functions.
- `env/types.py`
    - `TrafficTask` dataclass describing task-level configuration and grader hook.

### Graders package

- `graders/__init__.py`
    - Exposes `grade_easy`, `grade_medium`, `grade_hard` imports.
- `graders/common.py`
    - Shared score calculation and open-interval enforcement.
- `graders/grader.py`
    - Generic wrapper delegating to common grade.
- `graders/grader_easy.py`
    - Easy task wrapper around common grader with safety guards.
- `graders/grader_medium.py`
    - Medium task wrapper around common grader with safety guards.
- `graders/grader_hard.py`
    - Hard task wrapper around common grader with safety guards.
- `graders/__pycache__/__init__.cpython-314.pyc`
    - Compiled bytecode artifact (generated).
- `graders/__pycache__/common.cpython-314.pyc`
    - Compiled bytecode artifact (generated).
- `graders/__pycache__/grader.cpython-314.pyc`
    - Compiled bytecode artifact (generated).
- `graders/__pycache__/grader_easy.cpython-314.pyc`
    - Compiled bytecode artifact (generated).
- `graders/__pycache__/grader_medium.cpython-314.pyc`
    - Compiled bytecode artifact (generated).
- `graders/__pycache__/grader_hard.cpython-314.pyc`
    - Compiled bytecode artifact (generated).

### Server package

- `server/__init__.py`
    - Empty package marker.
- `server/app.py`
    - Alternate app launcher using `uvicorn.run()` and `HOST`/`PORT` env variables.

### Tasks package

- `tasks/__init__.py`
    - Package marker docstring.
- `tasks/task_easy.py`
    - Returns fixed-demand easy task config.
- `tasks/task_medium.py`
    - Returns dynamic-spike medium task config.
- `tasks/task_hard.py`
    - Returns hard task config with emergency + multi-intersection modifiers.

### SUMO assets

- `sumo/config.sumo.cfg`
    - SUMO config pointing to `net.xml` and `route.xml` with `step-length=1`.
- `sumo/net.xml`
    - Minimal network definition placeholder.
- `sumo/route.xml`
    - Vehicle type and route definitions placeholder.

### Tests

- `tests/test_env.py`
    - Covers reset shape, step tuple validity, action handling, invalid action errors,
        episode completion, waiting-time sanity, grader range, seeded reproducibility,
        and baseline rule-policy performance check.

### VS Code config

- `.vscode/settings.json`
    - Local terminal command auto-approval settings for specific commands/patterns.

## Local Development and Validation

### Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

### Run API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Quick endpoint checks

```bash
curl http://127.0.0.1:8000/reset
curl "http://127.0.0.1:8000/reset?task_id=medium_dynamic"
curl -X POST http://127.0.0.1:8000/step -H "Content-Type: application/json" -d '{"action":"SWITCH"}'
curl http://127.0.0.1:8000/state
```

### Run baseline client

```bash
ENV_URL=http://127.0.0.1:8000 python inference.py
```

### Tests and lint

```bash
ruff check .
black --check .
pytest -q
```

Optional auto-fix:

```bash
black .
ruff check . --fix
```

### Docker run

```bash
docker build -t traffic-env .
docker run --rm -p 7860:7860 traffic-env
curl http://127.0.0.1:7860/reset
```

### Submission check

```bash
./validate-submission.sh https://guuru-dev-traffic-signal-openenv.hf.space
```

## Deployment Notes

- Hugging Face Spaces expects port `7860`, already configured.
- Docker image starts with `uvicorn app.main:app` and `PYTHONPATH=/app`.
- `openenv.yaml` maps each task id to a fully-qualified grader function path.

## Practical Extension Points

- Add new task profile:
    1. create new `tasks/task_new.py`
    2. register in `TASK_BUILDERS` inside `env/traffic_env.py`
    3. add task entry in `openenv.yaml`
    4. add tests in `tests/test_env.py`
- Customize reward in `_reward()` for policy behavior changes.
- Swap grading weights or normalizers in `graders/common.py`.
- Replace baseline logic in `inference.py` with custom controller policy.

## Known Technical Notes

- SUMO files are present and configured, but current environment dynamics are computed in Python abstraction rather than direct runtime SUMO process control.
- Generated `.pyc` files are currently present under `graders/__pycache__`; they are build artifacts and generally should not be committed.
