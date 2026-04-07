from __future__ import annotations

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportMissingImports=false

import os
import json
from typing import Any, cast

import requests  # type: ignore[import-untyped]
from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "https://guuru-dev-traffic-signal-openenv.hf.space")


def log_event(event_type: str, data: dict[str, Any]) -> None:
    import json

    print(f"[{event_type}] " + json.dumps(data))


def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


def _observation_from_state(state: dict[str, Any] | None) -> dict[str, Any]:
    if not state:
        return {}

    observation = state.get("observation")
    if isinstance(observation, dict):
        return cast(dict[str, Any], observation)
    return state


def _rule_based_action(state: dict[str, Any] | None) -> str:
    """Simple heuristic: switch to the lane with the longest queue."""
    if state is None:
        return "KEEP"
    obs = _observation_from_state(state)
    queues = cast(list[float], obs.get("queue_lengths", [0.0, 0.0, 0.0, 0.0]))
    phase = int(obs.get("current_phase", 0))
    time_in_phase = int(obs.get("time_in_phase", 0))
    max_queue_lane = queues.index(max(queues))
    if max_queue_lane != phase and (time_in_phase >= 3 or queues[phase] < 2.0):
        return f"PHASE_{max_queue_lane}"
    return "KEEP"


def _select_action(step_index: int, state: dict[str, Any] | None = None) -> str:
    _ = step_index
    return _rule_based_action(state)


def _resolve_client() -> OpenAI | None:
    missing = []
    if not API_KEY:
        missing.append("OPENAI_API_KEY or HF_TOKEN")

    if missing:
        return None

    try:
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:  # pragma: no cover
        return None


def _action_from_llm(client: OpenAI, observation: dict[str, Any]) -> str:
    action = "KEEP"
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a traffic signal controller."},
                {"role": "user", "content": str(observation)},
            ],
        )
        content = response.choices[0].message.content or ""
        try:
            output = json.loads(content)
            action = str(output.get("action", "KEEP"))
        except Exception:
            pass
    except Exception:
        pass
    return action


def _request_json(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = requests.request(
        method=method,
        url=url,
        headers=headers,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object from {url}, got {type(data).__name__}")
    return cast(dict[str, Any], data)


def run() -> None:
    env_url = ENV_URL.rstrip("/")
    client = _resolve_client()
    task_id = None
    log_event("START", {"task_id": task_id})

    headers = _build_headers()
    try:
        state = _request_json("POST", f"{env_url}/reset", headers=headers)
    except (requests.RequestException, ValueError) as exc:
        log_event("END", {"total_steps": 0, "final_reward": 0.0, "done": False})
        return

    if isinstance(state, dict):
        task_id = state.get("task_id")

    total_score = 0.0
    total_throughput = 0
    steps = 30
    step_count = 0
    last_reward = 0.0
    last_done = False

    for step_index in range(steps):
        observation = _observation_from_state(state)
        if client is not None:
            action = _action_from_llm(client, observation)
        else:
            action = ""
        if not action:
            action = _select_action(step_index, state)
        try:
            result = _request_json(
                "POST",
                f"{env_url}/step",
                headers=headers,
                payload={"action": action},
            )
        except (requests.RequestException, ValueError) as exc:
            break

        state = result
        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", False))
        score = result.get("info", {}).get("score", 0.0)
        throughput = result.get("info", {}).get("throughput", 0)
        total_score += score
        total_throughput += throughput
        step_count = step_index + 1
        last_reward = reward
        last_done = done
        log_event(
            "STEP",
            {
                "step": step_count,
                "action": action,
                "reward": reward,
                "done": done,
            },
        )

    _ = total_score
    _ = total_throughput
    _ = steps
    log_event(
        "END",
        {
            "total_steps": step_count,
            "final_reward": last_reward,
            "done": last_done,
        },
    )


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:  # pragma: no cover
        pass
