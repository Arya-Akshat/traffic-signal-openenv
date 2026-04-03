from __future__ import annotations

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportMissingImports=false

import os
from typing import Any, cast

import requests  # type: ignore[import-untyped]


BASE_URL = os.getenv("BASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")


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


def run() -> None:
    if not BASE_URL:
        raise RuntimeError("BASE_URL must be set")

    headers = _build_headers()
    requests.get(f"{BASE_URL}/reset", headers=headers, timeout=30).json()
    state = requests.get(f"{BASE_URL}/state", headers=headers, timeout=30).json()

    total_score = 0.0
    total_throughput = 0
    steps = 30

    for step_index in range(steps):
        action = _select_action(step_index, state)
        response = requests.post(
            f"{BASE_URL}/step",
            json={"action": action},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        state = result
        score = result.get("info", {}).get("score", 0.0)
        throughput = result.get("info", {}).get("throughput", 0)
        total_score += score
        total_throughput += throughput
        print(
            f"step={step_index:02d} action={action:<10} score={score:.4f} throughput={throughput} avg_wait={result.get('info',{}).get('avg_wait',0):.2f}"
        )

    print("\n--- Episode Summary ---")
    print(f"Mean score:       {total_score / steps:.4f}")
    print(f"Total throughput: {total_throughput}")


if __name__ == "__main__":
    run()
