from __future__ import annotations

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportMissingImports=false

import os
from typing import Any, cast

import requests  # type: ignore[import-untyped]


BASE_URL = os.getenv("BASE_URL")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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


def _llm_action(step_index: int, state: dict[str, Any] | None = None) -> str:
    if not OPENAI_API_KEY:
        return _rule_based_action(state)

    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except Exception:
        return _rule_based_action(state)

    client: Any = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
    obs = _observation_from_state(state)
    queues = cast(list[float], obs.get("queue_lengths", [0.0, 0.0, 0.0, 0.0]))
    waits = cast(list[float], obs.get("waiting_times", [0.0, 0.0, 0.0, 0.0]))
    phase = int(obs.get("current_phase", 0))
    time_in_phase = int(obs.get("time_in_phase", 0))

    system_prompt = """You are a traffic signal controller for a 4-lane intersection.
Your goal is to minimize vehicle waiting time and queue lengths while maximizing throughput.

State fields:
- queue_lengths: [N, S, E, W] number of vehicles queued per lane (lower is better)
- waiting_times: [N, S, E, W] estimated wait in seconds per lane (lower is better)
- current_phase: which lane (0-3) currently has the green light
- time_in_phase: how many steps the current phase has been active

Actions available:
- KEEP: keep the current green phase active
- SWITCH: advance to the next phase (N->S->E->W->N), costs a small penalty
- PHASE_0 / PHASE_1 / PHASE_2 / PHASE_3: jump directly to that lane's green phase

Decision heuristics:
- If the current green lane queue is already low (<3) and another lane has high queue (>8), switch
- If time_in_phase > 5 and another lane has longer queues, consider switching
- Avoid switching too frequently — each switch has a small penalty
- Respond with ONLY one of: KEEP, SWITCH, PHASE_0, PHASE_1, PHASE_2, PHASE_3"""

    user_prompt = f"""Step {step_index}
Current green lane: {phase} (active for {time_in_phase} steps)
Queue lengths:  N={queues[0]:.1f}  S={queues[1]:.1f}  E={queues[2]:.1f}  W={queues[3]:.1f}
Waiting times:  N={waits[0]:.1f}  S={waits[1]:.1f}  E={waits[2]:.1f}  W={waits[3]:.1f}
Worst lane: {['N','S','E','W'][queues.index(max(queues))]} (queue={max(queues):.1f})
What action do you choose?"""

    response: Any = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.output_text.strip().upper()
    for action in ["PHASE_0", "PHASE_1", "PHASE_2", "PHASE_3", "SWITCH", "KEEP"]:
        if action in text:
            return action
    return "KEEP"


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
        action = _llm_action(step_index, state)
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
