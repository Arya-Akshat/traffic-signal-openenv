from __future__ import annotations

import os
import json
import argparse
from typing import Any, cast

import requests  # type: ignore[import-untyped]
from openai import OpenAI
from env.traffic_env import MIN_HOLD_STEPS, NODE_PERSONALITIES, ROUTES


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")


def log_event(event_type: str, data: dict[str, Any]) -> None:
    parts = [f"[{event_type}]"]
    for k, v in data.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts), flush=True)


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


# ---------------------------------------------------------------------------
# Section 2: Structured multi-term phase scoring
# ---------------------------------------------------------------------------

def score_phase(
    phase: int,
    node: str,
    local_qs: list[float],
    local_ws: list[float],
    time_in_phase: int,
    current_phase: int,
    policy: dict[str, float],
    obs: dict[str, Any],
) -> float:
    pers = NODE_PERSONALITIES.get(node, {})
    queue = float(local_qs[phase])
    wait = float(local_ws[phase])

    downstream_queue = 0.0
    if (node, phase) in ROUTES:
        down_node, down_lane = ROUTES[(node, phase)]
        q_dict = obs.get("queue_lengths", {})
        if isinstance(q_dict, dict) and down_node in q_dict:
            downstream_queue = float(q_dict[down_node][down_lane])

    network_queues = obs.get("queue_lengths", {})
    node_load = sum(local_qs)
    mean_load = 0.0
    if isinstance(network_queues, dict) and network_queues:
        mean_load = sum(sum(float(v) for v in lanes) for lanes in network_queues.values()) / len(network_queues)

    queue_term = min(1.0, queue / 18.0) * policy.get("queue_urgency_weight", 1.0) * float(pers.get("queue", 1.0))
    wait_term = min(1.0, wait / 28.0) * float(pers.get("wait", 1.0))
    throughput_term = min(queue, 7.0) / 7.0 * float(pers.get("throughput", 1.0))

    if phase != current_phase and time_in_phase < MIN_HOLD_STEPS:
        switch_term = -(policy.get("switch_penalty", 1.0) + 2.4)
    elif phase != current_phase:
        switch_term = -0.7 * policy.get("switch_penalty", 1.0)
    else:
        switch_term = 0.35 + min(0.25, time_in_phase * 0.04)

    downstream_term = -min(1.4, downstream_queue / 16.0) * float(pers.get("downstream", 1.0)) * policy.get("balance_penalty", 1.0)
    balance_term = -min(1.0, abs(node_load - mean_load) / 26.0) * policy.get("balance_penalty", 1.0) * 0.65

    corridor_term = 0.0
    if phase in {1, 3}:
        corridor_term = 0.55 * (policy.get("corridor_priority", 1.0) - 1.0) * (1.15 if node in {"NW", "NE"} else 0.9)

    emergency_term = 0.0
    if node == "SW" and phase == 1:
        emergency_term = policy.get("emergency_boost", 0.0) * float(pers.get("emergency", 1.0))

    return (
        1.55 * queue_term
        + 1.05 * wait_term
        + 0.9 * throughput_term
        + switch_term
        + corridor_term
        + emergency_term
        + downstream_term
        + balance_term
    )


def _rule_based_action(state: dict[str, Any] | None) -> dict[str, str]:
    """Structured local controller: evaluates all 4 phases per intersection
    using score_phase() and selects the highest-scoring one."""
    if state is None:
        return {}
    obs = _observation_from_state(state)
    q_dict = obs.get("queue_lengths", {})
    w_dict = obs.get("waiting_times", {})
    p_dict = obs.get("current_phase", {})
    t_dict = obs.get("time_in_phase", {})
    policy = obs.get("policy", {})

    local_actions: dict[str, str] = {}

    # Handle legacy flat format
    if isinstance(q_dict, list):
        q_dict = {"NW": q_dict}
        w_dict = {"NW": w_dict}
        p_dict = {"NW": p_dict}
        t_dict = {"NW": t_dict}

    for node, queues in q_dict.items():
        phase = int(p_dict.get(node, 0)) if isinstance(p_dict, dict) else int(p_dict)
        tip = int(t_dict.get(node, 0)) if isinstance(t_dict, dict) else int(t_dict)
        waits = w_dict.get(node, [0.0] * 4) if isinstance(w_dict, dict) else [0.0] * 4

        best_score = -9999.0
        best_phase = phase

        for p in range(4):
            s = score_phase(p, node, queues, waits, tip, phase, policy, obs)
            if s > best_score:
                best_score = s
                best_phase = p

        if best_phase != phase:
            local_actions[node] = f"PHASE_{best_phase}"
        else:
            local_actions[node] = "KEEP"

    return local_actions


def _resolve_client() -> OpenAI | None:
    if not HF_TOKEN:
        return None
    try:
        return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception:
        return None


def _action_from_llm(client: OpenAI, observation: dict[str, Any]) -> dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a traffic signal controller for a 2x2 grid. Output JSON with 'local_actions' (dict of node->action strings)."},
                {"role": "user", "content": json.dumps(observation)},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        output = json.loads(content)
        if "local_actions" in output:
            return {"local_actions": output["local_actions"], "central_action": output.get("central_action")}
    except Exception:
        pass
    return {}


def _request_json(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = requests.request(method=method, url=url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object from {url}, got {type(data).__name__}")
    return cast(dict[str, Any], data)


def run_episode(
    env_url: str,
    headers: dict[str, str],
    task_id: str,
    central_enabled: bool,
    client: OpenAI | None,
) -> dict[str, Any]:
    """Run a complete episode either via the server or locally."""
    try:
        state = _request_json(
            "POST",
            f"{env_url}/reset",
            headers=headers,
            payload={"task_id": task_id, "central_enabled": central_enabled},
        )
        max_steps = 300  # generous upper bound; episode terminates on done
        final_info: dict[str, Any] = {}

        for _ in range(max_steps):
            observation = _observation_from_state(state)
            action_payload: dict[str, Any] = {}
            if client is not None:
                action_payload = _action_from_llm(client, observation)
            if not action_payload:
                action_payload = {"local_actions": _rule_based_action(state)}

            result = _request_json("POST", f"{env_url}/step", headers=headers, payload=action_payload)
            state = result
            done = bool(result.get("done", False))
            final_info = result.get("info", {})
            if done:
                break

        return final_info

    except Exception:
        # Fall back to local environment if server is unavailable
        from env.traffic_env import TrafficEnv

        env = TrafficEnv(task=task_id)
        obs = env.reset(central_enabled=central_enabled)
        max_steps = env.task_config.max_steps  # run full episode
        final_info = {}
        state_dict: dict[str, Any] = {"observation": obs}

        for _ in range(max_steps):
            action_payload = {"local_actions": _rule_based_action(state_dict)}
            obs, reward, done, info = env.step(action_payload)
            state_dict = {"observation": obs}
            final_info = info
            if done:
                break

        return final_info


# ---------------------------------------------------------------------------
# Section 6: Demo comparison output
# ---------------------------------------------------------------------------

def _format_comparison(info_off: dict[str, Any], info_on: dict[str, Any]) -> None:
    """Print a clean, compelling ablation comparison table."""

    def g(src: dict, k: str) -> float:
        """Get a metric from summary or top-level info."""
        summary = src.get("summary", {})
        if k in summary:
            val = summary[k]
        elif k in src:
            val = src[k]
        else:
            val = 0.0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    rows = [
        ("Final Score",        "final_score",          "+"),
        ("Episode Reward",     "episode_reward",       "+"),
        ("Total Throughput",   "throughput",            "+"),
        ("Mean Queue",         "mean_queue",            "-"),
        ("Mean Wait",          "mean_wait",             "-"),
        ("Imbalance",          "imbalance",             "-"),
        ("Spillback Events",   "spillback_events_count","-"),
        ("Emergency Delay",    "emergency_delay",       "-"),
        ("Corridor Sync",      "corridor_sync_score",   "+"),
    ]

    W = 82
    print("\n" + "=" * W)
    print("  ABLATION COMPARISON REPORT".center(W))
    print("=" * W)
    print(f"  {'Metric':<22} | {'Central OFF':>13} | {'Central ON':>13} | {'Δ':>10}")
    print("-" * W)

    for label, key, polarity in rows:
        v_off = g(info_off, key)
        v_on = g(info_on, key)

        if polarity == "+":
            diff = v_on - v_off
        else:
            diff = v_off - v_on  # positive = improvement

        if abs(v_off) > 0.0001:
            pct = (diff / abs(v_off)) * 100
        else:
            pct = 0.0

        sign = "+" if diff >= 0 else ""
        print(f"  {label:<22} | {v_off:>13.3f} | {v_on:>13.3f} | {sign}{pct:>7.1f}%")

    print("=" * W)

    # One-line verdict
    score_off = g(info_off, "final_score")
    score_on = g(info_on, "final_score")
    wait_off = g(info_off, "mean_wait")
    wait_on = g(info_on, "mean_wait")

    score_pct = ((score_on - score_off) / max(score_off, 0.001)) * 100
    wait_diff = wait_off - wait_on

    print(f"\n  ✦ Central controller improved final score by {score_pct:+.1f}%"
          f" and reduced mean wait by {wait_diff:.1f} steps.\n")


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="hard_multi", help="Task ID to run")
    parser.add_argument("--compare", action="store_true", help="Run ablation comparison (central ON vs OFF)")
    args = parser.parse_args()

    env_url = ENV_URL.rstrip("/")
    client = _resolve_client()
    headers = _build_headers()

    if args.compare:
        print(f"\n>>> Running ablation on task [{args.task}] ...")

        print("  [1/2] Central OFF ...")
        info_off = run_episode(env_url, headers, args.task, central_enabled=False, client=client)

        print("  [2/2] Central ON  ...")
        info_on = run_episode(env_url, headers, args.task, central_enabled=True, client=client)

        _format_comparison(info_off, info_on)
    else:
        print(f"=== Running Task: {args.task} ===")
        info = run_episode(env_url, headers, args.task, central_enabled=True, client=client)
        summary = info.get("summary", info)
        for k, v in sorted(summary.items()):
            if k != "active_behaviors_log" and k != "text_obs":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback
        traceback.print_exc()
