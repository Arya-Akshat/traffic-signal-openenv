from __future__ import annotations

import os
import json
import argparse
from typing import Any, cast

import requests  # type: ignore[import-untyped]
from openai import OpenAI
# ---------------------------------------------------------------------------
# Section 1: Constants & Environment Config (Independent from Server)
# ---------------------------------------------------------------------------

MIN_HOLD_STEPS = 3
INTERSECTIONS = ("NW", "NE", "SW", "SE")

ROUTES = {
    ("NW", 3): ("NE", 3),
    ("NW", 0): ("SW", 0),
    ("NE", 1): ("NW", 1),
    ("NE", 0): ("SE", 0),
    ("SW", 3): ("SE", 3),
    ("SW", 2): ("NW", 2),
    ("SE", 1): ("SW", 1),
    ("SE", 2): ("NE", 2),
}

NODE_PERSONALITIES = {
    "NW": {"role": "corridor_entry", "queue": 1.25, "wait": 0.9, "throughput": 1.0, "downstream": 1.35, "emergency": 0.7},
    "NE": {"role": "bottleneck", "queue": 0.95, "wait": 1.0, "throughput": 0.9, "downstream": 1.7, "emergency": 0.8},
    "SW": {"role": "emergency_prone", "queue": 1.0, "wait": 1.15, "throughput": 0.95, "downstream": 1.0, "emergency": 2.3},
    "SE": {"role": "outflow_sink", "queue": 1.1, "wait": 0.95, "throughput": 1.2, "downstream": 1.5, "emergency": 0.9},
}


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
    
    # 1. One-step lookahead scoring
    current_q = float(local_qs[phase])
    # Estimate clearance: base (3.5) + green bonus (4.7) ≈ 8.2 if phase is current, else 0
    # Estimate arrivals: average across lanes ≈ 4.0
    expected_clearance = 8.2 if phase == current_phase else 0.0
    if obs.get("central_enabled"):
        expected_clearance *= 1.25 # Central coordination improves clearance efficiency
    expected_arrivals = 4.0
    simulated_next_queue = max(0.0, current_q - expected_clearance + expected_arrivals)
    
    queue_term = min(1.0, simulated_next_queue / 18.0) * policy.get("queue_urgency_weight", 1.0) * float(pers.get("queue", 1.0))
    wait = float(local_ws[phase])
    wait_term = min(1.0, wait / 28.0) * float(pers.get("wait", 1.0))
    throughput_term = min(current_q, 7.0) / 7.0 * float(pers.get("throughput", 1.0))

    # 2. Phase memory & oscillation penalty
    switch_penalty = policy.get("switch_penalty", 1.0)
    history = obs.get("phase_history", {}).get(node, [current_phase, current_phase, current_phase])
    oscillation_penalty = 0.0
    if len(history) >= 2 and phase != current_phase:
        # If we switch to 'p', and history was [p, current_phase, p]
        if history[-2] == phase:
            oscillation_penalty = switch_penalty * 1.5

    if phase != current_phase and time_in_phase < MIN_HOLD_STEPS:
        switch_term = -(switch_penalty + 2.4 + oscillation_penalty)
    elif phase != current_phase:
        switch_term = -(0.7 * switch_penalty + oscillation_penalty)
    else:
        switch_term = 0.35 + min(0.25, time_in_phase * 0.04)

    # 3. Fairness constraint & starvation prevention
    fairness_weight = 0.45
    if obs.get("central_enabled"):
        fairness_weight *= 1.8 # Central oversight prioritizes fairness and starvation prevention
    time_since_served = obs.get("time_since_served", {}).get(node, [0.0]*4)[phase]
    fairness_term = fairness_weight * max(0.0, float(time_since_served) - 8.0)

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
        + fairness_term
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
    export_metrics: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run a complete episode either via the server or locally."""
    episode_log: list[dict[str, Any]] = []
    try:
        state = _request_json(
            "POST",
            f"{env_url}/reset",
            headers=headers,
            payload={"task_id": task_id, "central_enabled": central_enabled},
        )
        max_steps = 300
        final_info: dict[str, Any] = {}

        for i in range(max_steps):
            observation = _observation_from_state(state)
            action_payload: dict[str, Any] = {}
            if client is not None:
                action_payload = _action_from_llm(client, observation)
            if not action_payload:
                action_payload = {"local_actions": _rule_based_action(state)}

            result = _request_json("POST", f"{env_url}/step", headers=headers, payload=action_payload)
            if export_metrics:
                episode_log.append({
                    "step": i,
                    "observation": observation,
                    "action": action_payload,
                    "reward": result.get("reward"),
                    "info": result.get("info"),
                    "policy": observation.get("policy")
                })
            state = result
            done = bool(result.get("done", False))
            final_info = result.get("info", {})
            if done:
                break

        return final_info, episode_log

    except Exception:
        from env.traffic_env import TrafficEnv
        env = TrafficEnv(task=task_id)
        obs = env.reset(central_enabled=central_enabled)
        max_steps = env.task_config.max_steps
        final_info = {}
        state_dict: dict[str, Any] = {"observation": obs}

        for i in range(max_steps):
            action_payload = {"local_actions": _rule_based_action(state_dict)}
            obs, reward, done, info = env.step(action_payload)
            if export_metrics:
                episode_log.append({
                    "step": i,
                    "observation": obs,
                    "action": action_payload,
                    "reward": reward,
                    "info": info,
                    "policy": obs.get("policy")
                })
            state_dict = {"observation": obs}
            final_info = info
            if done:
                break

        return final_info, episode_log


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
    parser.add_argument("--export-metrics", action="store_true", help="Export episode metrics to JSON/CSV")
    parser.add_argument("--curriculum", action="store_true", help="Run adaptive curriculum loop")
    args = parser.parse_args()

    env_url = ENV_URL.rstrip("/")
    client = _resolve_client()
    headers = _build_headers()

    if args.curriculum:
        _run_curriculum(env_url, headers, client)
    elif args.compare:
        print(f"\n>>> Running ablation on task [{args.task}] ...")

        print("  [1/2] Central OFF ...")
        info_off, _ = run_episode(env_url, headers, args.task, central_enabled=False, client=client, export_metrics=args.export_metrics)

        print("  [2/2] Central ON  ...")
        info_on, log_on = run_episode(env_url, headers, args.task, central_enabled=True, client=client, export_metrics=args.export_metrics)

        _format_comparison(info_off, info_on)
        if args.export_metrics and log_on:
            _do_export(log_on, args.task)
    else:
        print(f"=== Running Task: {args.task} ===")
        info, log = run_episode(env_url, headers, args.task, central_enabled=True, client=client, export_metrics=args.export_metrics)
        summary = info.get("summary", info)
        for k, v in sorted(summary.items()):
            if k != "active_behaviors_log" and k != "text_obs":
                print(f"  {k}: {v}")
        if args.export_metrics and log:
            _do_export(log, args.task)

def _do_export(log: list[dict[str, Any]], task_id: str) -> None:
    from env.metrics_exporter import (
        export_episode_to_json, export_episode_to_csv,
        export_policy_trace, export_queue_trace
    )
    os.makedirs("metrics", exist_ok=True)
    export_episode_to_json(log, f"metrics/{task_id}_log.json")
    
    csv_log = []
    for entry in log:
        row = {k: v for k, v in entry.items() if not isinstance(v, (dict, list))}
        if "info" in entry:
            row.update({k: v for k, v in entry["info"].items() if not isinstance(v, (dict, list))})
        csv_log.append(row)
    
    export_episode_to_csv(csv_log, f"metrics/{task_id}_metrics.csv")
    export_policy_trace(log, f"metrics/{task_id}_policy.csv")
    export_queue_trace(log, f"metrics/{task_id}_queues.csv")
    print(f"\n>>> Metrics exported to metrics/ directory.")

def _run_curriculum(env_url: str, headers: dict[str, str], client: OpenAI | None) -> None:
    tasks = [
        "easy_fixed", "medium_dynamic", "hard_multi", 
        "corridor_flow", "dynamic_demand", "incident_response", "gridlock_risk"
    ]
    current_idx = 0
    history = []
    
    print("\n" + "="*50)
    print(" ADAPTIVE CURRICULUM RUNNER ".center(50, "="))
    print("="*50)

    while current_idx < len(tasks):
        task = tasks[current_idx]
        print(f"\n>>> Current Level [{current_idx+1}/{len(tasks)}]: {task}")
        info, _ = run_episode(env_url, headers, task, central_enabled=True, client=client)
        score = float(info.get("summary", info).get("final_score", 0.0))
        history.append((task, score))
        
        print(f"    Score: {score:.4f}")
        
        if score > 0.8:
            print("    [!] Performance EXCELLENT. Advancing...")
            current_idx += 1
            if current_idx >= len(tasks):
                print("    [!] REACHED PEAK PERFORMANCE. Curriculum Complete.")
                break
        elif score < 0.4:
            print("    [!] Performance POOR. Stepping down...")
            current_idx = max(0, current_idx - 1)
        else:
            print("    [!] Performance STABLE. Staying at current level.")
            # To avoid infinite loops in stable state, we could advance after N tries
            # but for now we'll just advance if they get > 0.8
            # User didn't specify what to do if stable. I'll just end after 10 episodes total.
            if len(history) >= 10:
                break

    print("\n" + "="*50)
    print(" CURRICULUM PROGRESSION TABLE ".center(50, "-"))
    print("-" * 50)
    print(f"  {'#':<3} | {'Task ID':<20} | {'Final Score':>12}")
    print("-" * 50)
    for i, (t, s) in enumerate(history):
        print(f"  {i+1:<3} | {t:<20} | {s:>12.4f}")
    print("=" * 50 + "\n")



if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback
        traceback.print_exc()
