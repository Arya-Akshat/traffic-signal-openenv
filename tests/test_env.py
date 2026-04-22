from __future__ import annotations

import math

from fastapi.testclient import TestClient

from app.main import app
from env.traffic_env import DEFAULT_POLICY, INTERSECTIONS, TrafficEnv
from graders.grader import grade
from inference import _rule_based_action


REQUIRED_SUMMARY_KEYS = {
    "episode_reward",
    "mean_queue",
    "mean_wait",
    "throughput",
    "imbalance",
    "spillback_count",
    "emergency_delay",
    "central_enabled",
    "task_id",
    "policy_stability",
    "final_score",
    "step_count",
    "corridor_sync_score",
    "active_behaviors_log",
    "text_obs",
    "travel_time_mean",
    "travel_time_variance",
    "throughput_efficiency",
    "fairness_score",
    "stability_index",
    "recovery_time",
}

REQUIRED_STEP_INFO_KEYS = {
    "throughput",
    "avg_wait",
    "score",
    "task_id",
    "episode_throughput",
    "episode_avg_wait",
    "corridor_sync_score",
    "active_behaviors_log",
    "policy_stability",
    "spillback_count",
    "emergency_delay",
}


def _run_episode(task: str, central_enabled: bool, max_steps: int = 60) -> dict:
    env = TrafficEnv(task=task, max_steps=max_steps)
    observation = env.reset(central_enabled=central_enabled)
    state = {"observation": observation}
    info: dict = {}
    done = False
    while not done:
        action = {"local_actions": _rule_based_action(state)}
        observation, reward, done, info = env.step(action)
        assert -1.0 <= reward <= 1.0
        state = {"observation": observation}
    return info["summary"]


def test_reset_returns_network_observation():
    env = TrafficEnv(task="easy_fixed")
    obs = env.reset()
    assert set(obs["queue_lengths"]) == set(INTERSECTIONS)
    assert set(obs["waiting_times"]) == set(INTERSECTIONS)
    assert obs["current_phase"]["NW"] == 0
    assert set(obs["policy"]) == set(DEFAULT_POLICY)
    assert "text_obs" in obs


def test_state_contains_policy_and_central_flag():
    env = TrafficEnv(task="medium_dynamic")
    env.reset(central_enabled=True)
    snapshot = env.state()
    assert snapshot["central_enabled"] is True
    assert snapshot["observation"]["policy"]["queue_urgency_weight"] >= 0.5
    assert "policy=" in snapshot["observation"]["text_obs"]


def test_step_is_deterministic_for_same_seed_and_actions():
    env_one = TrafficEnv(task="medium_dynamic", max_steps=12)
    env_two = TrafficEnv(task="medium_dynamic", max_steps=12)
    obs_one = env_one.reset(central_enabled=True)
    obs_two = env_two.reset(central_enabled=True)
    state_one = {"observation": obs_one}
    state_two = {"observation": obs_two}

    rewards_one: list[float] = []
    rewards_two: list[float] = []
    for _ in range(12):
        action_one = {"local_actions": _rule_based_action(state_one)}
        action_two = {"local_actions": _rule_based_action(state_two)}
        obs_one, reward_one, done_one, _ = env_one.step(action_one)
        obs_two, reward_two, done_two, _ = env_two.step(action_two)
        rewards_one.append(reward_one)
        rewards_two.append(reward_two)
        state_one = {"observation": obs_one}
        state_two = {"observation": obs_two}
        assert obs_one == obs_two
        assert done_one == done_two

    assert rewards_one == rewards_two


def test_grader_returns_open_interval_score():
    score = grade(
        {
            "mean_wait": 8.0,
            "mean_queue": 24.0,
            "throughput": 30.0,
            "imbalance": 6.0,
            "spillback_count": 3.0,
            "emergency_delay": 20.0,
            "corridor_sync_score": 0.6,
            "policy_stability": 0.8,
        }
    )
    assert 0.0 < score < 1.0


def test_episode_summary_uses_required_schema():
    summary = _run_episode("easy_fixed", central_enabled=True, max_steps=12)
    assert set(summary) == REQUIRED_SUMMARY_KEYS


def test_fastapi_reset_step_state_contracts():
    client = TestClient(app)
    reset_response = client.post("/reset", json={"task_id": "medium_dynamic", "central_enabled": True})
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["task_id"] == "medium_dynamic"
    assert reset_payload["central_enabled"] is True

    action = {"local_actions": {node: "KEEP" for node in INTERSECTIONS}}
    step_response = client.post("/step", json=action)
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert set(step_payload) == {"observation", "reward", "done", "info"}

    state_response = client.get("/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["central_enabled"] is True
    assert "policy" in state_payload["observation"]
    assert REQUIRED_STEP_INFO_KEYS.issubset(step_payload["info"])


def test_central_policy_changes_under_hard_pressure():
    env = TrafficEnv(task="hard_multi", max_steps=25)
    obs = env.reset(central_enabled=True)
    state = {"observation": obs}
    for _ in range(10):
        obs, _, done, _ = env.step({"local_actions": _rule_based_action(state)})
        state = {"observation": obs}
        if done:
            break

    policy = obs["policy"]
    assert policy != DEFAULT_POLICY
    assert policy["emergency_boost"] > 0.0 or policy["corridor_priority"] != DEFAULT_POLICY["corridor_priority"]


def test_rule_based_controller_beats_keep_on_hard_task():
    env_rule = TrafficEnv(task="hard_multi", max_steps=40)
    env_keep = TrafficEnv(task="hard_multi", max_steps=40)

    obs_rule = env_rule.reset(central_enabled=True)
    state_rule = {"observation": obs_rule}
    done_rule = False
    info_rule: dict = {}
    while not done_rule:
        obs_rule, _, done_rule, info_rule = env_rule.step({"local_actions": _rule_based_action(state_rule)})
        state_rule = {"observation": obs_rule}

    env_keep.reset(central_enabled=True)
    done_keep = False
    info_keep: dict = {}
    while not done_keep:
        _, _, done_keep, info_keep = env_keep.step("KEEP")

    assert info_rule["summary"]["final_score"] > info_keep["summary"]["final_score"]


def test_hard_task_central_ablation_gap_exceeds_twenty_percent():
    summary_off = _run_episode("hard_multi", central_enabled=False, max_steps=200)
    summary_on = _run_episode("hard_multi", central_enabled=True, max_steps=200)

    score_off = float(summary_off["final_score"])
    score_on = float(summary_on["final_score"])
    improvement = (score_on - score_off) / max(score_off, 0.001)
    assert improvement > 0.20


def test_step_info_contains_training_keys_and_finite_values():
    env = TrafficEnv(task="hard_multi", max_steps=12)
    obs = env.reset(central_enabled=True)
    state = {"observation": obs}
    for _ in range(5):
        obs, reward, done, info = env.step({"local_actions": _rule_based_action(state)})
        assert REQUIRED_STEP_INFO_KEYS.issubset(info)
        numeric_keys = {
            "throughput",
            "avg_wait",
            "score",
            "episode_throughput",
            "episode_avg_wait",
            "corridor_sync_score",
            "policy_stability",
            "spillback_count",
            "emergency_delay",
        }
        for key in numeric_keys:
            assert math.isfinite(float(info[key]))
        assert -1.0 <= reward <= 1.0
        state = {"observation": obs}
        if done:
            break


def test_text_obs_is_compact_and_non_json():
    env = TrafficEnv(task="hard_multi", max_steps=20)
    obs = env.reset(central_enabled=True)
    text_obs = obs["text_obs"]
    assert isinstance(text_obs, str)
    assert len(text_obs) < 1200
    assert text_obs.count("\n") >= 4
    assert '"queue_lengths"' not in text_obs
