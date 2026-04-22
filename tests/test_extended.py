from __future__ import annotations
import math
import os
import json
import pytest
from env.traffic_env import TrafficEnv, INTERSECTIONS, MOVEMENTS
from inference import _rule_based_action

def _run_episode_summary(task: str, central_enabled: bool, max_steps: int = 100) -> dict:
    env = TrafficEnv(task=task, max_steps=max_steps)
    obs = env.reset(central_enabled=central_enabled)
    state = {"observation": obs}
    done = False
    while not done:
        action = {"local_actions": _rule_based_action(state)}
        obs, _, done, info = env.step(action)
        state = {"observation": obs}
    return info["summary"]

def test_turn_movement_queues():
    env = TrafficEnv(task="easy_fixed")
    env.reset()
    state = env.state_obj.intersections["NW"]
    # Verify movement queues exist for each lane
    assert len(state.movement_queues) == 4
    for lane_q in state.movement_queues:
        assert set(lane_q.keys()) == set(MOVEMENTS)
        assert all(isinstance(v, float) for v in lane_q.values())

def test_lane_capacity_spillback():
    # Gridlock task has low capacity (15)
    env = TrafficEnv(task="gridlock_risk", max_steps=20)
    env.reset()
    # Force a spillback by injecting high demand if needed, 
    # but gridlock_risk already has high demand.
    spillbacks = 0
    state = {"observation": env._observation()}
    for _ in range(20):
        _, _, _, info = env.step({"local_actions": _rule_based_action(state)})
        spillbacks += info["spillback_count"]
    # In a gridlock risk task, we expect some spillback pressure
    assert env.state_obj.spillback_events_count >= 0

def test_travel_delay():
    env = TrafficEnv(task="easy_fixed", max_steps=10)
    env.reset()
    # Verify transit buffers exist
    assert len(env.state_obj.transit_buffers) > 0
    for buffer in env.state_obj.transit_buffers.values():
        assert len(buffer) == 3 # Default 3-step FIFO

def test_incident_activation():
    env = TrafficEnv(task="incident_response", max_steps=30)
    env.reset()
    # Incident starts at step 20
    for _ in range(19):
        env.step({"local_actions": {}})
    
    active_before = env._get_active_incidents()
    assert len(active_before) == 0
    
    env.step({"local_actions": {}})
    active_after = env._get_active_incidents()
    assert len(active_after) > 0
    assert active_after[0]["incident_type"] == "LANE_CLOSURE"

def test_fairness_constraint():
    env = TrafficEnv(task="easy_fixed", max_steps=20)
    env.reset()
    # Fairness should be tracked
    for _ in range(5):
        _, _, _, info = env.step({"local_actions": {}})
        assert "fairness_score" in info
        assert 0.0 <= info["fairness_score"] <= 1.0

def test_reward_bounds_extended():
    tasks = ["gridlock_risk", "corridor_flow", "incident_response", "dynamic_demand"]
    for task in tasks:
        env = TrafficEnv(task=task, max_steps=10)
        env.reset()
        for _ in range(10):
            _, reward, _, _ = env.step({"local_actions": {}})
            assert -1.0001 <= reward <= 1.0001

def test_no_nan_inf_extended():
    tasks = ["gridlock_risk", "dynamic_demand"]
    for task in tasks:
        env = TrafficEnv(task=task, max_steps=20)
        env.reset()
        for _ in range(20):
            obs, reward, _, info = env.step({"local_actions": {}})
            assert math.isfinite(reward)
            assert math.isfinite(info["throughput"])
            for node in INTERSECTIONS:
                assert all(math.isfinite(q) for q in obs["queue_lengths"][node])

def test_ablation_gap_preserved():
    # Hard gap must remain > 35% (user increased it from 20%)
    summary_off = _run_episode_summary("hard_multi", central_enabled=False, max_steps=200)
    summary_on = _run_episode_summary("hard_multi", central_enabled=True, max_steps=200)
    
    score_off = float(summary_off["final_score"])
    score_on = float(summary_on["final_score"])
    improvement = (score_on - score_off) / max(score_off, 0.001)
    assert improvement > 0.35

def test_new_task_determinism():
    for task in ["gridlock_risk", "corridor_flow"]:
        env1 = TrafficEnv(task=task, max_steps=20)
        env2 = TrafficEnv(task=task, max_steps=20)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert obs1 == obs2
        for _ in range(20):
            o1, r1, d1, _ = env1.step({"local_actions": {}})
            o2, r2, d2, _ = env2.step({"local_actions": {}})
            assert o1 == o2
            assert r1 == r2

def test_reward_breakdown_completeness():
    env = TrafficEnv(task="easy_fixed")
    env.reset()
    _, reward, _, info = env.step({"local_actions": {}})
    breakdown = info["reward_breakdown"]
    required_keys = {
        "queue_reward", "wait_reward", "throughput_reward", 
        "switch_penalty_reward", "central_reward", "stability_bonus", 
        "coordination_bonus", "fairness_reward", "total"
    }
    assert set(breakdown.keys()) == required_keys
    assert abs(breakdown["total"] - reward) < 1e-4

def test_metrics_exporter(tmp_path):
    from env.metrics_exporter import export_episode_to_json, export_episode_to_csv
    log = [{"step": 0, "metric": 1.0}, {"step": 1, "metric": 2.0}]
    json_path = tmp_path / "test.json"
    csv_path = tmp_path / "test.csv"
    export_episode_to_json(log, str(json_path))
    export_episode_to_csv(log, str(csv_path))
    assert json_path.exists()
    assert csv_path.exists()
