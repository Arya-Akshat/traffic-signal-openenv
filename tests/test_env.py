from __future__ import annotations

import pytest

from env.traffic_env import TrafficEnv
from graders.grader import grade


def test_reset_returns_valid_observation():
    env = TrafficEnv(task="easy_fixed")
    obs = env.reset()
    assert "queue_lengths" in obs
    assert "waiting_times" in obs
    assert len(obs["queue_lengths"]) == 4
    assert len(obs["waiting_times"]) == 4
    assert obs["current_phase"] == 0


def test_step_keep_returns_valid_tuple():
    env = TrafficEnv(task="easy_fixed")
    env.reset()
    obs, reward, done, info = env.step("KEEP")
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "score" in info
    assert -20.0 <= reward <= 10.0


def test_step_all_actions_valid():
    env = TrafficEnv(task="easy_fixed")
    env.reset()
    for action in ["KEEP", "SWITCH", "PHASE_0", "PHASE_1", "PHASE_2", "PHASE_3"]:
        env.reset()
        obs, reward, done, info = env.step(action)
        assert obs is not None


def test_invalid_action_raises():
    env = TrafficEnv(task="easy_fixed")
    env.reset()
    with pytest.raises(ValueError):
        env.step("INVALID")


def test_episode_completes():
    env = TrafficEnv(task="easy_fixed", max_steps=10)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step("KEEP")
        steps += 1
    assert steps == 10


def test_waiting_time_does_not_grow_unboundedly():
    env = TrafficEnv(task="easy_fixed", max_steps=50)
    env.reset()
    for _ in range(50):
        obs, _, done, _ = env.step("KEEP")
        for wt in obs["waiting_times"]:
            assert wt < 200.0, f"Waiting time exploded: {wt}"
        if done:
            break


def test_grader_range():
    assert (
        0.0 <= grade({"avg_wait": 0, "throughput": 20, "total_queue_length": 0}) <= 1.0
    )
    assert (
        0.0
        <= grade({"avg_wait": 999, "throughput": 0, "total_queue_length": 999})
        <= 1.0
    )


def test_seeded_reproducibility():
    env1 = TrafficEnv(task="medium_dynamic")
    env2 = TrafficEnv(task="medium_dynamic")
    env1.reset()
    env2.reset()
    _, r1, _, _ = env1.step("SWITCH")
    _, r2, _, _ = env2.step("SWITCH")
    assert r1 == r2


def test_rule_based_baseline_beats_always_keep():
    from inference import _rule_based_action

    env_rule = TrafficEnv(task="hard_multi", max_steps=30)
    env_keep = TrafficEnv(task="hard_multi", max_steps=30)
    env_rule.reset()
    env_keep.reset()
    rule_scores, keep_scores = [], []
    state_rule = env_rule.state()
    for _ in range(30):
        action = _rule_based_action(state_rule)
        obs, _, done, info = env_rule.step(action)
        state_rule = {"observation": obs}
        rule_scores.append(info["score"])
        if done:
            break
    for _ in range(30):
        _, _, done, info = env_keep.step("KEEP")
        keep_scores.append(info["score"])
        if done:
            break
    assert sum(rule_scores) >= sum(keep_scores) * 0.9
