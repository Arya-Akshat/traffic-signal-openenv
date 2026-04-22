from __future__ import annotations
from env.types import TrafficTask

# dynamic_demand: Demand direction rotates every 25 steps.
# Tests the system's adaptability to shifting traffic patterns (e.g., morning vs. evening rush).
def dynamic_grader(summary: dict) -> float:
    # Reward for consistent throughput across shifts
    tp = summary.get("throughput", 0.0)
    wait = summary.get("mean_wait", 99.0)
    return max(0.01, min(0.99, (tp / 1000.0) + (1.0 - (wait / 50.0))))

def get_dynamic_task(max_steps: int = 150) -> TrafficTask:
    return TrafficTask(
        task_id="dynamic_demand",
        name="Time-varying demand patterns (every 25 steps)",
        max_steps=max_steps,
        seed=2024,
        arrival_base=(4.5, 4.5, 4.5, 4.5),
        arrival_jitter=(0.7, 0.7, 0.7, 0.7),
        directional_bias=(1.0, 1.0, 1.0, 1.0), # Rotated in TrafficSpawner
        multi_intersection=True,
        grader=dynamic_grader
    )
