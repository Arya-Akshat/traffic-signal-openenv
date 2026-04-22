from __future__ import annotations
from env.types import TrafficTask

# corridor_flow: Optimized for testing "green wave" synchronization. 
# Features heavily biased horizontal demand to demonstrate the effectiveness of corridor coordination.
def corridor_grader(summary: dict) -> float:
    # Reward for high sync score and low wait times on horizontal lanes
    sync = summary.get("corridor_sync_score", 0.0)
    wait = summary.get("mean_wait", 99.0)
    return max(0.01, min(0.99, (sync * 0.7) + (1.0 - (wait / 40.0)) * 0.3))

def get_corridor_task(max_steps: int = 150) -> TrafficTask:
    return TrafficTask(
        task_id="corridor_flow",
        name="Heavily biased horizontal demand for corridor optimization",
        max_steps=max_steps,
        seed=123,
        arrival_base=(4.2, 4.6, 3.7, 5.0),
        arrival_jitter=(0.5, 0.5, 0.5, 0.5),
        directional_bias=(0.5, 3.0, 0.5, 3.0),
        multi_intersection=True,
        total_priority_budget=5.0,
        grader=corridor_grader
    )
