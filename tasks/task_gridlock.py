from __future__ import annotations
from env.types import TrafficTask

# gridlock_risk: High-stress scenario with 1.8x base demand and reduced lane capacity (15 units).
# Designed to test system stability and throughput management under extreme load.
def gridlock_grader(summary: dict) -> float:
    # Continuous score based on queue management and throughput
    mean_queue = summary.get("mean_queue", 99.0)
    throughput = summary.get("throughput", 0.0)
    score = (throughput / 70.0) * 0.5 + (1.0 - mean_queue / 80.0) * 0.5
    return max(0.01, min(0.99, score))

def get_gridlock_task(max_steps: int = 100) -> TrafficTask:
    return TrafficTask(
        task_id="gridlock_risk",
        name="High-density gridlock risk with constrained capacity",
        max_steps=max_steps,
        seed=42,
        arrival_base=(7.56, 8.28, 6.66, 9.0),
        arrival_jitter=(0.8, 0.8, 0.8, 0.8),
        directional_bias=(1.0, 1.0, 1.0, 1.0),
        lane_capacity=15.0,
        service_base=4.0,
        green_bonus=5.0,
        red_penalty=1.5,
        multi_intersection=True,
        grader=gridlock_grader
    )
