from __future__ import annotations

from env.types import TrafficTask
from graders.grader_hard import grade


def get_hard_task(max_steps: int = 200) -> TrafficTask:
    return TrafficTask(
        task_id="hard_multi",
        name="Multi-intersection with emergency vehicles",
        max_steps=max_steps,
        seed=99,
        arrival_base=(4.4, 4.0, 4.2, 3.8),
        arrival_jitter=(1.2, 1.15, 1.1, 1.05),
        spike_steps=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150),
        spike_multipliers=(2.8, 3.0, 2.9, 2.7),
        emergency_step=18,
        emergency_lane=1,
        emergency_multiplier=4.5,
        multi_intersection=True,
        grader=grade,
    )
