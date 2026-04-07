from __future__ import annotations

from env.types import TrafficTask
from graders.grader_medium import grade


def get_medium_task(max_steps: int = 200) -> TrafficTask:
    return TrafficTask(
        task_id="medium_dynamic",
        name="Random traffic spikes",
        max_steps=max_steps,
        seed=21,
        arrival_base=(2.2, 1.9, 1.4, 2.1),
        arrival_jitter=(0.7, 0.5, 0.45, 0.65),
        spike_steps=(15, 30, 60, 90, 120),
        spike_multipliers=(1.8, 1.8, 1.6, 1.7),
        grader=grade,
    )
