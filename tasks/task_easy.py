from __future__ import annotations

from env.types import TrafficTask
from graders.grader_easy import grade


def get_easy_task(max_steps: int = 200) -> TrafficTask:
    return TrafficTask(
        task_id="easy_fixed",
        name="Fixed demand baseline",
        max_steps=max_steps,
        seed=7,
        arrival_base=(2.0, 1.5, 1.0, 1.8),
        arrival_jitter=(0.1, 0.1, 0.05, 0.08),
        grader=grade,
    )
