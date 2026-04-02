from __future__ import annotations

from env.types import TrafficTask


def get_hard_task(max_steps: int = 200) -> TrafficTask:
    return TrafficTask(
        task_id="hard_multi",
        name="Multi-intersection with emergency vehicles",
        max_steps=max_steps,
        seed=99,
        arrival_base=(2.8, 2.4, 2.6, 2.1),
        arrival_jitter=(0.9, 0.85, 0.8, 0.75),
        spike_steps=(20, 45, 70, 110, 150),
        spike_multipliers=(2.0, 2.2, 2.1, 2.0),
        emergency_step=42,
        emergency_lane=1,
        emergency_multiplier=3.0,
        multi_intersection=True,
    )
