from __future__ import annotations

from env.types import DemandPulse, TrafficTask
from graders.grader_hard import grade


def get_hard_task(max_steps: int = 200) -> TrafficTask:
    return TrafficTask(
        task_id="hard_multi",
        name="Asymmetric surge with emergency and spillback trap",
        max_steps=max_steps,
        seed=99,
        arrival_base=(4.2, 4.6, 3.7, 5.0),
        arrival_jitter=(0.95, 0.85, 0.75, 1.0),
        directional_bias=(1.4, 0.9, 0.8, 1.65),
        initial_queue_bounds=(7.0, 13.0),
        lane_capacity=23.0,
        service_base=3.5,
        green_bonus=4.7,
        red_penalty=1.8,
        route_transfer_ratio=1.0,
        transfer_delay_steps=3,
        demand_wave_period=4.5,
        node_demand_scale=(1.35, 1.45, 0.95, 1.2),
        spike_steps=(8, 14, 22, 29, 37, 45, 54, 63, 71, 80, 91, 104, 118, 134, 151, 167, 182),
        spike_multipliers=(2.15, 1.35, 1.2, 2.25),
        demand_pulses=(
            DemandPulse(6, 24, node="NW", lane=3, multiplier=2.5),
            DemandPulse(10, 26, node="NE", lane=0, multiplier=1.9),
            DemandPulse(18, 40, node="SE", lane=1, multiplier=2.2),
            DemandPulse(32, 54, node="NE", lane=1, multiplier=1.8),
            DemandPulse(48, 76, node="NW", lane=0, multiplier=1.9),
            DemandPulse(66, 88, node="SE", lane=2, multiplier=1.7),
            DemandPulse(96, 132, node="NE", lane=0, multiplier=2.1),
            DemandPulse(120, 160, node="NW", lane=3, multiplier=2.4),
        ),
        emergency_step=28,
        emergency_lane=1,
        emergency_multiplier=4.5,
        multi_intersection=True,
        grader=grade,
    )
