from __future__ import annotations

from env.types import DemandPulse, TrafficTask
from graders.grader_medium import grade


def get_medium_task(max_steps: int = 200) -> TrafficTask:
    return TrafficTask(
        task_id="medium_dynamic",
        name="Directional surges with corridor pressure",
        max_steps=max_steps,
        seed=21,
        arrival_base=(2.9, 2.4, 2.1, 2.9),
        arrival_jitter=(0.58, 0.48, 0.38, 0.54),
        directional_bias=(1.35, 0.92, 0.82, 1.45),
        initial_queue_bounds=(4.0, 9.0),
        lane_capacity=24.0,
        lane_capacities={
            "NW": (30.0, 30.0, 30.0, 30.0),
            "NE": (30.0, 30.0, 30.0, 30.0),
            "SW": (30.0, 30.0, 30.0, 30.0),
            "SE": (30.0, 30.0, 30.0, 30.0),
        },
        service_base=4.0,
        green_bonus=5.0,
        red_penalty=1.45,
        route_transfer_ratio=0.9,
        transfer_delay_steps=3,
        demand_wave_period=5.5,
        node_demand_scale=(1.32, 1.22, 0.84, 1.12),
        spike_steps=(12, 18, 36, 52, 76, 98, 124, 150, 174),
        spike_multipliers=(1.7, 1.25, 1.15, 1.8),
        demand_pulses=(
            DemandPulse(18, 38, node="NW", lane=3, multiplier=2.2),
            DemandPulse(24, 46, node="NE", lane=0, multiplier=1.9),
            DemandPulse(58, 84, node="SE", lane=1, multiplier=2.1),
            DemandPulse(104, 130, node="NW", lane=0, multiplier=1.7),
        ),
        turn_ratios={
            "NW": (0.27, 0.53, 0.20),
            "NE": (0.24, 0.56, 0.20),
            "SW": (0.25, 0.54, 0.21),
            "SE": (0.23, 0.56, 0.21),
        },
        multi_intersection=True,
        grader=grade,
    )
