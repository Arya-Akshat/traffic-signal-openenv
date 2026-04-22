from __future__ import annotations

from env.types import DemandPulse, TrafficTask
from graders.grader_easy import grade


def get_easy_task(max_steps: int = 200) -> TrafficTask:
    return TrafficTask(
        task_id="easy_fixed",
        name="Balanced commuter baseline",
        max_steps=max_steps,
        seed=7,
        arrival_base=(1.7, 1.4, 1.5, 1.3),
        arrival_jitter=(0.12, 0.08, 0.1, 0.08),
        directional_bias=(1.05, 0.95, 1.0, 1.0),
        initial_queue_bounds=(2.0, 6.0),
        lane_capacity=22.0,
        lane_capacities={
            "NW": (40.0, 40.0, 40.0, 40.0),
            "NE": (40.0, 40.0, 40.0, 40.0),
            "SW": (40.0, 40.0, 40.0, 40.0),
            "SE": (40.0, 40.0, 40.0, 40.0),
        },
        service_base=4.5,
        green_bonus=5.5,
        red_penalty=1.2,
        route_transfer_ratio=0.82,
        transfer_delay_steps=2,
        demand_wave_period=7.0,
        node_demand_scale=(0.95, 1.0, 0.9, 0.95),
        spike_steps=(40, 90, 140),
        spike_multipliers=(1.2, 1.1, 1.1, 1.15),
        demand_pulses=(
            DemandPulse(55, 65, node="NW", lane=3, multiplier=1.2),
            DemandPulse(80, 88, node="SE", lane=1, multiplier=1.15),
        ),
        turn_ratios={
            "NW": (0.25, 0.55, 0.20),
            "NE": (0.22, 0.58, 0.20),
            "SW": (0.24, 0.56, 0.20),
            "SE": (0.23, 0.57, 0.20),
        },
        grader=grade,
    )
