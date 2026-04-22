from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class DemandPulse:
    start_step: int
    end_step: int
    node: str | None = None
    lane: int | None = None
    multiplier: float = 1.0


@dataclass(frozen=True)
class Incident:
    intersection_id: str
    lane_id: int
    incident_type: str
    start_step: int
    duration: int
    severity: float = 1.0


@dataclass(frozen=True)
class TrafficTask:
    task_id: str
    name: str
    max_steps: int
    seed: int
    arrival_base: tuple[float, float, float, float]
    arrival_jitter: tuple[float, float, float, float]
    directional_bias: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    initial_queue_bounds: tuple[float, float] = (3.0, 10.0)
    lane_capacity: float = 24.0
    service_base: float = 4.2
    green_bonus: float = 5.4
    red_penalty: float = 1.4
    route_transfer_ratio: float = 0.9
    transfer_delay_steps: int = 2
    demand_wave_period: float = 5.0
    node_demand_scale: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    spike_steps: tuple[int, ...] = ()
    spike_multipliers: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    demand_pulses: tuple[DemandPulse, ...] = ()
    emergency_step: int | None = None
    emergency_lane: int | None = None
    emergency_multiplier: float = 1.0
    multi_intersection: bool = False
    turn_ratios: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    lane_capacities: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    incidents: tuple[Incident, ...] = ()
    total_priority_budget: float = 3.0
    grader: Callable[[dict], float] | None = None
