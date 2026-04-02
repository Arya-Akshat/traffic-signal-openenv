from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrafficTask:
    task_id: str
    name: str
    max_steps: int
    seed: int
    arrival_base: tuple[float, float, float, float]
    arrival_jitter: tuple[float, float, float, float]
    spike_steps: tuple[int, ...] = ()
    spike_multipliers: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    emergency_step: int | None = None
    emergency_lane: int | None = None
    emergency_multiplier: float = 1.0
    multi_intersection: bool = False
