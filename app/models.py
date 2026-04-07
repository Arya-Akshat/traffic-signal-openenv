from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ActionType = Literal["KEEP", "SWITCH", "PHASE_0", "PHASE_1", "PHASE_2", "PHASE_3"]


class StepRequest(BaseModel):
    action: ActionType = Field(..., description="KEEP, SWITCH, or PHASE_0 to PHASE_3")


class Observation(BaseModel):
    queue_lengths: list[float]
    waiting_times: list[float]
    current_phase: int
    time_in_phase: int


class ResetResponse(BaseModel):
    observation: Observation
    task_id: str


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


class StateResponse(BaseModel):
    task_id: str
    step_count: int
    observation: Observation
    metrics: dict
    episode_throughput: float
    episode_avg_wait: float
