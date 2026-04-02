from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ActionType = Literal["KEEP", "SWITCH"]


class StepRequest(BaseModel):
    action: ActionType = Field(..., description="KEEP or SWITCH")


class Observation(BaseModel):
    queue_lengths: list[float]
    waiting_times: list[float]
    current_phase: int
    time_in_phase: int


class ResetResponse(BaseModel):
    observation: Observation


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
