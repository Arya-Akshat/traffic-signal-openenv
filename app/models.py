from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal["KEEP", "SWITCH", "PHASE_0", "PHASE_1", "PHASE_2", "PHASE_3"]


class StepRequest(BaseModel):
    action: Optional[ActionType] = Field(None, description="KEEP, SWITCH, or PHASE_0 to PHASE_3 (Legacy)")
    local_actions: Optional[dict[str, ActionType]] = Field(None, description="Actions per intersection")
    central_action: Optional[dict[str, float]] = Field(None, description="Updates to policy vector")


class Observation(BaseModel):
    queue_lengths: dict[str, list[float]] | list[float]
    waiting_times: dict[str, list[float]] | list[float]
    current_phase: dict[str, int] | int
    time_in_phase: dict[str, int] | int
    policy: Optional[dict[str, float]] = None
    text_obs: Optional[str] = None


class ResetResponse(BaseModel):
    observation: Observation
    task_id: str
    central_enabled: bool = False


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
    central_enabled: bool = False
