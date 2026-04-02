from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.models import ResetResponse, StateResponse, StepRequest, StepResponse
from env.traffic_env import TrafficEnv


app = FastAPI(title="Traffic OpenEnv", version="1.0.0")
env = TrafficEnv(task=settings.task_id, max_steps=settings.max_steps)


@app.get("/reset", response_model=ResetResponse)
def reset() -> dict:
    observation = env.reset()
    return {"observation": observation}


@app.post("/step", response_model=StepResponse)
def step(payload: StepRequest) -> dict:
    try:
        observation, reward, done, info = env.step(payload.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", response_model=StateResponse)
def state() -> dict:
    return env.state()


@app.get("/")
def root() -> dict:
    return {"name": "traffic-openenv", "endpoints": ["/reset", "/step", "/state"]}
