from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Request

from app.config import settings
from app.models import ResetResponse, StateResponse, StepRequest, StepResponse
from env.traffic_env import TrafficEnv


app = FastAPI(title="Traffic OpenEnv", version="1.0.0")
env = TrafficEnv(task=settings.task_id, max_steps=settings.max_steps)


@app.api_route("/reset", methods=["GET", "POST"], response_model=ResetResponse)
async def reset(request: Request, task_id: Optional[str] = None, central_enabled: Optional[bool] = None) -> dict:
    if request.method == "POST":
        try:
            body = await request.json()
            task_id = body.get("task_id", task_id)
            if central_enabled is None:
                central_enabled = body.get("central_enabled", None)
        except Exception:
            pass
            
    if central_enabled is None:
        central_enabled = False
        
    try:
        observation = env.reset(task_id=task_id, central_enabled=central_enabled)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"observation": observation, "task_id": env.task_config.task_id, "central_enabled": env.central_enabled}


@app.post("/step", response_model=StepResponse)
def step(payload: StepRequest) -> dict:
    try:
        observation, reward, done, info = env.step(payload)
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


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "name": "traffic-signal-openenv",
        "description": "Deterministic traffic signal control environment for OpenEnv validation.",
    }


@app.get("/schema")
def schema() -> dict:
    return {
        "action": StepRequest.model_json_schema(),
        "observation": ResetResponse.model_json_schema(),
        "state": StateResponse.model_json_schema(),
    }


@app.post("/mcp")
def mcp() -> dict:
    return {"jsonrpc": "2.0", "result": {"status": "ok"}, "id": 1}
