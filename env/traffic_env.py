from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import random

from app.config import settings
from graders.grader import grade
from env.types import TrafficTask
from tasks.task_easy import get_easy_task
from tasks.task_hard import get_hard_task
from tasks.task_medium import get_medium_task


ACTION_KEEP = "KEEP"
ACTION_SWITCH = "SWITCH"


@dataclass
class TrafficState:
    queue_lengths: list[float]
    waiting_times: list[float]
    current_phase: int = 0
    time_in_phase: int = 0
    step_count: int = 0
    total_throughput: int = 0
    total_wait: float = 0.0
    done: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)


TASK_BUILDERS = {
    "easy_fixed": get_easy_task,
    "medium_dynamic": get_medium_task,
    "hard_multi": get_hard_task,
}


class TrafficEnv:
    def __init__(self, task: str = "easy_fixed", max_steps: int | None = None):
        if task not in TASK_BUILDERS:
            raise ValueError(f"Unknown task '{task}'. Expected one of {sorted(TASK_BUILDERS)}")

        self.task = task
        self.task_config = TASK_BUILDERS[task](max_steps=max_steps or settings.max_steps)
        self.random = random.Random(self.task_config.seed)
        self.state_obj: TrafficState | None = None
        self._real_sumo_enabled = settings.use_real_sumo
        self._traci = None

    def reset(self) -> dict[str, Any]:
        self.random.seed(self.task_config.seed)
        self.state_obj = TrafficState(
            queue_lengths=[12.0, 8.0, 6.0, 10.0],
            waiting_times=[24.0, 18.0, 12.0, 20.0],
            current_phase=0,
            time_in_phase=0,
        )
        self._start_sumo_if_available()
        return self._observation()

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._ensure_reset()
        action = action.upper().strip()
        if action not in {ACTION_KEEP, ACTION_SWITCH}:
            raise ValueError("action must be KEEP or SWITCH")

        assert self.state_obj is not None
        previous_phase = self.state_obj.current_phase
        switched = action == ACTION_SWITCH

        if switched:
            self.state_obj.current_phase = (self.state_obj.current_phase + 1) % 4
            self.state_obj.time_in_phase = 0
        else:
            self.state_obj.time_in_phase += 1

        arrivals = self._arrivals_for_step(self.state_obj.step_count)
        green_lane = self.state_obj.current_phase
        throughput = 0

        new_queues: list[float] = []
        new_waits: list[float] = []
        for lane, queue in enumerate(self.state_obj.queue_lengths):
            demand = arrivals[lane]
            service = self._service_rate(lane, green_lane, switched)
            next_queue = max(0.0, queue + demand - service)
            next_wait = max(0.0, self.state_obj.waiting_times[lane] + next_queue * 1.5 + demand * 2.0 - service)
            new_queues.append(next_queue)
            new_waits.append(next_wait)
            throughput += int(max(0.0, min(queue + demand, service)))

        self.state_obj.queue_lengths = new_queues
        self.state_obj.waiting_times = new_waits
        self.state_obj.step_count += 1
        self.state_obj.total_throughput += throughput
        self.state_obj.total_wait = sum(new_waits)
        self.state_obj.done = self.state_obj.step_count >= self.task_config.max_steps
        self.state_obj.history.append(
            {
                "action": action,
                "previous_phase": previous_phase,
                "switched": switched,
                "throughput": throughput,
            }
        )

        metrics = self._metrics(throughput=throughput, switched=switched)
        reward = self._reward(metrics, switched)
        observation = self._observation()
        info = {
            "throughput": throughput,
            "avg_wait": metrics["avg_wait"],
            "score": grade(metrics),
            "task_id": self.task_config.task_id,
        }
        return observation, reward, self.state_obj.done, info

    def state(self) -> dict[str, Any]:
        self._ensure_reset()
        assert self.state_obj is not None
        return {
            "task_id": self.task_config.task_id,
            "step_count": self.state_obj.step_count,
            "observation": self._observation(),
            "metrics": self._metrics(throughput=0, switched=False),
        }

    def close(self) -> None:
        if self._traci is not None:
            try:
                self._traci.close()
            finally:
                self._traci = None

    def _ensure_reset(self) -> None:
        if self.state_obj is None:
            self.reset()

    def _start_sumo_if_available(self) -> None:
        if not self._real_sumo_enabled:
            return
        try:
            import traci  # type: ignore
        except Exception:
            self._real_sumo_enabled = False
            return

        self._traci = traci

    def _observation(self) -> dict[str, Any]:
        assert self.state_obj is not None
        return {
            "queue_lengths": [round(value, 2) for value in self.state_obj.queue_lengths],
            "waiting_times": [round(value, 2) for value in self.state_obj.waiting_times],
            "current_phase": self.state_obj.current_phase,
            "time_in_phase": self.state_obj.time_in_phase,
        }

    def _arrivals_for_step(self, step_count: int) -> list[float]:
        arrivals = []
        spike_multiplier = 1.0
        if step_count in self.task_config.spike_steps:
            spike_multiplier = max(self.task_config.spike_multipliers)

        for lane, base in enumerate(self.task_config.arrival_base):
            jitter = self.task_config.arrival_jitter[lane]
            wave = math.sin((step_count + lane + 1) / 3.0) * jitter
            value = max(0.0, base + wave)
            value *= spike_multiplier
            if self.task_config.emergency_step is not None and step_count == self.task_config.emergency_step:
                if self.task_config.emergency_lane == lane:
                    value *= self.task_config.emergency_multiplier
            arrivals.append(round(value, 3))
        return arrivals

    def _service_rate(self, lane: int, green_lane: int, switched: bool) -> float:
        base_service = 6.5
        if lane == green_lane:
            base_service += 5.0
        else:
            base_service -= 2.0
        if switched:
            base_service -= 1.0
        if self.task_config.multi_intersection:
            base_service += 1.5 if lane in {0, 2} else 0.5
        return max(1.0, base_service)

    def _metrics(self, throughput: int, switched: bool) -> dict[str, float]:
        assert self.state_obj is not None
        avg_wait = sum(self.state_obj.waiting_times) / len(self.state_obj.waiting_times)
        total_queue = sum(self.state_obj.queue_lengths)
        switching_penalty = 2.5 if switched else 0.0
        return {
            "avg_wait": avg_wait,
            "total_waiting_time": sum(self.state_obj.waiting_times),
            "total_queue_length": total_queue,
            "throughput": float(throughput),
            "switching_penalty": switching_penalty,
        }

    def _reward(self, metrics: dict[str, float], switched: bool) -> float:
        switching_penalty = 2.5 if switched else 0.0
        reward = (
            -metrics["total_waiting_time"]
            - 0.5 * metrics["total_queue_length"]
            + metrics["throughput"] * 2.0
            - switching_penalty
        )
        return float(max(-500.0, min(100.0, reward)))
