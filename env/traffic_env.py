from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any

from app.config import settings
from tasks.task_easy import get_easy_task
from tasks.task_hard import get_hard_task
from tasks.task_medium import get_medium_task

ACTION_KEEP = "KEEP"
ACTION_SWITCH = "SWITCH"
ACTION_PHASE_0 = "PHASE_0"
ACTION_PHASE_1 = "PHASE_1"
ACTION_PHASE_2 = "PHASE_2"
ACTION_PHASE_3 = "PHASE_3"

VALID_ACTIONS = {ACTION_KEEP, ACTION_SWITCH, ACTION_PHASE_0, ACTION_PHASE_1, ACTION_PHASE_2, ACTION_PHASE_3}
PHASE_ACTIONS = {ACTION_PHASE_0: 0, ACTION_PHASE_1: 1, ACTION_PHASE_2: 2, ACTION_PHASE_3: 3}
INTERSECTIONS = ("NW", "NE", "SW", "SE")
NODE_INDEX = {node: idx for idx, node in enumerate(INTERSECTIONS)}

ROUTES: dict[tuple[str, int], tuple[str, int]] = {
    ("NW", 3): ("NE", 3),
    ("NW", 0): ("SW", 0),
    ("NE", 1): ("NW", 1),
    ("NE", 0): ("SE", 0),
    ("SW", 3): ("SE", 3),
    ("SW", 2): ("NW", 2),
    ("SE", 1): ("SW", 1),
    ("SE", 2): ("NE", 2),
}

MIN_HOLD_STEPS = 3
POLICY_SPECS = {
    "switch_penalty": (0.0, 4.0, 1.1),
    "queue_urgency_weight": (0.5, 3.5, 1.3),
    "emergency_boost": (0.0, 6.0, 0.0),
    "corridor_priority": (0.5, 3.5, 1.0),
    "balance_penalty": (0.5, 3.5, 1.0),
}
DEFAULT_POLICY = {name: spec[2] for name, spec in POLICY_SPECS.items()}
POLICY_ORDER = tuple(DEFAULT_POLICY)

NODE_PERSONALITIES: dict[str, dict[str, float | str]] = {
    "NW": {"role": "corridor_entry", "queue": 1.25, "wait": 0.9, "throughput": 1.0, "downstream": 1.35, "emergency": 0.7},
    "NE": {"role": "bottleneck", "queue": 0.95, "wait": 1.0, "throughput": 0.9, "downstream": 1.7, "emergency": 0.8},
    "SW": {"role": "emergency_prone", "queue": 1.0, "wait": 1.15, "throughput": 0.95, "downstream": 1.0, "emergency": 2.3},
    "SE": {"role": "outflow_sink", "queue": 1.1, "wait": 0.95, "throughput": 1.2, "downstream": 1.5, "emergency": 0.9},
}

TASK_BUILDERS = {
    "easy_fixed": get_easy_task,
    "medium_dynamic": get_medium_task,
    "hard_multi": get_hard_task,
}


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = _mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    steps = list(range(len(values)))
    x_mean = _mean([float(step) for step in steps])
    y_mean = _mean(values)
    denom = sum((step - x_mean) ** 2 for step in steps)
    if denom <= 0.0:
        return 0.0
    numer = sum((step - x_mean) * (value - y_mean) for step, value in zip(steps, values))
    return numer / denom


def _normalize_positive(value: float, bound: float) -> float:
    return _clip((value / max(bound, 1e-6)) * 2.0 - 1.0, -1.0, 1.0)


def _normalize_negative(value: float, bound: float) -> float:
    return _clip(1.0 - (2.0 * value / max(bound, 1e-6)), -1.0, 1.0)


@dataclass
class IntersectionState:
    queue_lengths: list[float]
    waiting_times: list[float]
    current_phase: int = 0
    time_in_phase: int = 0


@dataclass
class HistoryBuffer:
    queues: list[dict[str, list[float]]] = field(default_factory=list)
    waits: list[dict[str, list[float]]] = field(default_factory=list)
    throughputs: list[dict[str, float]] = field(default_factory=list)
    switches: list[dict[str, int]] = field(default_factory=list)
    policy_vectors: list[dict[str, float]] = field(default_factory=list)

    def add(
        self,
        queues: dict[str, list[float]],
        waits: dict[str, list[float]],
        throughputs: dict[str, float],
        switches: dict[str, int],
        policy: dict[str, float],
    ) -> None:
        self.queues.append(queues)
        self.waits.append(waits)
        self.throughputs.append(throughputs)
        self.switches.append(switches)
        self.policy_vectors.append(policy)
        while len(self.queues) > 6:
            self.queues.pop(0)
            self.waits.pop(0)
            self.throughputs.pop(0)
            self.switches.pop(0)
            self.policy_vectors.pop(0)


@dataclass
class CentralState:
    history: HistoryBuffer = field(default_factory=HistoryBuffer)
    detection_counts: dict[str, int] = field(
        default_factory=lambda: {
            "spillback_prevention": 0,
            "corridor_formation": 0,
            "load_balancing": 0,
            "emergency_routing": 0,
            "stability_control": 0,
        }
    )
    last_trend_features: dict[str, float] = field(default_factory=dict)
    last_policy_delta: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in POLICY_ORDER})
    last_policy_stability: float = 1.0


@dataclass
class TrafficState:
    intersections: dict[str, IntersectionState]
    policy: dict[str, float]
    transit_buffers: dict[tuple[str, int, str, int], list[float]]
    step_count: int = 0
    total_throughput: float = 0.0
    episode_wait_sum: float = 0.0
    episode_queue_sum: float = 0.0
    episode_reward_sum: float = 0.0
    done: bool = False
    spillback_events_count: int = 0
    emergency_delay: float = 0.0
    corridor_sync_score_sum: float = 0.0
    policy_change_sum: float = 0.0
    coordination_bonus_total: float = 0.0
    active_behaviors_log: list[str] = field(default_factory=list)
    phase_hold_bonus_steps: int = 0


class TrafficSpawner:
    def __init__(self, task_config: Any):
        self.config = task_config

    def arrivals_for_step(self, step_count: int) -> dict[str, list[float]]:
        arrivals: dict[str, list[float]] = {}
        spike_vector = [1.0, 1.0, 1.0, 1.0]
        if step_count in set(getattr(self.config, "spike_steps", ())):
            spike_vector = list(getattr(self.config, "spike_multipliers", (1.0, 1.0, 1.0, 1.0)))

        for node in INTERSECTIONS:
            node_idx = NODE_INDEX[node]
            node_scale = self.config.node_demand_scale[node_idx]
            node_arrivals: list[float] = []
            for lane in range(4):
                base = self.config.arrival_base[lane]
                jitter = self.config.arrival_jitter[lane]
                bias = self.config.directional_bias[lane]
                phase = (step_count + lane + node_idx) / self.config.demand_wave_period
                wave = 1.0 + math.sin(phase) * jitter
                value = base * bias * node_scale * spike_vector[lane] * wave

                for pulse in self.config.demand_pulses:
                    if pulse.start_step <= step_count <= pulse.end_step:
                        if pulse.node is not None and pulse.node != node:
                            continue
                        if pulse.lane is not None and pulse.lane != lane:
                            continue
                        value *= pulse.multiplier

                if (
                    self.config.emergency_step is not None
                    and self.config.emergency_lane is not None
                    and node == "SW"
                    and lane == self.config.emergency_lane
                    and self.config.emergency_step <= step_count <= self.config.emergency_step + 8
                ):
                    value *= self.config.emergency_multiplier

                node_arrivals.append(round(max(0.0, value), 3))
            arrivals[node] = node_arrivals
        return arrivals


def detect_spillback_risk(state_obj: TrafficState, lane_capacity: float) -> bool:
    for (up_node, up_lane), (down_node, down_lane) in ROUTES.items():
        upstream = state_obj.intersections[up_node].queue_lengths[up_lane]
        downstream = state_obj.intersections[down_node].queue_lengths[down_lane]
        if downstream > lane_capacity * 0.7 and upstream > lane_capacity * 0.24:
            return True
    return False


def detect_corridor_imbalance(state_obj: TrafficState) -> bool:
    ew = sum(state.queue_lengths[1] + state.queue_lengths[3] for state in state_obj.intersections.values())
    ns = sum(state.queue_lengths[0] + state.queue_lengths[2] for state in state_obj.intersections.values())
    return abs(ew - ns) > 14.0


def detect_congestion_growth(history: HistoryBuffer) -> bool:
    totals = [sum(sum(lanes) for lanes in snapshot.values()) for snapshot in history.queues]
    return len(totals) >= 3 and _slope(totals) > 2.0


def detect_starvation(history: HistoryBuffer) -> bool:
    if len(history.throughputs) < 3:
        return False
    for node in INTERSECTIONS:
        if all(snapshot.get(node, 0.0) < 2.0 for snapshot in history.throughputs[-3:]):
            return True
    return False


def detect_instability(history: HistoryBuffer) -> bool:
    totals = [float(sum(snapshot.values())) for snapshot in history.switches]
    return len(totals) >= 3 and _variance(totals[-3:]) > 1.2


def detect_emergency_dominance(state_obj: TrafficState, emergency_lane: int | None) -> bool:
    if emergency_lane is None:
        return False
    sw = state_obj.intersections["SW"]
    return sw.queue_lengths[emergency_lane] > 10.0 or sw.waiting_times[emergency_lane] > 18.0


class TrafficEnv:
    def __init__(self, task: str = "easy_fixed", max_steps: int | None = None):
        if task not in TASK_BUILDERS:
            raise ValueError(f"Unknown task '{task}'. Expected one of {sorted(TASK_BUILDERS)}")
        self.task = task
        self.max_steps = max_steps or settings.max_steps
        self.task_config = TASK_BUILDERS[task](max_steps=self.max_steps)
        self.random = random.Random(self.task_config.seed)
        self.spawner = TrafficSpawner(self.task_config)
        self.state_obj: TrafficState | None = None
        self.central_state: CentralState | None = None
        self.central_enabled = False

    def reset(self, task_id: str | None = None, central_enabled: bool = False) -> dict[str, Any]:
        self.central_enabled = central_enabled
        if task_id is not None:
            if task_id not in TASK_BUILDERS:
                raise ValueError(f"Unknown task '{task_id}'. Expected one of {sorted(TASK_BUILDERS)}")
            self.task = task_id
            self.task_config = TASK_BUILDERS[task_id](max_steps=self.max_steps)
            self.spawner = TrafficSpawner(self.task_config)

        self.random.seed(self.task_config.seed)
        low, high = self.task_config.initial_queue_bounds
        intersections: dict[str, IntersectionState] = {}
        for node in INTERSECTIONS:
            base_scale = self.task_config.node_demand_scale[NODE_INDEX[node]]
            queues = [
                round(self.random.uniform(low, high) * base_scale * self.task_config.directional_bias[lane], 2)
                for lane in range(4)
            ]
            waits = [round(queue * 1.25 + self.random.uniform(0.5, 2.0), 2) for queue in queues]
            intersections[node] = IntersectionState(queue_lengths=queues, waiting_times=waits)

        transit_buffers = {
            (up_node, up_lane, down_node, down_lane): [0.0] * self.task_config.transfer_delay_steps
            for (up_node, up_lane), (down_node, down_lane) in ROUTES.items()
        }
        self.state_obj = TrafficState(
            intersections=intersections,
            policy=DEFAULT_POLICY.copy(),
            transit_buffers=transit_buffers,
        )
        self.central_state = CentralState()
        return self._observation()

    def step(self, payload: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._ensure_reset()
        assert self.state_obj is not None
        assert self.central_state is not None

        local_actions, central_action = self._parse_payload(payload)
        active_behaviors = self._update_central_policy(central_action)
        self.state_obj.active_behaviors_log.append(
            f"{self.state_obj.step_count}:{'|'.join(active_behaviors) if active_behaviors else 'none'}"
        )

        switched_map = self._apply_local_actions(local_actions)
        released_transfers = self._release_transfers()
        external_arrivals = self.spawner.arrivals_for_step(self.state_obj.step_count)

        total_throughput = 0.0
        node_throughputs = {node: 0.0 for node in INTERSECTIONS}
        spillback_this_step = 0

        for node in INTERSECTIONS:
            state = self.state_obj.intersections[node]
            green_lane = state.current_phase
            switched = switched_map[node]
            next_queues: list[float] = []
            next_waits: list[float] = []

            for lane in range(4):
                incoming = external_arrivals[node][lane] + released_transfers[node][lane]
                queue_before = state.queue_lengths[lane] + incoming
                service = self._service_rate(node, lane, green_lane, switched)

                route = ROUTES.get((node, lane))
                spillback_factor = 1.0
                if route is not None:
                    down_node, down_lane = route
                    downstream_queue = self.state_obj.intersections[down_node].queue_lengths[down_lane]
                    capacity_ratio = downstream_queue / self.task_config.lane_capacity
                    if capacity_ratio > 0.68:
                        spillback_factor = max(0.05, 1.0 - (capacity_ratio - 0.68) * 2.3)
                        spillback_this_step += 1

                cleared = min(queue_before, service * spillback_factor)
                if route is not None and cleared > 0.0:
                    self._push_transfer(node, lane, cleared * self.task_config.route_transfer_ratio)

                next_queue = max(0.0, queue_before - cleared)
                prev_wait = state.waiting_times[lane]
                if lane == green_lane:
                    clearance_ratio = cleared / max(queue_before, 1e-6)
                    next_wait = (
                        prev_wait * max(0.18, 0.72 - 0.55 * clearance_ratio)
                        + next_queue * 0.42
                        - cleared * 0.18
                        - (0.6 if not switched else 0.2)
                    )
                    if next_queue < 0.5:
                        next_wait *= 0.2
                    next_wait = max(0.0, next_wait)
                else:
                    next_wait = min(150.0, prev_wait + 0.48 * queue_before + 0.22 * incoming)

                if (
                    node == "SW"
                    and self.task_config.emergency_step is not None
                    and self.task_config.emergency_lane is not None
                    and lane == self.task_config.emergency_lane
                    and self.state_obj.step_count >= self.task_config.emergency_step
                ):
                    self.state_obj.emergency_delay += next_wait

                next_queue = min(next_queue, self.task_config.lane_capacity * 1.65)
                next_queues.append(next_queue)
                next_waits.append(next_wait)
                total_throughput += cleared
                node_throughputs[node] += cleared

            state.queue_lengths = next_queues
            state.waiting_times = next_waits

        self.state_obj.spillback_events_count += spillback_this_step
        if not any(switched_map.values()):
            self.state_obj.phase_hold_bonus_steps += 1

        self.state_obj.step_count += 1
        self.state_obj.total_throughput += total_throughput

        step_metrics = self._metrics(total_throughput, switched_map, spillback_this_step)
        reward = self._reward(step_metrics)
        self.state_obj.episode_reward_sum += reward
        self.state_obj.episode_wait_sum += step_metrics["mean_wait"]
        self.state_obj.episode_queue_sum += step_metrics["mean_queue"]
        self.state_obj.done = self.state_obj.step_count >= self.task_config.max_steps

        self._record_history(node_throughputs, switched_map)

        summary = self.episode_summary()
        score = self.task_config.grader(summary) if self.task_config.grader is not None else 0.5
        score = max(0.01, min(0.99, float(score)))

        observation = self._observation()
        info: dict[str, Any] = {
            "throughput": round(total_throughput, 3),
            "avg_wait": round(step_metrics["mean_wait"], 3),
            "score": score,
            "task_id": self.task_config.task_id,
            "episode_throughput": round(self.state_obj.total_throughput, 3),
            "episode_avg_wait": round(self.state_obj.episode_wait_sum / max(self.state_obj.step_count, 1), 3),
            "corridor_sync_score": round(step_metrics["corridor_sync_score"], 3),
            "active_behaviors_log": list(self.state_obj.active_behaviors_log[-3:]),
            "policy_stability": round(self.central_state.last_policy_stability, 3),
            "spillback_count": self.state_obj.spillback_events_count,
            "emergency_delay": round(self.state_obj.emergency_delay, 3),
        }
        if self.state_obj.done:
            summary["final_score"] = score
            info["summary"] = summary
        return observation, reward, self.state_obj.done, info

    def state(self) -> dict[str, Any]:
        self._ensure_reset()
        assert self.state_obj is not None
        summary = self.episode_summary()
        return {
            "task_id": self.task_config.task_id,
            "step_count": self.state_obj.step_count,
            "observation": self._observation(),
            "metrics": summary,
            "episode_throughput": round(self.state_obj.total_throughput, 3),
            "episode_avg_wait": round(self.state_obj.episode_wait_sum / max(self.state_obj.step_count, 1), 3),
            "central_enabled": self.central_enabled,
        }

    def close(self) -> None:
        return None

    def _ensure_reset(self) -> None:
        if self.state_obj is None:
            self.reset()

    def _parse_payload(self, payload: Any) -> tuple[dict[str, str], dict[str, float] | None]:
        local_actions: dict[str, str] = {}
        central_action: dict[str, float] | None = None

        if hasattr(payload, "local_actions") and payload.local_actions:
            local_actions = dict(payload.local_actions)
            central_action = getattr(payload, "central_action", None)
        elif isinstance(payload, dict) and "local_actions" in payload:
            local_actions = dict(payload["local_actions"])
            raw_central = payload.get("central_action")
            central_action = dict(raw_central) if isinstance(raw_central, dict) else None
        else:
            if hasattr(payload, "action"):
                fallback = str(payload.action or "KEEP")
            elif isinstance(payload, dict):
                fallback = str(payload.get("action", "KEEP"))
            else:
                fallback = str(payload)
            local_actions = {node: fallback for node in INTERSECTIONS}
        return local_actions, central_action

    def _apply_local_actions(self, local_actions: dict[str, str]) -> dict[str, bool]:
        assert self.state_obj is not None
        switched_map: dict[str, bool] = {}
        for node in INTERSECTIONS:
            action = str(local_actions.get(node, ACTION_KEEP)).upper().strip()
            if action not in VALID_ACTIONS:
                action = ACTION_KEEP

            state = self.state_obj.intersections[node]
            previous_phase = state.current_phase
            switched = False

            if action in PHASE_ACTIONS:
                target_phase = PHASE_ACTIONS[action]
                switched = target_phase != previous_phase
                state.current_phase = target_phase
                state.time_in_phase = 0 if switched else state.time_in_phase + 1
            elif action == ACTION_SWITCH:
                state.current_phase = (previous_phase + 1) % 4
                state.time_in_phase = 0
                switched = True
            else:
                state.time_in_phase += 1

            switched_map[node] = switched
        return switched_map

    def _release_transfers(self) -> dict[str, list[float]]:
        assert self.state_obj is not None
        released = {node: [0.0, 0.0, 0.0, 0.0] for node in INTERSECTIONS}
        for key, buffer in self.state_obj.transit_buffers.items():
            _, _, down_node, down_lane = key
            arriving = buffer.pop(0)
            buffer.append(0.0)
            released[down_node][down_lane] += arriving
        return released

    def _push_transfer(self, node: str, lane: int, amount: float) -> None:
        assert self.state_obj is not None
        route = ROUTES.get((node, lane))
        if route is None:
            return
        down_node, down_lane = route
        key = (node, lane, down_node, down_lane)
        self.state_obj.transit_buffers[key][-1] += amount

    def _record_history(self, node_throughputs: dict[str, float], switched_map: dict[str, bool]) -> None:
        assert self.state_obj is not None
        assert self.central_state is not None
        queues = {
            node: [round(value, 3) for value in self.state_obj.intersections[node].queue_lengths]
            for node in INTERSECTIONS
        }
        waits = {
            node: [round(value, 3) for value in self.state_obj.intersections[node].waiting_times]
            for node in INTERSECTIONS
        }
        switches = {node: 1 if switched_map[node] else 0 for node in INTERSECTIONS}
        self.central_state.history.add(queues, waits, node_throughputs, switches, self.state_obj.policy.copy())

    def _compute_trend_features(self) -> dict[str, float]:
        assert self.central_state is not None
        history = self.central_state.history
        total_queues = [sum(sum(lanes) for lanes in snapshot.values()) for snapshot in history.queues]
        total_waits = [sum(sum(lanes) for lanes in snapshot.values()) / 16.0 for snapshot in history.waits]
        total_throughput = [sum(snapshot.values()) for snapshot in history.throughputs]
        switch_totals = [float(sum(snapshot.values())) for snapshot in history.switches]

        features = {
            "queue_growth_rate": _slope(total_queues),
            "wait_growth_rate": _slope(total_waits),
            "throughput_trend": _slope(total_throughput),
            "switching_instability": _variance(switch_totals[-4:]) if switch_totals else 0.0,
        }
        self.central_state.last_trend_features = features
        return features

    def _estimate_risks(self, trend_features: dict[str, float]) -> dict[str, float]:
        assert self.state_obj is not None
        lane_capacity = self.task_config.lane_capacity

        ew = sum(state.queue_lengths[1] + state.queue_lengths[3] for state in self.state_obj.intersections.values())
        ns = sum(state.queue_lengths[0] + state.queue_lengths[2] for state in self.state_obj.intersections.values())
        corridor_gap = abs(ew - ns)

        max_downstream_ratio = 0.0
        for (_, _), (down_node, down_lane) in ROUTES.items():
            queue = self.state_obj.intersections[down_node].queue_lengths[down_lane]
            max_downstream_ratio = max(max_downstream_ratio, queue / lane_capacity)

        emergency_pressure = 0.0
        if self.task_config.emergency_lane is not None:
            emergency_pressure = (
                self.state_obj.intersections["SW"].queue_lengths[self.task_config.emergency_lane] / lane_capacity
            )

        throughput_decline = _clip((-trend_features["throughput_trend"]) / 6.0, 0.0, 1.0)
        queue_growth = _clip(trend_features["queue_growth_rate"] / 10.0, 0.0, 1.0)
        wait_growth = _clip(trend_features["wait_growth_rate"] / 5.0, 0.0, 1.0)

        return {
            "spillback": _clip((max_downstream_ratio - 0.65) / 0.3 + queue_growth * 0.4, 0.0, 1.0),
            "corridor": _clip(corridor_gap / 40.0 + throughput_decline * 0.25, 0.0, 1.0),
            "load_balance": _clip(wait_growth * 0.35 + queue_growth * 0.45 + corridor_gap / 60.0, 0.0, 1.0),
            "emergency": _clip(max(0.0, emergency_pressure - 0.25) / 0.75 + wait_growth * 0.2, 0.0, 1.0),
            "instability": _clip(trend_features["switching_instability"] / 4.0, 0.0, 1.0),
        }

    def _update_central_policy(self, central_action: dict[str, float] | None) -> list[str]:
        assert self.state_obj is not None
        assert self.central_state is not None
        if not self.central_enabled:
            self.central_state.last_policy_delta = {name: 0.0 for name in POLICY_ORDER}
            self.central_state.last_policy_stability = 1.0
            return []

        trend_features = self._compute_trend_features()
        risks = self._estimate_risks(trend_features)
        raw_flags = {
            "spillback_prevention": detect_spillback_risk(self.state_obj, self.task_config.lane_capacity) or risks["spillback"] > 0.55,
            "corridor_formation": detect_corridor_imbalance(self.state_obj) or risks["corridor"] > 0.5,
            "load_balancing": detect_congestion_growth(self.central_state.history) or detect_starvation(self.central_state.history) or risks["load_balance"] > 0.52,
            "emergency_routing": detect_emergency_dominance(self.state_obj, self.task_config.emergency_lane) or risks["emergency"] > 0.4,
            "stability_control": detect_instability(self.central_state.history) or risks["instability"] > 0.45,
        }

        active_behaviors: list[str] = []
        for behavior, active in raw_flags.items():
            if active:
                self.central_state.detection_counts[behavior] += 1
            else:
                self.central_state.detection_counts[behavior] = max(0, self.central_state.detection_counts[behavior] - 1)
            if self.central_state.detection_counts[behavior] >= 2 or (behavior == "emergency_routing" and active):
                active_behaviors.append(behavior)

        target = DEFAULT_POLICY.copy()
        throughput_decline = _clip((-trend_features["throughput_trend"]) / 6.0, 0.0, 1.0)

        if "spillback_prevention" in active_behaviors:
            target["queue_urgency_weight"] += 0.3 + risks["spillback"] * 0.9
            target["balance_penalty"] += 0.8 + risks["spillback"] * 1.5
            target["corridor_priority"] -= risks["spillback"] * 0.2
        if "corridor_formation" in active_behaviors:
            target["corridor_priority"] += 0.4 + risks["corridor"] * 1.7 + throughput_decline * 0.35
            target["switch_penalty"] += 0.15 + risks["corridor"] * 0.35
        if "load_balancing" in active_behaviors:
            target["queue_urgency_weight"] += 0.35 + risks["load_balance"] * 1.0
            target["balance_penalty"] += 0.25 + risks["load_balance"] * 0.7
        if "emergency_routing" in active_behaviors:
            target["emergency_boost"] += 1.5 + risks["emergency"] * 4.5
            target["switch_penalty"] -= 0.3 + risks["emergency"] * 0.6
            target["queue_urgency_weight"] += risks["emergency"] * 0.25
        if "stability_control" in active_behaviors:
            target["switch_penalty"] += 0.55 + risks["instability"] * 1.15
            target["corridor_priority"] -= risks["instability"] * 0.15

        if central_action:
            for name, delta in central_action.items():
                if name in target:
                    target[name] += float(delta)

        deltas: dict[str, float] = {}
        total_change = 0.0
        total_range = 0.0
        for name in POLICY_ORDER:
            low, high, _ = POLICY_SPECS[name]
            target[name] = _clip(target[name], low, high)
            current = self.state_obj.policy[name]
            policy_range = high - low
            max_delta = policy_range * 0.10
            delta = _clip(target[name] - current, -max_delta, max_delta)
            updated = _clip(current + delta, low, high)
            deltas[name] = updated - current
            total_change += abs(deltas[name])
            total_range += policy_range
            self.state_obj.policy[name] = updated

        self.central_state.last_policy_delta = deltas
        normalized_change = total_change / max(total_range, 1e-6)
        self.central_state.last_policy_stability = 1.0 - _clip(normalized_change * 5.0, 0.0, 1.0)
        self.state_obj.policy_change_sum += normalized_change
        return active_behaviors

    def _service_rate(self, node: str, lane: int, green_lane: int, switched: bool) -> float:
        assert self.state_obj is not None
        policy = self.state_obj.policy
        personality = NODE_PERSONALITIES[node]
        queue = self.state_obj.intersections[node].queue_lengths[lane]
        downstream_penalty = 0.0
        route = ROUTES.get((node, lane))
        if route is not None:
            down_node, down_lane = route
            downstream_queue = self.state_obj.intersections[down_node].queue_lengths[down_lane]
            downstream_penalty = (downstream_queue / self.task_config.lane_capacity) * float(personality["downstream"])

        service = self.task_config.service_base
        service += 0.18 * min(queue, 12.0) * policy["queue_urgency_weight"] * float(personality["queue"])

        if lane == green_lane:
            service += self.task_config.green_bonus
            if lane in {1, 3}:
                service += 0.8 * policy["corridor_priority"]
        else:
            service -= self.task_config.red_penalty

        if switched:
            service -= policy["switch_penalty"] * 0.75

        if route is not None:
            service -= downstream_penalty * policy["balance_penalty"]

        if (
            node == "SW"
            and self.task_config.emergency_lane is not None
            and lane == self.task_config.emergency_lane
            and self.state_obj.step_count >= (self.task_config.emergency_step or 10**9)
        ):
            service += policy["emergency_boost"] * float(personality["emergency"])

        if self.task_config.multi_intersection and lane in {0, 3}:
            service += 0.35

        return max(0.35, service)

    def _metrics(self, throughput: float, switched_map: dict[str, bool], spillback_this_step: int) -> dict[str, float]:
        assert self.state_obj is not None
        total_queue = sum(sum(state.queue_lengths) for state in self.state_obj.intersections.values())
        total_wait = sum(sum(state.waiting_times) for state in self.state_obj.intersections.values())
        node_queue_totals = [sum(state.queue_lengths) for state in self.state_obj.intersections.values()]
        mean_queue = total_queue / len(INTERSECTIONS)
        mean_wait = total_wait / 16.0
        imbalance = math.sqrt(_mean([(value - _mean(node_queue_totals)) ** 2 for value in node_queue_totals]))
        corridor_sync = self._corridor_sync_score()
        self.state_obj.corridor_sync_score_sum += corridor_sync
        policy_delta = sum(abs(self.central_state.last_policy_delta[name]) for name in POLICY_ORDER) if self.central_state else 0.0

        return {
            "mean_queue": mean_queue,
            "mean_wait": mean_wait,
            "throughput": throughput,
            "imbalance": imbalance,
            "spillback_count": float(spillback_this_step),
            "emergency_delay_step": self._current_emergency_wait(),
            "corridor_sync_score": corridor_sync,
            "policy_delta": policy_delta,
            "switches_count": float(sum(1 for switched in switched_map.values() if switched)),
        }

    def _reward(self, metrics: dict[str, float]) -> float:
        queue_term = 0.24 * _normalize_negative(metrics["mean_queue"], self.task_config.lane_capacity * 4.0)
        wait_term = 0.22 * _normalize_negative(metrics["mean_wait"], 35.0)
        throughput_term = 0.18 * _normalize_positive(metrics["throughput"], 34.0)
        imbalance_term = 0.12 * _normalize_negative(metrics["imbalance"], 22.0)
        emergency_term = 0.08 * _normalize_negative(metrics["emergency_delay_step"], 40.0)
        spillback_term = 0.08 * _normalize_negative(metrics["spillback_count"], 8.0)
        stability_bonus = 0.04 * _normalize_negative(metrics["policy_delta"], 1.2)
        coordination_bonus = 0.04 * (2.0 * metrics["corridor_sync_score"] - 1.0)

        reward = (
            queue_term
            + wait_term
            + throughput_term
            + imbalance_term
            + emergency_term
            + spillback_term
            + stability_bonus
            + coordination_bonus
        )
        self.state_obj.coordination_bonus_total += coordination_bonus
        return float(_clip(reward, -1.0, 1.0))

    def _corridor_sync_score(self) -> float:
        assert self.state_obj is not None
        nw_phase = self.state_obj.intersections["NW"].current_phase
        ne_phase = self.state_obj.intersections["NE"].current_phase
        sw_phase = self.state_obj.intersections["SW"].current_phase
        se_phase = self.state_obj.intersections["SE"].current_phase
        ew_aligned = 1.0 if nw_phase in {1, 3} and ne_phase in {1, 3} else 0.0
        diagonal_aligned = 1.0 if (nw_phase % 2) == (se_phase % 2) else 0.0
        south_band = 1.0 if (sw_phase % 2) == (se_phase % 2) else 0.0
        return round((0.45 * ew_aligned) + (0.3 * diagonal_aligned) + (0.25 * south_band), 3)

    def _current_emergency_wait(self) -> float:
        assert self.state_obj is not None
        lane = self.task_config.emergency_lane
        if lane is None:
            return 0.0
        return self.state_obj.intersections["SW"].waiting_times[lane]

    def _observation(self) -> dict[str, Any]:
        assert self.state_obj is not None
        return {
            "queue_lengths": {
                node: [round(value, 2) for value in self.state_obj.intersections[node].queue_lengths]
                for node in INTERSECTIONS
            },
            "waiting_times": {
                node: [round(value, 2) for value in self.state_obj.intersections[node].waiting_times]
                for node in INTERSECTIONS
            },
            "current_phase": {node: self.state_obj.intersections[node].current_phase for node in INTERSECTIONS},
            "time_in_phase": {node: self.state_obj.intersections[node].time_in_phase for node in INTERSECTIONS},
            "policy": {name: round(value, 3) for name, value in self.state_obj.policy.items()},
            "text_obs": self.format_central_llm_prompt(),
        }

    def format_central_llm_prompt(self) -> str:
        assert self.state_obj is not None
        assert self.central_state is not None
        trends = self.central_state.last_trend_features or {
            "queue_growth_rate": 0.0,
            "wait_growth_rate": 0.0,
            "throughput_trend": 0.0,
            "switching_instability": 0.0,
        }
        lines = [
            f"step={self.state_obj.step_count}",
            f"central_enabled={self.central_enabled}",
            "policy="
            + ", ".join(f"{name}:{self.state_obj.policy[name]:.2f}" for name in POLICY_ORDER),
            "trends="
            + ", ".join(f"{name}:{value:.2f}" for name, value in trends.items()),
        ]
        if self.state_obj.active_behaviors_log:
            lines.append(f"active_behaviors={self.state_obj.active_behaviors_log[-1].split(':', 1)[-1]}")
        for node in INTERSECTIONS:
            state = self.state_obj.intersections[node]
            lines.append(
                f"{node} role={NODE_PERSONALITIES[node]['role']} phase={state.current_phase} hold={state.time_in_phase} "
                f"queues={[round(value, 1) for value in state.queue_lengths]} waits={[round(value, 1) for value in state.waiting_times]}"
            )
        return "\n".join(lines)

    def episode_summary(self) -> dict[str, Any]:
        assert self.state_obj is not None
        steps = max(self.state_obj.step_count, 1)
        node_queue_totals = [sum(state.queue_lengths) for state in self.state_obj.intersections.values()]
        imbalance = math.sqrt(_mean([(value - _mean(node_queue_totals)) ** 2 for value in node_queue_totals]))
        policy_stability = 1.0 - _clip(self.state_obj.policy_change_sum / max(steps * 0.12, 1e-6), 0.0, 1.0)
        return {
            "episode_reward": round(self.state_obj.episode_reward_sum, 4),
            "mean_queue": round(self.state_obj.episode_queue_sum / steps, 4),
            "mean_wait": round(self.state_obj.episode_wait_sum / steps, 4),
            "throughput": round(self.state_obj.total_throughput / steps, 4),
            "imbalance": round(imbalance, 4),
            "spillback_count": self.state_obj.spillback_events_count,
            "emergency_delay": round(self.state_obj.emergency_delay, 4),
            "central_enabled": self.central_enabled,
            "task_id": self.task_config.task_id,
            "policy_stability": round(policy_stability, 4),
            "final_score": 0.5,
            "step_count": self.state_obj.step_count,
            "corridor_sync_score": round(self.state_obj.corridor_sync_score_sum / steps, 4),
            "active_behaviors_log": list(self.state_obj.active_behaviors_log),
            "text_obs": self.format_central_llm_prompt(),
        }
