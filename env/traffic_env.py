from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any

from app.config import settings
from tasks.task_easy import get_easy_task
from tasks.task_hard import get_hard_task
from tasks.task_medium import get_medium_task
from tasks.task_gridlock import get_gridlock_task
from tasks.task_corridor import get_corridor_task
from tasks.task_incident import get_incident_task
from tasks.task_dynamic import get_dynamic_task

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
MOVEMENTS = ("left", "straight", "right")
TURN_SERVICE_MULTIPLIER = {"left": 0.6, "straight": 1.0, "right": 1.3}

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
PHASE_MOVEMENT_MAP = {
    node: {phase: MOVEMENTS for phase in range(4)}
    for node in INTERSECTIONS
}

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
    "gridlock_risk": get_gridlock_task,
    "corridor_flow": get_corridor_task,
    "incident_response": get_incident_task,
    "dynamic_demand": get_dynamic_task,
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


def _round_movement_dict(values: dict[str, float]) -> dict[str, float]:
    return {movement: round(float(values.get(movement, 0.0)), 2) for movement in MOVEMENTS}


def _empty_movement_queue() -> dict[str, float]:
    return {movement: 0.0 for movement in MOVEMENTS}


def _sum_movements(values: dict[str, float]) -> float:
    return sum(float(values.get(movement, 0.0)) for movement in MOVEMENTS)


@dataclass
class IntersectionState:
    movement_queues: list[dict[str, float]]
    waiting_times: list[float]
    current_phase: int = 0
    time_in_phase: int = 0
    lane_blocked: list[bool] = field(default_factory=lambda: [False, False, False, False])
    blockage_durations: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    phase_history: list[int] = field(default_factory=lambda: [0, 0, 0])
    time_since_served: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    @property
    def queue_lengths(self) -> list[float]:
        return [_sum_movements(movement_queue) for movement_queue in self.movement_queues]

    def total_queue(self, lane: int) -> float:
        return _sum_movements(self.movement_queues[lane])

    def set_lane_total(self, lane: int, movement_queue: dict[str, float]) -> None:
        self.movement_queues[lane] = {movement: max(0.0, float(movement_queue.get(movement, 0.0))) for movement in MOVEMENTS}


@dataclass
class HistoryBuffer:
    queues: list[dict[str, list[float]]] = field(default_factory=list)
    waits: list[dict[str, list[float]]] = field(default_factory=list)
    throughputs: list[dict[str, float]] = field(default_factory=list)
    lane_throughputs: list[dict[str, list[float]]] = field(default_factory=list)
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
        if hasattr(self, 'lane_throughputs'):
            # This is a bit of a hack to handle the new field in existing objects if any
            # But in this env it should be fine as it's recreated on reset
            pass
        
    def add(
        self,
        queues: dict[str, list[float]],
        waits: dict[str, list[float]],
        throughputs: dict[str, float],
        lane_throughputs: dict[str, list[float]],
        switches: dict[str, int],
        policy: dict[str, float],
    ) -> None:
        self.queues.append(queues)
        self.waits.append(waits)
        self.throughputs.append(throughputs)
        self.lane_throughputs.append(lane_throughputs)
        self.switches.append(switches)
        self.policy_vectors.append(policy)
        while len(self.queues) > 6:
            self.queues.pop(0)
            self.waits.pop(0)
            self.throughputs.pop(0)
            self.lane_throughputs.pop(0)
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
    oversight_interventions_count: int = 0
    last_rationale: str = ""
    active_corridors: list[str] = field(default_factory=list)
    budget_remaining: float = 3.0
    anticipatory_spillback_scores: dict[str, float] = field(default_factory=dict)


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
    total_in_transit: float = 0.0
    total_demand_spawned: float = 0.0
    blockage_duration_sum: float = 0.0
    oversight_interventions_total: int = 0
    max_starvation_time: float = 0.0
    incident_start_steps: dict[str, int] = field(default_factory=dict)
    incident_response_latency: dict[str, int] = field(default_factory=dict)
    efficiency_baseline: float = 0.0
    recovery_tracking_active: bool = False
    recovery_timer: int = 0
    final_recovery_time: int = 0


class TrafficSpawner:
    def __init__(self, task_config: Any):
        self.config = task_config

    def arrivals_for_step(self, step_count: int, split_fn: Any) -> dict[str, list[dict[str, float]]]:
        arrivals: dict[str, list[dict[str, float]]] = {}
        spike_vector = [1.0, 1.0, 1.0, 1.0]
        if step_count in set(getattr(self.config, "spike_steps", ())):
            spike_vector = list(getattr(self.config, "spike_multipliers", (1.0, 1.0, 1.0, 1.0)))

        step_rng = random.Random(self.config.seed + step_count)
        for node in INTERSECTIONS:
            node_idx = NODE_INDEX[node]
            node_scale = self.config.node_demand_scale[node_idx]
            node_arrivals: list[dict[str, float]] = []
            for lane in range(4):
                base = self.config.arrival_base[lane]
                jitter = self.config.arrival_jitter[lane]
                
                # Phase 5: Dynamic demand rotation
                if self.config.task_id == "dynamic_demand":
                    rotation_cycle = (step_count // 25) % 4
                    if rotation_cycle == 0: # Horizontal
                        bias = 2.0 if lane in {1, 3} else 0.5
                    elif rotation_cycle == 1: # Vertical
                        bias = 2.0 if lane in {0, 2} else 0.5
                    elif rotation_cycle == 2: # Balanced
                        bias = 1.0
                    else: # Asymmetric
                        bias = 1.5 if lane in {0, 3} else 0.7
                else:
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

                for incident in getattr(self.config, "incidents", ()):
                    if (
                        incident.incident_type == "DEMAND_SURGE"
                        and incident.intersection_id == node
                        and incident.lane_id == lane
                        and incident.start_step <= step_count < incident.start_step + incident.duration
                    ):
                        value *= 1.5

                noise = 1.0 + step_rng.uniform(-0.1, 0.1)
                node_arrivals.append(split_fn(node, max(0.0, value * noise)))
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
        self.normalize_obs = False

    def reset(self, task_id: str | None = None, central_enabled: bool = False, normalize_obs: bool = False) -> dict[str, Any]:
        self.central_enabled = central_enabled
        self.normalize_obs = normalize_obs
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
            lane_movements: list[dict[str, float]] = []
            waits: list[float] = []
            lane_blocked: list[bool] = []
            blockage_durations: list[int] = []
            for lane in range(4):
                queue_total = round(self.random.uniform(low, high) * base_scale * self.task_config.directional_bias[lane], 2)
                movement_queue = self._split_total_to_movements(node, queue_total)
                lane_movements.append(movement_queue)
                waits.append(round(queue_total * 1.25 + self.random.uniform(0.5, 2.0), 2))
                cap = self._lane_capacity(node, lane)
                blocked = queue_total >= cap
                lane_blocked.append(blocked)
                blockage_durations.append(1 if blocked else 0)
            intersections[node] = IntersectionState(
                movement_queues=lane_movements,
                waiting_times=waits,
                lane_blocked=lane_blocked,
                blockage_durations=blockage_durations,
                phase_history=[0, 0, 0],
                time_since_served=[0.0, 0.0, 0.0, 0.0]
            )

        delay_steps = max(3, int(self.task_config.transfer_delay_steps))
        transit_buffers = {
            (up_node, up_lane, down_node, down_lane): [0.0] * delay_steps
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

        blocked_before = {
            node: list(self.state_obj.intersections[node].lane_blocked)
            for node in INTERSECTIONS
        }
        switched_map = self._apply_local_actions(local_actions)
        released_transfers = self._release_transfers(blocked_before)
        external_arrivals = self.spawner.arrivals_for_step(self.state_obj.step_count, self._split_total_to_movements)

        total_throughput = 0.0
        node_throughputs = {node: 0.0 for node in INTERSECTIONS}
        lane_throughputs = {node: [0.0 for _ in range(4)] for node in INTERSECTIONS}
        spillback_this_step = 0
        new_breach_count = 0
        capacity_breach_event: list[str] = []

        spawned_total = sum(
            _sum_movements(external_arrivals[node][lane]) for node in INTERSECTIONS for lane in range(4)
        )
        self.state_obj.total_demand_spawned += spawned_total

        # Phase 5: Incident response latency tracking
        active_incidents = self._get_active_incidents()
        for inc in active_incidents:
            inc_key = f"{inc['intersection_id']}:{inc['lane_id']}:{inc['incident_type']}"
            if inc_key not in self.state_obj.incident_start_steps:
                self.state_obj.incident_start_steps[inc_key] = self.state_obj.step_count
        
        for node in INTERSECTIONS:
            state = self.state_obj.intersections[node]
            green_lane = state.current_phase
            switched = switched_map[node]

            for lane in range(4):
                incoming_movements = self._add_movement_dicts(external_arrivals[node][lane], released_transfers[node][lane])
                queue_before_movements = self._add_movement_dicts(state.movement_queues[lane], incoming_movements)
                queue_before = _sum_movements(queue_before_movements)
                service = self._service_rate(node, lane, green_lane, switched)

                route = ROUTES.get((node, lane))
                spillback_factor = 1.0
                if route is not None:
                    down_node, down_lane = route
                    downstream_queue = self.state_obj.intersections[down_node].queue_lengths[down_lane]
                    downstream_capacity = self._lane_capacity(down_node, down_lane)
                    capacity_ratio = downstream_queue / max(downstream_capacity, 1e-6)
                    if blocked_before[down_node][down_lane]:
                        spillback_factor *= 0.5
                        spillback_this_step += 1
                    elif capacity_ratio > 0.68:
                        spillback_factor *= max(0.05, 1.0 - (capacity_ratio - 0.68) * 2.3)
                        spillback_this_step += 1

                allowed_movements = PHASE_MOVEMENT_MAP[node][green_lane] if lane == green_lane else ()
                cleared_movements = self._clear_movements(queue_before_movements, service * spillback_factor, allowed_movements)
                cleared = _sum_movements(cleared_movements)
                if route is not None and cleared > 0.0:
                    self._push_transfer(node, lane, cleared * self.task_config.route_transfer_ratio)

                next_movements = {
                    movement: max(0.0, queue_before_movements[movement] - cleared_movements[movement])
                    for movement in MOVEMENTS
                }
                next_queue = _sum_movements(next_movements)
                prev_wait = state.waiting_times[lane]
                incoming = _sum_movements(incoming_movements)
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

                lane_capacity = self._lane_capacity(node, lane)
                breached = next_queue >= lane_capacity
                if breached:
                    if not state.lane_blocked[lane]:
                        new_breach_count += 1
                        capacity_breach_event.append(f"{node}:{lane}")
                    next_movements = self._scale_to_capacity(next_movements, lane_capacity)
                    next_queue = lane_capacity
                    state.lane_blocked[lane] = True
                    state.blockage_durations[lane] += 1
                else:
                    state.lane_blocked[lane] = False
                    state.blockage_durations[lane] = 0

                state.set_lane_total(lane, next_movements)
                state.waiting_times[lane] = next_wait
                total_throughput += cleared
                node_throughputs[node] += cleared
                lane_throughputs[node][lane] = cleared
                
                # Update phase memory and fairness counters
                if lane == green_lane and cleared > 0.5:
                    state.time_since_served[lane] = 0.0
                else:
                    state.time_since_served[lane] += 1.0

        for node in INTERSECTIONS:
            state = self.state_obj.intersections[node]
            if state.phase_history[-1] != state.current_phase:
                state.phase_history.append(state.current_phase)
                while len(state.phase_history) > 3:
                    state.phase_history.pop(0)

        current_max_starvation = max(
            max(state.time_since_served) for state in self.state_obj.intersections.values()
        )
        self.state_obj.max_starvation_time = max(self.state_obj.max_starvation_time, current_max_starvation)

        blocked_duration_pressure = sum(
            sum(state.blockage_durations) for state in self.state_obj.intersections.values()
        )
        self.state_obj.blockage_duration_sum += blocked_duration_pressure
        self.state_obj.spillback_events_count += spillback_this_step + new_breach_count
        if not any(switched_map.values()):
            self.state_obj.phase_hold_bonus_steps += 1

        self.state_obj.step_count += 1
        self.state_obj.total_throughput += total_throughput
        self.state_obj.total_in_transit = self._total_in_transit()

        # Phase 5: Calculate response latency
        if active_behaviors and self.state_obj.incident_start_steps:
            for inc_key, start_step in list(self.state_obj.incident_start_steps.items()):
                if inc_key not in self.state_obj.incident_response_latency:
                    latency = self.state_obj.step_count - start_step
                    self.state_obj.incident_response_latency[inc_key] = latency

        step_metrics = self._metrics(total_throughput, switched_map, spillback_this_step + new_breach_count)
        adv_metrics = self._advanced_metrics(total_throughput, switched_map)
        reward, breakdown = self._reward(step_metrics, adv_metrics)
        self.state_obj.episode_reward_sum += reward
        self.state_obj.episode_wait_sum += step_metrics["mean_wait"]
        self.state_obj.episode_queue_sum += step_metrics["mean_queue"]
        self.state_obj.done = self.state_obj.step_count >= self.task_config.max_steps

        self._record_history(node_throughputs, lane_throughputs, switched_map)

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
            "capacity_breach_event": capacity_breach_event,
            "total_in_transit": round(self.state_obj.total_in_transit, 3),
            "active_incidents": self._get_active_incidents(),
            "active_corridors": list(self.central_state.active_corridors),
            "budget_remaining": round(self.central_state.budget_remaining, 3),
            "rationale": self.central_state.last_rationale,
            "oversight_interventions_count": self.central_state.oversight_interventions_count,
            "max_starvation_time": self.state_obj.max_starvation_time,
            "incident_response_latency": self.state_obj.incident_response_latency,
            "anticipatory_spillback_score": {
                node: max([v for k, v in self.central_state.anticipatory_spillback_scores.items() if k.startswith(node)] or [0.0])
                for node in INTERSECTIONS
            }
        }
        
        # Phase 6: Advanced Metrics Integration
        info.update({k: round(v, 4) for k, v in adv_metrics.items()})
        info["reward_breakdown"] = breakdown

        # Phase 6: Recovery Time Logic
        current_eff = adv_metrics["throughput_efficiency"]
        if self.state_obj.step_count <= 10:
            # Establishing baseline
            self.state_obj.efficiency_baseline = (self.state_obj.efficiency_baseline * (self.state_obj.step_count - 1) + current_eff) / self.state_obj.step_count
        
        if active_incidents and not self.state_obj.recovery_tracking_active:
            self.state_obj.recovery_tracking_active = True
            self.state_obj.recovery_timer = 0
        
        if self.state_obj.recovery_tracking_active:
            if not active_incidents:
                self.state_obj.recovery_timer += 1
                if current_eff >= self.state_obj.efficiency_baseline * 0.9:
                    if self.state_obj.final_recovery_time == 0:
                        self.state_obj.final_recovery_time = self.state_obj.recovery_timer
                    self.state_obj.recovery_tracking_active = False
            else:
                # Reset timer if a new incident is still active
                self.state_obj.recovery_timer = 0
        
        info["recovery_time"] = self.state_obj.final_recovery_time

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

    def _release_transfers(self, blocked_before: dict[str, list[bool]]) -> dict[str, list[dict[str, float]]]:
        assert self.state_obj is not None
        released = {node: [_empty_movement_queue() for _ in range(4)] for node in INTERSECTIONS}
        for key, buffer in self.state_obj.transit_buffers.items():
            _, _, down_node, down_lane = key
            arriving = buffer.pop(0)
            buffer.append(0.0)
            if blocked_before[down_node][down_lane]:
                buffer[0] += arriving
                continue
            released[down_node][down_lane] = self._add_movement_dicts(
                released[down_node][down_lane],
                self._split_total_to_movements(down_node, arriving),
            )
        return released

    def _push_transfer(self, node: str, lane: int, amount: float) -> None:
        assert self.state_obj is not None
        route = ROUTES.get((node, lane))
        if route is None:
            return
        down_node, down_lane = route
        key = (node, lane, down_node, down_lane)
        self.state_obj.transit_buffers[key][-1] += amount

    def _record_history(self, node_throughputs: dict[str, float], lane_throughputs: dict[str, list[float]], switched_map: dict[str, bool]) -> None:
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
        self.central_state.history.add(queues, waits, node_throughputs, lane_throughputs, switches, self.state_obj.policy.copy())

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
            max_downstream_ratio = max(max_downstream_ratio, queue / max(self._lane_capacity(down_node, down_lane), 1e-6))

        emergency_pressure = 0.0
        if self.task_config.emergency_lane is not None:
            emergency_pressure = (
                self.state_obj.intersections["SW"].queue_lengths[self.task_config.emergency_lane] / max(lane_capacity, 1e-6)
            )
        blocked_lane_fraction = sum(
            1.0
            for node in INTERSECTIONS
            for blocked in self.state_obj.intersections[node].lane_blocked
            if blocked
        ) / 16.0

        throughput_decline = _clip((-trend_features["throughput_trend"]) / 6.0, 0.0, 1.0)
        queue_growth = _clip(trend_features["queue_growth_rate"] / 10.0, 0.0, 1.0)
        wait_growth = _clip(trend_features["wait_growth_rate"] / 5.0, 0.0, 1.0)

        return {
            "spillback": _clip((max_downstream_ratio - 0.65) / 0.3 + queue_growth * 0.4 + blocked_lane_fraction * 0.9, 0.0, 1.0),
            "corridor": _clip(corridor_gap / 40.0 + throughput_decline * 0.25, 0.0, 1.0),
            "load_balance": _clip(wait_growth * 0.35 + queue_growth * 0.45 + corridor_gap / 60.0, 0.0, 1.0),
            "emergency": _clip(max(0.0, emergency_pressure - 0.25) / 0.75 + wait_growth * 0.2, 0.0, 1.0),
            "instability": _clip(trend_features["switching_instability"] / 4.0, 0.0, 1.0),
        }

    def _detect_active_corridors(self) -> list[str]:
        assert self.central_state is not None
        history = self.central_state.history
        if not history.lane_throughputs:
            return []
        last_snapshot = history.lane_throughputs[-1]
        all_flows = [flow for node_flows in last_snapshot.values() for flow in node_flows]
        mean_flow = sum(all_flows) / 16.0 if all_flows else 0.0
        threshold = 1.3 * mean_flow
        corridors = []
        for (up_node, up_lane), (down_node, down_lane) in ROUTES.items():
            flow = last_snapshot[up_node][up_lane]
            if flow > threshold and flow > 0.5:
                corridors.append(f"{up_node}{up_lane}->{down_node}{down_lane}")
        corridors.sort(key=lambda x: last_snapshot[x[:2]][int(x[2])], reverse=True)
        return corridors[:2]

    def _compute_anticipatory_scores(self) -> dict[str, float]:
        assert self.state_obj is not None
        assert self.central_state is not None
        history = self.central_state.history
        if not history.lane_throughputs:
            return {}
        scores = {}
        last_flows = history.lane_throughputs[-1]
        for (up_node, up_lane), (down_node, down_lane) in ROUTES.items():
            down_state = self.state_obj.intersections[down_node]
            down_q = down_state.queue_lengths[down_lane]
            down_cap = self._lane_capacity(down_node, down_lane)
            up_flow = last_flows[up_node][up_lane]
            score = (down_q / max(down_cap, 1.0)) * up_flow
            scores[f"{up_node}:{up_lane}"] = round(score, 3)
        return scores

    def _update_central_policy(self, central_action: dict[str, float] | None) -> list[str]:
        assert self.state_obj is not None
        assert self.central_state is not None
        if not self.central_enabled:
            self.central_state.last_policy_delta = {name: 0.0 for name in POLICY_ORDER}
            self.central_state.last_policy_stability = 1.0
            self.central_state.last_rationale = "Central controller disabled."
            return []

        trend_features = self._compute_trend_features()
        risks = self._estimate_risks(trend_features)
        self.central_state.active_corridors = self._detect_active_corridors()
        self.central_state.anticipatory_spillback_scores = self._compute_anticipatory_scores()

        blocked_lanes_present = any(any(self.state_obj.intersections[node].lane_blocked) for node in INTERSECTIONS)
        raw_flags = {
            "spillback_prevention": blocked_lanes_present or detect_spillback_risk(self.state_obj, self.task_config.lane_capacity) or risks["spillback"] > 0.55,
            "corridor_formation": detect_corridor_imbalance(self.state_obj) or risks["corridor"] > 0.5 or bool(self.central_state.active_corridors),
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

        # 1. Base Logic
        if "spillback_prevention" in active_behaviors:
            target["queue_urgency_weight"] += 0.65 + risks["spillback"] * 1.35
            target["balance_penalty"] += 1.5 + risks["spillback"] * 2.5
        if "corridor_formation" in active_behaviors:
            corridor_boost = 1.0 + risks["corridor"] * 2.5 + throughput_decline * 0.6
            if self.central_state.active_corridors:
                # Divide priority proportionally (simplified: full boost if active)
                target["corridor_priority"] += corridor_boost
            else:
                target["corridor_priority"] += corridor_boost * 0.5
        if "load_balancing" in active_behaviors:
            target["queue_urgency_weight"] += 0.5 + risks["load_balance"] * 1.2
            target["balance_penalty"] += 0.4 + risks["load_balance"] * 0.9
        if "emergency_routing" in active_behaviors:
            target["emergency_boost"] += 1.5 + risks["emergency"] * 4.5
        if "stability_control" in active_behaviors:
            target["switch_penalty"] += 0.55 + risks["instability"] * 1.15

        # 2. Anticipatory spillback score adjustment
        max_pred_score = max(self.central_state.anticipatory_spillback_scores.values()) if self.central_state.anticipatory_spillback_scores else 0.0
        if max_pred_score > 0.7:
            # We used to reduce global priority here, but it's too blunt.
            # We'll handle it locally in _service_rate for the upstream intersection.
            if "spillback_prevention" not in active_behaviors:
                active_behaviors.append("anticipatory_spillback_mitigation")

        # 3. Rationale Generation
        rationale_parts = []
        if active_behaviors:
            rationale_parts.append(f"Addressing {', '.join(active_behaviors[:2])}")
        if self.central_state.active_corridors:
            rationale_parts.append(f"prioritizing corridors {', '.join(self.central_state.active_corridors)}")
        if max_pred_score > 0.7:
            rationale_parts.append("mitigating anticipatory spillback")
        self.central_state.last_rationale = "; ".join(rationale_parts) + "." if rationale_parts else "Maintained stable policy."

        if central_action:
            for name, delta in central_action.items():
                if name in target:
                    target[name] += float(delta)

        # 4. Priority Budget Constraint
        budget = getattr(self.task_config, "total_priority_budget", 6.0)
        boosts = {name: max(0.0, target[name] - DEFAULT_POLICY[name]) for name in POLICY_ORDER}
        total_boost = sum(boosts.values())
        if total_boost > budget:
            # Protect emergency_boost, reduce others proportionally
            non_emergency_boost = total_boost - boosts.get("emergency_boost", 0.0)
            reduction_needed = total_boost - budget
            if non_emergency_boost > 0:
                scale = max(0.0, (non_emergency_boost - reduction_needed) / non_emergency_boost)
                for name in boosts:
                    if name != "emergency_boost":
                        target[name] = DEFAULT_POLICY[name] + boosts[name] * scale
            else:
                # Even emergency boost might need reduction if it alone exceeds budget? 
                # User says "reduce lower-priority", implying emergency is high.
                # If emergency alone > budget, we clip it to budget.
                if boosts.get("emergency_boost", 0.0) > budget:
                    target["emergency_boost"] = DEFAULT_POLICY["emergency_boost"] + budget
        self.central_state.budget_remaining = max(0.0, budget - sum(max(0.0, target[n] - DEFAULT_POLICY[n]) for n in POLICY_ORDER))

        # 5. Safety Oversight Monitor
        oversight_triggered = False
        if "emergency_routing" in active_behaviors and target["emergency_boost"] < 3.0:
            target["emergency_boost"] = 4.0
            oversight_triggered = True
        if self.central_state.budget_remaining < -0.01: # Small epsilon
            oversight_triggered = True # Budget logic above should prevent this, but monitor flags it
        
        if oversight_triggered:
            self.central_state.oversight_interventions_count += 1
            self.state_obj.oversight_interventions_total += 1

        # 6. Apply with Delta Clipping
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
            downstream_capacity = self._lane_capacity(down_node, down_lane)
            downstream_penalty = (downstream_queue / max(downstream_capacity, 1e-6)) * float(personality["downstream"])

        service = self.task_config.service_base * (1.1 if self.central_enabled else 1.0)
        service += 0.14 * min(queue, 12.0) * float(personality["queue"])
        service += (
            0.24
            * min(queue, 12.0)
            * max(0.0, policy["queue_urgency_weight"] - DEFAULT_POLICY["queue_urgency_weight"])
            * float(personality["queue"])
        )

        if lane == green_lane:
            service += self.task_config.green_bonus * (1.3 if self.central_enabled else 1.0)
            # Phase 3: Selective corridor priority with starvation prevention
            boost = max(0.0, policy["corridor_priority"] - DEFAULT_POLICY["corridor_priority"])
            if lane in {1, 3}:
                # Original base boost for E-W
                service += 0.45
            
            if boost > 0:
                active = self.central_state.active_corridors if self.central_state else []
                if active:
                    # Additional boost divided between base E-W and active corridors
                    if lane in {1, 3}:
                        service += 3.5 * boost * 0.6
                    if any(c.startswith(f"{node}{lane}") for c in active):
                        service += 3.5 * boost * 0.6
                elif lane in {1, 3}:
                    # Fallback to full boost for E-W
                    service += 3.5 * boost

            # Phase 3: Anticipatory spillback mitigation (preemptive throttling)
            scores = self.central_state.anticipatory_spillback_scores if self.central_state else {}
            if scores.get(f"{node}:{lane}", 0.0) > 0.7:
                service *= 0.85 # Throttling upstream outflow to prevent downstream spillback
        else:
            service -= self.task_config.red_penalty

        if switched:
            service -= policy["switch_penalty"] * 0.75

        if route is not None:
            service -= downstream_penalty * policy["balance_penalty"] * 1.25
            down_node, down_lane = route
            if self.state_obj.intersections[down_node].lane_blocked[down_lane]:
                service *= 0.5

        if self.state_obj.intersections[node].lane_blocked[lane] and lane == green_lane:
            service += 0.2 + 3.2 * max(0.0, policy["queue_urgency_weight"] - DEFAULT_POLICY["queue_urgency_weight"])

        if (
            node == "SW"
            and self.task_config.emergency_lane is not None
            and lane == self.task_config.emergency_lane
            and self.state_obj.step_count >= (self.task_config.emergency_step or 10**9)
        ):
            service += policy["emergency_boost"] * float(personality["emergency"])

        if self.task_config.multi_intersection and lane in {0, 3}:
            service += 0.35

        # Phase 2: Blockage handling
        for incident in getattr(self.task_config, "incidents", ()):
            if (
                incident.incident_type == "BLOCKAGE"
                and incident.intersection_id == node
                and incident.lane_id == lane
                and incident.start_step <= self.state_obj.step_count < incident.start_step + incident.duration
            ):
                return 0.0

        node_rng = random.Random(self.task_config.seed + self.state_obj.step_count + NODE_INDEX[node])
        service_noise = 1.0
        for _ in range(lane + 1):
            service_noise = 1.0 + node_rng.uniform(-0.05, 0.05)

        return max(0.35, service * service_noise)

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
        blocked_pressure = sum(sum(state.blockage_durations) for state in self.state_obj.intersections.values())

        # Phase 2: Incident-aware weighted queue
        active_incidents = self._get_active_incidents(self.state_obj.step_count - 1)
        active_set = {(inc["intersection_id"], inc["lane_id"]) for inc in active_incidents}
        weighted_sum = 0.0
        for node in INTERSECTIONS:
            for lane in range(4):
                q = self.state_obj.intersections[node].queue_lengths[lane]
                weight = 2.5 if (node, lane) in active_set else 1.0
                weighted_sum += q * weight
        weighted_mean_queue = weighted_sum / 16.0

        return {
            "mean_queue": mean_queue,
            "incident_weighted_mean_queue": weighted_mean_queue,
            "mean_wait": mean_wait,
            "throughput": throughput,
            "imbalance": imbalance,
            "spillback_count": float(spillback_this_step),
            "emergency_delay_step": self._current_emergency_wait(),
            "corridor_sync_score": corridor_sync,
            "policy_delta": policy_delta,
            "switches_count": float(sum(1 for switched in switched_map.values() if switched)),
            "blocked_pressure": float(blocked_pressure),
        }

    def _advanced_metrics(self, throughput: float, switched_map: dict[str, bool]) -> dict[str, float]:
        assert self.state_obj is not None
        all_waits = [w for state in self.state_obj.intersections.values() for w in state.waiting_times]
        max_wait = max(all_waits) if all_waits else 0.0
        mean_wait = _mean(all_waits)
        fairness = 1.0 - (max_wait - mean_wait) / max(max_wait, 1e-6) if max_wait > 1e-6 else 1.0

        efficiency = self.state_obj.total_throughput / max(self.state_obj.total_demand_spawned, 1e-6)
        
        service_est = self.task_config.service_base + self.task_config.green_bonus / 4.0
        travel_times = [q / max(service_est, 1e-6) for state in self.state_obj.intersections.values() for q in state.queue_lengths]
        tt_mean = _mean(travel_times)
        tt_var = _variance(travel_times)

        policy_vals = list(self.state_obj.policy.values())
        policy_var = _variance(policy_vals)
        switches = [1.0 if v else 0.0 for v in switched_map.values()]
        switch_var = _variance(switches)
        stability = 1.0 - (policy_var + switch_var) / 2.0

        return {
            "travel_time_mean": tt_mean,
            "travel_time_variance": tt_var,
            "throughput_efficiency": efficiency,
            "fairness_score": fairness,
            "stability_index": stability,
        }

    def _reward(self, metrics: dict[str, float], adv_metrics: dict[str, float]) -> tuple[float, dict[str, float]]:
        # Phase 8: Detailed Reward Breakdown with fairness and stability
        queue_val = metrics.get("incident_weighted_mean_queue", metrics["mean_queue"])
        queue_term = 0.24 * _normalize_negative(queue_val, self.task_config.lane_capacity * 4.0)
        wait_term = 0.22 * _normalize_negative(metrics["mean_wait"], 35.0)
        throughput_term = 0.18 * _normalize_positive(metrics["throughput"], 34.0)
        imbalance_term = 0.10 * _normalize_negative(metrics["imbalance"], 22.0)
        emergency_term = 0.06 * _normalize_negative(metrics["emergency_delay_step"], 40.0)
        spillback_term = 0.04 * _normalize_negative(metrics["spillback_count"], 8.0)
        blockage_term = 0.03 * _normalize_negative(metrics["blocked_pressure"], 12.0)
        
        stability_bonus = 0.04 * _normalize_negative(metrics["policy_delta"], 1.2)
        coordination_bonus = 0.04 * (2.0 * metrics["corridor_sync_score"] - 1.0)
        fairness_reward = 0.05 * (2.0 * adv_metrics["fairness_score"] - 1.0)
        
        # Central reward as oversight components
        central_reward = imbalance_term + emergency_term

        total = (
            queue_term + wait_term + throughput_term + 
            spillback_term + blockage_term + stability_bonus + 
            coordination_bonus + fairness_reward + central_reward
        )
        # Ensure strict bounding
        total = float(_clip(total, -1.0, 1.0))
        
        breakdown = {
            "queue_reward": round(queue_term, 4),
            "wait_reward": round(wait_term, 4),
            "throughput_reward": round(throughput_term, 4),
            "switch_penalty_reward": round(spillback_term + blockage_term, 4),
            "central_reward": round(central_reward, 4),
            "stability_bonus": round(stability_bonus, 4),
            "coordination_bonus": round(coordination_bonus, 4),
            "fairness_reward": round(fairness_reward, 4),
            "total": round(total, 4)
        }
        self.state_obj.coordination_bonus_total += coordination_bonus
        return total, breakdown

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
            "movement_queue_lengths": {
                node: [_round_movement_dict(values) for values in self.state_obj.intersections[node].movement_queues]
                for node in INTERSECTIONS
            },
            "lane_status": {
                node: [
                    {
                        "blocked": self.state_obj.intersections[node].lane_blocked[lane],
                        "blockage_duration": self.state_obj.intersections[node].blockage_durations[lane],
                        "total_queue": round(self.state_obj.intersections[node].queue_lengths[lane], 2),
                        "capacity": round(self._lane_capacity(node, lane), 2),
                    }
                    for lane in range(4)
                ]
                for node in INTERSECTIONS
            },
            "transit_buffers": {
                f"{up_node}:{up_lane}->{down_node}:{down_lane}": [round(value, 2) for value in buffer]
                for (up_node, up_lane, down_node, down_lane), buffer in self.state_obj.transit_buffers.items()
            },
            "waiting_times": {
                node: [round(value, 2) for value in self.state_obj.intersections[node].waiting_times]
                for node in INTERSECTIONS
            },
            "current_phase": {node: self.state_obj.intersections[node].current_phase for node in INTERSECTIONS},
            "phase_history": {node: list(self.state_obj.intersections[node].phase_history) for node in INTERSECTIONS},
            "time_since_served": {node: list(self.state_obj.intersections[node].time_since_served) for node in INTERSECTIONS},
            "time_in_phase": {node: self.state_obj.intersections[node].time_in_phase for node in INTERSECTIONS},
            "policy": {name: round(value, 3) for name, value in self.state_obj.policy.items()},
            "text_obs": self.format_central_llm_prompt(),
        }

    def format_central_llm_prompt(self) -> str:
        assert self.state_obj is not None
        assert self.central_state is not None
        norm = self.normalize_obs
        cap = self.task_config.lane_capacity
        
        def n(val: float, scale: float) -> str:
            if norm:
                return f"{min(1.0, max(0.0, val / scale)):.2f}"
            return f"{val:.1f}"

        trends = self.central_state.last_trend_features or {
            "queue_growth_rate": 0.0,
            "wait_growth_rate": 0.0,
            "throughput_trend": 0.0,
            "switching_instability": 0.0,
        }
        lines = [
            f"step: {self.state_obj.step_count}",
            f"central_enabled: {str(self.central_enabled).lower()}",
            "policy="
            + ", ".join(f"{name}:{self.state_obj.policy[name]:.2f}" for name in POLICY_ORDER),
            "trends="
            + ", ".join(
                f"{name.split('_')[0]}:{n(value, 2.5) if 'instability' in name else n(value, 5.0)}"
                for name, value in trends.items()
            ),
        ]
        if self.state_obj.active_behaviors_log:
            lines.append(f"active: {self.state_obj.active_behaviors_log[-1].split(':', 1)[-1]}")
        for node in INTERSECTIONS:
            state = self.state_obj.intersections[node]
            lane_parts: list[str] = []
            for lane in range(4):
                mv_vals = "/".join(n(state.movement_queues[lane][mv], cap) for mv in MOVEMENTS)
                blocked = "B" if state.lane_blocked[lane] else "O"
                lane_parts.append(f"{lane}:{mv_vals}:{blocked}")
            lines.append(
                f"{node}: ph={state.current_phase} hold={state.time_in_phase} q="
                + ",".join(n(v, cap) for v in state.queue_lengths)
                + " m="
                + "|".join(lane_parts)
            )
        buffer_parts = [
            f"{up_node}{up_lane}>{down_node}{down_lane}={'/'.join(n(v, 8.0) for v in buffer)}"
            for (up_node, up_lane, down_node, down_lane), buffer in self.state_obj.transit_buffers.items()
        ]
        lines.append(f"transit_total: {n(self._total_in_transit(), cap * 4)}")
        lines.append("buffers: " + "; ".join(buffer_parts))
        text_obs = "\n".join(lines)
        if len(text_obs) <= 995:
            return text_obs
        compact_buffers = [
            f"{up_node}{up_lane}>{down_node}{down_lane}:{sum(buffer):.1f}"
            for (up_node, up_lane, down_node, down_lane), buffer in self.state_obj.transit_buffers.items()
        ]
        lines[-1] = "buffers: " + "; ".join(compact_buffers)
        return "\n".join(lines)[:995]

    def episode_summary(self) -> dict[str, Any]:
        assert self.state_obj is not None
        steps = max(self.state_obj.step_count, 1)
        node_queue_totals = [sum(state.queue_lengths) for state in self.state_obj.intersections.values()]
        imbalance = math.sqrt(_mean([(value - _mean(node_queue_totals)) ** 2 for value in node_queue_totals]))
        policy_stability = 1.0 - _clip(self.state_obj.policy_change_sum / max(steps * 0.12, 1e-6), 0.0, 1.0)
        # Episode-level advanced metrics
        adv = self._advanced_metrics(self.state_obj.total_throughput / steps, {n: False for n in INTERSECTIONS})
        
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
            "travel_time_mean": round(adv["travel_time_mean"], 4),
            "travel_time_variance": round(adv["travel_time_variance"], 4),
            "throughput_efficiency": round(adv["throughput_efficiency"], 4),
            "fairness_score": round(adv["fairness_score"], 4),
            "stability_index": round(adv["stability_index"], 4),
            "recovery_time": self.state_obj.final_recovery_time,
            "text_obs": self.format_central_llm_prompt(),
        }

    def _turn_ratios(self, node: str) -> tuple[float, float, float]:
        ratios = getattr(self.task_config, "turn_ratios", {}).get(node, (0.25, 0.55, 0.2))
        left, straight, right = ratios
        total = max(left + straight + right, 1e-6)
        return (left / total, straight / total, right / total)

    def _split_total_to_movements(self, node: str, total: float) -> dict[str, float]:
        left, straight, right = self._turn_ratios(node)
        left_value = max(0.0, total * left)
        straight_value = max(0.0, total * straight)
        right_value = max(0.0, total - left_value - straight_value)
        return {
            "left": round(left_value, 6),
            "straight": round(straight_value, 6),
            "right": round(right_value, 6),
        }

    def _lane_capacity(self, node: str, lane: int) -> float:
        capacities = getattr(self.task_config, "lane_capacities", {})
        base_cap = float(self.task_config.lane_capacity)
        if node in capacities and lane < len(capacities[node]):
            base_cap = float(capacities[node][lane])
        
        # Phase 2: Incident capacity modification
        step = self.state_obj.step_count if self.state_obj else 0
        for incident in getattr(self.task_config, "incidents", ()):
            if incident.intersection_id == node and incident.lane_id == lane:
                if incident.start_step <= step < incident.start_step + incident.duration:
                    if incident.incident_type == "LANE_CLOSURE":
                        base_cap *= 0.5
                    elif incident.incident_type == "BLOCKAGE":
                        base_cap *= 0.1 # Very low capacity to trigger blockage faster, but flow is zero anyway
        return base_cap

    def _get_active_incidents(self, step: int | None = None) -> list[dict[str, Any]]:
        if self.state_obj is None and step is None:
            return []
        current_step = step if step is not None else self.state_obj.step_count
        active = []
        for inc in getattr(self.task_config, "incidents", ()):
            if inc.start_step <= current_step < inc.start_step + inc.duration:
                active.append({
                    "intersection_id": inc.intersection_id,
                    "lane_id": inc.lane_id,
                    "incident_type": inc.incident_type,
                    "severity": inc.severity,
                    "start_step": inc.start_step,
                    "duration": inc.duration
                })
        return active

    def _add_movement_dicts(self, left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
        return {
            movement: float(left.get(movement, 0.0)) + float(right.get(movement, 0.0))
            for movement in MOVEMENTS
        }

    def _clear_movements(
        self,
        queue_before: dict[str, float],
        clearance_budget: float,
        allowed_movements: tuple[str, ...] | list[str],
    ) -> dict[str, float]:
        if clearance_budget <= 0.0 or not allowed_movements:
            return _empty_movement_queue()

        weighted_total = sum(
            queue_before[movement] * TURN_SERVICE_MULTIPLIER[movement]
            for movement in allowed_movements
        )
        if weighted_total <= 0.0:
            return _empty_movement_queue()

        cleared = _empty_movement_queue()
        remaining_budget = clearance_budget
        remaining_movements = list(allowed_movements)
        while remaining_budget > 1e-6 and remaining_movements:
            weight_sum = sum(
                max(queue_before[movement] - cleared[movement], 0.0) * TURN_SERVICE_MULTIPLIER[movement]
                for movement in remaining_movements
            )
            if weight_sum <= 1e-6:
                break
            progressed = False
            next_remaining: list[str] = []
            for movement in remaining_movements:
                remaining_queue = max(queue_before[movement] - cleared[movement], 0.0)
                if remaining_queue <= 1e-6:
                    continue
                share = remaining_budget * (
                    remaining_queue * TURN_SERVICE_MULTIPLIER[movement] / weight_sum
                )
                cleared_amount = min(remaining_queue, share)
                if cleared_amount > 1e-6:
                    progressed = True
                cleared[movement] += cleared_amount
                if remaining_queue - cleared_amount > 1e-6:
                    next_remaining.append(movement)
            remaining_budget = max(0.0, clearance_budget - _sum_movements(cleared))
            if not progressed:
                break
            remaining_movements = next_remaining
        return cleared

    def _scale_to_capacity(self, movement_queue: dict[str, float], lane_capacity: float) -> dict[str, float]:
        total = _sum_movements(movement_queue)
        if total <= lane_capacity:
            return movement_queue
        ratio = lane_capacity / max(total, 1e-6)
        return {movement: movement_queue[movement] * ratio for movement in MOVEMENTS}

    def _total_in_transit(self) -> float:
        assert self.state_obj is not None
        return sum(sum(buffer) for buffer in self.state_obj.transit_buffers.values())
