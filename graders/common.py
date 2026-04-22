from __future__ import annotations

def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))

def _positive_score(value: float, bound: float) -> float:
    return _clamp(value / max(bound, 1e-6))

def _negative_score(value: float, bound: float) -> float:
    return 1.0 - _clamp(value / max(bound, 1e-6))

def compute_detailed_rubrics(metrics: dict) -> dict[str, float]:
    # 1. Local Efficiency (0.25)
    mean_wait = float(metrics.get("mean_wait", 0.0) or 0.0)
    mean_queue = float(metrics.get("mean_queue", 0.0) or 0.0)
    local_eff = 0.5 * _negative_score(mean_wait, 120.0) + 0.5 * _negative_score(mean_queue, 100.0)
    
    # 2. Global Coordination (0.25)
    imbalance = float(metrics.get("imbalance", 0.0) or 0.0)
    sync = float(metrics.get("corridor_sync_score", 0.0) or 0.0)
    spillback = float(metrics.get("spillback_count", 0.0) or 0.0)
    global_coord = 0.4 * _negative_score(imbalance, 20.0) + 0.4 * _clamp(sync) + 0.2 * _negative_score(spillback, 30.0)
    
    # 3. Throughput (0.20)
    efficiency = float(metrics.get("throughput_efficiency", 0.0) or 0.0)
    throughput = _clamp(efficiency)
    
    # 4. Emergency Response (0.15)
    emergency_delay = float(metrics.get("emergency_delay", 0.0) or 0.0)
    emergency = _negative_score(emergency_delay, 15000.0)
    
    # 5. Stability (0.10)
    policy_stab = float(metrics.get("policy_stability", 0.0) or 0.0)
    phase_stab = float(metrics.get("stability_index", 0.0) or 0.0)
    stability = 0.5 * _clamp(policy_stab) + 0.5 * _clamp(phase_stab)
    
    # 6. Fairness (0.05)
    fairness_score = float(metrics.get("fairness_score", 0.0) or 0.0)
    max_starvation = float(metrics.get("max_starvation_time", 0.0) or 0.0)
    fairness = 0.6 * _clamp(fairness_score) + 0.4 * _negative_score(max_starvation, 45.0)
    
    return {
        "rubric_local_efficiency": round(local_eff, 4),
        "rubric_global_coordination": round(global_coord, 4),
        "rubric_throughput": round(throughput, 4),
        "rubric_emergency_response": round(emergency, 4),
        "rubric_stability": round(stability, 4),
        "rubric_fairness": round(fairness, 4),
    }

def compute_score(metrics: dict) -> float:
    rubrics = compute_detailed_rubrics(metrics)
    score = (
        0.25 * rubrics["rubric_local_efficiency"] +
        0.25 * rubrics["rubric_global_coordination"] +
        0.20 * rubrics["rubric_throughput"] +
        0.15 * rubrics["rubric_emergency_response"] +
        0.10 * rubrics["rubric_stability"] +
        0.05 * rubrics["rubric_fairness"]
    )
    return max(0.01, min(0.99, float(score)))

def grade(metrics: dict) -> float:
    try:
        score = float(compute_score(metrics))
    except Exception:
        score = 0.5
    if score != score or score in {float("inf"), float("-inf")}:
        score = 0.5
    return max(0.01, min(0.99, score))
