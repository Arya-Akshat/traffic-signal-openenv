from __future__ import annotations


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _positive_score(value: float, bound: float) -> float:
    return _clamp(value / max(bound, 1e-6))


def _negative_score(value: float, bound: float) -> float:
    return 1.0 - _clamp(value / max(bound, 1e-6))


def compute_score(metrics: dict) -> float:
    mean_wait = float(metrics.get("mean_wait", metrics.get("avg_wait", 0.0)) or 0.0)
    mean_queue = float(metrics.get("mean_queue", metrics.get("total_queue_length", 0.0)) or 0.0)
    throughput = float(metrics.get("throughput", 0.0) or 0.0)
    imbalance = float(metrics.get("imbalance", 0.0) or 0.0)
    emergency_delay = float(metrics.get("emergency_delay", 0.0) or 0.0)
    spillback_events = float(
        metrics.get("spillback_count", metrics.get("spillback_events_count", 0.0)) or 0.0
    )
    corridor_sync = float(metrics.get("corridor_sync_score", 0.0) or 0.0)
    policy_stability = float(metrics.get("policy_stability", 0.0) or 0.0)

    wait_score = _negative_score(mean_wait, 170.0)
    queue_score = _negative_score(mean_queue, 130.0)
    throughput_score = _positive_score(throughput, 140.0)
    imbalance_score = _negative_score(imbalance, 25.0)
    emergency_score = _negative_score(emergency_delay, 30000.0)
    spillback_score = _negative_score(spillback_events, 60.0)
    corridor_score = _clamp(corridor_sync)
    stability_score = _clamp(policy_stability)

    score = (
        0.20 * wait_score
        + 0.18 * queue_score
        + 0.18 * throughput_score
        + 0.10 * imbalance_score
        + 0.18 * emergency_score
        + 0.06 * spillback_score
        + 0.06 * corridor_score
        + 0.04 * stability_score
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
