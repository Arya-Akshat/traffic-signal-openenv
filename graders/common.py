from __future__ import annotations


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def compute_score(
    metrics: dict,
    wait_norm: float,
    throughput_norm: float,
    queue_norm: float,
) -> float:
    avg_wait = float(metrics.get("avg_wait", 0.0))
    throughput = float(metrics.get("throughput", 0.0))
    total_queue_length = float(metrics.get("total_queue_length", 0.0))

    normalized_wait = 1.0 - _clamp(avg_wait / wait_norm)
    throughput_score = _clamp(throughput / throughput_norm)
    queue_score = 1.0 - _clamp(total_queue_length / queue_norm)

    score = 0.5 * normalized_wait + 0.3 * throughput_score + 0.2 * queue_score

    # Clamp strictly inside (0, 1) for validator compatibility.
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99

    return float(round(score, 4))
