from __future__ import annotations


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def grade(metrics: dict) -> float:
    avg_wait = float(metrics.get("avg_wait", 0.0))
    throughput = float(metrics.get("throughput", 0.0))
    total_queue_length = float(metrics.get("total_queue_length", 0.0))

    normalized_wait = 1.0 - _clamp(avg_wait / 100.0)

    throughput_denom = max(throughput + total_queue_length, 1.0)
    throughput_score = _clamp(throughput / throughput_denom)

    queue_score = 1.0 - _clamp(total_queue_length / 50.0)

    score = (
        0.5 * normalized_wait
        + 0.3 * throughput_score
        + 0.2 * queue_score
    )
    return round(_clamp(score), 4)
