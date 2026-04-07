from __future__ import annotations


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def compute_score(
    metrics: dict,
    wait_norm: float = 40.0,
    throughput_norm: float = 20.0,
    queue_norm: float = 30.0,
) -> float:
    avg_wait = float(metrics.get("avg_wait", 0.0))
    throughput = float(metrics.get("throughput", 0.0))
    total_queue_length = float(metrics.get("total_queue_length", 0.0))

    normalized_wait = 1.0 - _clamp(avg_wait / wait_norm)
    throughput_score = _clamp(throughput / throughput_norm)
    queue_score = 1.0 - _clamp(total_queue_length / queue_norm)

    score = 0.5 * normalized_wait + 0.3 * throughput_score + 0.2 * queue_score

    score = float(score)

    # STRICT clamp
    score = max(0.01, min(0.99, score))
    assert 0 < score < 1, f"Invalid score: {score}"

    return score


def grade(metrics):
    try:
        score = compute_score(metrics)
    except Exception:
        score = 0.5

    # ensure float
    try:
        score = float(score)
    except Exception:
        score = 0.5

    # handle NaN
    if score != score:
        score = 0.5

    # handle inf
    if score == float("inf") or score == float("-inf"):
        score = 0.5

    # STRICT RANGE ENFORCEMENT
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99

    return score
