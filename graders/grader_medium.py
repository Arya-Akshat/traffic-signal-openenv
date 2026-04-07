from __future__ import annotations

from graders.common import compute_score


def grade(metrics: dict) -> float:
    score = compute_score(
        metrics,
        wait_norm=40.0,
        throughput_norm=20.0,
        queue_norm=30.0,
    )
    score = float(score)
    score = max(0.01, min(0.99, score))
    assert 0 < score < 1, f"Invalid score: {score}"
    return score
