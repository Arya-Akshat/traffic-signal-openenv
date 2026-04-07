from __future__ import annotations

from graders.common import compute_score


def grade(metrics: dict) -> float:
    score = compute_score(
        metrics,
        wait_norm=35.0,
        throughput_norm=18.0,
        queue_norm=28.0,
    )
    score = float(score)
    score = max(0.01, min(0.99, score))
    assert 0 < score < 1, f"Invalid score: {score}"
    return score
