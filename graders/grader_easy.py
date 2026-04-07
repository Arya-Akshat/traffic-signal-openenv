from __future__ import annotations

from graders.common import compute_score


def grade(metrics: dict) -> float:
    return compute_score(metrics, wait_norm=35.0, throughput_norm=18.0, queue_norm=28.0)
