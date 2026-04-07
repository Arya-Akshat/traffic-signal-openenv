from __future__ import annotations

from graders.common import compute_score


def grade(metrics: dict) -> float:
    return compute_score(metrics, wait_norm=40.0, throughput_norm=20.0, queue_norm=30.0)
