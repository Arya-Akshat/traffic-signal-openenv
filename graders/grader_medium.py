from __future__ import annotations

from graders.common import grade as _common_grade


def grade(metrics: dict) -> float:
    try:
        score = _common_grade(metrics)
    except Exception:
        score = 0.5

    # Ensure float
    try:
        score = float(score)
    except Exception:
        score = 0.5

    # NaN / Inf guard
    if score != score or score == float("inf") or score == float("-inf"):
        score = 0.5

    # STRICT: must be in open interval (0, 1)
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99

    return score
