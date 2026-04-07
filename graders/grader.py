from __future__ import annotations

from graders.common import grade as common_grade


def grade(metrics: dict) -> float:
    return common_grade(metrics)
