from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    task_id: str = os.getenv("TASK_ID", "easy_fixed")
    max_steps: int = int(os.getenv("MAX_STEPS", "200"))
    use_real_sumo: bool = os.getenv("USE_REAL_SUMO", "0") == "1"
    sumo_home: str | None = os.getenv("SUMO_HOME")


settings = Settings()
