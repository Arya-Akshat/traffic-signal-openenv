from __future__ import annotations
from env.types import TrafficTask, Incident

# incident_response: Scheduled disruptions including lane closures and surges.
# Evaluates the system's ability to reroute traffic and recover from transient blockages.
def incident_grader(summary: dict) -> float:
    # Penalize high latency and spillbacks, reward throughput
    latencies = summary.get("incident_response_latency", {})
    avg_latency = sum(latencies.values()) / len(latencies) if latencies else 0.0
    tp = summary.get("throughput", 0.0)
    wait = summary.get("mean_wait", 99.0)
    score = (tp / 80.0) * 0.4 + (1.0 - wait / 100.0) * 0.4 + (1.0 - avg_latency / 60.0) * 0.2
    return max(0.01, min(0.99, score))

def get_incident_task(max_steps: int = 200) -> TrafficTask:
    return TrafficTask(
        task_id="incident_response",
        name="Sequential incidents requiring rapid coordination",
        max_steps=max_steps,
        seed=77,
        arrival_base=(4.0, 4.0, 4.0, 4.0),
        arrival_jitter=(0.6, 0.6, 0.6, 0.6),
        incidents=(
            Incident("NE", 0, "LANE_CLOSURE", start_step=20, duration=30, severity=0.7),
            Incident("SE", 1, "DEMAND_SURGE", start_step=50, duration=30, severity=1.0),
            Incident("NW", 2, "BLOCKAGE", start_step=80, duration=30, severity=1.0),
        ),
        multi_intersection=True,
        grader=incident_grader
    )
