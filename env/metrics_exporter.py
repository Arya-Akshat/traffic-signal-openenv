import json
import csv
import os
from typing import Any

def export_episode_to_json(episode_log: list[dict[str, Any]], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(episode_log, f, indent=2)

def export_episode_to_csv(episode_log: list[dict[str, Any]], filepath: str) -> None:
    if not episode_log:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    keys = episode_log[0].keys()
    with open(filepath, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(episode_log)

def export_policy_trace(episode_log: list[dict[str, Any]], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # Policy is usually inside 'observation' or in the log top-level if we saved it
        # Assuming episode_log entries have a 'policy' key or it's in 'info'
        first_entry = episode_log[0]
        # Check if 'policy' is in info
        info = first_entry.get("info", {})
        # Wait, usually policy is a dict. Let's find policy keys.
        # Actually, let's just use whatever policy we can find.
        policy_keys = []
        if "policy" in first_entry:
            policy_keys = list(first_entry["policy"].keys())
        elif "policy" in info:
            policy_keys = list(info["policy"].keys())
        
        if not policy_keys:
            return

        writer.writerow(["step"] + policy_keys)
        for i, entry in enumerate(episode_log):
            p = entry.get("policy") or entry.get("info", {}).get("policy")
            if p:
                writer.writerow([i] + [p.get(k, 0) for k in policy_keys])

def export_queue_trace(episode_log: list[dict[str, Any]], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # Queues are usually in observation
        # Structure: observation["queue_lengths"][node][lane]
        first_entry = episode_log[0]
        obs = first_entry.get("observation", {})
        q_dict = obs.get("queue_lengths", {})
        if not q_dict:
            return
        
        header = ["step"]
        nodes = sorted(q_dict.keys())
        for node in nodes:
            for lane in range(len(q_dict[node])):
                header.append(f"{node}_L{lane}")
        
        writer.writerow(header)
        for i, entry in enumerate(episode_log):
            obs = entry.get("observation", {})
            qs = obs.get("queue_lengths", {})
            if not qs:
                continue
            row = [i]
            for node in nodes:
                node_qs = qs.get(node, [0]*4)
                row.extend(node_qs)
            writer.writerow(row)
