#!/usr/bin/env python3

import argparse
import os

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="Guuru-DEV/OpenEnv-traffic-signal-3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found. Please run: export HF_TOKEN=your_token")

    api = HfApi(token=hf_token)
    runtime = api.get_space_runtime(repo_id=args.space)
    stage = str(getattr(runtime, "stage", "")).upper()
    if stage in {"PAUSED", "STOPPED"}:
        print(f"Space {args.space} is already paused.")
        return

    confirm = input("Are you sure you want to pause the Space and stop billing? (yes/no): ")
    if confirm.lower() == "yes":
        api.pause_space(repo_id=args.space)
        print("Space paused. Billing stopped.")
    else:
        print("Pause cancelled.")


if __name__ == "__main__":
    main()
