#!/usr/bin/env python3
# Deployment helper script — not part of the environment

import os

from huggingface_hub import HfApi


SPACE_REPO_ID = "Guuru-DEV/OpenEnv-traffic-signal-3"
FILES_TO_UPLOAD = [
    "notebooks/train_colab_SMOKE.ipynb",
    "notebooks/train_colab_FULL.ipynb",
]


def main() -> None:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found. Please run: export HF_TOKEN=your_token")

    api = HfApi(token=hf_token)
    for path in FILES_TO_UPLOAD:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path,
            repo_id=SPACE_REPO_ID,
            repo_type="space",
        )
        print(f"Uploaded: {path}")

    repo_files = set(api.list_repo_files(repo_id=SPACE_REPO_ID, repo_type="space"))
    for path in FILES_TO_UPLOAD:
        if path in repo_files:
            print(f"Verified on Hub: {path}")
        else:
            raise RuntimeError(f"Upload verification failed for: {path}")

    print("IMPORTANT: Set the following as Space Secrets in HF Space settings UI before running:")
    print("  - HF_TOKEN")
    print("  - WANDB_API_KEY")
    print("  - ENV_URL (set to your live traffic signal HF Space URL)")
    print("Go to: https://huggingface.co/spaces/Guuru-DEV/OpenEnv-traffic-signal-3/settings")


if __name__ == "__main__":
    main()
