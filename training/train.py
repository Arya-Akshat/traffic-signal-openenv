"""Hardened GRPO training runner for traffic OpenEnv."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import time
from typing import Any

import numpy as np
import requests
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL

try:
    import wandb
except Exception:
    wandb = None

DEFAULT_ENV_URL = "https://guuru-dev-traffic-signal-openenv-2.hf.space"
ENV_URL = DEFAULT_ENV_URL


def find_local_env_port() -> str:
    for port in [7860, 8000, 8080, 3000]:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=3)
            if resp.status_code == 200:
                print(f"Found local environment on port {port}")
                return f"http://localhost:{port}"
        except Exception:
            pass
    raise RuntimeError("No local environment found on ports 7860, 8000, 8080, 3000")


def safe_post(url: str, payload: dict[str, Any], retries: int = 16, timeout: int = 60) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = min(45, 2 * (attempt + 1)) + random.uniform(0.0, 1.0)
                print(f"HTTP {resp.status_code}. Waiting {wait:.1f}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            wait = min(30, 2 + attempt)
            print(f"Network/timeout error on attempt {attempt + 1}/{retries}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def parse_action(completion: str) -> dict[str, Any]:
    valid = {"KEEP", "SWITCH", "PHASE_0", "PHASE_1", "PHASE_2", "PHASE_3"}
    base = {"NW": "KEEP", "NE": "KEEP", "SW": "KEEP", "SE": "KEEP"}
    try:
        action = json.loads(completion)
        if isinstance(action, dict):
            raw_local = action.get("local_actions", {})
            clean_local: dict[str, str] = {}
            for key in ("NW", "NE", "SW", "SE"):
                raw_val = str(raw_local.get(key, "KEEP")).upper().strip() if isinstance(raw_local, dict) else "KEEP"
                clean_local[key] = raw_val if raw_val in valid else "KEEP"
            return {"local_actions": clean_local, "central_action": {}}
    except Exception:
        pass

    try:
        match = re.search(r"\{.*\}", completion, re.DOTALL)
        if match:
            action = json.loads(match.group())
            if isinstance(action, dict):
                return parse_action(json.dumps(action))
    except Exception:
        pass

    return {"local_actions": base, "central_action": {}}


# Anti-reward-hacking properties of this environment:
# 1. Deterministic seeded transitions
# 2. 6 independent rubric components
# 3. Reward clipped to [-1.0, 1.0]
# 4. total_priority_budget constraint prevents all-boost exploitation
# 5. Episode-level final_score prevents short-term gaming
def reward_fn(prompts, completions, **kwargs):  # type: ignore[no-untyped-def]
    rewards: list[float] = []
    task_id = kwargs.get("task_id", "hard_multi")
    use_wandb = kwargs.get("use_wandb", False)

    for episode, (_, completion) in enumerate(zip(prompts, completions), start=1):
        safe_post(
            f"{ENV_URL}/reset",
            {"task_id": task_id, "central_enabled": True},
        )

        action = parse_action(completion)
        episode_reward = 0.0
        done = False
        step_count = 0
        info: dict[str, Any] = {}
        latency_ms = 0.0

        while not done and step_count < 100:
            t0 = time.time()
            try:
                step_res = safe_post(f"{ENV_URL}/step", action)
            except requests.HTTPError as exc:
                if getattr(exc.response, "status_code", None) == 422:
                    action = {
                        "local_actions": {"NW": "KEEP", "NE": "KEEP", "SW": "KEEP", "SE": "KEEP"},
                        "central_action": {},
                    }
                    step_res = safe_post(f"{ENV_URL}/step", action)
                else:
                    raise
            latency_ms = (time.time() - t0) * 1000
            data = step_res.json()
            info = data.get("info", {})
            episode_reward += float(data.get("reward", 0.0))
            done = data.get("done", False)
            step_count += 1
            time.sleep(0.05)

        log_data = {
            "episode_reward": episode_reward,
            "mean_queue": info.get("mean_queue", 0.0),
            "final_score": info.get("final_score", 0.0),
            "throughput": info.get("throughput", 0.0),
            "step_count": step_count,
            "step_latency_ms": latency_ms,
        }
        if use_wandb and wandb:
            wandb.log(log_data)

        if episode % 2 == 0:
            print(f"\n=== Episode {episode} ===")
            print(f"  Reward   : {episode_reward:.4f}")
            print(f"  Score    : {info.get('final_score', 'N/A')}")
            print(f"  Queue    : {info.get('mean_queue', 'N/A')}")
            print(f"  Steps    : {step_count}")
            print(f"  Latency  : {latency_ms:.1f}ms")
            print(f"  Action   : {action}")
            # WARNING: if action is always identical across episodes,
            # reward hacking may be occurring. Stop and inspect.

        rewards.append(episode_reward)
        time.sleep(0.2)

    return rewards


def train(args: argparse.Namespace) -> None:
    global ENV_URL

    if args.local_env:
        ENV_URL = find_local_env_port()
    else:
        ENV_URL = os.getenv("ENV_URL", DEFAULT_ENV_URL)

    if args.batch_size * args.gradient_accumulation_steps < 4:
        raise ValueError(
            "per_device_train_batch_size * gradient_accumulation_steps "
            "must be >= num_generations (4)"
        )

    if args.wandb and wandb:
        wandb.init(project="traffic-signal-rl", name="script-run")

    r = requests.get(f"{ENV_URL}/health", timeout=30)
    r.raise_for_status()
    print("Space status:", r.json().get("status"))

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    PatchFastRL("GRPO", FastLanguageModel)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=1024,
        dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    train_dataset = Dataset.from_dict(
        {
            "prompt": [
                "You are a traffic controller. Output a JSON object with keys "
                "'local_actions' (dict mapping NW/NE/SW/SE to one of KEEP/SWITCH/"
                "PHASE_0/PHASE_1/PHASE_2/PHASE_3) and 'central_action' (empty dict)."
            ]
            * 20
        }
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = GRPOConfig(
        output_dir="./outputs",
        learning_rate=5e-6,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=3,
        max_steps=args.max_steps,
        max_prompt_length=512,
        max_completion_length=128,
        num_generations=4,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="wandb" if args.wandb else "none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=lambda p, c: reward_fn(p, c, use_wandb=args.wandb, task_id=args.task),
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    torch.cuda.empty_cache()
    gc.collect()
    print("Trainer ready. Launching training...")
    trainer.train()
    print("\nTraining complete.")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    model.save_pretrained("outputs/traffic-lora")
    tokenizer.save_pretrained("outputs/traffic-lora")
    print("Adapter weights saved to outputs/traffic-lora")

    if args.wandb and wandb:
        wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2, dest="gradient_accumulation_steps")
    parser.add_argument("--max-steps", type=int, default=100, dest="max_steps")
    parser.add_argument("--task", type=str, default="hard_multi")
    parser.add_argument("--local-env", action="store_true", dest="local_env")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
