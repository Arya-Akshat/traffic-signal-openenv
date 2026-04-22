"""
Traffic Signal OpenEnv: LLM-Based Reinforcement Learning Training Script

This script implements a training pipeline for fine-tuning an LLM-based Central Controller
using Reinforcement Learning (GRPO) via Unsloth and TRL. It interactively steps the 
environment via the HTTP API.
"""

import os
import sys
import argparse
import requests
import numpy as np
from typing import Any, List, Dict

try:
    import wandb
    import torch
    from trl import GRPOTrainer, GRPOConfig
    from unsloth import FastLanguageModel
except ImportError:
    print("Warning: RL dependencies (trl, unsloth, wandb) not found. Install with: pip install trl unsloth wandb")

# Configuration
ENV_URL = os.getenv("ENV_URL", "https://guuru-dev-traffic-signal-openenv-2.hf.space")

def reward_fn(prompts, completions, **kwargs):
    """Reward function that interacts with the OpenEnv API."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            # 1. Reset
            res = requests.post(f"{ENV_URL}/reset", json={"task_id": "hard_multi", "central_enabled": True})
            # 2. Step
            step_res = requests.post(f"{ENV_URL}/step", json={"action": completion})
            data = step_res.json()
            
            # 3. Telemetry
            reward = float(data.get("reward", 0.0))
            if kwargs.get("use_wandb"):
                info = data.get("info", {})
                wandb.log({
                    "episode_reward": reward,
                    "final_score": info.get("final_score", 0.0),
                    "throughput": info.get("throughput", 0.0)
                })
            rewards.append(reward)
        except Exception as e:
            print(f"API Error: {e}")
            rewards.append(0.0)
    return rewards

def train(args):
    if args.wandb:
        wandb.init(project="traffic-openenv-rl")

    # Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = 2048,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    training_args = GRPOConfig(
        output_dir="./outputs",
        learning_rate=5e-6,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=lambda p, c: reward_fn(p, c, use_wandb=args.wandb),
        args=training_args,
        train_dataset=None, # Load your dataset here
        processing_class=tokenizer,
    )

    trainer.train()
    
    # Save & Plot
    model.save_pretrained("traffic-llm-checkpoint")
    if args.plot:
        from env.metrics_exporter import generate_training_plots
        # generate_training_plots(collected_data, "plots")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--plot", action="store_true", help="Generate training plots at the end")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    
    try:
        train(args)
    except Exception as e:
        print(f"Training failed: {e}")
