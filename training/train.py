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

# Anti-reward-hacking properties of this environment:
# 1. Deterministic seeded transitions: the model cannot manipulate environment state
#    because every transition is a pure function of (state, action, seed).
#    There is no mutable global state the model can exploit.
# 2. Rubric-based multi-component reward: the reward is a weighted combination of
#    6 independent rubric components (local_efficiency, global_coordination,
#    throughput, emergency_response, stability, fairness).
#    Optimizing one component alone cannot dominate the total reward.
# 3. Reward clipping: all step rewards are clipped to [-1.0, 1.0].
#    Extreme outlier actions cannot produce runaway reward signals.
# 4. Priority budget constraint: the central controller cannot boost all intersections
#    simultaneously — total_priority_budget caps the sum of active boosts.
# 5. Episode-level grading: final_score is computed over the full episode,
#    not just the last step, so short-term exploitation does not persist.
def reward_fn(prompts, completions, **kwargs):
    """Reward function that interacts with the OpenEnv API."""
    rewards = []
    log_buffer = kwargs.get("log_buffer")
    for episode, (prompt, completion) in enumerate(zip(prompts, completions), start=1):
        try:
            # 1. Reset
            res = requests.post(f"{ENV_URL}/reset", json={"task_id": "hard_multi", "central_enabled": True})
            # 2. Step
            action = completion
            step_res = requests.post(f"{ENV_URL}/step", json={"action": action})
            data = step_res.json()
            
            # 3. Telemetry
            reward = float(data.get("reward", 0.0))
            info = data.get("info", {})
            if isinstance(log_buffer, list):
                log_buffer.append(
                    {
                        "episode_reward": reward,
                        "final_score": info.get("final_score", 0.0),
                        "throughput": info.get("throughput", 0.0),
                        "mean_queue": info.get("mean_queue", 0.0),
                        "reward_breakdown": info.get("reward_breakdown", {}),
                    }
                )
            if kwargs.get("use_wandb"):
                wandb.log({
                    "episode_reward": reward,
                    "final_score": info.get("final_score", 0.0),
                    "throughput": info.get("throughput", 0.0),
                    "mean_queue": info.get("mean_queue", 0.0),
                })

            if episode % 10 == 0:
                print(f"\n=== Episode {episode} Sample Inspection ===")
                print(f"  Action sent     : {action}")
                print(f"  Step reward     : {reward:.4f}")
                print(f"  Final score     : {info.get('final_score', 'N/A')}")
                print(f"  Active behaviors: {info.get('active_behaviors_log', [])}")
                print(f"  Mean queue      : {info.get('mean_queue', 'N/A')}")
                print("=" * 48)
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

    collected_data: list[dict[str, Any]] = []
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=lambda p, c: reward_fn(p, c, use_wandb=args.wandb, log_buffer=collected_data),
        args=training_args,
        train_dataset=None,  # Intentional: this is online RL. Data is collected live from the environment.
        processing_class=tokenizer,
    )

    trainer.train()
    
    # Save & Plot
    # Save LoRA adapter weights only (correct Unsloth pattern for 4-bit quantized models)
    model.save_pretrained("outputs/traffic-lora")
    tokenizer.save_pretrained("outputs/traffic-lora")
    print("Adapter weights saved to outputs/traffic-lora")
    if args.plot:
        if collected_data:
            from env.metrics_exporter import generate_training_plots
            generate_training_plots(collected_data, "plots")
            print("Training plots saved to plots/")

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
