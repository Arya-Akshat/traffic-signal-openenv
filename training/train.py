"""
Traffic Signal OpenEnv: LLM-Based Hierarchical Training Script (Colab Optimized)

This script implements a training pipeline for fine-tuning an LLM-based Central Controller
using Reinforcement Learning (GRPO/PPO) via Unsloth and TRL. It interacts with the
Traffic Signal OpenEnv API to orchestrate grid-level policy decisions.

Theme: Multi-Agent Interactions & Self-Improvement
"""

# [Step 1: Dependency Installation]
# Run this in a Colab cell:
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps trl peft accelerate bitsandbytes
# !pip install matplotlib numpy requests

import os
import sys
import json
import requests
import numpy as np
from typing import Any, List, Dict

# Set your Environment URL (Hugging Face Space or Local)
ENV_URL = "https://YOUR_HF_SPACE_URL.hf.space"
TRAINING_DIR = "./training_outputs"
os.makedirs(TRAINING_DIR, exist_ok=True)

# [Step 2: Model Setup (Unsloth)]
# We use Unsloth for 2x faster training and 70% less memory usage.
try:
    from unsloth import FastLanguageModel
    import torch
    
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = 2048
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )

    # Add LoRA adapters for parameter-efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
except ImportError:
    print("Unsloth not found. Running in Simulation Mode (Mock LLM).")
    model, tokenizer = None, None

# [Step 3: Environment Interaction Utilities]
def env_reset(task_id: str = "hard_multi") -> Dict[str, Any]:
    """Resets the environment via API and returns the initial observation."""
    response = requests.post(f"{ENV_URL}/reset", json={
        "task_id": task_id,
        "central_enabled": True,
        "normalize_obs": True
    })
    return response.json()

def env_step(action: str) -> Dict[str, Any]:
    """Sends a policy vector or action switch to the environment."""
    # Note: In hierarchical mode, Central Controller often sends a policy override
    # For simplicity, this skeleton uses the standard step API
    response = requests.post(f"{ENV_URL}/step", json={"action": action})
    return response.json()

# [Step 4: Training Loop Skeleton]
def train_llm_agent(num_episodes: int = 50):
    training_log = []
    print(f"Starting Training on {ENV_URL}...")

    for episode in range(num_episodes):
        state = env_reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 1. Extract Text Observation for the LLM
            # The environment provides a YAML-like structured prompt
            obs_text = state.get("observation", {}).get("text_obs", "No observation found.")
            
            # 2. LLM Inference (Generate Policy Rationale and Action)
            # In a real GRPO/PPO loop, TRL handles the rollout and reward calculation
            # Here we show the manual interaction pattern:
            prompt = f"### System State:\n{obs_text}\n\n### Task: Select next Traffic Control Action (KEEP, SWITCH, PHASE_0-3).\n### Response:"
            
            # Mocking action for the skeleton (replace with actual model.generate)
            action = "KEEP" 
            
            # 3. Execute Step
            state = env_step(action)
            episode_reward += state.get("reward", 0.0)
            done = state.get("done", False)

        # 4. Episode Telemetry
        info = state.get("info", {})
        summary = {
            "episode": episode,
            "episode_reward": episode_reward,
            "final_score": info.get("final_score", 0.0),
            "throughput": info.get("throughput", 0.0),
            "reward_breakdown": info.get("reward_breakdown", {})
        }
        training_log.append(summary)
        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Score: {summary['final_score']:.3f}")

    # [Step 5: Post-Training Analysis]
    # Import the visualization utility we built in env/metrics_exporter.py
    try:
        # Assuming the repo is cloned in the current directory
        sys.path.append(os.getcwd())
        from env.metrics_exporter import generate_training_plots
        generate_training_plots(training_log, TRAINING_DIR)
        print(f"Training plots saved to {TRAINING_DIR}")
    except Exception as e:
        print(f"Could not generate plots: {e}")

if __name__ == "__main__":
    train_llm_agent()
