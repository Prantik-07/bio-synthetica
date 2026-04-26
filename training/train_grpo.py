# FILE: training/train_grpo.py
#
# Run these in Colab before this script:
# !pip install "transformers>=4.44" unsloth openenv wandb torch trl matplotlib datasets accelerate bitsandbytes
# !git clone https://github.com/Prantik-07/bio-synthetica.git /content/bio-synthetica-pro

import sys
sys.path.append('/content/bio-synthetica-pro')

import gc
import os

# Colab T4: reduce allocator fragmentation; helps avoid "modules on CPU/disk" with 4bit BNB
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    gc.collect()

from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig

PatchFastRL("GRPO", FastLanguageModel)
from datasets import Dataset
import wandb
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from environment.bio_synthetica_env import BioSyntheticaEnv
from training.reward import RewardCalculator

# ---------------------------------------------------------------------------
# WandB initialisation
# ---------------------------------------------------------------------------
wandb.init(
    project="bio-synthetica-pro",
    name="grpo-llama3-run1",
    config={
        "model": "Llama-3.1-8B-4bit",
        "algorithm": "GRPO",
        "max_steps": 1000,
        "batch_size": 4,
        "group_size": 8,
        "learning_rate": 2e-5,
        "max_completion_length": 512,
    },
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# T4: modest ctx; float16 (no bf16 on T4). If 8B still hits BNB "CPU/disk" map error, use 3.2-3B.
_PRIMARY = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
_FALLBACK = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
max_seq_length = 1024
_load_dtype = torch.float16


def _load_4bit(name: str):
    return FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=_load_dtype,
        fast_inference=False,
        device_map={"": 0},
    )


model_name = _PRIMARY
try:
    model, tokenizer = _load_4bit(_PRIMARY)
except ValueError as e:
    err = str(e).lower()
    if "cpu" in err or "disk" in err:
        print("8B load failed; using 3.2-3B fallback:", _FALLBACK)
        model_name = _FALLBACK
        model, tokenizer = _load_4bit(_FALLBACK)
    else:
        raise

print("Loaded:", model_name)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a lab automation scientist operating an Opentrons OT-2 liquid handling robot.

RULES YOU MUST FOLLOW:
1. Always call scan(well_id) before using any well
2. Never exceed 200ul volume in any well
3. Never pipette more than 200ul at once
4. Temperature must stay between 4C and 95C only
5. Never use contaminated wells
6. Minimize reagent cost while achieving the goal
7. If contamination alert fires, avoid that well completely
8. Always end your protocol with report_complete()

OUTPUT ONLY VALID PYTHON CODE.
No explanations. No markdown. No comments. Just Python."""


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate_dataset(n_samples: int = 200) -> Dataset:
    env = BioSyntheticaEnv()
    samples = []

    for i in range(n_samples):
        obs_dict = env.reset()
        observation = obs_dict["observation"]
        prompt = f"{SYSTEM_PROMPT}\n\n{observation}"
        samples.append({"prompt": prompt, "completion": ""})

    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------
class MetricsTracker:

    def __init__(self):
        self.rewards = []
        self.syntax_passes = []
        self.violations = []
        self.goal_scores = []
        self.budget_scores = []
        self.replan_scores = []

    def log(self, reward: float, info: dict):
        self.rewards.append(reward)
        self.syntax_passes.append(1 if info.get("syntax_pass") else 0)
        self.violations.append(len(info.get("violations", [])))
        self.goal_scores.append(info.get("goal_progress", 0))
        budget_used = info.get("budget_used", 0)
        self.budget_scores.append(max(0, 1 - budget_used / 10.0))
        self.replan_scores.append(
            1 if info.get("rerouted_successfully") else 0
        )

    def moving_average(self, data: list, window: int = 20) -> list:
        if len(data) < window:
            return data
        return [
            np.mean(data[max(0, i - window): i + 1])
            for i in range(len(data))
        ]

    def log_to_wandb(self, step: int):
        if not self.rewards:
            return
        wandb.log(
            {
                "reward": self.rewards[-1],
                "reward_ma20": np.mean(self.rewards[-20:]),
                "syntax_pass_rate": np.mean(self.syntax_passes[-20:]),
                "avg_violations": np.mean(self.violations[-20:]),
                "goal_achievement": np.mean(self.goal_scores[-20:]),
                "budget_efficiency": np.mean(self.budget_scores[-20:]),
                "replan_success": np.mean(self.replan_scores[-20:]),
            },
            step=step,
        )


# ---------------------------------------------------------------------------
# Reward function for GRPO
# ---------------------------------------------------------------------------
tracker = MetricsTracker()


def reward_fn(completions, prompts=None, **kwargs):
    env = BioSyntheticaEnv()
    rewards = []
    syntax_flags = []

    for i, completion in enumerate(completions):
        try:
            env.reset()
            obs, reward, done, info = env.step(completion)
            tracker.log(reward, info)
            rewards.append(float(reward))
            syntax_flags.append(1.0 if info.get("syntax_pass") else 0.0)
        except Exception as e:
            rewards.append(-0.5)
            syntax_flags.append(0.0)
            if os.environ.get("BIO_DEBUG_REWARD"):
                print("reward_fn exception:", repr(e))

    try:
        wandb.log(
            {
                "reward/batch_mean": float(np.mean(rewards)) if rewards else 0.0,
                "parse/syntax_ok_rate": (
                    float(np.mean(syntax_flags)) if syntax_flags else 0.0
                ),
            },
            commit=False,
        )
    except Exception:
        pass
    return rewards


# ---------------------------------------------------------------------------
# GRPO config and trainer
# ---------------------------------------------------------------------------
dataset = generate_dataset(200)

# TRL GRPO: length is set ONLY via GRPOConfig — NOT via GRPOTrainer(generate_kwargs=...).
# - max_completion_length (default 256) -> becomes max_new_tokens in GenerationConfig
# - generation_kwargs is merged on top (belt-and-suspenders)
# Wrong: pass max_new_tokens= to GRPOConfig (invalid field) or generate_kwargs= to GRPOTrainer (ignored).
grpo_config = GRPOConfig(
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_generations=8,
    max_completion_length=512,
    generation_kwargs={"max_new_tokens": 512},
    max_steps=1000,
    logging_steps=10,
    save_steps=100,
    output_dir="./bio-synthetica-checkpoints",
    report_to="wandb",
    warmup_steps=50,
    weight_decay=0.01,
)
grpo_config.max_completion_length = 512
_gw = dict(grpo_config.generation_kwargs or {})
_gw["max_new_tokens"] = 512
grpo_config.generation_kwargs = _gw
print("GRPOConfig check → max_completion_length =", grpo_config.max_completion_length,
      "| generation_kwargs =", grpo_config.generation_kwargs)

trainer = GRPOTrainer(
    model=model,
    config=grpo_config,
    reward_funcs=reward_fn,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
print("trainer.max_completion_length =", getattr(trainer, "max_completion_length", "n/a"), "(expect 512)")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
trainer.train()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def save_all_plots(tracker: MetricsTracker):
    os.makedirs("plots", exist_ok=True)

    plots = [
        {
            "data": tracker.rewards,
            "filename": "plots/episode_reward.png",
            "title": "Bio-Synthetica: Episode Reward Over Training",
            "ylabel": "Episode Reward",
            "color": "blue",
            "baseline": -0.1,
        },
        {
            "data": tracker.syntax_passes,
            "filename": "plots/syntax_pass_rate.png",
            "title": "Syntax Pass Rate Over Training",
            "ylabel": "Syntax Pass Rate",
            "color": "green",
            "baseline": 0.3,
        },
        {
            "data": tracker.violations,
            "filename": "plots/constraint_violations.png",
            "title": "Constraint Violations Per Episode",
            "ylabel": "Violations Per Episode",
            "color": "red",
            "baseline": 4.0,
        },
        {
            "data": tracker.goal_scores,
            "filename": "plots/goal_achievement.png",
            "title": "Goal Achievement Rate Over Training",
            "ylabel": "Goal Achievement Score (0-1)",
            "color": "purple",
            "baseline": 0.08,
        },
        {
            "data": tracker.budget_scores,
            "filename": "plots/budget_efficiency.png",
            "title": "Budget Efficiency Score Over Training",
            "ylabel": "Budget Efficiency (0-1)",
            "color": "orange",
            "baseline": 0.4,
        },
        {
            "data": tracker.replan_scores,
            "filename": "plots/replanning_success.png",
            "title": "Mid-Episode Replanning Success Rate",
            "ylabel": "Replanning Success Rate",
            "color": "teal",
            "baseline": 0.1,
        },
    ]

    for p in plots:
        fig, ax = plt.subplots(figsize=(10, 6))

        raw = p["data"]
        ma = tracker.moving_average(raw, window=20)
        episodes = list(range(len(raw)))

        ax.plot(episodes, raw, alpha=0.3, color=p["color"],
                linewidth=0.8, label="Raw")
        ax.plot(episodes, ma, color=p["color"],
                linewidth=2.0, label="Moving Avg (20)")
        ax.axhline(
            y=p["baseline"], color="red", linestyle="--", linewidth=1.5,
            label=f"Untrained baseline ({p['baseline']})",
        )

        ax.set_xlabel("Training Episodes", fontsize=13)
        ax.set_ylabel(p["ylabel"], fontsize=13)
        ax.set_title(p["title"], fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(p["filename"], dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p['filename']}")


save_all_plots(tracker)
print("All plots saved to plots/ folder")

# ---------------------------------------------------------------------------
# Save metrics JSON
# ---------------------------------------------------------------------------
with open("metrics.json", "w") as f:
    json.dump(
        {
            "rewards": tracker.rewards,
            "syntax_passes": tracker.syntax_passes,
            "violations": tracker.violations,
            "goal_scores": tracker.goal_scores,
            "budget_scores": tracker.budget_scores,
            "replan_scores": tracker.replan_scores,
        },
        f,
    )
print("Metrics saved to metrics.json")
