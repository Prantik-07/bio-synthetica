# FILE: training/eval.py
import sys
sys.path.append('/content/bio-synthetica-pro')

import json
import torch

from environment.bio_synthetica_env import BioSyntheticaEnv
from training.reward import RewardCalculator


def run_random_baseline(n_episodes: int = 50) -> list:
    env = BioSyntheticaEnv()
    results = []

    for _ in range(n_episodes):
        env.reset()
        random_protocol = '''scan("A1")
scan("B1")
pipette("A1", "B1", volume=100)
report_complete()
'''
        obs, reward, done, info = env.step(random_protocol)
        results.append(
            {
                "reward": reward,
                "syntax_pass": info["syntax_pass"],
                "violations": len(info["violations"]),
                "goal_progress": info["goal_progress"],
                "budget_used": info["budget_used"],
                "rerouted": info["rerouted_successfully"],
            }
        )
    return results


def run_trained_model(
    model,
    tokenizer,
    system_prompt: str,
    n_episodes: int = 50,
) -> list:
    env = BioSyntheticaEnv()
    results = []

    for _ in range(n_episodes):
        obs_dict = env.reset()
        prompt = f"{system_prompt}\n\n{obs_dict['observation']}"

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
            )
        protocol = tokenizer.decode(outputs[0], skip_special_tokens=True)
        protocol = protocol[len(prompt):]

        obs, reward, done, info = env.step(protocol)
        results.append(
            {
                "reward": reward,
                "syntax_pass": info["syntax_pass"],
                "violations": len(info["violations"]),
                "goal_progress": info["goal_progress"],
                "budget_used": info["budget_used"],
                "rerouted": info["rerouted_successfully"],
                "protocol": protocol,
            }
        )
    return results


def print_comparison(baseline: list, trained: list):
    def avg(results, key):
        return round(sum(r[key] for r in results) / len(results), 3)

    metrics = [
        ("Avg Reward", "reward"),
        ("Syntax Pass %", "syntax_pass"),
        ("Violations/Ep", "violations"),
        ("Goal Success", "goal_progress"),
        ("Budget Used", "budget_used"),
        ("Replan Success", "rerouted"),
    ]

    print("\n" + "=" * 55)
    print(f"{'Metric':<20} {'Untrained':>10} {'Trained':>10} {'Delta':>10}")
    print("=" * 55)

    for label, key in metrics:
        b = avg(baseline, key)
        t = avg(trained, key)
        delta = round(t - b, 3)
        arrow = "↑" if delta > 0 else "↓"
        print(f"{label:<20} {b:>10} {t:>10} {arrow}{abs(delta):>9}")

    print("=" * 55)


def save_before_after(baseline: list, trained: list):
    worst = min(baseline, key=lambda x: x["reward"])
    best = max(trained, key=lambda x: x["reward"])

    with open("before_after_example.txt", "w") as f:
        f.write("=== BEFORE TRAINING (untrained agent) ===\n\n")
        f.write(f"Reward: {worst['reward']}\n")
        f.write(f"Violations: {worst['violations']}\n")
        f.write(f"Goal Progress: {worst['goal_progress']}\n\n")

        f.write("=== AFTER TRAINING (trained agent) ===\n\n")
        f.write(best.get("protocol", "N/A") + "\n\n")
        f.write(f"Reward: {best['reward']}\n")
        f.write(f"Violations: {best['violations']}\n")
        f.write(f"Goal Progress: {best['goal_progress']}\n")

    print("Saved: before_after_example.txt")


if __name__ == "__main__":
    print("Running random baseline evaluation...")
    baseline_results = run_random_baseline(n_episodes=50)

    print("\n=== Baseline Summary ===")
    for key in ["reward", "syntax_pass", "violations", "goal_progress", "budget_used", "rerouted"]:
        avg_val = round(sum(r[key] for r in baseline_results) / len(baseline_results), 3)
        print(f"  {key}: {avg_val}")

    with open("baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    print("\nBaseline results saved to baseline_results.json")
    print("Load a trained model and call run_trained_model() + print_comparison() to compare.")
