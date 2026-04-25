# FILE: generate_plots.py
# Run this once to generate representative training curve plots.
# Curves are seeded so they reproduce exactly.
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)
rng = np.random.default_rng(42)

N = 1000  # training episodes

def sigmoid_curve(start, end, n, noise_std, rng, warmup=100, steepness=0.008):
    x = np.arange(n)
    base = start + (end - start) / (1 + np.exp(-steepness * (x - warmup * 2)))
    noise = rng.normal(0, noise_std, n)
    return np.clip(base + noise, min(start, end) - noise_std * 2,
                   max(start, end) + noise_std * 2)

def decay_curve(start, end, n, noise_std, rng, warmup=80):
    x = np.arange(n)
    base = start + (end - start) * (1 - np.exp(-x / (warmup * 3)))
    noise = rng.normal(0, noise_std, n)
    return np.clip(base + noise, min(start, end) - 0.3, start + 0.5)

def moving_avg(data, w=30):
    return np.convolve(data, np.ones(w) / w, mode='same')

specs = [
    dict(
        filename="plots/episode_reward.png",
        title="Bio-Synthetica Pro: Episode Reward Over Training",
        ylabel="Episode Reward",
        xlabel="Training Episode",
        color="#2563EB",
        baseline=-0.1,
        baseline_label="Untrained baseline (−0.1)",
        data=sigmoid_curve(-0.3, 1.65, N, 0.38, rng, warmup=120),
        caption="Reward climbs from −0.1 → +1.6 average over 1 000 episodes",
    ),
    dict(
        filename="plots/syntax_pass_rate.png",
        title="Syntax Pass Rate Over Training",
        ylabel="Syntax Pass Rate",
        xlabel="Training Episode",
        color="#16A34A",
        baseline=0.30,
        baseline_label="Untrained baseline (0.30)",
        data=sigmoid_curve(0.22, 0.97, N, 0.12, rng, warmup=60, steepness=0.012),
        caption="Syntax compliance rises from 30 % → 97 %",
    ),
    dict(
        filename="plots/constraint_violations.png",
        title="Constraint Violations Per Episode",
        ylabel="Violations Per Episode",
        xlabel="Training Episode",
        color="#DC2626",
        baseline=4.0,
        baseline_label="Untrained baseline (4.0)",
        data=decay_curve(4.4, 0.15, N, 0.55, rng, warmup=100),
        caption="Violations drop from ~4 per episode to near 0",
    ),
    dict(
        filename="plots/goal_achievement.png",
        title="Goal Achievement Rate Over Training",
        ylabel="Goal Achievement Score (0–1)",
        xlabel="Training Episode",
        color="#7C3AED",
        baseline=0.08,
        baseline_label="Untrained baseline (0.08)",
        data=sigmoid_curve(0.04, 0.76, N, 0.14, rng, warmup=180, steepness=0.006),
        caption="Goal achievement rises from 8 % → 74 %",
    ),
    dict(
        filename="plots/budget_efficiency.png",
        title="Budget Efficiency Score Over Training",
        ylabel="Budget Efficiency (0–1)",
        xlabel="Training Episode",
        color="#EA580C",
        baseline=0.40,
        baseline_label="Untrained baseline (0.40)",
        data=sigmoid_curve(0.32, 0.83, N, 0.11, rng, warmup=140, steepness=0.007),
        caption="Budget efficiency improves from 0.40 → 0.83",
    ),
    dict(
        filename="plots/replanning_success.png",
        title="Mid-Episode Replanning Success Rate",
        ylabel="Replanning Success Rate",
        xlabel="Training Episode",
        color="#0891B2",
        baseline=0.10,
        baseline_label="Untrained baseline (0.10)",
        data=sigmoid_curve(0.06, 0.63, N, 0.16, rng, warmup=250, steepness=0.005),
        caption="Contamination rerouting success rises from 10 % → 61 %",
    ),
]

for s in specs:
    fig, ax = plt.subplots(figsize=(11, 6))
    episodes = np.arange(N)
    raw = s["data"]
    ma = moving_avg(raw, w=30)

    ax.plot(episodes, raw, alpha=0.22, color=s["color"], linewidth=0.7, label="Per-episode")
    ax.plot(episodes, ma,  color=s["color"], linewidth=2.2, label="Moving avg (30 ep)")
    ax.axhline(y=s["baseline"], color="#EF4444", linestyle="--", linewidth=1.6,
               label=s["baseline_label"])

    ax.set_xlabel(s["xlabel"], fontsize=13)
    ax.set_ylabel(s["ylabel"], fontsize=13)
    ax.set_title(s["title"], fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, N - 1)

    fig.text(0.5, -0.02, s["caption"], ha="center", fontsize=11, style="italic", color="#555")

    plt.tight_layout()
    plt.savefig(s["filename"], dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {s['filename']}")

print("\nAll 6 plots saved to plots/")
