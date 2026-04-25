---
title: "Bio-Synthetica Pro: Training an AI Scientist to Respect Physics"
thumbnail: https://raw.githubusercontent.com/Prantik-07/bio-synthetica/main/plots/episode_reward.png
authors:
  - user: Prantik-07
  - user: shivaansh0610-LUFFY
  - user: ZehaanArshad
---

# Bio-Synthetica Pro: Training an AI Scientist to Respect Physics

LLMs confidently generate lab protocols that would destroy real equipment. Ask any frontier model to design an Opentrons OT-2 protocol and it will overflow wells, pipette from unscanned sources, and exceed hardware limits — not because it lacks knowledge, but because no training environment has ever penalised it for doing so.

## What We Built

Bio-Synthetica Pro is an [OpenEnv](https://github.com/openenv/openenv) reinforcement learning environment where an LLM agent learns to write physically valid Opentrons OT-2 liquid-handling protocols. The agent operates a simulated 16-well plate, writes Python code as its action, and receives a structured six-tier reward signal that enforces real laboratory physics. We trained **Llama-3.1-8B** (4-bit quantised via Unsloth) using **GRPO** on a single Google Colab T4 GPU — free tier.

**Try it:** [🧬 HuggingFace Space](https://huggingface.co/spaces/Prantik-07/bio-synthetica-pro) · [📓 Colab Notebook](https://github.com/Prantik-07/bio-synthetica/blob/main/train_grpo.ipynb) · [💻 GitHub](https://github.com/Prantik-07/bio-synthetica)

## The Three Challenges That Make This Hard

**1. Partial Observability.** The agent cannot see all 16 wells at once. Every well must be `scan()`ed before use — otherwise the simulator returns a violation. This forces the model to build an internal world model of the plate state before acting, rather than blindly issuing pipette commands. It is a direct analogue of real robot perception uncertainty.

**2. Dynamic Replanning.** Between steps 3 and 7, a contamination alert fires with 30% probability, randomly marking one well as unsafe. The agent must detect this mid-episode and reroute its protocol without restarting. Successfully avoiding the contaminated well earns a **+0.5 replanning bonus**. This tests long-horizon contingency planning — the exact capability that makes AI useful in real automated labs.

**3. Multi-Objective Reward.** Six reward tiers run simultaneously across every episode:

| Tier | Condition | Score |
|---|---|---|
| Syntax | Valid Python generated | +0.1 |
| Compliance | Zero constraint violations | +0.3 |
| Goal | Target concentration reached | +1.0 scaled |
| Efficiency | Completed in <8 steps | +0.3 |
| Budget | Reagent cost minimised | +0.3 |
| Replanning | Contamination avoided mid-episode | +0.5 |

Maximum possible reward per episode: **+2.5**. The agent cannot exploit one dimension without solving the others.

## Results After 1,000 GRPO Steps

![Episode Reward](https://raw.githubusercontent.com/Prantik-07/bio-synthetica/main/plots/episode_reward.png)
*Episode reward climbs from −0.1 → +1.6 average*

| Metric | Untrained | Trained | Change |
|---|---|---|---|
| Avg episode reward | −0.10 | +1.60 | ↑ 1.70 |
| Syntax pass rate | 30% | 97% | ↑ 67pp |
| Violations per episode | 4.0 | ~0.2 | ↓ 95% |
| Goal achievement | 8% | 74% | ↑ 66pp |
| Budget efficiency | 0.40 | 0.83 | ↑ 0.43 |
| Replanning success | 10% | 61% | ↑ 51pp |

The learning progression follows a predictable curriculum: syntax compliance first (steps 0–50), scan discipline next (50–200), cost-aware goal planning (200–600), with replanning emerging last as the hardest compositional skill.

## Key Insight

The agent did not learn laboratory chemistry. It learned the **structure of physical constraints** — that actions have preconditions, that resources are finite, and that plans must remain contingent on the environment's state. This is the core transferable capability that makes AI trustworthy in real automated research workflows.

## Before vs After

**Before training:**
```python
pipette("A1", "B1", volume=250)   # ❌ B1 not scanned
                                   # ❌ 250ul > 200ul max
# Reward: -0.5
```

**After training:**
```python
scan("A1")
scan("B1")
pipette("A1", "B1", volume=150)
mix("B1", volume=50, repetitions=3)
report_complete()
# Reward: +1.7
```

## Resources

| Resource | Link |
|---|---|
| 🤗 HuggingFace Space | [bio-synthetica-pro](https://huggingface.co/spaces/Prantik-07/bio-synthetica-pro) |
| 📓 Colab Notebook | [train_grpo.ipynb](https://github.com/Prantik-07/bio-synthetica/blob/main/train_grpo.ipynb) |
| 💻 GitHub | [Prantik-07/bio-synthetica](https://github.com/Prantik-07/bio-synthetica) |
| 📝 Writeup (story) | [writeup.md](https://github.com/Prantik-07/bio-synthetica/blob/main/writeup.md) |

*Built at OpenEnv Hackathon India 2026 by Prantik-07, Shivaansh Pandey, and Zehaan Asgar.*
