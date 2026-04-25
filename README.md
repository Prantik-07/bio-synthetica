# Bio-Synthetica Pro 🧬
### Teaching AI to Think Like a Scientist — Under Real Constraints

> "An AI that learned the laws of physics by breaking them thousands of times in simulation."

## The Problem

LLMs confidently generate lab protocols that would destroy real equipment. They overflow wells, use contaminated samples, and exceed hardware limits — because no training environment has ever punished them for it. Bio-Synthetica Pro is the first RL environment that does.

## What Makes This Hard

| Challenge | What It Means |
|---|---|
| Partial Observability | Agent cannot see all wells. Must `scan()` before acting |
| Dynamic Replanning | Mid-episode contamination alerts force adaptation |
| Multi-Objective | Must achieve goal AND minimize reagent cost |

## The Environment

**Agent sees:** partial 16-well lab plate with 5% sensor noise

**Agent does:** writes Python protocols using `scan()`, `pipette()`, `mix()`, `set_temperature()`, `aspirate()`, `dispense()`, `report_complete()`

**Simulator enforces (hard constraints):**
- Well volume cannot exceed 200ul
- Must `scan()` before any operation on a well
- Pipette max volume: 200ul
- Temperature: 4°C to 95°C only
- No operations on contaminated wells
- Tips cannot go below 0

**Reward tiers:**

| Tier | Condition | Score |
|---|---|---|
| Syntax pass | Valid Python generated | +0.1 |
| No violations | Zero constraint breaks | +0.3 |
| Goal achievement | Target concentration reached | +1.0 (scaled) |
| Step efficiency | Completed in <8 steps | +0.3 |
| Budget efficiency | Budget conserved | +0.3 |
| Replanning bonus | Avoided contaminated well mid-episode | +0.5 |
| **Max possible** | | **+2.5** |

## Training

| Setting | Value |
|---|---|
| Model | Llama-3.1-8B (4-bit, Unsloth) |
| Algorithm | GRPO |
| Hardware | T4 GPU (Google Colab free tier) |
| Episodes | 1000 |
| Batch size | 4 |
| Group size | 8 |
| Learning rate | 2e-5 |

## Results

![Episode Reward](plots/episode_reward.png)
*Episode reward climbs from baseline −0.1 to average +1.6*

![Constraint Violations](plots/constraint_violations.png)
*Violations drop from ~4 per episode to near 0*

![Goal Achievement](plots/goal_achievement.png)
*Goal achievement rises from 8% to 74%*

### Before vs After Training

**BEFORE (untrained agent):**
```python
pipette("A1", "B1", volume=250)
# VIOLATION: B1 not scanned
# VIOLATION: 250ul exceeds 200ul max
# Reward: -0.5
```

**AFTER (trained agent):**
```python
scan("A1")
scan("B1")
pipette("A1", "B1", volume=150)
mix("B1", volume=50, repetitions=3)
report_complete()
# Reward: +1.7
```

## Project Structure

```
bio-synthetica-pro/
├── environment/
│   ├── __init__.py
│   ├── lab_simulator.py          # Physics engine & constraint checker
│   ├── observation_generator.py  # Partial obs + noise + goal gen
│   └── bio_synthetica_env.py     # OpenEnv-compliant env wrapper
├── training/
│   ├── __init__.py
│   ├── reward.py                 # 6-tier reward calculator
│   ├── train_grpo.py             # Unsloth GRPO training script
│   └── eval.py                   # Baseline vs trained comparison
├── demo/
│   ├── __init__.py
│   └── app.py                    # Gradio HuggingFace Space demo
├── plots/                        # Training curve PNGs (auto-generated)
├── openenv.yaml                  # OpenEnv environment descriptor
├── requirements.txt
├── blog_post.md
├── video_script.md
└── README.md
```

## Quick Start

```bash
git clone https://github.com/Prantik-07/bio-synthetica.git
cd bio-synthetica
pip install -r requirements.txt

# Run the Gradio demo locally
python demo/app.py

# Run baseline evaluation
python training/eval.py

# Full training (requires Colab T4 or equivalent GPU)
python training/train_grpo.py
```

## Why It Matters

Every robotics lab, biotech startup, and automated research facility needs AI that understands physical constraints. Bio-Synthetica Pro is the first benchmark for training exactly that capability — using real hardware specs, real reagent economics, and real replanning pressure.

## Resources

| Resource | Link |
|---|---|
| HuggingFace Space | [ADD LINK] |
| Training Notebook | [ADD LINK] |
| WandB Run | [ADD LINK] |
| Mini Blog | [ADD LINK] |
| YouTube Demo | [ADD LINK] |

## Team

Built at **OpenEnv Hackathon India 2026**
