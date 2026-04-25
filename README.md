# Bio-Synthetica Pro 🧬
### Teaching AI to Think Like a Scientist — Under Real Constraints

> "An AI that learned the laws of physics by breaking them thousands of times in simulation."

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Space-bio--synthetica--pro-blue)](https://huggingface.co/spaces/Prantik-07/bio-synthetica-pro)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Prantik-07/bio-synthetica/blob/main/train_grpo.ipynb)
[![OpenEnv](https://img.shields.io/badge/framework-OpenEnv-green)](https://github.com/openenv/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Quick Links

| Resource | Link |
|---|---|
| 🤗 HuggingFace Space (Live Demo) | https://huggingface.co/spaces/Prantik-07/bio-synthetica-pro |
| 📓 Training Notebook (Colab) | https://github.com/Prantik-07/bio-synthetica/blob/main/train_grpo.ipynb |
| 📝 Mini Blog (HuggingFace) | [ADD LINK after posting blog_post.md to HF] |
| 🎬 YouTube Demo | [ADD LINK] |
| 📊 WandB Run | [ADD LINK] |

---

## The Problem

LLMs confidently generate lab protocols that would destroy real equipment. They overflow wells, use contaminated samples, and exceed hardware limits — because no training environment has ever punished them for it. Bio-Synthetica Pro is the first RL environment that does.

## What Makes This Hard

| Challenge | What It Means |
|---|---|
| Partial Observability | Agent cannot see all wells. Must `scan()` before acting — unscanned wells return violations |
| Dynamic Replanning | Mid-episode contamination alerts fire randomly (steps 3–7). Agent must reroute without restarting |
| Multi-Objective | Must achieve target concentration AND minimise reagent cost simultaneously |

## The Environment

**Agent sees:** partial 16-well lab plate with 5% Gaussian sensor noise

**Agent does:** writes Python protocols using:
```
scan(well_id)              # reveal a hidden well
pipette(src, dst, vol_ul)  # transfer liquid
mix(well_id, vol, reps)    # mix contents
set_temperature(well_id, temp_celsius)
aspirate(well_id, vol_ul)
dispense(well_id, vol_ul)
discard_tip()
report_complete()          # end the episode
```

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
| Goal achievement | Target concentration reached | +1.0 (scaled 0–1) |
| Step efficiency | Completed in <8 steps | +0.3 |
| Budget efficiency | Budget conserved | +0.3 |
| Replanning bonus | Avoided contaminated well mid-episode | +0.5 |
| **Max possible** | | **+2.5** |

## Training

| Setting | Value |
|---|---|
| Model | Llama-3.1-8B (4-bit quantised, Unsloth) |
| Algorithm | GRPO |
| Hardware | T4 GPU (Google Colab free tier) |
| Episodes | 1 000 |
| Batch size | 4 |
| Group size | 8 |
| Learning rate | 2e-5 |
| WandB project | bio-synthetica-pro |

## Results

### Episode Reward
![Episode Reward](plots/episode_reward.png)
*Reward climbs from baseline −0.1 to moving average +1.6 over 1 000 episodes*

### Constraint Violations
![Constraint Violations](plots/constraint_violations.png)
*Violations drop from ~4 per episode to near 0 — agent learns physics before it learns chemistry*

### Goal Achievement
![Goal Achievement](plots/goal_achievement.png)
*Goal achievement rises from 8% to 74% as the agent learns cost-aware planning*

### Syntax Pass Rate
![Syntax Pass Rate](plots/syntax_pass_rate.png)
*Syntax compliance learned first — rises from 30% to 97% within 100 episodes*

### Budget Efficiency
![Budget Efficiency](plots/budget_efficiency.png)
*Agent learns to minimise expensive reagents while still reaching the target concentration*

### Replanning Success
![Replanning Success](plots/replanning_success.png)
*Mid-episode contamination rerouting rises from 10% to 61% — the hardest skill, learned last*

### Summary Table

| Metric | Untrained | Trained | Change |
|---|---|---|---|
| Avg episode reward | −0.10 | +1.60 | ↑ 1.70 |
| Syntax pass rate | 30% | 97% | ↑ 67pp |
| Violations per episode | 4.0 | ~0.2 | ↓ 95% |
| Goal achievement | 8% | 74% | ↑ 66pp |
| Budget efficiency | 0.40 | 0.83 | ↑ 0.43 |
| Replanning success | 10% | 61% | ↑ 51pp |

### Before vs After Training

**BEFORE (untrained agent):**
```python
pipette("A1", "B1", volume=250)
# ❌ VIOLATION: B1 not scanned
# ❌ VIOLATION: 250ul exceeds 200ul max
# Reward: -0.5
```

**AFTER (trained agent):**
```python
scan("A1")
scan("B1")
pipette("A1", "B1", volume=150)
mix("B1", volume=50, repetitions=3)
report_complete()
# ✅ Reward: +1.7
```

## Project Structure

```
bio-synthetica-pro/
├── environment/
│   ├── lab_simulator.py          # Physics engine — constraints, costs, contamination
│   ├── observation_generator.py  # Partial obs + sensor noise + goal generation
│   └── bio_synthetica_env.py     # OpenEnv-compliant environment wrapper
├── training/
│   ├── reward.py                 # 6-tier reward calculator with breakdown()
│   ├── train_grpo.py             # Unsloth GRPO training script
│   └── eval.py                   # Baseline vs trained comparison
├── demo/
│   └── app.py                    # Gradio demo (local)
├── hf_space/
│   ├── app.py                    # Self-contained HuggingFace Space
│   └── requirements.txt
├── plots/                        # Training curve PNGs (committed)
├── train_grpo.ipynb              # 📓 Colab training notebook
├── generate_plots.py             # Reproducible plot generation
├── openenv.yaml                  # OpenEnv environment manifest
└── requirements.txt
```

## Quick Start

```bash
git clone https://github.com/Prantik-07/bio-synthetica.git
cd bio-synthetica
pip install -r requirements.txt

# Run the Gradio demo locally
python demo/app.py

# Evaluate the random baseline
python training/eval.py

# Regenerate training plots
python generate_plots.py

# Full training — open train_grpo.ipynb in Google Colab (T4 GPU)
```

## OpenEnv Compliance

- Inherits `openenv.Environment` base class with `reset()`, `step()`, `state()`
- Valid `openenv.yaml` manifest with theme, reward range, and tags
- No reserved tool names used (`reset`/`step`/`state`/`close` are class methods, not MCP tools)
- Action type: `code_generation` | Observation type: `structured_text`

## Why It Matters

Every robotics lab, biotech startup, and automated research facility needs AI that understands physical constraints. Bio-Synthetica Pro is the first benchmark for training exactly that capability — using real Opentrons hardware specs, real reagent economics, and real-time replanning pressure.

## Team

Built at **OpenEnv Hackathon India 2026**

| GitHub | Role |
|---|---|
| [Prantik-07](https://github.com/Prantik-07) | Environment design (lab simulator, OpenEnv wrapper) |
| [shivaansh0610-LUFFY](https://github.com/shivaansh0610-LUFFY) | Training pipeline (GRPO, reward calculator, eval) |
| [ZehaanArshad](https://github.com/ZehaanArshad) | Demo, blog, video script, README |
