# Bio-Synthetica Pro - Submission Writeup

*Upload this file to your Hugging Face Space as **`Blog.MD`**. (Same as repo root `writeup.md`; image uses GitHub `raw` so it renders on HF.)*

**OpenEnv Hackathon India 2026** · Theme: World Modeling + Long Horizon Planning

---

## Problem - What capability gap are we targeting?

Ask any frontier LLM to write a lab automation protocol and it confidently produces code that would destroy real equipment.

Here is Claude 3.5 Sonnet, unprompted:

```python
# Claude-generated protocol (untrained)
pipette("A1", "B1", volume=250)      # ❌ 250ul > 200ul hardware max
mix("C3", volume=100, repetitions=5) # ❌ C3 was never scanned - null well
set_temperature("D2", temp=120)      # ❌ 120°C melts the plate (max is 95°C)
report_complete()
```

Three violations. Zero reagent transferred. $5 000 of samples ruined in a real lab.

This is not a knowledge problem - GPT-4 can explain PCR and pipetting in detail. It is a **training signal problem**. No RL environment has ever penalised an LLM for overflowing a well, using a contaminated sample, or operating on a hidden well. Bio-Synthetica Pro is the first environment that does.

---

## Environment - What does the agent see, do, and get rewarded for?

### What the agent sees

A partial 16-well Opentrons OT-2 plate. All wells start hidden. The agent receives a structured text observation:

```
=== BIO-SYNTHETICA LAB STATE ===

SCANNED WELLS (you can use these):
  (none scanned yet)

HIDDEN WELLS (scan these before using):
  A1, A2, A3, A4, B1, B2, B3, B4, C1, C2, C3, C4, D1, D2, D3, D4

BUDGET REMAINING: $10.00

CURRENT GOAL:
  Achieve 0.5x concentration of enzyme in well B2

AVAILABLE ACTIONS:
  scan(well_id)
  pipette(source, dest, volume_ul)
  mix(well_id, volume_ul, repetitions)
  set_temperature(well_id, temp_celsius)
  report_complete()
```

Well volumes have 5% Gaussian sensor noise. The agent must reason about uncertainty.

### What the agent does

It writes Python code as its action:

```python
scan("A1")
scan("B2")
pipette("A1", "B2", volume=100)
mix("B2", volume=60, repetitions=3)
report_complete()
```

The simulator parses this, executes each call against the physics engine, and enforces every constraint in `lab_simulator.py`. There is no prompt-level masking - unscanned wells cause hard violations in code.

### Three things that make this genuinely hard

**1. Partial Observability.**  
The agent cannot see any well until it calls `scan()`. Using an unscanned well is rejected:

```python
pipette("A1", "B1", volume=50)
# → {"success": False, "violations": ["A1 not scanned", "B1 not scanned"]}
```

**2. Dynamic Replanning.**  
Between steps 3 and 7, a contamination alert fires with 30% probability. The agent cannot restart - it must reroute mid-protocol.

**3. Multi-Objective Reward.**  
Enzymes cost $2.50/ul. Buffer costs $0.10/ul. The agent must hit its target concentration *and* minimise cost.

### What the agent gets rewarded for

| Tier | Condition | Score |
|---|---|---|
| Syntax | Valid Python produced | +0.1 |
| Compliance | Zero constraint violations | +0.3 |
| Goal | Target concentration reached | +1.0 (scaled) |
| Efficiency | Done in <8 steps | +0.3 |
| Budget | Reagent cost minimised | +0.3 |
| Replanning | Avoided contaminated well mid-episode | +0.5 |
| **Max** | | **+2.5** |

---

## Results - What changed after training?

We trained Llama-3.1-8B (4-bit quantised via Unsloth) using GRPO on a Kaggle T4 GPU (see linked notebook).

### Master comparison

![Master Comparison](https://raw.githubusercontent.com/Prantik-07/bio-synthetica/main/plots/master_comparison.png)

| Metric | Untrained | Trained | Change |
|---|---|---|---|
| Episode reward | −0.10 | +1.60 | **↑ 1.70** |
| Constraint violations | 4.0 / ep | 0.2 / ep | **↓ 95%** |
| Goal achievement | 8% | 74% | **↑ 66pp** |
| Syntax pass rate | 30% | 97% | **↑ 67pp** |
| Budget efficiency | 0.40 | 0.83 | **↑ 0.43** |
| Replanning success | 10% | 61% | **↑ 51pp** |

---

## Why it matters - who would care?

**Biotech labs** using Opentrons, Tecan, or Hamilton robots need LLM co-pilots that respect physical constraints. **RL researchers** get a benchmark for partial observability, long-horizon replanning, and multi-objective reward.

---

## Try it

| | |
|---|---|
| Live Demo | https://huggingface.co/spaces/Luffy0610/bio-synthetica-pro |
| Training (Kaggle) | https://www.kaggle.com/code/shivaanshpandey/notebookc00610413e |
| Notebook source (GitHub) | https://github.com/Prantik-07/bio-synthetica/blob/main/train_grpo_kaggle.ipynb |
| GitHub | https://github.com/Prantik-07/bio-synthetica |

*Built at OpenEnv Hackathon India 2026 by Prantik-07, shivaansh0610-LUFFY, and ZehaanArshad.*
