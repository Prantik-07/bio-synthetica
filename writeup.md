# Bio-Synthetica Pro - Submission Writeup

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

The agent must decide: scan everything upfront (safe, slow) or scan on-demand (risky, efficient). The trained agent learns the second strategy.

**2. Dynamic Replanning.**
Between steps 3 and 7, a contamination alert fires with 30% probability:

```
ACTIVE ALERT: Well B2 is contaminated. Avoid it.
```

The agent cannot restart. It must reroute mid-protocol to a clean well. This is long-horizon planning under surprise - exactly what real lab robots encounter.

**3. Multi-Objective Reward.**
Enzymes cost $2.50/ul. Buffer costs $0.10/ul. The agent must hit its target concentration *and* minimise cost. It cannot greedily pipette expensive reagents without losing budget points.

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

### Master comparison - all 6 metrics

![Master Comparison](plots/master_comparison.png)
*Red dashed line = untrained baseline. Coloured line = trained agent. Arrow = improvement.*

### Numbers

| Metric | Untrained | Trained | Change |
|---|---|---|---|
| Episode reward | −0.10 | +1.60 | **↑ 1.70** |
| Constraint violations | 4.0 / ep | 0.2 / ep | **↓ 95%** |
| Goal achievement | 8% | 74% | **↑ 66pp** |
| Syntax pass rate | 30% | 97% | **↑ 67pp** |
| Budget efficiency | 0.40 | 0.83 | **↑ 0.43** |
| Replanning success | 10% | 61% | **↑ 51pp** |

### How learning progressed

The agent learned in a predictable curriculum that mirrors how a human student would approach the same problem:

1. **Steps 0–100:** Syntax compliance first. The model stops generating broken Python.
2. **Steps 100–300:** Scan discipline. The model learns to call `scan()` before every well.
3. **Steps 300–600:** Goal-directed planning. The model learns which wells to use and in what volumes.
4. **Steps 600–1000:** Replanning. The hardest skill - the model learns to reroute around contamination alerts.

### Before vs after (same task)

**Before training:**
```python
# Episode 1 - untrained agent
pipette("A1", "B2", volume=250)
mix("C3", volume=100, repetitions=5)
report_complete()
# Violations: 4   Goal: 0%   Reward: -0.50
```

**After training:**
```python
# Episode 1000 - trained agent
scan("A1")
scan("B2")
pipette("A1", "B2", volume=100)
mix("B2", volume=60, repetitions=3)
report_complete()
# Violations: 0   Goal: 74%   Reward: +1.72
```

---

## Why it matters - who would care?

**Biotech labs** using Opentrons, Tecan, or Hamilton liquid-handling robots increasingly want LLM co-pilots that can draft protocols from natural language. Today those LLMs are unreliable because they were never trained on the physics of lab work. Bio-Synthetica Pro is the training environment that closes that gap.

**RL researchers** get a clean benchmark for:
- Partial observability in physical domains
- Long-horizon planning with mid-episode surprises  
- Multi-objective reward under resource constraints

**The broader point:** Bio-Synthetica Pro shows that an LLM does not need to be retrained on domain data to become safe in a new domain - it just needs an RL environment that enforces the right constraints. The same approach could be applied to surgical robotics, chemical synthesis, and any domain where physical violations are costly and irreversible.

---

## Try it

| | |
|---|---|
| 🤗 Live Demo | https://huggingface.co/spaces/Luffy0610/bio-synthetica-pro |
| Training (Kaggle) | https://www.kaggle.com/code/shivaanshpandey/notebookc00610413e |
| Notebook source (GitHub) | https://github.com/Prantik-07/bio-synthetica/blob/main/train_grpo_kaggle.ipynb |
| 💻 GitHub | https://github.com/Prantik-07/bio-synthetica |

*Built at OpenEnv Hackathon India 2026 by Prantik-07, shivaansh0610-LUFFY, and ZehaanArshad.*
