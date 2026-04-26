---
title: Bio-Synthetica Pro
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
license: mit
short_description: "OpenEnv + GRPO: valid Opentrons OT-2 protocols"
tags:
  - reinforcement-learning
  - openenv
  - scientific-workflow
  - llm-training
  - gradio
---

# Bio-Synthetica Pro

**OpenEnv Hackathon India 2026** - Teaching an LLM to write **physically valid** Opentrons OT-2 protocols under **partial observability**, **dynamic contamination alerts**, and a **multi-objective reward**.

## Hugging Face Space (this page)

**Live demo:** use the Gradio UI above to compare model outputs.

## OpenEnv

Built on **[OpenEnv](https://github.com/openenv/openenv)** (`Environment`: `reset` / `step` / `state`). The simulator enforces scans before well use, volume limits, temperature bounds, contamination, and budget.

## Training (re-runnable on Kaggle)

| Platform | Link |
|----------|------|
| Kaggle (team notebook) | [notebook on Kaggle](https://www.kaggle.com/code/shivaanshpandey/notebookc00610413e) |
| All-in-one notebook (GitHub) | [train_grpo_kaggle.ipynb](https://github.com/Prantik-07/bio-synthetica/blob/main/train_grpo_kaggle.ipynb) |

**Stack:** Unsloth + **TRL GRPO**, Llama 3.x 4-bit, W&B logging.

## Training evidence

Judges can verify training via **W&B metrics** (curves per run) and **static plots** in the GitHub repo (no need to upload images to this Space).

- **W&B:** [project `huggingface`](https://wandb.ai/shivaansh0610-polaris-school-of-technology/huggingface)
- **Plots on GitHub:** [plots folder](https://github.com/Prantik-07/bio-synthetica/tree/main/plots), [master comparison PNG](https://raw.githubusercontent.com/Prantik-07/bio-synthetica/main/plots/master_comparison.png)

## Write-ups and links

| | |
|--|--|
| **GitHub repo** | https://github.com/Prantik-07/bio-synthetica |
| **Full writeup** | [writeup.md](https://github.com/Prantik-07/bio-synthetica/blob/main/writeup.md) |
| **Mini-blog (this Space)** | **[Blog.MD](./Blog.MD)** |

## Files in this Space

| File | Purpose |
|------|---------|
| `app.py` | Gradio demo |
| `requirements.txt` | Dependencies |
| `Blog.MD` | Round-2 mini-blog |
| `README.md` | This card (YAML + description) |

Training code and notebooks stay on **GitHub and Kaggle** to keep the Space small.
