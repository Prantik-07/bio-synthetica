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
short_description: OpenEnv RL env — GRPO-trained LLMs write valid Opentrons lab protocols
tags:
  - reinforcement-learning
  - openenv
  - scientific-workflow
  - llm-training
  - gradio
---

# Bio-Synthetica Pro

**OpenEnv Hackathon India 2026** — Teaching an LLM to write **physically valid** Opentrons OT-2 protocols under **partial observability**, **dynamic contamination alerts**, and a **multi-objective reward**.

## Hugging Face Space (this page)

**Live demo:** use the Gradio UI above to compare model outputs.

## OpenEnv

Built on **[OpenEnv](https://github.com/openenv/openenv)** (`Environment`: `reset` / `step` / `state`). The simulator enforces scans before well use, volume limits, temperature bounds, contamination, and budget.

## Training (re-runnable)

| Platform | Link |
|----------|------|
| Colab | [train_grpo.ipynb](https://colab.research.google.com/github/Prantik-07/bio-synthetica/blob/main/train_grpo.ipynb) |
| Kaggle (team notebook) | [notebook on Kaggle](https://www.kaggle.com/code/shivaanshpandey/notebookc00610413e) |
| All-in-one Kaggle `.ipynb` (GitHub) | [train_grpo_kaggle.ipynb](https://github.com/Prantik-07/bio-synthetica/blob/main/train_grpo_kaggle.ipynb) |

**Stack:** Unsloth + **TRL GRPO**, Llama 3.x 4-bit, W&B logging.

## Evidence of training

- **Weights & Biases:** [project `huggingface`](https://wandb.ai/shivaansh0610-polaris-school-of-technology/huggingface)
- **Plots (GitHub, no large files in Space):** [plots folder](https://github.com/Prantik-07/bio-synthetica/tree/main/plots) · [master comparison PNG](https://raw.githubusercontent.com/Prantik-07/bio-synthetica/main/plots/master_comparison.png)

## Write-ups & links

| | |
|--|--|
| **GitHub repo** | https://github.com/Prantik-07/bio-synthetica |
| **Full writeup** | [writeup.md](https://github.com/Prantik-07/bio-synthetica/blob/main/writeup.md) |
| **Mini-blog (this Space)** | **[Blog.MD](./Blog.MD)** |
| **Video** | *Public YouTube URL in GitHub README when published* |
| **Judge rubric** | [Google Doc](https://docs.google.com/document/d/1Odznuzwtb1ecDOm2t6ToZd4MuMXXfO6vWUGcxbC6mFs/edit?tab=t.0#bookmark=kix.2dz0x0nie3me) |

## Files in this Space

| File | Purpose |
|------|---------|
| `app.py` | Gradio demo |
| `requirements.txt` | Dependencies |
| `Blog.MD` | Round-2 mini-blog |
| `README.md` | This card (YAML + description) |

Training code and notebooks stay on **GitHub / Colab / Kaggle** to keep the Space small, per organizer guidance.
