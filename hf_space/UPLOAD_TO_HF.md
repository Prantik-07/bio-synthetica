# What to upload to your Hugging Face Space

## Required (from this `hf_space/` folder)

| File | Purpose |
|------|---------|
| `app.py` | Gradio demo (mandatory) |
| `requirements.txt` | Python deps for the Space build |
| `README.md` | Space card (YAML `---` header + short description) |

## Round 2 write-up (separate from Space `README.md`)

| Source in repo | On HuggingFace Space |
|----------------|----------------------|
| **`writeup.md`** (repo root) | Upload the same file and name it **`Blog.MD`** in the Space file browser |

**Why one file:** The story already lives in `writeup.md` on GitHub. Duplicates (`blog_post.md`, extra `Blog.MD` copies) were removed to avoid confusion.

**Steps:** Space → **Files and versions** → **Add file** → upload `writeup.md` from the repo → **rename to `Blog.MD`** (exact casing) so judges see it in the Space.

---

**Do not** upload: training code, plots, `train_grpo.ipynb`, or the full repo (keeps the Space small per hackathon rules).
