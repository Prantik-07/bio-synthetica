# Bio-Synthetica Pro — YouTube Video Script
## Runtime: ~2 minutes | Word count: ~260

---

**[0:00–0:15 — Hook]**
*(Show on screen: broken protocol with red violation warnings)*

Watch this AI try to run a lab experiment. It just overflowed a well, pipetted from an unscanned source, and blew past the 200-microliter hardware limit — all in three lines of code. This is what every LLM does today when you ask it to automate a biology lab.

---

**[0:15–0:35 — Problem]**
*(Show: real Opentrons OT-2 robot footage)*

Opentrons robots are used in real research labs. A bad protocol doesn't just fail — it destroys samples, contaminates wells, and wastes thousands of dollars of reagents. The problem isn't that the AI is dumb. It's that no training environment has ever taught it what physics actually costs.

---

**[0:35–1:00 — Environment]**
*(Show: diagram of 16-well plate, scan action, contamination alert)*

We built Bio-Synthetica Pro — an OpenEnv RL environment that simulates a 16-well lab plate with three research-grade challenges. First: partial observability. The agent must scan each well before touching it. Second: dynamic replanning. A contamination alert fires mid-episode and the agent must reroute. Third: multi-objective reward — achieve the goal and stay within budget. No shortcuts.

---

**[1:00–1:30 — Training]**
*(Show: reward curve climbing, WandB dashboard)*

We trained Llama-3.1-8B using GRPO on a free Colab T4 GPU. Six reward tiers run simultaneously — syntax, constraint compliance, goal achievement, step efficiency, budget, and replanning. The model starts at minus-0.1 average reward and reaches plus-1.6 after 1,000 episodes.

---

**[1:30–1:50 — Results]**
*(Show: side-by-side before/after protocols)*

Before training: three violations, zero goal progress, negative reward. After training: clean scans, valid transfers, contamination rerouting, reward plus-1.7. Violations drop from four per episode to near zero. Goal achievement jumps from 8% to 74%.

---

**[1:50–2:00 — Call to action]**
*(Show: HuggingFace Space demo)*

Try it yourself — link in the description. All code is open source on GitHub. Built for OpenEnv Hackathon India 2026.

---
*Total words: ~262*
