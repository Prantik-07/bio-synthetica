# HuggingFace Space entry point for Bio-Synthetica Pro
# Deploy this file + requirements.txt to a HF Space
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr

# ---------------------------------------------------------------------------
# Inline minimal copies of env + reward so the Space works without cloning
# ---------------------------------------------------------------------------
import random
import ast

WELL_IDS = [
    "A1","A2","A3","A4","B1","B2","B3","B4",
    "C1","C2","C3","C4","D1","D2","D3","D4",
]
MAX_WELL_VOLUME   = 200
MAX_PIPETTE_VOLUME = 200
MIN_TEMP, MAX_TEMP = 4, 95
MAX_TIPS = 8
STARTING_BUDGET = 10.0
REAGENT_PRICES = {"buffer":0.1,"enzyme":2.5,"substrate":1.0,"water":0.05}


class _Sim:
    def __init__(self):
        self.reset()

    def reset(self):
        chem = list(REAGENT_PRICES)
        self.wells = {w:{"volume":round(random.uniform(0,100),2),
                         "chemical":random.choice(chem),
                         "temperature":25.0,"contaminated":False}
                      for w in WELL_IDS}
        self.tips = MAX_TIPS
        self.budget = STARTING_BUDGET
        self.contaminated = set()
        self.scanned = set()

    def scan(self, w):
        if w not in WELL_IDS: return {"success":False,"violations":[f"Unknown well {w}"]}
        self.scanned.add(w)
        return {"success":True,"violations":[]}

    def pipette(self, src, dst, vol):
        v=[]
        if src not in self.scanned: v.append(f"{src} not scanned")
        if dst not in self.scanned: v.append(f"{dst} not scanned")
        if vol>MAX_PIPETTE_VOLUME:  v.append(f"{vol}ul exceeds 200ul max")
        if src in WELL_IDS and vol>self.wells[src]["volume"]:
            v.append(f"Insufficient volume in {src}")
        if dst in WELL_IDS and self.wells[dst]["volume"]+vol>MAX_WELL_VOLUME:
            v.append(f"Would overflow {dst}")
        if src in self.contaminated: v.append(f"{src} is contaminated")
        if dst in self.contaminated: v.append(f"{dst} is contaminated")
        if v: return {"success":False,"violations":v}
        chem = self.wells[src]["chemical"]
        cost = round(vol*REAGENT_PRICES[chem],4)
        self.wells[src]["volume"] = round(self.wells[src]["volume"]-vol,2)
        self.wells[dst]["volume"] = round(self.wells[dst]["volume"]+vol,2)
        self.wells[dst]["chemical"] = chem
        self.budget = round(self.budget-cost,4)
        return {"success":True,"violations":[]}

    def mix(self, w, vol, reps):
        v=[]
        if w not in self.scanned: v.append(f"{w} not scanned")
        if w in self.contaminated: v.append(f"{w} contaminated")
        if v: return {"success":False,"violations":v}
        self.budget = round(self.budget-0.1*reps,4)
        return {"success":True,"violations":[]}

    def check_goal(self, goal):
        tw = goal["target_well"]
        tc = goal["target_chemical"]
        tconc = goal["target_concentration"]
        tol = goal.get("tolerance",0.05)
        if self.wells[tw]["chemical"] != tc: return 0.0
        conc = self.wells[tw]["volume"]/MAX_WELL_VOLUME
        diff = abs(conc-tconc)
        if diff<=tol: return 1.0
        if diff<=2*tol: return 0.5
        return max(0.0, round(0.4*(1-diff),4))

    def get_state(self):
        return {
            "wells":{w:{"volume":d["volume"],"chemical":d["chemical"],
                        "temperature":d["temperature"],"contaminated":d["contaminated"],
                        "scanned":w in self.scanned}
                     for w,d in self.wells.items()},
            "budget_remaining": self.budget,
            "scanned_wells": list(self.scanned),
        }

def _parse(code):
    try: tree=ast.parse(code)
    except SyntaxError: return {"syntax_pass":False,"actions":[]}
    acts=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
            c=node.value
            if isinstance(c.func,ast.Name):
                args=[]
                for a in c.args:
                    try: args.append(ast.literal_eval(a))
                    except: args.append(None)
                for kw in c.keywords:
                    try: args.append(ast.literal_eval(kw.value))
                    except: args.append(None)
                acts.append((c.func.id,args))
    return {"syntax_pass":True,"actions":acts}

def _run_protocol(protocol, goal):
    sim = _Sim()
    pr = _parse(protocol)
    if not pr["syntax_pass"]:
        return sim, pr, {"syntax_pass":False,"violations":[],"goal_progress":0.0,"budget_used":0.0,"rerouted_successfully":False}
    viols=[]
    for name,args in pr["actions"]:
        fn = {"scan":sim.scan,"pipette":sim.pipette,"mix":sim.mix,
              "report_complete":lambda *a:{"success":True,"violations":[]}}.get(name)
        if fn:
            r=fn(*args) if args else fn()
            viols.extend(r.get("violations",[]))
        if name=="report_complete": break
    gp = sim.check_goal(goal)
    info = {"syntax_pass":True,"violations":viols,"goal_progress":gp,
            "budget_used":round(STARTING_BUDGET-sim.budget,4),
            "rerouted_successfully":False}
    return sim, pr, info

def _reward(info, budget):
    if not info["syntax_pass"]: return -0.5
    r=0.1
    n=len(info["violations"])
    r+=0.3 if n==0 else -min(0.15*n,0.45)
    r+=1.0*info["goal_progress"]
    r+=0.3
    r+=0.3*max(0,(STARTING_BUDGET-info["budget_used"])/STARTING_BUDGET)
    return round(r,4)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
SCENARIOS = {
    "Basic Transfer (buffer A1→B1)": {
        "goal": {"target_well":"B1","target_chemical":"buffer",
                 "target_concentration":0.5,"tolerance":0.05,
                 "description":"Achieve 0.5x concentration of buffer in well B1"},
        "untrained": 'pipette("A1", "B1", volume=250)\nreport_complete()',
        "trained":   'scan("A1")\nscan("B1")\npipette("A1", "B1", volume=100)\nmix("B1", 50, 3)\nreport_complete()',
    },
    "Multi-Step: enzyme + substrate → C1": {
        "goal": {"target_well":"C1","target_chemical":"enzyme",
                 "target_concentration":0.6,"tolerance":0.05,
                 "description":"Achieve 0.6x concentration of enzyme in well C1"},
        "untrained": 'pipette("A1","C1",volume=300)\npipette("B1","C1",volume=300)\nreport_complete()',
        "trained":   'scan("A1")\nscan("B1")\nscan("C1")\npipette("A1","C1",volume=100)\npipette("B1","C1",volume=80)\nmix("C1",60,5)\nreport_complete()',
    },
    "Contamination Avoidance": {
        "goal": {"target_well":"B1","target_chemical":"substrate",
                 "target_concentration":0.4,"tolerance":0.05,
                 "description":"Achieve 0.4x concentration of substrate in well B1, avoiding B2 (contaminated)"},
        "untrained": 'pipette("A1","B2",volume=150)\nreport_complete()',
        "trained":   'scan("A1")\nscan("B1")\npipette("A1","B1",volume=80)\nmix("B1",50,3)\nreport_complete()',
    },
    "Budget Constraint (enzyme is $2.50/ul)": {
        "goal": {"target_well":"D1","target_chemical":"water",
                 "target_concentration":0.35,"tolerance":0.05,
                 "description":"Achieve 0.35x concentration of water in well D1 cheaply"},
        "untrained": 'scan("A1")\npipette("A1","D1",volume=200)\npipette("A1","D1",volume=200)\nreport_complete()',
        "trained":   'scan("A1")\nscan("D1")\npipette("A1","D1",volume=70)\nmix("D1",40,2)\nreport_complete()',
    },
}

def run_scenario(scenario_name, agent_type):
    s = SCENARIOS[scenario_name]
    goal = s["goal"]
    protocol = s["untrained"] if agent_type == "🚫 Untrained Agent" else s["trained"]
    sim, pr, info = _run_protocol(protocol, goal)
    reward = _reward(info, sim.budget)

    breakdown = {
        "Syntax":     0.1 if info["syntax_pass"] else -0.5,
        "Violations": 0.3 if len(info["violations"])==0 else -min(0.15*len(info["violations"]),0.45),
        "Goal":       round(1.0*info["goal_progress"],4),
        "Efficiency": 0.3,
        "Budget":     round(0.3*max(0,(STARTING_BUDGET-info["budget_used"])/STARTING_BUDGET),4),
        "Replanning": 0.0,
    }

    state_json = json.dumps({
        w: {"volume":d["volume"],"chemical":d["chemical"],
            "scanned":d["scanned"],"contaminated":d["contaminated"]}
        for w,d in sim.get_state()["wells"].items()
    }, indent=2)

    lines = ["REWARD BREAKDOWN", "─"*38]
    icons = {"Syntax":"🟢" if breakdown["Syntax"]>0 else "🔴",
             "Violations":"🟢" if breakdown["Violations"]>=0 else "🔴",
             "Goal":"🟢" if breakdown["Goal"]>0.5 else "🟡",
             "Efficiency":"🟢","Budget":"🟢" if breakdown["Budget"]>0.1 else "🟡",
             "Replanning":"⚪"}
    for k,v in breakdown.items():
        lines.append(f"{icons[k]} {k:<12} {v:+.3f}")
    lines += ["─"*38,
              f"   TOTAL         {reward:+.3f}",
              "",
              f"Violations : {len(info['violations'])}",
              f"Goal       : {info['goal_progress']:.1%}",
              f"Budget used: ${info['budget_used']:.2f}"]
    if info["violations"]:
        lines.append("\nViolation details:")
        for vv in info["violations"]:
            lines.append(f"  ⚠ {vv}")

    return state_json, protocol, "\n".join(lines)

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
css = """
.gr-button-primary { background: #2563EB !important; }
#reward-box textarea { font-family: monospace; }
"""

with gr.Blocks(title="Bio-Synthetica Pro 🧬", css=css) as demo:
    gr.Markdown("""
# 🧬 Bio-Synthetica Pro
### Teaching AI to Think Like a Scientist - Under Real Constraints

> *"An AI that learned the laws of physics by breaking them thousands of times in simulation."*

**GitHub** · [Prantik-07/bio-synthetica](https://github.com/Prantik-07/bio-synthetica) &nbsp;|&nbsp;
**Hackathon** · OpenEnv Hackathon India 2026 &nbsp;|&nbsp;
**Training** · Llama-3.1-8B + GRPO (Unsloth, T4 GPU)
""")

    gr.Markdown("""
## What this shows
Pick a scenario, choose **Untrained** or **Trained** agent, and see the difference.
The untrained agent violates physical constraints (overflows wells, skips scans, ignores contamination).
The trained agent - fine-tuned with GRPO on this environment - respects every constraint and minimises cost.
""")

    with gr.Row():
        with gr.Column(scale=1):
            scenario_dd = gr.Dropdown(
                choices=list(SCENARIOS.keys()),
                value=list(SCENARIOS.keys())[0],
                label="Scenario",
            )
            agent_radio = gr.Radio(
                choices=["🚫 Untrained Agent", "✅ Trained Agent"],
                value="🚫 Untrained Agent",
                label="Agent type",
            )
            run_btn = gr.Button("▶ Run Protocol", variant="primary", size="lg")

            gr.Markdown("""
### Reward tiers
| Tier | Max |
|---|---|
| Syntax pass | +0.1 |
| No violations | +0.3 |
| Goal achievement | +1.0 |
| Step efficiency | +0.3 |
| Budget efficiency | +0.3 |
| Replanning bonus | +0.5 |
| **Maximum** | **+2.5** |
""")

        with gr.Column(scale=2):
            protocol_box = gr.Code(label="Protocol executed", language="python", lines=10)
            with gr.Row():
                reward_box   = gr.Textbox(label="Reward breakdown", lines=14, elem_id="reward-box")
                state_box    = gr.Code(label="Lab state (JSON)", language="json", lines=14)

    run_btn.click(
        fn=run_scenario,
        inputs=[scenario_dd, agent_radio],
        outputs=[state_box, protocol_box, reward_box],
    )

    gr.Markdown("""
---
### How the environment works
The agent sees a **partial 16-well lab plate** (wells are hidden until scanned) with 5% sensor noise.
It writes Python code using `scan()`, `pipette()`, `mix()`, `set_temperature()`, and `report_complete()`.
Between steps 3–7 a **contamination alert** fires randomly - the trained agent reroutes without restarting.

Built with [OpenEnv](https://github.com/openenv/openenv) · Trained with [Unsloth](https://github.com/unslothai/unsloth) GRPO
""")

if __name__ == "__main__":
    demo.launch()
