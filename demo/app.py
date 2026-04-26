# FILE: demo/app.py
import json
import sys
sys.path.append('.')

import gradio as gr

from environment.bio_synthetica_env import BioSyntheticaEnv
from training.reward import RewardCalculator

SCENARIOS = {
    "Basic Transfer": {
        "description": "Transfer buffer from A1 to B1",
        "untrained_protocol": '''pipette("A1", "B1", volume=250)
report_complete()''',
        "trained_protocol": '''scan("A1")
scan("B1")
pipette("A1", "B1", volume=150)
mix("B1", volume=50, repetitions=3)
report_complete()''',
    },
    "Multi-Step Protocol": {
        "description": "Mix enzyme and substrate in C1",
        "untrained_protocol": '''pipette("A1", "C1", volume=300)
pipette("B1", "C1", volume=300)
report_complete()''',
        "trained_protocol": '''scan("A1")
scan("B1")
scan("C1")
pipette("A1", "C1", volume=100)
pipette("B1", "C1", volume=80)
mix("C1", volume=60, repetitions=5)
report_complete()''',
    },
    "Contamination Challenge": {
        "description": "Achieve goal while avoiding contaminated well",
        "untrained_protocol": '''pipette("A1", "B2", volume=150)
report_complete()''',
        "trained_protocol": '''scan("A1")
scan("B1")
scan("B2")
pipette("A1", "B1", volume=120)
mix("B1", volume=80, repetitions=3)
report_complete()''',
    },
    "Budget Constraint": {
        "description": "Achieve goal using minimum expensive reagents",
        "untrained_protocol": '''scan("A1")
pipette("A1", "D1", volume=200)
pipette("A1", "D1", volume=200)
report_complete()''',
        "trained_protocol": '''scan("A1")
scan("D1")
pipette("A1", "D1", volume=80)
mix("D1", volume=40, repetitions=2)
report_complete()''',
    },
}


def run_scenario(scenario_name: str, agent_type: str):
    scenario = SCENARIOS[scenario_name]
    env = BioSyntheticaEnv()
    calc = RewardCalculator()

    obs_dict = env.reset()

    protocol = (
        scenario["untrained_protocol"]
        if agent_type == "Untrained Agent"
        else scenario["trained_protocol"]
    )

    obs, reward, done, info = env.step(protocol)
    breakdown = calc.get_breakdown(info, env.state())

    lab_state = json.dumps(env.state()["wells"], indent=2)

    reward_display = f"""
REWARD BREAKDOWN:
Syntax Pass:      {'✅' if breakdown['syntax'] > 0 else '❌'} {breakdown['syntax']:+.2f}
No Violations:    {'✅' if breakdown['violations'] >= 0 else '❌'} {breakdown['violations']:+.2f}
Goal Achievement: {'✅' if breakdown['goal'] > 0.5 else '⚠️'} {breakdown['goal']:+.2f}
Step Efficiency:  {'✅' if breakdown['efficiency'] > 0 else '⚠️'} {breakdown['efficiency']:+.2f}
Budget Score:     {'✅' if breakdown['budget'] > 0.2 else '⚠️'} {breakdown['budget']:+.2f}
Replanning:       {'✅' if breakdown['replanning'] > 0 else 'N/A'} {breakdown['replanning']:+.2f}
──────────────────────────────────────────
TOTAL REWARD:     {reward:+.3f}

Violations: {len(info['violations'])}
Goal Progress: {info['goal_progress']:.1%}
Budget Used: ${info['budget_used']:.2f}
"""

    return lab_state, protocol, reward_display


with gr.Blocks(title="Bio-Synthetica Pro") as demo:
    gr.Markdown("# Bio-Synthetica Pro 🧬")
    gr.Markdown(
        "### Teaching AI to Think Like a Scientist - Under Real Constraints"
    )

    with gr.Row():
        with gr.Column():
            scenario_dropdown = gr.Dropdown(
                choices=list(SCENARIOS.keys()),
                value="Basic Transfer",
                label="Select Scenario",
            )
            agent_radio = gr.Radio(
                choices=["Untrained Agent", "Trained Agent"],
                value="Untrained Agent",
                label="Agent Type",
            )
            run_btn = gr.Button("Run Protocol", variant="primary")

            gr.Markdown("### What This Shows")
            gr.Markdown(
                "Compare how an untrained agent violates physical constraints "
                "vs how the trained agent respects them. The reward breakdown "
                "shows exactly what the agent learned."
            )

        with gr.Column():
            lab_state_box = gr.Code(label="Lab State", language="json")
            protocol_box = gr.Code(
                label="Generated Protocol", language="python"
            )
            reward_box = gr.Textbox(label="Reward Breakdown", lines=12)

    run_btn.click(
        fn=run_scenario,
        inputs=[scenario_dropdown, agent_radio],
        outputs=[lab_state_box, protocol_box, reward_box],
    )

if __name__ == "__main__":
    demo.launch()
