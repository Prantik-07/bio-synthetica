# FILE: environment/observation_generator.py
import random
from environment.lab_simulator import WELL_IDS, REAGENT_PRICES


class ObservationGenerator:

    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level

    def add_noise(self, value: float) -> float:
        noisy = value * (1 + random.gauss(0, self.noise_level))
        return round(noisy, 2)

    def generate_goal(self, simulator) -> dict:
        target_well = random.choice(WELL_IDS)
        target_chemical = random.choice(list(REAGENT_PRICES.keys()))
        target_concentration = round(random.uniform(0.3, 0.8), 2)
        return {
            "target_well": target_well,
            "target_chemical": target_chemical,
            "target_concentration": target_concentration,
            "tolerance": 0.05,
            "description": (
                f"Achieve {target_concentration}x concentration of "
                f"{target_chemical} in well {target_well}"
            ),
        }

    def should_trigger_event(self, step: int, already_triggered: bool) -> bool:
        if already_triggered:
            return False
        if step < 3:
            return False
        if step > 7:
            return False
        return random.random() < 0.3

    def generate_observation(self, simulator, goal: dict, event=None) -> str:
        state = simulator.get_full_state()
        wells = state["wells"]
        scanned = set(state["scanned_wells"])
        unscanned = [w for w in WELL_IDS if w not in scanned]

        scanned_lines = []
        for well_id in sorted(scanned):
            w = wells[well_id]
            noisy_vol = self.add_noise(w["volume"])
            contamination_tag = "  ⚠️ CONTAMINATED" if w["contaminated"] else ""
            scanned_lines.append(
                f"  {well_id}: volume={noisy_vol}ul, "
                f"chemical={w['chemical']}, temp={w['temperature']}C"
                f"{contamination_tag}"
            )

        scanned_block = (
            "\n".join(scanned_lines) if scanned_lines else "  (none scanned yet)"
        )
        hidden_block = (
            ", ".join(unscanned) if unscanned else "(all wells scanned)"
        )

        alert_block = ""
        if event:
            alert_block = f"\nACTIVE ALERT: {event['message']}\n"

        obs = f"""=== BIO-SYNTHETICA LAB STATE ===

SCANNED WELLS (you can use these):
{scanned_block}

HIDDEN WELLS (scan these before using):
{hidden_block}

PIPETTE STATUS:
  Max volume: 200ul
  Tips remaining: {state['pipette']['tips_remaining']}

REAGENT COSTS (per ul):
  buffer: $0.10 | enzyme: $2.50 | substrate: $1.00 | water: $0.05

BUDGET REMAINING: ${state['budget_remaining']:.2f}

CURRENT GOAL:
  {goal['description']}
{alert_block}
AVAILABLE ACTIONS:
  scan(well_id)
  pipette(source_well, dest_well, volume_ul)
  mix(well_id, volume_ul, repetitions)
  set_temperature(well_id, temp_celsius)
  aspirate(well_id, volume_ul)
  dispense(well_id, volume_ul)
  discard_tip()
  report_complete()

Write a Python protocol to achieve the goal.
Output ONLY Python code. No explanations."""
        return obs
