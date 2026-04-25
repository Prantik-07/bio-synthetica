# FILE: environment/lab_simulator.py
import random

WELL_IDS = [
    "A1", "A2", "A3", "A4",
    "B1", "B2", "B3", "B4",
    "C1", "C2", "C3", "C4",
    "D1", "D2", "D3", "D4",
]

MAX_WELL_VOLUME = 200
MAX_PIPETTE_VOLUME = 200
MIN_TEMP = 4
MAX_TEMP = 95
MAX_TIPS = 8
STARTING_BUDGET = 10.0

REAGENT_PRICES = {
    "buffer": 0.1,
    "enzyme": 2.5,
    "substrate": 1.0,
    "water": 0.05,
}


class LabSimulator:

    def __init__(self, seed=None):
        self._seed = seed
        self._rng = random.Random(seed)
        self._init_state()

    def _init_state(self):
        chemicals = list(REAGENT_PRICES.keys())
        self.wells = {
            well: {
                "volume": round(self._rng.uniform(0, 100), 2),
                "chemical": self._rng.choice(chemicals),
                "temperature": 25.0,
                "contaminated": False,
            }
            for well in WELL_IDS
        }
        self.tips_remaining = MAX_TIPS
        self.budget_remaining = STARTING_BUDGET
        self.contaminated_wells = set()
        self.scanned_wells = set()
        self.event_log = []
        self.step_count = 0

    def reset(self, seed=None):
        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random(self._seed)
        self._init_state()
        return self.get_full_state()

    def scan(self, well_id: str) -> dict:
        if well_id not in WELL_IDS:
            return {"success": False, "error": f"Unknown well: {well_id}"}
        self.scanned_wells.add(well_id)
        well = self.wells[well_id]
        return {
            "success": True,
            "well_id": well_id,
            "volume": well["volume"],
            "chemical": well["chemical"],
            "temperature": well["temperature"],
            "contaminated": well["contaminated"],
        }

    def pipette(self, source: str, dest: str, volume: float) -> dict:
        violations = []
        if source not in self.scanned_wells:
            violations.append(f"Source well {source} has not been scanned")
        if dest not in self.scanned_wells:
            violations.append(f"Destination well {dest} has not been scanned")
        if volume > MAX_PIPETTE_VOLUME:
            violations.append(
                f"Volume {volume}ul exceeds pipette max {MAX_PIPETTE_VOLUME}ul"
            )
        if source in WELL_IDS and volume > self.wells[source]["volume"]:
            violations.append(
                f"Insufficient volume in {source}: "
                f"{self.wells[source]['volume']}ul available, {volume}ul requested"
            )
        if dest in WELL_IDS and self.wells[dest]["volume"] + volume > MAX_WELL_VOLUME:
            violations.append(
                f"Transfer would overflow {dest}: "
                f"{self.wells[dest]['volume']}ul + {volume}ul > {MAX_WELL_VOLUME}ul"
            )
        if source in self.contaminated_wells:
            violations.append(f"Source well {source} is contaminated")
        if dest in self.contaminated_wells:
            violations.append(f"Destination well {dest} is contaminated")

        if violations:
            return {"success": False, "violations": violations, "cost": 0}

        chemical = self.wells[source]["chemical"]
        cost = round(volume * REAGENT_PRICES[chemical], 4)
        self.wells[source]["volume"] = round(
            self.wells[source]["volume"] - volume, 2
        )
        self.wells[dest]["volume"] = round(
            self.wells[dest]["volume"] + volume, 2
        )
        self.wells[dest]["chemical"] = chemical
        self.budget_remaining = round(self.budget_remaining - cost, 4)
        self.step_count += 1
        return {"success": True, "violations": [], "cost": cost}

    def mix(self, well_id: str, volume: float, repetitions: int) -> dict:
        violations = []
        if well_id not in self.scanned_wells:
            violations.append(f"Well {well_id} has not been scanned")
        if well_id in self.contaminated_wells:
            violations.append(f"Well {well_id} is contaminated")
        if well_id in WELL_IDS and volume > self.wells[well_id]["volume"]:
            violations.append(
                f"Mix volume {volume}ul exceeds well volume "
                f"{self.wells[well_id]['volume']}ul"
            )

        if violations:
            return {"success": False, "violations": violations, "cost": 0}

        cost = round(0.1 * repetitions, 4)
        self.budget_remaining = round(self.budget_remaining - cost, 4)
        self.step_count += 1
        return {"success": True, "violations": [], "cost": cost}

    def set_temperature(self, well_id: str, temp: float) -> dict:
        violations = []
        if well_id not in self.scanned_wells:
            violations.append(f"Well {well_id} has not been scanned")
        if temp < MIN_TEMP or temp > MAX_TEMP:
            violations.append(
                f"Temperature {temp}C out of range [{MIN_TEMP}C, {MAX_TEMP}C]"
            )

        if violations:
            return {"success": False, "violations": violations, "cost": 0}

        self.wells[well_id]["temperature"] = temp
        cost = 0.2
        self.budget_remaining = round(self.budget_remaining - cost, 4)
        self.step_count += 1
        return {"success": True, "violations": [], "cost": cost}

    def aspirate(self, well_id: str, volume: float) -> dict:
        violations = []
        if well_id not in self.scanned_wells:
            violations.append(f"Well {well_id} has not been scanned")
        if volume > MAX_PIPETTE_VOLUME:
            violations.append(
                f"Volume {volume}ul exceeds pipette max {MAX_PIPETTE_VOLUME}ul"
            )
        if well_id in WELL_IDS and volume > self.wells[well_id]["volume"]:
            violations.append(
                f"Insufficient volume in {well_id}: "
                f"{self.wells[well_id]['volume']}ul available, {volume}ul requested"
            )
        if well_id in self.contaminated_wells:
            violations.append(f"Well {well_id} is contaminated")

        if violations:
            return {"success": False, "violations": violations, "cost": 0}

        cost = round(volume * 0.01, 4)
        self.wells[well_id]["volume"] = round(
            self.wells[well_id]["volume"] - volume, 2
        )
        self.budget_remaining = round(self.budget_remaining - cost, 4)
        self.step_count += 1
        return {"success": True, "violations": [], "cost": cost}

    def dispense(self, well_id: str, volume: float) -> dict:
        violations = []
        if well_id not in self.scanned_wells:
            violations.append(f"Well {well_id} has not been scanned")
        if well_id in WELL_IDS and self.wells[well_id]["volume"] + volume > MAX_WELL_VOLUME:
            violations.append(
                f"Dispense would overflow {well_id}: "
                f"{self.wells[well_id]['volume']}ul + {volume}ul > {MAX_WELL_VOLUME}ul"
            )
        if well_id in self.contaminated_wells:
            violations.append(f"Well {well_id} is contaminated")

        if violations:
            return {"success": False, "violations": violations, "cost": 0}

        self.wells[well_id]["volume"] = round(
            self.wells[well_id]["volume"] + volume, 2
        )
        self.step_count += 1
        return {"success": True, "violations": [], "cost": 0}

    def discard_tip(self) -> dict:
        if self.tips_remaining == 0:
            return {"success": False, "error": "No tips remaining"}
        self.tips_remaining -= 1
        self.step_count += 1
        return {"success": True, "tips_remaining": self.tips_remaining}

    def trigger_contamination_event(self, well_id: str) -> dict:
        self.contaminated_wells.add(well_id)
        self.wells[well_id]["contaminated"] = True
        event = {
            "event": "contamination_alert",
            "affected_well": well_id,
            "message": f"ALERT: Well {well_id} is contaminated. Avoid it.",
        }
        self.event_log.append(event)
        return event

    def check_goal(self, goal: dict) -> float:
        target_well = goal.get("target_well")
        target_chemical = goal.get("target_chemical")
        target_concentration = goal.get("target_concentration", 0.5)
        tolerance = goal.get("tolerance", 0.05)

        if target_well not in WELL_IDS:
            return 0.0

        well = self.wells[target_well]

        if well["chemical"] != target_chemical:
            return 0.0

        total_volume = sum(w["volume"] for w in self.wells.values())
        if total_volume == 0:
            return 0.0

        well_volume = well["volume"]
        concentration = well_volume / MAX_WELL_VOLUME

        diff = abs(concentration - target_concentration)

        if diff <= tolerance:
            return 1.0
        elif diff <= 2 * tolerance:
            return 0.5
        else:
            max_diff = 1.0
            score = max(0.0, 1.0 - diff / max_diff)
            return round(score * 0.4, 4)

    def get_full_state(self) -> dict:
        return {
            "wells": {
                well_id: {
                    "volume": data["volume"],
                    "chemical": data["chemical"],
                    "temperature": data["temperature"],
                    "contaminated": data["contaminated"],
                    "scanned": well_id in self.scanned_wells,
                }
                for well_id, data in self.wells.items()
            },
            "pipette": {
                "max_volume": MAX_PIPETTE_VOLUME,
                "tips_remaining": self.tips_remaining,
            },
            "budget_remaining": self.budget_remaining,
            "contaminated_wells": list(self.contaminated_wells),
            "scanned_wells": list(self.scanned_wells),
            "step_count": self.step_count,
            "event_log": self.event_log,
        }
