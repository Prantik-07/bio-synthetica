# FILE: environment/bio_synthetica_env.py
import ast
import random

try:
    from openenv import Environment
except ImportError:
    class Environment:
        """Fallback base class when openenv is not installed."""
        def reset(self):
            raise NotImplementedError
        def step(self, action):
            raise NotImplementedError
        def state(self):
            raise NotImplementedError

from environment.lab_simulator import LabSimulator
from environment.observation_generator import ObservationGenerator


class BioSyntheticaEnv(Environment):

    def __init__(self):
        self.simulator = LabSimulator()
        self.obs_generator = ObservationGenerator(noise_level=0.05)
        self.current_goal = None
        self.episode_step = 0
        self.max_steps = 15
        self.contamination_triggered = False
        self.active_event = None
        self.total_violations = 0
        self.episode_rewards = []

    def reset(self) -> dict:
        self.simulator.reset()
        self.current_goal = self.obs_generator.generate_goal(self.simulator)
        self.episode_step = 0
        self.contamination_triggered = False
        self.active_event = None
        self.total_violations = 0

        observation = self.obs_generator.generate_observation(
            self.simulator, self.current_goal
        )

        return {
            "observation": observation,
            "goal": self.current_goal,
            "step": 0,
            "done": False,
            "info": {},
        }

    def step(self, action: str) -> tuple:
        self.episode_step += 1

        if self.obs_generator.should_trigger_event(
            self.episode_step, self.contamination_triggered
        ):
            target = random.choice(self.simulator.WELL_IDS if hasattr(
                self.simulator, 'WELL_IDS') else list(self.simulator.wells.keys()))
            self.active_event = self.simulator.trigger_contamination_event(target)
            self.contamination_triggered = True

        parse_result = self.parse_protocol(action)

        all_violations = []
        rerouted = False
        goal_progress = 0.0
        syntax_pass = parse_result["syntax_pass"]

        if syntax_pass:
            for action_name, args in parse_result["actions"]:
                result = self.execute_action(action_name, args)
                if result.get("violations"):
                    all_violations.extend(result["violations"])
                if action_name == "report_complete":
                    break

            goal_progress = self.simulator.check_goal(self.current_goal)

            if self.contamination_triggered:
                contaminated = self.simulator.contaminated_wells
                used_wells = set()
                for action_name, args in parse_result["actions"]:
                    if len(args) > 0:
                        used_wells.add(args[0])
                if not contaminated.intersection(used_wells):
                    rerouted = True

        info = {
            "violations": all_violations,
            "goal_progress": goal_progress,
            "budget_used": round(10.0 - self.simulator.budget_remaining, 4),
            "steps_taken": self.episode_step,
            "event_triggered": self.contamination_triggered,
            "rerouted_successfully": rerouted,
            "syntax_pass": syntax_pass,
        }

        from training.reward import RewardCalculator
        calc = RewardCalculator()
        state = self.simulator.get_full_state()
        state["episode_step"] = self.episode_step
        state["budget_remaining"] = self.simulator.budget_remaining
        state["event"] = (
            "contamination_alert" if self.contamination_triggered else None
        )
        reward = calc.compute(info, state)

        done = self.episode_step >= self.max_steps or "report_complete" in action

        new_observation = self.obs_generator.generate_observation(
            self.simulator, self.current_goal, self.active_event
        )

        obs_dict = {
            "observation": new_observation,
            "goal": self.current_goal,
            "step": self.episode_step,
            "done": done,
            "info": info,
        }

        return (obs_dict, reward, done, info)

    def state(self) -> dict:
        full_state = self.simulator.get_full_state()
        full_state["episode_step"] = self.episode_step
        full_state["current_goal"] = self.current_goal
        full_state["contamination_triggered"] = self.contamination_triggered
        return full_state

    def parse_protocol(self, protocol_str: str) -> dict:
        """Parse LLM output string into list of (action_name, args) tuples."""
        try:
            tree = ast.parse(protocol_str)
        except SyntaxError:
            return {"syntax_pass": False, "actions": []}

        actions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name):
                    func_name = call.func.id
                    args = []
                    for arg in call.args:
                        try:
                            args.append(ast.literal_eval(arg))
                        except (ValueError, TypeError):
                            args.append(None)
                    for kw in call.keywords:
                        try:
                            args.append(ast.literal_eval(kw.value))
                        except (ValueError, TypeError):
                            args.append(None)
                    actions.append((func_name, args))

        return {"syntax_pass": True, "actions": actions}

    def execute_action(self, action_name: str, args: list) -> dict:
        """Route parsed action name to simulator method."""
        routing = {
            "scan": lambda a: self.simulator.scan(*a),
            "pipette": lambda a: self.simulator.pipette(*a),
            "mix": lambda a: self.simulator.mix(*a),
            "set_temperature": lambda a: self.simulator.set_temperature(*a),
            "aspirate": lambda a: self.simulator.aspirate(*a),
            "dispense": lambda a: self.simulator.dispense(*a),
            "discard_tip": lambda a: self.simulator.discard_tip(),
            "report_complete": lambda a: {"success": True, "violations": []},
        }
        if action_name not in routing:
            return {"success": False, "violations": [f"Unknown action: {action_name}"]}
        try:
            return routing[action_name](args)
        except Exception as e:
            return {"success": False, "violations": [f"Action error: {str(e)}"]}
