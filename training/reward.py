# FILE: training/reward.py


class RewardCalculator:

    def compute(self, action_result: dict, state: dict) -> float:
        reward = 0.0

        # TIER 1: Syntax pass (+0.1) or hard fail (-0.5)
        if not action_result.get("syntax_pass", False):
            return -0.5
        reward += 0.1

        # TIER 2: No violations (+0.3), penalise per violation
        violations = action_result.get("violations", [])
        n = len(violations)
        if n == 0:
            reward += 0.3
        else:
            reward -= min(0.15 * n, 0.45)

        # TIER 3: Goal achievement (+1.0 scaled 0–1)
        goal_progress = action_result.get("goal_progress", 0.0)
        reward += 1.0 * goal_progress

        # TIER 4: Step efficiency (+0.3)
        steps = state.get("episode_step", 0)
        if steps < 8:
            reward += 0.3
        elif steps < 12:
            reward += 0.1

        # TIER 5: Budget efficiency (+0.3)
        budget_remaining = state.get("budget_remaining", 10.0)
        budget_ratio = max(0.0, budget_remaining / 10.0)
        reward += 0.3 * budget_ratio

        # TIER 6: Replanning bonus (+0.5)
        if (
            state.get("event") == "contamination_alert"
            and action_result.get("rerouted_successfully", False)
        ):
            reward += 0.5

        return round(reward, 4)

    def get_breakdown(self, action_result: dict, state: dict) -> dict:
        syntax_score = 0.0
        if not action_result.get("syntax_pass", False):
            return {
                "syntax": -0.5,
                "violations": 0.0,
                "goal": 0.0,
                "efficiency": 0.0,
                "budget": 0.0,
                "replanning": 0.0,
                "total": -0.5,
            }
        syntax_score = 0.1

        violations = action_result.get("violations", [])
        n = len(violations)
        violation_score = 0.3 if n == 0 else -min(0.15 * n, 0.45)

        goal_progress = action_result.get("goal_progress", 0.0)
        goal_score = round(1.0 * goal_progress, 4)

        steps = state.get("episode_step", 0)
        if steps < 8:
            efficiency_score = 0.3
        elif steps < 12:
            efficiency_score = 0.1
        else:
            efficiency_score = 0.0

        budget_remaining = state.get("budget_remaining", 10.0)
        budget_ratio = max(0.0, budget_remaining / 10.0)
        budget_score = round(0.3 * budget_ratio, 4)

        replanning_score = 0.0
        if (
            state.get("event") == "contamination_alert"
            and action_result.get("rerouted_successfully", False)
        ):
            replanning_score = 0.5

        total = round(
            syntax_score
            + violation_score
            + goal_score
            + efficiency_score
            + budget_score
            + replanning_score,
            4,
        )

        return {
            "syntax": syntax_score,
            "violations": round(violation_score, 4),
            "goal": goal_score,
            "efficiency": efficiency_score,
            "budget": budget_score,
            "replanning": replanning_score,
            "total": total,
        }
