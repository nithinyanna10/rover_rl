from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RewardComponents:
    """Decomposed reward terms for easier logging and testing."""

    progress: float
    collision_penalty: float
    goal_bonus: float
    action_mag_penalty: float
    jerk_penalty: float

    def total(self) -> float:
        """Return weighted sum (the final reward)."""
        return (
            1.0 * self.progress
            + 10.0 * self.goal_bonus
            + 5.0 * self.collision_penalty
            + self.action_mag_penalty
            + self.jerk_penalty
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "progress": float(self.progress),
            "collision_penalty": float(self.collision_penalty),
            "goal_bonus": float(self.goal_bonus),
            "action_mag_penalty": float(self.action_mag_penalty),
            "jerk_penalty": float(self.jerk_penalty),
        }


def compute_reward(
    *,
    prev_distance: float,
    current_distance: float,
    collided: bool,
    reached_goal: bool,
    action_norm: float,
    jerk_norm: float,
) -> RewardComponents:
    """Compute shaped reward for rover navigation.

    Parameters
    ----------
    prev_distance : float
        Distance to goal at the previous step.
    current_distance : float
        Distance to goal after the current transition.
    collided : bool
        Whether a collision occurred at this step.
    reached_goal : bool
        Whether the goal was reached at this step.
    action_norm : float
        L2 norm of the current action.
    jerk_norm : float
        L2 norm of action difference from previous step.
    """
    progress = prev_distance - current_distance
    collision_penalty = -1.0 if collided else 0.0
    goal_bonus = 1.0 if reached_goal else 0.0
    action_mag_penalty = -0.01 * action_norm
    jerk_penalty = -0.01 * jerk_norm

    return RewardComponents(
        progress=progress,
        collision_penalty=collision_penalty,
        goal_bonus=goal_bonus,
        action_mag_penalty=action_mag_penalty,
        jerk_penalty=jerk_penalty,
    )

