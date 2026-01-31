from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


# Thresholds for "close to wall" (normalized lidar: 0 = at wall, 1 = max range)
CLOSE_TURN_THRESHOLD = 0.25   # penalize fast turning when min_lidar_norm < this
CLOSE_SPEED_THRESHOLD = 0.35  # penalize high speed when min_lidar_norm < this


@dataclass
class RewardComponents:
    """Decomposed reward terms for easier logging and testing."""

    progress: float
    collision_penalty: float
    goal_bonus: float
    action_mag_penalty: float
    jerk_penalty: float
    close_turn_penalty: float = 0.0
    close_speed_penalty: float = 0.0

    def total(self) -> float:
        """Return weighted sum (the final reward)."""
        return (
            1.0 * self.progress
            + 10.0 * self.goal_bonus
            + 5.0 * self.collision_penalty
            + self.action_mag_penalty
            + self.jerk_penalty
            + self.close_turn_penalty
            + self.close_speed_penalty
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "progress": float(self.progress),
            "collision_penalty": float(self.collision_penalty),
            "goal_bonus": float(self.goal_bonus),
            "action_mag_penalty": float(self.action_mag_penalty),
            "jerk_penalty": float(self.jerk_penalty),
            "close_turn_penalty": float(self.close_turn_penalty),
            "close_speed_penalty": float(self.close_speed_penalty),
        }


def compute_reward(
    *,
    prev_distance: float,
    current_distance: float,
    collided: bool,
    reached_goal: bool,
    action_norm: float,
    jerk_norm: float,
    min_lidar_normalized: float | None = None,
    linear_velocity: float | None = None,
    angular_velocity: float | None = None,
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
    min_lidar_normalized : float, optional
        Min LiDAR range / max_range (0 = at wall, 1 = max range). Used for
        close-turn and close-speed penalties.
    linear_velocity : float, optional
        Current linear velocity command (v). Used for close-speed penalty.
    angular_velocity : float, optional
        Current angular velocity command (w). Used for close-turn penalty.
    """
    progress = prev_distance - current_distance
    collision_penalty = -1.0 if collided else 0.0
    goal_bonus = 1.0 if reached_goal else 0.0
    action_mag_penalty = -0.01 * action_norm
    jerk_penalty = -0.01 * jerk_norm

    # Encourage "slow and turn carefully" when close to walls
    close_turn_penalty = 0.0
    close_speed_penalty = 0.0
    if (
        min_lidar_normalized is not None
        and angular_velocity is not None
        and linear_velocity is not None
    ):
        if min_lidar_normalized < CLOSE_TURN_THRESHOLD:
            # Penalize fast turning in tight spaces (s_curve, zigzag, maze)
            close_turn_penalty = -0.03 * abs(angular_velocity)
        if min_lidar_normalized < CLOSE_SPEED_THRESHOLD:
            # Penalize high speed when near walls
            close_speed_penalty = -0.015 * abs(linear_velocity)

    return RewardComponents(
        progress=progress,
        collision_penalty=collision_penalty,
        goal_bonus=goal_bonus,
        action_mag_penalty=action_mag_penalty,
        jerk_penalty=jerk_penalty,
        close_turn_penalty=close_turn_penalty,
        close_speed_penalty=close_speed_penalty,
    )

