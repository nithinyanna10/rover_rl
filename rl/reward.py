from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


# Barrier-style proximity: only when VERY close
PROXIMITY_SAFE = 0.12
PROXIMITY_SCALE = 0.02

# Clearance: small reward for keeping min_lidar above d_safe (reduces hugging / collisions in narrow)
CLEARANCE_SAFE = 0.15   # normalized; reward when min_lidar > this
CLEARANCE_SCALE = 0.02  # r_clear = CLEARANCE_SCALE * (min_lidar - CLEARANCE_SAFE) clipped

# Stuck: small penalty when progress over last N steps is below threshold (stops thrashing)
STUCK_PROGRESS_THRESHOLD = 0.05  # min progress (normalized) over window to avoid penalty
STUCK_PENALTY = 0.01


@dataclass
class RewardComponents:
    """Decomposed reward terms for easier logging and testing."""

    progress: float
    collision_penalty: float
    goal_bonus: float
    action_mag_penalty: float
    jerk_penalty: float
    proximity_penalty: float = 0.0
    clearance_reward: float = 0.0
    stuck_penalty: float = 0.0

    def total(self) -> float:
        """Return weighted sum (the final reward)."""
        return (
            1.0 * self.progress
            + 10.0 * self.goal_bonus
            + 5.0 * self.collision_penalty
            + self.action_mag_penalty
            + self.jerk_penalty
            + self.proximity_penalty
            + self.clearance_reward
            + self.stuck_penalty
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "progress": float(self.progress),
            "collision_penalty": float(self.collision_penalty),
            "goal_bonus": float(self.goal_bonus),
            "action_mag_penalty": float(self.action_mag_penalty),
            "jerk_penalty": float(self.jerk_penalty),
            "proximity_penalty": float(self.proximity_penalty),
            "clearance_reward": float(self.clearance_reward),
            "stuck_penalty": float(self.stuck_penalty),
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
    progress_over_last_N: float | None = None,
    lidar_max_range: float = 10.0,
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
        Min LiDAR range / max_range. Used for proximity penalty and clearance reward.
    linear_velocity : float, optional
        Unused; kept for API compatibility.
    angular_velocity : float, optional
        Unused; kept for API compatibility.
    progress_over_last_N : float, optional
        Distance progress over last N steps (positive = got closer). Used for stuck penalty.
    lidar_max_range : float
        Used to normalize progress for stuck threshold.
    """
    progress = prev_distance - current_distance
    collision_penalty = -1.0 if collided else 0.0
    goal_bonus = 1.0 if reached_goal else 0.0
    action_mag_penalty = -0.01 * action_norm
    jerk_penalty = -0.01 * jerk_norm

    proximity_penalty = 0.0
    if min_lidar_normalized is not None and min_lidar_normalized < PROXIMITY_SAFE:
        proximity_penalty = -PROXIMITY_SCALE * (PROXIMITY_SAFE - min_lidar_normalized)

    # Clearance: small reward for keeping distance from walls (topology maps)
    clearance_reward = 0.0
    if min_lidar_normalized is not None and min_lidar_normalized > CLEARANCE_SAFE:
        clearance_reward = CLEARANCE_SCALE * (min_lidar_normalized - CLEARANCE_SAFE)

    # Stuck: small penalty when progress over last N steps is below threshold
    stuck_penalty = 0.0
    if progress_over_last_N is not None and lidar_max_range > 0:
        progress_norm = progress_over_last_N / lidar_max_range
        if progress_norm < STUCK_PROGRESS_THRESHOLD:
            stuck_penalty = -STUCK_PENALTY

    return RewardComponents(
        progress=progress,
        collision_penalty=collision_penalty,
        goal_bonus=goal_bonus,
        action_mag_penalty=action_mag_penalty,
        jerk_penalty=jerk_penalty,
        proximity_penalty=proximity_penalty,
        clearance_reward=clearance_reward,
        stuck_penalty=stuck_penalty,
    )

