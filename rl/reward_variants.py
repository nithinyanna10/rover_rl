"""
Alternative reward functions and reward analysis for rover RL.

Provides sparse, dense, hybrid, and curriculum reward variants,
plus utilities to log and compare reward components across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
import math
import numpy as np

from rl.reward import (
    RewardComponents,
    compute_reward,
    PROXIMITY_SAFE,
    PROXIMITY_SCALE,
    CLEARANCE_SAFE,
    CLEARANCE_SCALE,
    STUCK_PROGRESS_THRESHOLD,
    STUCK_PENALTY,
)


# ---------------------------------------------------------------------------
# Sparse reward: only goal and collision
# ---------------------------------------------------------------------------


def compute_reward_sparse(
    *,
    prev_distance: float,
    current_distance: float,
    collided: bool,
    reached_goal: bool,
    action_norm: float = 0.0,
    jerk_norm: float = 0.0,
    min_lidar_normalized: Optional[float] = None,
    linear_velocity: Optional[float] = None,
    angular_velocity: Optional[float] = None,
    progress_over_last_N: Optional[float] = None,
    lidar_max_range: float = 10.0,
) -> RewardComponents:
    """Sparse reward: progress=0, only goal bonus and collision penalty."""
    progress = 0.0
    collision_penalty = -1.0 if collided else 0.0
    goal_bonus = 1.0 if reached_goal else 0.0
    return RewardComponents(
        progress=progress,
        collision_penalty=collision_penalty,
        goal_bonus=goal_bonus,
        action_mag_penalty=0.0,
        jerk_penalty=0.0,
        proximity_penalty=0.0,
        clearance_reward=0.0,
        stuck_penalty=0.0,
    )


# ---------------------------------------------------------------------------
# Dense progress-only (no action/jerk/clearance)
# ---------------------------------------------------------------------------


def compute_reward_dense_progress(
    *,
    prev_distance: float,
    current_distance: float,
    collided: bool,
    reached_goal: bool,
    action_norm: float = 0.0,
    jerk_norm: float = 0.0,
    min_lidar_normalized: Optional[float] = None,
    linear_velocity: Optional[float] = None,
    angular_velocity: Optional[float] = None,
    progress_over_last_N: Optional[float] = None,
    lidar_max_range: float = 10.0,
) -> RewardComponents:
    """Dense reward: progress + goal + collision only (no action/jerk/clearance/stuck)."""
    progress = prev_distance - current_distance
    collision_penalty = -1.0 if collided else 0.0
    goal_bonus = 1.0 if reached_goal else 0.0
    return RewardComponents(
        progress=progress,
        collision_penalty=collision_penalty,
        goal_bonus=goal_bonus,
        action_mag_penalty=0.0,
        jerk_penalty=0.0,
        proximity_penalty=0.0,
        clearance_reward=0.0,
        stuck_penalty=0.0,
    )


# ---------------------------------------------------------------------------
# Time-penalty variant (small negative reward per step)
# ---------------------------------------------------------------------------


TIME_PENALTY_PER_STEP = -0.001


@dataclass
class RewardComponentsWithTime(RewardComponents):
    """Extended reward components with optional time penalty."""

    time_penalty: float = 0.0

    def total(self) -> float:
        return (
            super().total()
            + self.time_penalty
        )


def compute_reward_time_penalty(
    *,
    prev_distance: float,
    current_distance: float,
    collided: bool,
    reached_goal: bool,
    action_norm: float,
    jerk_norm: float,
    min_lidar_normalized: Optional[float] = None,
    linear_velocity: Optional[float] = None,
    angular_velocity: Optional[float] = None,
    progress_over_last_N: Optional[float] = None,
    lidar_max_range: float = 10.0,
    time_penalty: float = TIME_PENALTY_PER_STEP,
) -> RewardComponentsWithTime:
    """Standard shaped reward plus a small per-step time penalty."""
    base = compute_reward(
        prev_distance=prev_distance,
        current_distance=current_distance,
        collided=collided,
        reached_goal=reached_goal,
        action_norm=action_norm,
        jerk_norm=jerk_norm,
        min_lidar_normalized=min_lidar_normalized,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        progress_over_last_N=progress_over_last_N,
        lidar_max_range=lidar_max_range,
    )
    return RewardComponentsWithTime(
        progress=base.progress,
        collision_penalty=base.collision_penalty,
        goal_bonus=base.goal_bonus,
        action_mag_penalty=base.action_mag_penalty,
        jerk_penalty=base.jerk_penalty,
        proximity_penalty=base.proximity_penalty,
        clearance_reward=base.clearance_reward,
        stuck_penalty=base.stuck_penalty,
        time_penalty=time_penalty,
    )


# ---------------------------------------------------------------------------
# Potential-based shaping (difference of potential)
# ---------------------------------------------------------------------------


def potential_distance(distance_to_goal: float, scale: float = 1.0) -> float:
    """Potential = -scale * distance (so decreasing distance increases potential)."""
    return -scale * distance_to_goal


def compute_reward_potential_shaping(
    *,
    prev_distance: float,
    current_distance: float,
    collided: bool,
    reached_goal: bool,
    action_norm: float = 0.0,
    jerk_norm: float = 0.0,
    min_lidar_normalized: Optional[float] = None,
    linear_velocity: Optional[float] = None,
    angular_velocity: Optional[float] = None,
    progress_over_last_N: Optional[float] = None,
    lidar_max_range: float = 10.0,
    potential_scale: float = 1.0,
) -> RewardComponents:
    """
    Reward = (potential(s') - potential(s)) * gamma_term + goal/collision.
    This preserves optimal policy when potential is bounded (we use distance).
    """
    progress = prev_distance - current_distance  # same as potential difference with scale 1
    shaped = potential_scale * progress
    collision_penalty = -1.0 if collided else 0.0
    goal_bonus = 1.0 if reached_goal else 0.0
    return RewardComponents(
        progress=shaped,
        collision_penalty=collision_penalty,
        goal_bonus=goal_bonus,
        action_mag_penalty=0.0,
        jerk_penalty=0.0,
        proximity_penalty=0.0,
        clearance_reward=0.0,
        stuck_penalty=0.0,
    )


# ---------------------------------------------------------------------------
# Aggressive clearance (encourage staying away from walls)
# ---------------------------------------------------------------------------


CLEARANCE_AGGRESSIVE_SAFE = 0.25
CLEARANCE_AGGRESSIVE_SCALE = 0.05


def compute_reward_aggressive_clearance(
    *,
    prev_distance: float,
    current_distance: float,
    collided: bool,
    reached_goal: bool,
    action_norm: float,
    jerk_norm: float,
    min_lidar_normalized: Optional[float] = None,
    linear_velocity: Optional[float] = None,
    angular_velocity: Optional[float] = None,
    progress_over_last_N: Optional[float] = None,
    lidar_max_range: float = 10.0,
) -> RewardComponents:
    """Standard reward with stronger clearance bonus (larger safe distance and scale)."""
    progress = prev_distance - current_distance
    collision_penalty = -1.0 if collided else 0.0
    goal_bonus = 1.0 if reached_goal else 0.0
    action_mag_penalty = -0.01 * action_norm
    jerk_penalty = -0.01 * jerk_norm

    proximity_penalty = 0.0
    if min_lidar_normalized is not None and min_lidar_normalized < PROXIMITY_SAFE:
        proximity_penalty = -PROXIMITY_SCALE * (PROXIMITY_SAFE - min_lidar_normalized)

    clearance_reward = 0.0
    if min_lidar_normalized is not None and min_lidar_normalized > CLEARANCE_AGGRESSIVE_SAFE:
        clearance_reward = CLEARANCE_AGGRESSIVE_SCALE * (
            min_lidar_normalized - CLEARANCE_AGGRESSIVE_SAFE
        )

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


# ---------------------------------------------------------------------------
# Reward function registry and selector
# ---------------------------------------------------------------------------


REWARD_FN_TYPE = Callable[..., RewardComponents]

_REWARD_REGISTRY: Dict[str, REWARD_FN_TYPE] = {
    "default": compute_reward,
    "sparse": compute_reward_sparse,
    "dense_progress": compute_reward_dense_progress,
    "potential": compute_reward_potential_shaping,
    "aggressive_clearance": compute_reward_aggressive_clearance,
}


def get_reward_function(name: str) -> REWARD_FN_TYPE:
    """Return the reward function for the given name."""
    if name not in _REWARD_REGISTRY:
        raise KeyError(f"Unknown reward: {name}. Available: {list(_REWARD_REGISTRY.keys())}")
    return _REWARD_REGISTRY[name]


def list_reward_functions() -> List[str]:
    """Return list of registered reward function names."""
    return list(_REWARD_REGISTRY.keys())


def register_reward_function(name: str, fn: REWARD_FN_TYPE) -> None:
    """Register a custom reward function."""
    _REWARD_REGISTRY[name] = fn


# ---------------------------------------------------------------------------
# Reward analysis and logging
# ---------------------------------------------------------------------------


@dataclass
class RewardEpisodeSummary:
    """Per-episode reward summary for analysis."""

    episode_id: int
    total_reward: float
    steps: int
    reached_goal: bool
    collided: bool
    timeout: bool
    component_means: Dict[str, float] = field(default_factory=dict)
    component_sums: Dict[str, float] = field(default_factory=dict)


def summarize_episode_rewards(
    episode_id: int,
    step_rewards: List[float],
    step_components: List[Dict[str, float]],
    reached_goal: bool,
    collided: bool,
    timeout: bool,
) -> RewardEpisodeSummary:
    """Build a summary from a list of per-step rewards and optional component dicts."""
    total = sum(step_rewards)
    component_means: Dict[str, float] = {}
    component_sums: Dict[str, float] = {}
    if step_components:
        keys = step_components[0].keys() if step_components else []
        for k in keys:
            vals = [c[k] for c in step_components if k in c]
            if vals:
                component_means[k] = float(np.mean(vals))
                component_sums[k] = float(np.sum(vals))
    return RewardEpisodeSummary(
        episode_id=episode_id,
        total_reward=total,
        steps=len(step_rewards),
        reached_goal=reached_goal,
        collided=collided,
        timeout=timeout,
        component_means=component_means,
        component_sums=component_sums,
    )


def compare_reward_components(
    summaries: List[RewardEpisodeSummary],
    component_key: str,
) -> Dict[str, float]:
    """Compute mean/std/min/max for a single component across episodes."""
    vals = [
        s.component_means.get(component_key, 0.0)
        for s in summaries
        if component_key in s.component_means
    ]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "count": len(vals),
    }


def reward_components_from_info(info: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Extract reward_components dict from step info if present."""
    return info.get("reward_components")


def aggregate_reward_stats(
    summaries: List[RewardEpisodeSummary],
) -> Dict[str, Any]:
    """Aggregate stats over many episode summaries."""
    if not summaries:
        return {}
    total_rewards = [s.total_reward for s in summaries]
    steps_list = [s.steps for s in summaries]
    success_count = sum(1 for s in summaries if s.reached_goal)
    collision_count = sum(1 for s in summaries if s.collided)
    timeout_count = sum(1 for s in summaries if s.timeout)
    n = len(summaries)
    return {
        "num_episodes": n,
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "success_rate": success_count / n,
        "collision_rate": collision_count / n,
        "timeout_rate": timeout_count / n,
    }
