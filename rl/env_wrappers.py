"""
Gymnasium environment wrappers for rover RL.

Provides observation preprocessing, frame stacking, action scaling,
reward scaling, and episode recording for training and debugging.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np


# ---------------------------------------------------------------------------
# Observation normalization wrapper
# ---------------------------------------------------------------------------


class ObsNormalizeWrapper(gym.ObservationWrapper):
    """
    Normalize observations to zero mean and unit variance using running statistics.
    Optionally clip normalized values.
    """

    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 1e-8,
        clip_obs: float = 10.0,
        update_rms: bool = True,
    ):
        super().__init__(env)
        self.epsilon = epsilon
        self.clip_obs = clip_obs
        self.update_rms = update_rms
        self._count = 0
        self._mean = np.zeros(env.observation_space.shape, dtype=np.float64)
        self._var = np.ones(env.observation_space.shape, dtype=np.float64)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self.update_rms:
            self._update_rms(obs)
        normalized = (obs - self._mean) / np.sqrt(self._var + self.epsilon)
        if self.clip_obs > 0:
            normalized = np.clip(normalized, -self.clip_obs, self.clip_obs)
        return normalized.astype(np.float32)

    def _update_rms(self, obs: np.ndarray) -> None:
        self._count += 1
        delta = obs - self._mean
        self._mean += delta / self._count
        delta2 = obs - self._mean
        self._var += (delta * delta2 - self._var) / self._count

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info


# ---------------------------------------------------------------------------
# Frame stack wrapper
# ---------------------------------------------------------------------------


class FrameStackWrapper(gym.ObservationWrapper):
    """Stack the last k observations along the last axis."""

    def __init__(self, env: gym.Env, n_stack: int = 2):
        super().__init__(env)
        self.n_stack = n_stack
        low = np.repeat(env.observation_space.low[np.newaxis, ...], n_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], n_stack, axis=0)
        self.observation_space = spaces.Box(
            low=low.min(axis=0),
            high=high.max(axis=0),
            shape=(env.observation_space.shape[0] * n_stack,),
            dtype=env.observation_space.dtype,
        )
        self._frames: List[np.ndarray] = []

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._frames.append(obs)
        if len(self._frames) > self.n_stack:
            self._frames.pop(0)
        while len(self._frames) < self.n_stack:
            self._frames.insert(0, obs)
        return np.concatenate(self._frames, axis=-1)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._frames = []
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.n_stack - 1):
            self._frames.append(obs)
        self._frames.append(obs)
        return self.observation(obs), info


# ---------------------------------------------------------------------------
# Action scaling (e.g. policy outputs [-1,1] -> env action range)
# ---------------------------------------------------------------------------


class ActionScaleWrapper(gym.ActionWrapper):
    """Scale and shift actions from [-1, 1] to env action space bounds."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = np.asarray(env.action_space.low, dtype=np.float32)
        high = np.asarray(env.action_space.high, dtype=np.float32)
        self._mid = (low + high) / 2.0
        self._half_range = (high - low) / 2.0

    def action(self, action: np.ndarray) -> np.ndarray:
        # Assume action in [-1, 1]
        scaled = self._mid + self._half_range * np.clip(action, -1.0, 1.0)
        return scaled.astype(np.float32)


# ---------------------------------------------------------------------------
# Reward scale wrapper
# ---------------------------------------------------------------------------


class RewardScaleWrapper(gym.RewardWrapper):
    """Scale rewards by a constant factor."""

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        return reward * self.scale


# ---------------------------------------------------------------------------
# Episode info recorder (for logging)
# ---------------------------------------------------------------------------


class EpisodeInfoRecorderWrapper(gym.Wrapper):
    """
    Record per-step info and episode summary (rewards, steps, termination reason).
    Access via get_episode_summary() after reset or at end of episode.
    """

    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self._episode_rewards: List[float] = []
        self._episode_infos: List[Dict[str, Any]] = []
        self._episode_summary: Optional[Dict[str, Any]] = None

    def step(
        self,
        action: Union[int, np.ndarray],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_rewards.append(float(reward))
        self._episode_infos.append(dict(info))
        if terminated or truncated:
            self._episode_summary = {
                "total_reward": sum(self._episode_rewards),
                "steps": len(self._episode_rewards),
                "terminated": terminated,
                "truncated": truncated,
                "reached_goal": info.get("goal_reached", False),
                "collision": info.get("collision", False),
                "timeout": info.get("timeout", False),
                "distance_to_goal_final": info.get("distance_to_goal"),
            }
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._episode_rewards = []
        self._episode_infos = []
        self._episode_summary = None
        return self.env.reset(seed=seed, options=options)

    def get_episode_summary(self) -> Optional[Dict[str, Any]]:
        """Return summary of the last completed episode, or None."""
        return self._episode_summary

    def get_episode_rewards(self) -> List[float]:
        """Return list of rewards for the current (or last) episode."""
        return list(self._episode_rewards)

    def get_episode_infos(self) -> List[Dict[str, Any]]:
        """Return list of info dicts for the current (or last) episode."""
        return list(self._episode_infos)


# ---------------------------------------------------------------------------
# Flatten dict observation (if env ever returns Dict observation)
# ---------------------------------------------------------------------------


class FlattenDictObsWrapper(gym.ObservationWrapper):
    """Flatten a Dict observation space into a single vector (key order fixed)."""

    def __init__(self, env: gym.Env, key_order: Optional[List[str]] = None):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError("FlattenDictObsWrapper expects Dict observation space")
        self.key_order = key_order or list(env.observation_space.spaces.keys())
        low_list = []
        high_list = []
        for k in self.key_order:
            sp = env.observation_space.spaces[k]
            low_list.append(np.ravel(sp.low))
            high_list.append(np.ravel(sp.high))
        low = np.concatenate(low_list)
        high = np.concatenate(high_list)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        parts = [np.ravel(obs[k]) for k in self.key_order]
        return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# Clip action wrapper (redundant if env already clips; for safety)
# ---------------------------------------------------------------------------


class ClipActionWrapper(gym.ActionWrapper):
    """Clip action to the environment's action space bounds."""

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(
            action,
            self.env.action_space.low,
            self.env.action_space.high,
        ).astype(self.env.action_space.dtype)


# ---------------------------------------------------------------------------
# Time limit wrapper (convenience)
# ---------------------------------------------------------------------------


def make_rover_env_with_wrappers(
    base_env: gym.Env,
    obs_normalize: bool = False,
    n_frame_stack: int = 0,
    reward_scale: float = 1.0,
    record_episode: bool = False,
    clip_action: bool = False,
) -> gym.Env:
    """
    Wrap a rover env with optional observation normalization, frame stack,
    reward scaling, episode recording, and action clipping.
    """
    env = base_env
    if clip_action:
        env = ClipActionWrapper(env)
    if reward_scale != 1.0:
        env = RewardScaleWrapper(env, scale=reward_scale)
    if n_frame_stack > 1:
        env = FrameStackWrapper(env, n_stack=n_frame_stack)
    if obs_normalize:
        env = ObsNormalizeWrapper(env)
    if record_episode:
        env = EpisodeInfoRecorderWrapper(env)
    return env
