"""
Training and evaluation metrics for rover RL.

Provides episode aggregators, per-map statistics, time series logging,
and report generation compatible with TensorBoard and JSON outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import time
import numpy as np


# ---------------------------------------------------------------------------
# Episode-level metrics
# ---------------------------------------------------------------------------


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    episode_id: int
    total_reward: float
    steps: int
    reached_goal: bool
    collided: bool
    timeout: bool
    distance_to_goal_final: Optional[float] = None
    path_length: Optional[float] = None
    map_name: Optional[str] = None
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "total_reward": float(self.total_reward),
            "steps": int(self.steps),
            "reached_goal": self.reached_goal,
            "collided": self.collided,
            "timeout": self.timeout,
            "distance_to_goal_final": self.distance_to_goal_final,
            "path_length": self.path_length,
            "map_name": self.map_name,
            "seed": self.seed,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Rolling aggregators
# ---------------------------------------------------------------------------


class RollingEpisodeAggregator:
    """Maintain rolling statistics over the last N episodes."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._rewards: List[float] = []
        self._steps: List[int] = []
        self._successes: List[bool] = []
        self._collisions: List[bool] = []
        self._timeouts: List[bool] = []

    def add(self, metrics: EpisodeMetrics) -> None:
        self._rewards.append(metrics.total_reward)
        self._steps.append(metrics.steps)
        self._successes.append(metrics.reached_goal)
        self._collisions.append(metrics.collided)
        self._timeouts.append(metrics.timeout)
        if len(self._rewards) > self.window_size:
            self._rewards.pop(0)
            self._steps.pop(0)
            self._successes.pop(0)
            self._collisions.pop(0)
            self._timeouts.pop(0)

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self._rewards)) if self._rewards else 0.0

    @property
    def mean_steps(self) -> float:
        return float(np.mean(self._steps)) if self._steps else 0.0

    @property
    def success_rate(self) -> float:
        return float(np.mean(self._successes)) if self._successes else 0.0

    @property
    def collision_rate(self) -> float:
        return float(np.mean(self._collisions)) if self._collisions else 0.0

    @property
    def timeout_rate(self) -> float:
        return float(np.mean(self._timeouts)) if self._timeouts else 0.0

    @property
    def count(self) -> int:
        return len(self._rewards)

    def summary(self) -> Dict[str, float]:
        return {
            "mean_reward": self.mean_reward,
            "mean_steps": self.mean_steps,
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "timeout_rate": self.timeout_rate,
            "count": float(self.count),
        }


# ---------------------------------------------------------------------------
# Per-map aggregator
# ---------------------------------------------------------------------------


class PerMapAggregator:
    """Aggregate metrics per map name."""

    def __init__(self):
        self._by_map: Dict[str, List[EpisodeMetrics]] = {}

    def add(self, metrics: EpisodeMetrics) -> None:
        key = metrics.map_name or "unknown"
        if key not in self._by_map:
            self._by_map[key] = []
        self._by_map[key].append(metrics)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for map_name, episodes in self._by_map.items():
            rewards = [e.total_reward for e in episodes]
            steps = [e.steps for e in episodes]
            successes = [e.reached_goal for e in episodes]
            collisions = [e.collided for e in episodes]
            n = len(episodes)
            out[map_name] = {
                "num_episodes": float(n),
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "std_reward": float(np.std(rewards)) if len(rewards) > 1 else 0.0,
                "mean_steps": float(np.mean(steps)) if steps else 0.0,
                "success_rate": float(np.mean(successes)) if successes else 0.0,
                "collision_rate": float(np.mean(collisions)) if collisions else 0.0,
            }
        return out

    def map_names(self) -> List[str]:
        return list(self._by_map.keys())


# ---------------------------------------------------------------------------
# Training time series (for TensorBoard-style logging)
# ---------------------------------------------------------------------------


class TrainingMetricsLogger:
    """Append-only logger for training step/epoch metrics."""

    def __init__(self, log_dir: Optional[str] = None, prefix: str = "train"):
        self.log_dir = log_dir
        self.prefix = prefix
        self._step: List[int] = []
        self._scalars: Dict[str, List[float]] = {}
        self._start_time = time.time()

    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        if step is None:
            step = len(self._step)
        if not self._step or step > self._step[-1]:
            self._step.append(step)
        if key not in self._scalars:
            self._scalars[key] = []
        self._scalars[key].append(float(value))

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self.log_scalar(k, v, step)

    def get_series(self, key: str) -> Tuple[List[int], List[float]]:
        steps = self._step[:]
        vals = self._scalars.get(key, [])
        if len(vals) < len(steps):
            vals = vals + [0.0] * (len(steps) - len(vals))
        return steps[: len(vals)], vals[: len(steps)]

    def save_json(self, path: str) -> None:
        data = {
            "steps": self._step,
            "scalars": self._scalars,
            "elapsed_seconds": time.time() - self._start_time,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._step = data.get("steps", [])
        self._scalars = data.get("scalars", {})


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """Full evaluation report with overall and per-map stats."""

    timestamp: str
    model_path: Optional[str] = None
    vecnormalize_path: Optional[str] = None
    overall_success_rate: float = 0.0
    overall_collision_rate: float = 0.0
    overall_timeout_rate: float = 0.0
    mean_episode_reward: float = 0.0
    mean_steps_to_goal: Optional[float] = None
    num_episodes: int = 0
    per_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model_path": self.model_path,
            "vecnormalize_path": self.vecnormalize_path,
            "overall_success_rate": self.overall_success_rate,
            "overall_collision_rate": self.overall_collision_rate,
            "overall_timeout_rate": self.overall_timeout_rate,
            "mean_episode_reward": self.mean_episode_reward,
            "mean_steps_to_goal": self.mean_steps_to_goal,
            "num_episodes": self.num_episodes,
            "per_map": self.per_map,
            "config_snapshot": self.config_snapshot,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EvalReport":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            timestamp=d.get("timestamp", ""),
            model_path=d.get("model_path"),
            vecnormalize_path=d.get("vecnormalize_path"),
            overall_success_rate=float(d.get("overall_success_rate", 0)),
            overall_collision_rate=float(d.get("overall_collision_rate", 0)),
            overall_timeout_rate=float(d.get("overall_timeout_rate", 0)),
            mean_episode_reward=float(d.get("mean_episode_reward", 0)),
            mean_steps_to_goal=d.get("mean_steps_to_goal"),
            num_episodes=int(d.get("num_episodes", 0)),
            per_map=d.get("per_map", {}),
            config_snapshot=d.get("config_snapshot", {}),
        )


def build_eval_report(
    episodes: List[EpisodeMetrics],
    model_path: Optional[str] = None,
    vecnormalize_path: Optional[str] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> EvalReport:
    """Build an EvalReport from a list of episode metrics."""
    if not episodes:
        return EvalReport(
            timestamp=time.strftime("%Y-%m-%d_%H-%M-%S"),
            model_path=model_path,
            vecnormalize_path=vecnormalize_path,
            config_snapshot=config_snapshot or {},
        )
    n = len(episodes)
    success_rate = sum(1 for e in episodes if e.reached_goal) / n
    collision_rate = sum(1 for e in episodes if e.collided) / n
    timeout_rate = sum(1 for e in episodes if e.timeout) / n
    mean_reward = float(np.mean([e.total_reward for e in episodes]))
    steps_to_goal = [e.steps for e in episodes if e.reached_goal]
    mean_steps = float(np.mean(steps_to_goal)) if steps_to_goal else None

    per_map_agg = PerMapAggregator()
    for e in episodes:
        per_map_agg.add(e)
    per_map = per_map_agg.summary()

    return EvalReport(
        timestamp=time.strftime("%Y-%m-%d_%H-%M-%S"),
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
        overall_success_rate=success_rate,
        overall_collision_rate=collision_rate,
        overall_timeout_rate=timeout_rate,
        mean_episode_reward=mean_reward,
        mean_steps_to_goal=mean_steps,
        num_episodes=n,
        per_map=per_map,
        config_snapshot=config_snapshot or {},
    )


# ---------------------------------------------------------------------------
# Text report formatting
# ---------------------------------------------------------------------------


def format_eval_report_table(report: EvalReport) -> str:
    """Produce a human-readable table string for the report."""
    lines = [
        "=" * 72,
        "EVALUATION REPORT",
        "=" * 72,
        f"  Timestamp:     {report.timestamp}",
        f"  Model:         {report.model_path or 'N/A'}",
        f"  Episodes:      {report.num_episodes}",
        "",
        "  Overall",
        "  - Success rate:   {:.1%}".format(report.overall_success_rate),
        "  - Collision rate: {:.1%}".format(report.overall_collision_rate),
        "  - Timeout rate:   {:.1%}".format(report.overall_timeout_rate),
        "  - Mean reward:    {:.3f}".format(report.mean_episode_reward),
        "  - Mean steps (success): {}".format(
            f"{report.mean_steps_to_goal:.0f}" if report.mean_steps_to_goal is not None else "N/A"
        ),
        "",
        "  Per-map",
        "-" * 72,
        f"  {'Map':<24} {'Success':>10} {'Collision':>10} {'Reward':>10} {'Steps':>8}",
        "-" * 72,
    ]
    for map_name, m in report.per_map.items():
        sr = m.get("success_rate", 0)
        cr = m.get("collision_rate", 0)
        rw = m.get("mean_reward", 0)
        st = m.get("mean_steps", 0)
        lines.append(f"  {map_name[:22]:<24} {sr:>9.1%} {cr:>9.1%} {rw:>10.2f} {st:>8.1f}")
    lines.append("-" * 72)
    return "\n".join(lines)
