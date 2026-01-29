from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
import math
import os
import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .world import World
from .rover import Rover
from .sensors import LidarConfig, LidarSensor
from rl.reward import compute_reward


@dataclass
class EnvConfig:
    dt: float
    max_steps: int
    rover_radius: float
    max_linear_speed: float
    max_angular_speed: float
    linear_accel_limit: float
    angular_accel_limit: float
    lidar_num_rays: int
    lidar_fov_deg: float
    lidar_max_range: float
    lidar_noise_std: float
    front_sector_deg: float
    goal_radius: float


class RoverEnv(gym.Env):
    """Gymnasium-compatible environment for RL rover navigation."""

    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    def __init__(
        self,
        world: World,
        config: EnvConfig,
        domain_randomization_cfg: Dict[str, Any],
        seed: int = 0,
        curriculum_scale: float = 1.0,
        random_obstacles_cfg: Optional[Dict[str, Any]] = None,
        fixed_maps_dir: Optional[str] = None,
        fixed_map_names: Optional[List[str]] = None,
        fixed_map_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.world = world
        self.cfg = config
        self.random_obstacles_cfg = random_obstacles_cfg
        self.domain_randomization_cfg = domain_randomization_cfg
        self.curriculum_scale = curriculum_scale
        self.fixed_maps_dir = fixed_maps_dir
        self.fixed_map_names = fixed_map_names or []
        self.fixed_map_prob = fixed_map_prob

        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.rng = random.Random(int(seed))

        self.rover = Rover(
            radius=self.cfg.rover_radius,
            max_linear_speed=self.cfg.max_linear_speed,
            max_angular_speed=self.cfg.max_angular_speed,
            linear_accel_limit=self.cfg.linear_accel_limit,
            angular_accel_limit=self.cfg.angular_accel_limit,
            rng=self.rng,
        )
        self.lidar = LidarSensor(
            LidarConfig(
                num_rays=self.cfg.lidar_num_rays,
                fov_deg=self.cfg.lidar_fov_deg,
                max_range=self.cfg.lidar_max_range,
                noise_std=self.cfg.lidar_noise_std,
            ),
            rng=self.rng,
        )

        self._step_count = 0
        self._last_distance_to_goal = 0.0
        self._last_action = np.zeros(2, dtype=np.float32)

        # Observation: lidar + goal vector (2) + (v, w) (2)
        obs_dim = self.cfg.lidar_num_rays + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: [linear_vel, angular_vel]
        self.action_space = spaces.Box(
            low=np.array(
                [-self.cfg.max_linear_speed, -self.cfg.max_angular_speed],
                dtype=np.float32,
            ),
            high=np.array(
                [self.cfg.max_linear_speed, self.cfg.max_angular_speed],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            self.rng.seed(int(seed))

        self._step_count = 0

        # With fixed_map_prob, load a fixed complex map (maze, zigzag, etc.) so policy sees hard layouts
        use_fixed_map = (
            self.fixed_map_names
            and self.fixed_maps_dir
            and self.fixed_map_prob > 0.0
            and self.rng.random() < self.fixed_map_prob
        )
        if use_fixed_map:
            map_name = self.rng.choice(self.fixed_map_names)
            map_path = os.path.join(self.fixed_maps_dir, f"{map_name}.json")
            if os.path.isfile(map_path):
                loaded = World.from_map_file(self.world.width, self.world.height, map_path)
                self.world.clear_obstacles()
                for o in loaded.obstacles:
                    self.world.add_obstacle(o)
                self.world.goal = loaded.goal
        # Else: generate random obstacles for curriculum (training only)
        elif self.random_obstacles_cfg is not None and self.curriculum_scale > 0.0:
            ro = self.random_obstacles_cfg
            num_min = int(ro["num_min"])
            num_max = int(ro["num_max"])
            n = num_min + int(self.curriculum_scale * max(0, num_max - num_min))
            n = max(num_min, min(num_max, n))
            min_size = tuple(ro["min_size"])
            max_size = tuple(ro["max_size"])
            self.world.clear_obstacles()
            self.world.generate_random_obstacles(
                num_min=n,
                num_max=n,
                min_size=min_size,
                max_size=max_size,
                rng=self.rng,
                margin=1.0,
            )
            self.world.goal = (self.world.width - 2.0, self.world.height - 2.0)

        # Domain randomization values
        dr_cfg = self.domain_randomization_cfg
        friction_low, friction_high = dr_cfg["friction_scale_range"]
        lidar_noise_low, lidar_noise_high = dr_cfg["lidar_noise_scale_range"]
        slip_std_lin = dr_cfg["slip_std_linear"]
        slip_std_ang = dr_cfg["slip_std_angular"]
        lat_min, lat_max = dr_cfg["latency_steps_range"]
        self.friction_scale = self.rng.uniform(friction_low, friction_high)
        self.lidar_noise_scale = self.rng.uniform(lidar_noise_low, lidar_noise_high)
        self.slip_std_linear = slip_std_lin * self.curriculum_scale
        self.slip_std_angular = slip_std_ang * self.curriculum_scale
        self.actuation_latency_steps = self.rng.randint(lat_min, lat_max)
        self._action_buffer: List[np.ndarray] = []

        # Randomize obstacles if requested via curriculum_scale (handled outside).
        # Here we assume world already has correct obstacles for this episode.

        # Sample start pose not colliding.
        for _ in range(100):
            x = self.rng.uniform(
                self.cfg.rover_radius, self.world.width - self.cfg.rover_radius
            )
            y = self.rng.uniform(
                self.cfg.rover_radius, self.world.height - self.cfg.rover_radius
            )
            if not self.world.check_collision(x, y, self.cfg.rover_radius):
                break
        else:
            # Fallback: center
            x = self.world.width * 0.1
            y = self.world.height * 0.1

        yaw = self.rng.uniform(-math.pi, math.pi)

        self.rover.reset(
            x=x,
            y=y,
            yaw=yaw,
            friction_scale=self.friction_scale,
            slip_std_linear=self.slip_std_linear,
            slip_std_angular=self.slip_std_angular,
        )

        self._last_distance_to_goal = self.world.distance_to_goal(x, y)
        self._last_action = np.zeros(2, dtype=np.float32)

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1

        # Apply action with simple actuation latency
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._action_buffer.append(action)
        if len(self._action_buffer) <= self.actuation_latency_steps:
            effective_action = np.zeros_like(action)
        else:
            effective_action = self._action_buffer.pop(0)

        v_cmd = float(effective_action[0])
        w_cmd = float(effective_action[1])

        # Integrate rover dynamics
        self.rover.step(v_cmd, w_cmd, self.cfg.dt)
        state = self.rover.get_state()

        # Collision and termination checks
        collided = self.world.check_collision(
            state.x, state.y, self.cfg.rover_radius
        )
        distance_to_goal = self.world.distance_to_goal(state.x, state.y)
        reached_goal = distance_to_goal <= self.cfg.goal_radius
        timeout = self._step_count >= self.cfg.max_steps

        terminated = bool(collided or reached_goal)
        truncated = bool(timeout and not terminated)

        # Reward shaping via shared reward module
        action_norm = float(np.linalg.norm(effective_action, ord=2))
        jerk_norm = float(np.linalg.norm(effective_action - self._last_action, ord=2))
        reward_components = compute_reward(
            prev_distance=self._last_distance_to_goal,
            current_distance=distance_to_goal,
            collided=collided,
            reached_goal=reached_goal,
            action_norm=action_norm,
            jerk_norm=jerk_norm,
        )
        progress = reward_components.progress
        self._last_distance_to_goal = distance_to_goal
        self._last_action = effective_action.copy()

        reward = reward_components.total()

        obs = self._get_obs()

        info: Dict[str, Any] = {
            "distance_to_goal": distance_to_goal,
            "progress": progress,
            "collision": collided,
            "goal_reached": reached_goal,
            "timeout": timeout,
            "reward_components": reward_components.as_dict(),
        }

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        state = self.rover.get_state()
        lidar_ranges = self.lidar.scan(
            self.world,
            pose=(state.x, state.y, state.yaw),
            noise_scale=self.lidar_noise_scale,
        )
        lidar_array = np.asarray(lidar_ranges, dtype=np.float32) / self.cfg.lidar_max_range
        lidar_array = np.clip(lidar_array, 0.0, 1.0)

        # Goal vector in rover frame (dx, dy), normalized
        dx_body, dy_body = self.rover.goal_relative_vector(self.world.goal)
        goal_vec = np.array([dx_body, dy_body], dtype=np.float32)
        goal_norm = np.linalg.norm(goal_vec)
        if goal_norm > 1e-6:
            goal_vec /= goal_norm

        vel = np.array([state.v, state.w], dtype=np.float32)

        obs = np.concatenate([lidar_array, goal_vec, vel], axis=0)
        return obs

    # ------------------------------------------------------------------
    # Curriculum control
    # ------------------------------------------------------------------
    def set_curriculum_scale(self, scale: float) -> None:
        """Adjust difficulty scaling (used for domain randomization and obstacle count)."""
        self.curriculum_scale = float(scale)

