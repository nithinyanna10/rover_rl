from __future__ import annotations

from typing import Any, Dict

import yaml

from rover_sim.env import RoverEnv, EnvConfig
from rover_sim.world import World


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_env_one_episode_smoke() -> None:
    cfg = load_yaml("configs/sim.yaml")
    sim_cfg = cfg["sim"]
    rover_cfg = cfg["rover"]
    lidar_cfg = cfg["lidar"]
    domain_rand_cfg = cfg["domain_randomization"]

    world = World(width=sim_cfg["world_width"], height=sim_cfg["world_height"], obstacles=[], goal=(5.0, 5.0))

    env_cfg = EnvConfig(
        dt=float(sim_cfg["dt"]),
        max_steps=50,
        rover_radius=float(sim_cfg["rover_radius"]),
        max_linear_speed=float(rover_cfg["max_linear_speed"]),
        max_angular_speed=float(rover_cfg["max_angular_speed"]),
        linear_accel_limit=float(rover_cfg["linear_accel_limit"]),
        angular_accel_limit=float(rover_cfg["angular_accel_limit"]),
        lidar_num_rays=int(lidar_cfg["num_rays"]),
        lidar_fov_deg=float(lidar_cfg["fov_deg"]),
        lidar_max_range=float(lidar_cfg["max_range"]),
        lidar_noise_std=float(lidar_cfg["noise_std"]),
        front_sector_deg=float(lidar_cfg["front_sector_deg"]),
        goal_radius=float(sim_cfg["goal_radius"]),
    )

    env = RoverEnv(world=world, config=env_cfg, domain_randomization_cfg=domain_rand_cfg, seed=cfg.get("seed", 0))

    obs, _ = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]

    done = False
    truncated = False
    steps = 0
    while not (done or truncated) and steps < 50:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    assert steps > 0

