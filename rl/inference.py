from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import yaml

from rover_sim.world import World
from rover_sim.env import RoverEnv, EnvConfig
from rover_sim.render import PygameRenderer
from rover_sim.sensors import LidarConfig, LidarSensor
from rl.shield import LidarSafetyShield, ShieldConfig
from telemetry.logger import TelemetryLogger


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_and_renderer(sim_cfg_path: str) -> tuple[RoverEnv, PygameRenderer, LidarSensor]:
    sim_cfg = load_yaml(sim_cfg_path)["sim"]

    world_width = float(sim_cfg["world_width"])
    world_height = float(sim_cfg["world_height"])
    goal_radius = float(sim_cfg["goal_radius"])

    world = World.from_map_file(
        width=world_width,
        height=world_height,
        path=os.path.join("rover_sim", "maps", f"{sim_cfg['maps']['default_fixed_map']}.json"),
    )

    env_cfg = EnvConfig(
        dt=float(sim_cfg["dt"]),
        max_steps=int(sim_cfg["max_steps"]),
        rover_radius=float(sim_cfg["rover_radius"]),
        max_linear_speed=float(sim_cfg["rover"]["max_linear_speed"]),
        max_angular_speed=float(sim_cfg["rover"]["max_angular_speed"]),
        linear_accel_limit=float(sim_cfg["rover"]["linear_accel_limit"]),
        angular_accel_limit=float(sim_cfg["rover"]["angular_accel_limit"]),
        lidar_num_rays=int(sim_cfg["lidar"]["num_rays"]),
        lidar_fov_deg=float(sim_cfg["lidar"]["fov_deg"]),
        lidar_max_range=float(sim_cfg["lidar"]["max_range"]),
        lidar_noise_std=float(sim_cfg["lidar"]["noise_std"]),
        front_sector_deg=float(sim_cfg["lidar"]["front_sector_deg"]),
        goal_radius=goal_radius,
    )
    domain_rand_cfg = sim_cfg["domain_randomization"]

    env = RoverEnv(world=world, config=env_cfg, domain_randomization_cfg=domain_rand_cfg, seed=sim_cfg.get("seed", 0))

    render_cfg = sim_cfg["render"]
    renderer = PygameRenderer(
        world=world,
        window_width=int(render_cfg["window_width"]),
        window_height=int(render_cfg["window_height"]),
        show_lidar=bool(render_cfg.get("show_lidar", True)),
        show_trail=bool(render_cfg.get("show_trail", True)),
        trail_max_length=int(render_cfg.get("trail_max_length", 500)),
    )

    lidar_sensor = LidarSensor(
        LidarConfig(
            num_rays=int(sim_cfg["lidar"]["num_rays"]),
            fov_deg=float(sim_cfg["lidar"]["fov_deg"]),
            max_range=float(sim_cfg["lidar"]["max_range"]),
            noise_std=float(sim_cfg["lidar"]["noise_std"]),
        ),
        rng=np.random.RandomState(sim_cfg.get("seed", 0)),
    )  # type: ignore[arg-type]

    return env, renderer, lidar_sensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained policy in sim.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval.yaml",
        help="Path to eval config (used to resolve sim config).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained PPO checkpoint (.zip).",
    )
    parser.add_argument(
        "--telemetry-path",
        type=str,
        default=None,
        help="Optional JSONL file to log telemetry.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    env_cfg_root = cfg["env"]
    sim_cfg_path = env_cfg_root["config_path"]
    sim_cfg = load_yaml(sim_cfg_path)["sim"]

    env, renderer, lidar_sensor = make_env_and_renderer(sim_cfg_path)

    model = PPO.load(args.model_path)

    shield_cfg = ShieldConfig(
        min_distance=float(sim_cfg["shield"]["min_distance"]),
        turn_angular_speed=float(sim_cfg["shield"]["turn_angular_speed"]),
    )
    shield = LidarSafetyShield(shield_cfg)

    telemetry_logger = None
    if args.telemetry_path:
        os.makedirs(os.path.dirname(args.telemetry_path), exist_ok=True)
        telemetry_logger = TelemetryLogger(args.telemetry_path)

    obs, _ = env.reset()
    done = False
    truncated = False

    clock = gym.utils.seeding.np_random(sim_cfg.get("seed", 0))[0]
    step_idx = 0
    while not (done or truncated):
        step_start = time.time()

        action, _ = model.predict(obs, deterministic=True)

        # Compute LiDAR for shield (can reuse env's sensor via env._get_obs, but here we simply
        # trust that obs contains normalized ranges)
        state = env.rover.get_state()
        lidar_ranges = lidar_sensor.scan(
            world=env.world,
            pose=(state.x, state.y, state.yaw),
            noise_scale=1.0,
        )
        shielded_action, overridden = shield.filter_action(
            action, lidar_ranges, lidar_fov_deg=sim_cfg["lidar"]["fov_deg"]
        )

        obs, reward, done, truncated, info = env.step(shielded_action)

        # Render
        fps = renderer.tick(sim_cfg["fps"])
        renderer.draw(
            rover_state=env.rover.get_state(),
            lidar_ranges=lidar_ranges,
            lidar_fov_deg=sim_cfg["lidar"]["fov_deg"],
            dt=env.cfg.dt,
            fps=fps,
        )

        step_time = time.time() - step_start

        if telemetry_logger is not None:
            telemetry_logger.log_step(
                {
                    "step": step_idx,
                    "pose": env.rover.to_dict(),
                    "action": shielded_action.tolist(),
                    "action_overridden": overridden,
                    "reward": reward,
                    "reward_components": info.get("reward_components", {}),
                    "lidar": {
                        "ranges": lidar_ranges,
                        "min_range": float(np.min(lidar_ranges)),
                        "max_range": float(np.max(lidar_ranges)),
                    },
                    "goal": {"x": env.world.goal[0], "y": env.world.goal[1]},
                    "episode_done": done or truncated,
                    "fps": fps,
                    "step_time": step_time,
                }
            )

        step_idx += 1

    renderer.close()
    if telemetry_logger is not None:
        telemetry_logger.close()


if __name__ == "__main__":
    main()

