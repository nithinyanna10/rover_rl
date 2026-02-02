from __future__ import annotations

import argparse
import os
import random
import time
from typing import Any, Dict

import pygame
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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
    cfg = load_yaml(sim_cfg_path)
    sim_cfg = cfg["sim"]
    rover_cfg = cfg["rover"]
    lidar_cfg = cfg["lidar"]
    domain_rand_cfg = cfg["domain_randomization"]
    render_cfg = cfg["render"]
    maps_cfg = cfg.get("maps", {})

    world_width = float(sim_cfg["world_width"])
    world_height = float(sim_cfg["world_height"])
    goal_radius = float(sim_cfg["goal_radius"])
    default_map = maps_cfg.get("default_fixed_map", "corridor_map")

    world = World.from_map_file(
        width=world_width,
        height=world_height,
        path=os.path.join("rover_sim", "maps", f"{default_map}.json"),
    )

    env_cfg = EnvConfig(
        dt=float(sim_cfg["dt"]),
        max_steps=int(sim_cfg["max_steps"]),
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
        goal_radius=goal_radius,
    )

    env = RoverEnv(world=world, config=env_cfg, domain_randomization_cfg=domain_rand_cfg, seed=cfg.get("seed", 0))

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
            num_rays=int(lidar_cfg["num_rays"]),
            fov_deg=float(lidar_cfg["fov_deg"]),
            max_range=float(lidar_cfg["max_range"]),
            noise_std=float(lidar_cfg["noise_std"]),
        ),
        rng=random.Random(cfg.get("seed", 0)),
    )

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
    parser.add_argument(
        "--vecnormalize-path",
        type=str,
        default=None,
        help="Path to vecnormalize.pkl from training (required if model was trained with VecNormalize).",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    env_cfg_root = cfg["env"]
    sim_cfg_path = env_cfg_root["config_path"]
    sim_full_cfg = load_yaml(sim_cfg_path)
    sim_cfg = sim_full_cfg["sim"]
    shield_cfg_dict = sim_full_cfg["shield"]
    lidar_cfg_dict = sim_full_cfg["lidar"]

    env, renderer, lidar_sensor = make_env_and_renderer(sim_cfg_path)
    if args.vecnormalize_path and os.path.isfile(args.vecnormalize_path):
        venv = DummyVecEnv([lambda e=env: e])  # wrap same env so render/rover/world stay in sync
        venv = VecNormalize.load(args.vecnormalize_path, venv)
        venv.training = False
        venv.norm_reward = False
        step_env = venv
    else:
        step_env = None

    model = PPO.load(args.model_path)

    shield_cfg = ShieldConfig(
        min_distance=float(shield_cfg_dict["min_distance"]),
        turn_angular_speed=float(shield_cfg_dict["turn_angular_speed"]),
    )
    shield = LidarSafetyShield(shield_cfg)

    telemetry_logger = None
    if args.telemetry_path:
        os.makedirs(os.path.dirname(args.telemetry_path), exist_ok=True)
        telemetry_logger = TelemetryLogger(args.telemetry_path)

    active = step_env if step_env is not None else env
    if step_env is not None:
        obs = active.reset()
        infos = getattr(active, "reset_infos", None) or []
        obs, info = obs[0], (infos[0] if infos else {})
    else:
        obs, info = active.reset()
    done = False
    truncated = False

    # Debug: print starting state
    state = env.rover.get_state()
    print(f"Starting episode:")
    print(f"  Rover at ({state.x:.2f}, {state.y:.2f}), yaw={state.yaw:.2f}")
    print(f"  Goal at ({env.world.goal[0]:.2f}, {env.world.goal[1]:.2f})")
    print(f"  Distance to goal: {env.world.distance_to_goal(state.x, state.y):.2f}m")

    # Draw initial frame so the window appears immediately
    lidar_ranges_init = lidar_sensor.scan(
        world=env.world,
        pose=(state.x, state.y, state.yaw),
        noise_scale=1.0,
    )
    renderer.draw(
        rover_state=state,
        lidar_ranges=lidar_ranges_init,
        lidar_fov_deg=lidar_cfg_dict["fov_deg"],
        dt=env.cfg.dt,
        fps=0.0,
    )
    renderer.tick(sim_cfg["fps"])

    step_idx = 0
    while not (done or truncated):
        # Process events so the window stays responsive and visible (important on macOS)
        pygame.event.pump()

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
            action, lidar_ranges, lidar_fov_deg=lidar_cfg_dict["fov_deg"]
        )

        if step_env is not None:
            shielded_action = np.expand_dims(shielded_action, 0)
            step_obs, rewards, dones, infos = active.step(shielded_action)
            obs, reward, done, info = step_obs[0], float(rewards[0]), bool(dones[0]), (infos[0] if infos else {})
            truncated = done
        else:
            obs, reward, done, truncated, info = active.step(shielded_action)

        # Debug: print termination reason
        if done or truncated:
            if info.get("goal_reached", False):
                print(f"✓ Goal reached in {step_idx + 1} steps!")
            elif info.get("collision", False):
                print(f"✗ Collision at step {step_idx + 1}")
            elif info.get("timeout", False):
                print(f"⏱ Timeout at step {step_idx + 1}")

        # Render
        fps = renderer.tick(sim_cfg["fps"])
        renderer.draw(
            rover_state=env.rover.get_state(),
            lidar_ranges=lidar_ranges,
            lidar_fov_deg=lidar_cfg_dict["fov_deg"],
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

    # Keep window visible for a moment so the user can see the final state
    print("Episode finished. Closing in 3 seconds...")
    for _ in range(int(sim_cfg["fps"] * 3)):
        pygame.event.pump()
        renderer.tick(sim_cfg["fps"])

    renderer.close()
    if telemetry_logger is not None:
        telemetry_logger.close()


if __name__ == "__main__":
    main()

