from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import yaml

from rover_sim.world import World
from rover_sim.env import RoverEnv, EnvConfig


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_eval_env(sim_cfg: Dict[str, Any], map_name: str, seed: int) -> gym.Env:
    world_width = float(sim_cfg["world_width"])
    world_height = float(sim_cfg["world_height"])
    goal_radius = float(sim_cfg["goal_radius"])

    maps_cfg = sim_cfg["maps"]
    maps_dir = os.path.join(os.path.dirname(__file__), "..", "rover_sim", "maps")
    map_file = os.path.join(maps_dir, f"{map_name}.json")
    world = World.from_map_file(width=world_width, height=world_height, path=map_file)

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

    env = RoverEnv(world=world, config=env_cfg, domain_randomization_cfg=domain_rand_cfg, seed=seed)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval.yaml",
        help="Path to evaluation YAML config.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained PPO .zip checkpoint.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    eval_cfg = cfg["eval"]
    env_cfg_root = cfg["env"]
    sim_cfg_path = env_cfg_root["config_path"]
    sim_cfg = load_yaml(sim_cfg_path)["sim"]

    seed = int(cfg.get("seed", 0))
    episodes_per_map = int(eval_cfg.get("episodes_per_map", 20))
    map_names: List[str] = list(eval_cfg["maps"])

    model = PPO.load(args.model_path)

    all_episode_rewards: List[float] = []
    results_by_map: Dict[str, Dict[str, Any]] = {}

    for map_name in map_names:
        print(f"Evaluating on map '{map_name}'...")
        env = make_eval_env(sim_cfg, map_name, seed=seed)

        successes = 0
        collisions = 0
        time_to_goal: List[int] = []
        path_lengths: List[float] = []
        ep_rewards: List[float] = []

        for ep in range(episodes_per_map):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            truncated = False
            ep_reward = 0.0
            last_pos = None
            path_len = 0.0
            step_idx = 0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += float(reward)
                step_idx += 1

                # Approximate path length by distance between successive positions
                state = env.rover.get_state()
                pos = (state.x, state.y)
                if last_pos is not None:
                    dx = pos[0] - last_pos[0]
                    dy = pos[1] - last_pos[1]
                    path_len += float(np.hypot(dx, dy))
                last_pos = pos

                if done or truncated:
                    if info.get("goal_reached", False):
                        successes += 1
                        time_to_goal.append(step_idx)
                    if info.get("collision", False):
                        collisions += 1

            path_lengths.append(path_len)
            ep_rewards.append(ep_reward)
            all_episode_rewards.append(ep_reward)

        env.close()

        results_by_map[map_name] = {
            "successes": successes,
            "collisions": collisions,
            "episodes": episodes_per_map,
            "success_rate": successes / max(1, episodes_per_map),
            "collision_rate": collisions / max(1, episodes_per_map),
            "avg_time_to_goal": float(np.mean(time_to_goal)) if time_to_goal else None,
            "avg_path_length": float(np.mean(path_lengths)) if path_lengths else None,
            "avg_episode_reward": float(np.mean(ep_rewards)) if ep_rewards else None,
        }

    total_episodes = len(all_episode_rewards)
    overall_successes = sum(r["successes"] for r in results_by_map.values())
    overall_collisions = sum(r["collisions"] for r in results_by_map.values())

    metrics = {
        "success_rate": overall_successes / max(1, total_episodes),
        "collision_rate": overall_collisions / max(1, total_episodes),
        "avg_episode_reward": float(np.mean(all_episode_rewards)) if all_episode_rewards else None,
        "per_map": results_by_map,
    }

    eval_root = cfg["logging"].get("eval_root", "runs")
    os.makedirs(eval_root, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(eval_root, f"eval_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    main()

