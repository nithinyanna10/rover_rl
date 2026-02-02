from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

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


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_eval_env(cfg: Dict[str, Any], map_name: str, seed: int) -> gym.Env:
    sim_cfg = cfg["sim"]
    rover_cfg = cfg["rover"]
    lidar_cfg = cfg["lidar"]
    domain_rand_cfg = cfg["domain_randomization"]
    maps_cfg = cfg.get("maps", {})

    world_width = float(sim_cfg["world_width"])
    world_height = float(sim_cfg["world_height"])
    goal_radius = float(sim_cfg["goal_radius"])

    maps_dir = os.path.join(os.path.dirname(__file__), "..", "rover_sim", "maps")
    map_file = os.path.join(maps_dir, f"{map_name}.json")
    world = World.from_map_file(width=world_width, height=world_height, path=map_file)

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

    # Disable domain randomization for evaluation (use minimal/noise for deterministic results)
    eval_domain_rand = {
        "friction_scale_range": [1.0, 1.0],  # No friction variation
        "slip_std_linear": 0.0,
        "slip_std_angular": 0.0,
        "lidar_noise_scale_range": [1.0, 1.0],  # Fixed noise
        "latency_steps_range": [0, 0],  # No latency
    }

    env = RoverEnv(world=world, config=env_cfg, domain_randomization_cfg=eval_domain_rand, seed=seed, curriculum_scale=0.0)
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
    parser.add_argument(
        "--vecnormalize-path",
        type=str,
        default=None,
        help="Path to vecnormalize.pkl from training (required if model was trained with VecNormalize).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show pygame visualization (renders first episode of each map).",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    eval_cfg = cfg["eval"]
    env_cfg_root = cfg["env"]
    sim_cfg_path = env_cfg_root["config_path"]
    sim_full_cfg = load_yaml(sim_cfg_path)

    seed = int(cfg.get("seed", 0))
    episodes_per_map = int(eval_cfg.get("episodes_per_map", 20))
    map_names: List[str] = list(eval_cfg["maps"])

    model = PPO.load(args.model_path)

    # Setup renderer if requested
    renderer = None
    lidar_sensor = None
    if args.render:
        sim_cfg = sim_full_cfg["sim"]
        render_cfg = sim_full_cfg["render"]
        lidar_cfg = sim_full_cfg["lidar"]
        import random
        lidar_sensor = LidarSensor(
            LidarConfig(
                num_rays=int(lidar_cfg["num_rays"]),
                fov_deg=float(lidar_cfg["fov_deg"]),
                max_range=float(lidar_cfg["max_range"]),
                noise_std=float(lidar_cfg["noise_std"]),
            ),
            rng=random.Random(seed),
        )

    all_episode_rewards: List[float] = []
    results_by_map: Dict[str, Dict[str, Any]] = {}

    for map_name in map_names:
        print(f"Evaluating on map '{map_name}'...")
        env = make_eval_env(sim_full_cfg, map_name, seed=seed)
        if args.vecnormalize_path and os.path.isfile(args.vecnormalize_path):
            venv = DummyVecEnv([lambda m=map_name: make_eval_env(sim_full_cfg, m, seed=seed)])
            venv = VecNormalize.load(args.vecnormalize_path, venv)
            venv.training = False
            venv.norm_reward = False  # eval rewards unscaled
            step_env = venv
            env_for_render = venv.envs[0]
        else:
            step_env = None
            env_for_render = env

        # Setup renderer for this map if rendering
        if args.render and renderer is None:
            sim_cfg = sim_full_cfg["sim"]
            render_cfg = sim_full_cfg["render"]
            renderer = PygameRenderer(
                world=env_for_render.world,
                window_width=int(render_cfg["window_width"]),
                window_height=int(render_cfg["window_height"]),
                show_lidar=bool(render_cfg.get("show_lidar", True)),
                show_trail=bool(render_cfg.get("show_trail", True)),
                trail_max_length=int(render_cfg.get("trail_max_length", 500)),
            )

        successes = 0
        collisions = 0
        time_to_goal: List[int] = []
        path_lengths: List[float] = []
        ep_rewards: List[float] = []

        use_venv = step_env is not None
        for ep in range(episodes_per_map):
            active = step_env if use_venv else env
            if use_venv:
                # SB3 VecEnv.reset() returns only obs; info is in vec_env.reset_infos
                obs = active.reset()
                infos = getattr(active, "reset_infos", None) or []
                obs, info = obs[0], (infos[0] if infos else {})
            else:
                obs, info = active.reset(seed=seed + ep)
            done = False
            truncated = False
            ep_reward = 0.0
            last_pos = None
            path_len = 0.0
            step_idx = 0
            
            # Debug: print starting position for first episode
            if ep == 0:
                start_state = env_for_render.rover.get_state()
                print(f"  Episode {ep+1}: Start at ({start_state.x:.2f}, {start_state.y:.2f}), goal at ({env_for_render.world.goal[0]:.2f}, {env_for_render.world.goal[1]:.2f})")

            # Only render first episode of each map
            render_this_episode = args.render and ep == 0
            if render_this_episode and renderer is not None:
                renderer.world = env_for_render.world  # Update world reference
                state = env_for_render.rover.get_state()
                lidar_ranges_init = lidar_sensor.scan(
                    world=env_for_render.world,
                    pose=(state.x, state.y, state.yaw),
                    noise_scale=1.0,
                )
                renderer.draw(
                    rover_state=state,
                    lidar_ranges=lidar_ranges_init,
                    lidar_fov_deg=sim_full_cfg["lidar"]["fov_deg"],
                    dt=sim_full_cfg["sim"]["dt"],
                    fps=0.0,
                )
                renderer.tick(sim_full_cfg["sim"]["fps"])

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                if use_venv:
                    action = np.expand_dims(action, 0)
                # Debug: print first action for first episode
                if ep == 0 and step_idx == 0:
                    print(f"    First action: v={float(action.flat[0]):.3f}, w={float(action.flat[1]):.3f}")
                if use_venv:
                    step_obs, rewards, dones, infos = active.step(action)
                    obs, reward, done, info = step_obs[0], float(rewards[0]), bool(dones[0]), infos[0] if infos else {}
                    truncated = done
                else:
                    obs, reward, done, truncated, info = active.step(action)
                ep_reward += float(reward)
                step_idx += 1

                # Render if enabled for this episode
                if render_this_episode and renderer is not None:
                    pygame.event.pump()
                    state = env_for_render.rover.get_state()
                    lidar_ranges = lidar_sensor.scan(
                        world=env_for_render.world,
                        pose=(state.x, state.y, state.yaw),
                        noise_scale=1.0,
                    )
                    fps = renderer.tick(sim_full_cfg["sim"]["fps"])
                    renderer.draw(
                        rover_state=state,
                        lidar_ranges=lidar_ranges,
                        lidar_fov_deg=sim_full_cfg["lidar"]["fov_deg"],
                        dt=sim_full_cfg["sim"]["dt"],
                        fps=fps,
                    )

                # Approximate path length by distance between successive positions
                state = env_for_render.rover.get_state()
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
                        if ep == 0:
                            print(f"    ✓ Goal reached in {step_idx} steps")
                    if info.get("collision", False):
                        collisions += 1
                        if ep == 0:
                            print(f"    ✗ Collision at step {step_idx}")
                    if info.get("timeout", False):
                        if ep == 0:
                            print(f"    ⏱ Timeout at step {step_idx}")

            # Pause briefly after rendering an episode
            if render_this_episode and renderer is not None:
                for _ in range(int(sim_full_cfg["sim"]["fps"] * 2)):
                    pygame.event.pump()
                    renderer.tick(sim_full_cfg["sim"]["fps"])

            path_lengths.append(path_len)
            ep_rewards.append(ep_reward)
            all_episode_rewards.append(ep_reward)

        if step_env is not None:
            step_env.close()
        else:
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
    print()
    print("=" * 72)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 72)
    print(f"  Overall success rate:    {metrics['success_rate']:.1%}")
    print(f"  Overall collision rate:  {metrics['collision_rate']:.1%}")
    print(f"  Avg episode reward:      {metrics['avg_episode_reward']:.2f}" if metrics.get('avg_episode_reward') is not None else "  Avg episode reward:      N/A")
    print()
    print("Per-map results:")
    print("-" * 72)
    print(f"  {'Map':<22} {'Success':>8} {'Collision':>10} {'Avg time':>10} {'Avg reward':>10}")
    print("-" * 72)
    for map_name, r in metrics["per_map"].items():
        sr = r["success_rate"]
        cr = r["collision_rate"]
        t2g = r.get("avg_time_to_goal")
        t2g_str = f"{t2g:.0f}" if t2g is not None else "N/A"
        rew = r.get("avg_episode_reward")
        rew_str = f"{rew:.2f}" if rew is not None else "N/A"
        print(f"  {map_name:<22} {sr:>7.1%} {cr:>9.1%} {t2g_str:>10} {rew_str:>10}")
    print("-" * 72)
    print()

    if renderer is not None:
        renderer.close()


if __name__ == "__main__":
    main()

