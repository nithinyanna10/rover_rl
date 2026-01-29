from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import yaml

from rover_sim.world import World
from rover_sim.env import RoverEnv, EnvConfig


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_fn(
    sim_cfg_path: str,
    seed: int,
    rank: int,
    curriculum_phases: list[Dict[str, Any]],
) -> Callable[[], gym.Env]:
    """Factory to create RoverEnv instances for vectorized training."""

    def _init() -> gym.Env:
        sim_cfg = load_yaml(sim_cfg_path)["sim"]

        world_width = float(sim_cfg["world_width"])
        world_height = float(sim_cfg["world_height"])
        goal_radius = float(sim_cfg["goal_radius"])

        # Base world with no obstacles; training code will randomize.
        world = World(width=world_width, height=world_height, obstacles=[], goal=(18.0, 18.0))

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

        env = RoverEnv(
            world=world,
            config=env_cfg,
            domain_randomization_cfg=domain_rand_cfg,
            seed=seed + rank,
        )
        env = Monitor(env)
        return env

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO for rover navigation.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_ppo.yaml",
        help="Path to training YAML config.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total training timesteps (for smoke tests).",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg["train"]
    env_cfg_root = cfg["env"]

    sim_cfg_path = env_cfg_root["config_path"]
    curriculum_cfg = env_cfg_root.get("curriculum", {})
    curriculum_phases = curriculum_cfg.get("phases", [])

    n_envs = int(train_cfg.get("n_envs", 4))
    seed = int(cfg.get("seed", 0))

    # Vectorized environments
    env_fns = [
        make_env_fn(sim_cfg_path=sim_cfg_path, seed=seed, rank=i, curriculum_phases=curriculum_phases)
        for i in range(n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    run_root = train_cfg.get("tensorboard_log_dir", "runs")
    os.makedirs(run_root, exist_ok=True)

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(run_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    total_timesteps = args.total_timesteps or int(train_cfg.get("total_timesteps", 500000))

    model = PPO(
        policy=train_cfg.get("policy", "MlpPolicy"),
        env=vec_env,
        n_steps=int(train_cfg.get("n_steps", 1024)),
        batch_size=int(train_cfg.get("batch_size", 256)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
        clip_range=float(train_cfg.get("clip_range", 0.2)),
        ent_coef=float(train_cfg.get("ent_coef", 0.0)),
        vf_coef=float(train_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
        tensorboard_log=run_root,
        seed=seed,
        verbose=1,
    )

    # Callbacks: checkpointing and evaluation
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=int(cfg["logging"].get("save_freq", 50000)) // n_envs,
        save_path=checkpoints_dir,
        name_prefix="ppo_rover",
    )

    # Simple eval env (single)
    eval_env_fns = [make_env_fn(sim_cfg_path=sim_cfg_path, seed=seed + 1000, rank=0, curriculum_phases=curriculum_phases)]
    eval_vec_env = SubprocVecEnv(eval_env_fns)
    eval_vec_env = VecMonitor(eval_vec_env)

    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=checkpoints_dir,
        log_path=eval_dir,
        eval_freq=int(cfg["logging"].get("eval_freq", 50000)) // n_envs,
        n_eval_episodes=int(cfg["logging"].get("eval_episodes", 10)),
        deterministic=True,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model_path = os.path.join(checkpoints_dir, "final_model.zip")
    model.save(model_path)
    print(f"Training complete. Final model saved to {model_path}")


if __name__ == "__main__":
    main()

