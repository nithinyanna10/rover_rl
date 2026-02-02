from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
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
    fixed_map_names_override: list[str] | None = None,
) -> Callable[[], gym.Env]:
    """Factory to create RoverEnv instances for vectorized training."""

    def _init() -> gym.Env:
        cfg = load_yaml(sim_cfg_path)
        sim_cfg = cfg["sim"]
        rover_cfg = cfg["rover"]
        lidar_cfg = cfg["lidar"]
        domain_rand_cfg = cfg["domain_randomization"]

        world_width = float(sim_cfg["world_width"])
        world_height = float(sim_cfg["world_height"])
        goal_radius = float(sim_cfg["goal_radius"])
        maps_cfg = cfg.get("maps", {})
        random_obstacles_cfg = maps_cfg.get("random_obstacles")
        fixed_maps_for_training = fixed_map_names_override or maps_cfg.get("fixed_maps_for_training", [])
        fixed_map_prob = float(maps_cfg.get("fixed_map_prob", 0.25))
        easy_map_names = maps_cfg.get("easy_maps", [])
        min_easy_map_prob = float(maps_cfg.get("min_easy_map_prob", 0.0))
        maps_dir = os.path.join(os.path.dirname(__file__), "..", "rover_sim", "maps")

        # Base world; obstacles are generated on reset (random or from fixed map).
        world = World(width=world_width, height=world_height, obstacles=[], goal=(world_width - 2.0, world_height - 2.0))

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

        env = RoverEnv(
            world=world,
            config=env_cfg,
            domain_randomization_cfg=domain_rand_cfg,
            seed=seed + rank,
            curriculum_scale=0.3,
            random_obstacles_cfg=random_obstacles_cfg,
            fixed_maps_dir=maps_dir if fixed_maps_for_training else None,
            fixed_map_names=fixed_maps_for_training or None,
            fixed_map_prob=fixed_map_prob,
            easy_map_names=easy_map_names or None,
            min_easy_map_prob=min_easy_map_prob,
        )
        env = Monitor(env)
        return env

    return _init


class CurriculumCallback(BaseCallback):
    """Updates curriculum_scale, fixed_map_prob, and fixed_map_names on all envs based on current timestep."""

    def __init__(self, total_timesteps: int, phases: list, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.phases = phases

    def _on_step(self) -> bool:
        if self.n_calls == 0:
            return True
        if self.n_calls % 1000 != 0:
            return True
        t = self.num_timesteps
        scale = self.phases[0]["random_obstacles_scale"] if self.phases else 0.0
        fixed_map_prob = self.phases[0].get("fixed_map_prob", 0.0) if self.phases else 0.0
        fixed_map_names = self.phases[0].get("fixed_map_names") if self.phases else None
        for p in self.phases:
            if t >= p["max_steps"]:
                scale = p["random_obstacles_scale"]
                fixed_map_prob = p.get("fixed_map_prob", fixed_map_prob)
                if "fixed_map_names" in p:
                    fixed_map_names = p["fixed_map_names"]
        try:
            self.training_env.env_method("set_curriculum_scale", scale)
            self.training_env.env_method("set_fixed_map_prob", fixed_map_prob)
            if fixed_map_names is not None:
                self.training_env.env_method("set_fixed_map_names", fixed_map_names)
        except Exception:
            pass
        return True


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

    # VecNormalize before model (policy sees normalized obs)
    use_vec_normalize = train_cfg.get("use_vec_normalize", True)
    if use_vec_normalize:
        # Obs-only normalization (norm_reward=False avoids advantage swing / action saturation)
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            gamma=float(train_cfg.get("gamma", 0.99)),
        )

    run_root = train_cfg.get("tensorboard_log_dir", "runs")
    os.makedirs(run_root, exist_ok=True)

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(run_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    total_timesteps = args.total_timesteps or int(train_cfg.get("total_timesteps", 500000))

    policy_kwargs = train_cfg.get("policy_kwargs") or {}

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
        policy_kwargs=policy_kwargs,
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

    curriculum_callback = CurriculumCallback(
        total_timesteps=total_timesteps,
        phases=curriculum_phases,
    )

    if use_vec_normalize:
        eval_vec_env = VecNormalize(
            eval_vec_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )

    class EvalCallbackSyncNorm(EvalCallback):
        """Sync VecNormalize from training env to eval env before each evaluation."""

        def _on_step(self) -> bool:
            if use_vec_normalize and hasattr(self.model.get_env(), "obs_rms") and self.model.get_env().obs_rms is not None:
                if hasattr(self.eval_env, "obs_rms"):
                    self.eval_env.obs_rms = self.model.get_env().obs_rms
            return super()._on_step()

    eval_callback = EvalCallbackSyncNorm(
        eval_vec_env,
        best_model_save_path=checkpoints_dir,
        log_path=eval_dir,
        eval_freq=int(cfg["logging"].get("eval_freq", 50000)) // n_envs,
        n_eval_episodes=int(cfg["logging"].get("eval_episodes", 10)),
        deterministic=True,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_callback, checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model_path = os.path.join(checkpoints_dir, "final_model.zip")
    model.save(model_path)
    if use_vec_normalize:
        norm_path = os.path.join(run_dir, "vecnormalize.pkl")
        vec_env.save(norm_path)
        print(f"VecNormalize stats saved to {norm_path}")
    print(f"Training complete. Final model saved to {model_path}")


if __name__ == "__main__":
    main()

