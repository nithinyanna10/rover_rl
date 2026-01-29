from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on path when running this script directly
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pygame
import yaml

from rover_sim.world import World
from rover_sim.env import RoverEnv, EnvConfig
from rover_sim.render import PygameRenderer
from rover_sim.sensors import LidarConfig, LidarSensor


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual rover sim with keyboard teleop.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sim.yaml",
        help="Path to sim YAML config.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    sim_cfg = cfg["sim"]
    rover_cfg = cfg["rover"]
    lidar_cfg = cfg["lidar"]
    render_cfg = cfg["render"]
    domain_rand_cfg = cfg["domain_randomization"]
    maps_cfg = cfg.get("maps", {})

    world_width = float(sim_cfg["world_width"])
    world_height = float(sim_cfg["world_height"])
    goal_radius = float(sim_cfg["goal_radius"])

    # Default map
    default_map = maps_cfg.get("default_fixed_map", "corridor_map")
    map_path = os.path.join("rover_sim", "maps", f"{default_map}.json")
    world = World.from_map_file(width=world_width, height=world_height, path=map_path)

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
        rng=env.rng,
    )

    obs, _ = env.reset()
    v_cmd = 0.0
    w_cmd = 0.0

    print("Keyboard teleop: W/S forward/back, A/D turn, SPACE stop, ESC to quit.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    v_cmd = 0.0
                    w_cmd = 0.0
                elif event.key == pygame.K_w:
                    v_cmd += 0.1
                elif event.key == pygame.K_s:
                    v_cmd -= 0.1
                elif event.key == pygame.K_a:
                    w_cmd += 0.1
                elif event.key == pygame.K_d:
                    w_cmd -= 0.1

        # Step env
        obs, reward, done, truncated, info = env.step([v_cmd, w_cmd])

        # Render
        lidar_ranges = lidar_sensor.scan(
            world=env.world,
            pose=(env.rover.state.x, env.rover.state.y, env.rover.state.yaw),
            noise_scale=1.0,
        )
        fps = renderer.tick(sim_cfg["fps"])
        renderer.draw(
            rover_state=env.rover.get_state(),
            lidar_ranges=lidar_ranges,
            lidar_fov_deg=lidar_cfg["fov_deg"],
            dt=env.cfg.dt,
            fps=fps,
        )

        if done or truncated:
            obs, _ = env.reset()
            v_cmd = 0.0
            w_cmd = 0.0

    renderer.close()


if __name__ == "__main__":
    main()

