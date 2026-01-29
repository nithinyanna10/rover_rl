from __future__ import annotations

import argparse
import os
from typing import Any, Dict

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

    sim_cfg = load_yaml(args.config)["sim"]

    world_width = float(sim_cfg["world_width"])
    world_height = float(sim_cfg["world_height"])
    goal_radius = float(sim_cfg["goal_radius"])

    # Default map
    maps_cfg = sim_cfg["maps"]
    default_map = maps_cfg["default_fixed_map"]
    map_path = os.path.join("rover_sim", "maps", f"{default_map}.json")
    world = World.from_map_file(width=world_width, height=world_height, path=map_path)

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
            lidar_fov_deg=sim_cfg["lidar"]["fov_deg"],
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

