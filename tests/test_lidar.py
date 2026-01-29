from __future__ import annotations

import math
import random

from rover_sim.sensors import LidarConfig, LidarSensor
from rover_sim.world import World, Obstacle


def test_lidar_hits_simple_wall() -> None:
    world = World(width=10.0, height=10.0, obstacles=[], goal=(9.0, 5.0))
    # Vertical wall in front of rover at x=5
    world.add_obstacle(Obstacle(x=5.0, y=5.0, w=0.1, h=10.0))

    rng = random.Random(0)
    cfg = LidarConfig(num_rays=1, fov_deg=1.0, max_range=10.0, noise_std=0.0)
    lidar = LidarSensor(cfg, rng)

    # Rover at x=1, facing +x
    ranges = lidar.scan(world, pose=(1.0, 5.0, 0.0), noise_scale=1.0)
    assert len(ranges) == 1
    dist = ranges[0]
    assert math.isclose(dist, 4.0, rel_tol=1e-3)

