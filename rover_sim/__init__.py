"""
Top-level package for the 2D rover simulator.

Components:
- world: map, obstacles, collision checking
- rover: differential drive kinematics and dynamics
- sensors: LiDAR and related sensor models
- render: pygame-based visualization
- env: Gymnasium-compatible RL environment
- geometry_utils: angle/point/segment/polygon helpers
- map_generator: procedural map generation (corridors, clutter, rooms, maze)
"""

from .world import World, Obstacle
from .rover import RoverState, Rover
from .sensors import LidarConfig, LidarSensor

__all__ = [
    "World",
    "Obstacle",
    "RoverState",
    "Rover",
    "LidarConfig",
    "LidarSensor",
]

