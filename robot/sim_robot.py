from __future__ import annotations

from typing import Any, Dict, Tuple

from rover_sim.env import RoverEnv
from rover_sim.sensors import LidarSensor, LidarConfig
from rover_sim.world import World
from rover_sim.rover import Rover
from robot.api import RobotAPI


class SimRobot(RobotAPI):
    """RobotAPI wrapper around the RoverEnv simulator."""

    def __init__(self, env: RoverEnv, lidar_sensor: LidarSensor) -> None:
        self.env = env
        self.lidar_sensor = lidar_sensor
        self.latest_obs = None

    def reset(self, goal: Tuple[float, float]) -> None:
        # Update world goal and reset env
        self.env.world.goal = goal
        self.latest_obs, _ = self.env.reset()

    def read_sensors(self) -> Dict[str, Any]:
        state = self.env.rover.get_state()
        lidar_ranges = self.lidar_sensor.scan(
            world=self.env.world,
            pose=(state.x, state.y, state.yaw),
            noise_scale=1.0,
        )
        return {
            "pose": self.env.rover.to_dict(),
            "lidar_ranges": lidar_ranges,
            "goal": {"x": self.env.world.goal[0], "y": self.env.world.goal[1]},
        }

    def get_state(self) -> Dict[str, float]:
        return self.env.rover.to_dict()

    def send_command(self, v: float, w: float) -> None:
        action = [float(v), float(w)]
        self.latest_obs, _, _, _, _ = self.env.step(action)

    def stop(self) -> None:
        # Send zero velocities
        self.send_command(0.0, 0.0)

