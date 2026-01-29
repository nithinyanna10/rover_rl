from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class RobotAPI(ABC):
    """Abstract interface for simulated and real rovers."""

    @abstractmethod
    def reset(self, goal: Tuple[float, float]) -> None:
        """Reset robot state and set a new goal in world coordinates."""

    @abstractmethod
    def read_sensors(self) -> Dict[str, Any]:
        """Return sensor readings (e.g., LiDAR, IMU, odometry)."""

    @abstractmethod
    def get_state(self) -> Dict[str, float]:
        """Return high-level state: position, orientation, velocities."""

    @abstractmethod
    def send_command(self, v: float, w: float) -> None:
        """Send linear and angular velocity command to the robot."""

    @abstractmethod
    def stop(self) -> None:
        """Immediately stop the robot (safe state)."""

