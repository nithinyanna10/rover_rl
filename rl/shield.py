from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np


@dataclass
class ShieldConfig:
    """Configuration for the LiDAR-based safety shield."""

    min_distance: float
    turn_angular_speed: float


class LidarSafetyShield:
    """Simple action shield that overrides unsafe actions based on LiDAR."""

    def __init__(self, cfg: ShieldConfig) -> None:
        self.cfg = cfg

    def filter_action(
        self,
        action: np.ndarray,
        lidar_ranges: Sequence[float],
        lidar_fov_deg: float,
    ) -> Tuple[np.ndarray, bool]:
        """Return possibly modified action and whether it was overridden.

        Strategy:
        - If minimum LiDAR distance is above threshold, do nothing.
        - If below threshold:
          - Stop forward motion (v = 0).
          - Turn away from the closest side: positive or negative angular velocity.
        """
        arr = np.asarray(action, dtype=np.float32).copy()
        if len(lidar_ranges) == 0:
            return arr, False

        ranges = np.asarray(lidar_ranges, dtype=np.float32)
        min_dist = float(np.min(ranges))
        if not np.isfinite(min_dist) or min_dist >= self.cfg.min_distance:
            return arr, False

        # Danger zone: decide turn direction based on left vs right distances
        num = len(ranges)
        if num < 4:
            # Fallback: just stop and turn left
            arr[0] = 0.0
            arr[1] = self.cfg.turn_angular_speed
            return arr, True

        mid = num // 2
        left_avg = float(np.mean(ranges[mid:]))
        right_avg = float(np.mean(ranges[:mid]))

        arr[0] = 0.0  # no forward motion
        if left_avg > right_avg:
            # More space on left; turn left (positive w)
            arr[1] = self.cfg.turn_angular_speed
        else:
            arr[1] = -self.cfg.turn_angular_speed
        return arr, True

