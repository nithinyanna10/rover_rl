from __future__ import annotations

from typing import List


class LidarDriver:
    """Stub LiDAR driver (e.g., RPLIDAR, Hokuyo).

    TODO:
    - Implement connection and scan acquisition from your LiDAR hardware.
    - Map raw scans into a fixed number of evenly spaced rays.
    """

    def __init__(self, num_rays: int, fov_deg: float, max_range: float) -> None:
        self.num_rays = num_rays
        self.fov_deg = fov_deg
        self.max_range = max_range

    def get_scan(self) -> List[float]:
        """Return a single LiDAR scan as a list of distances (meters)."""
        # TODO: Replace with real scan logic. For now, return max_range values.
        return [self.max_range] * self.num_rays

