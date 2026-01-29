from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
import random

from .world import World, Obstacle


@dataclass
class LidarConfig:
    """Configuration for the 2D LiDAR sensor."""

    num_rays: int
    fov_deg: float
    max_range: float
    noise_std: float


class LidarSensor:
    """Simple 2D LiDAR ray caster in a rectangular world with AABB obstacles."""

    def __init__(self, config: LidarConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng

    def scan(
        self,
        world: World,
        pose: Tuple[float, float, float],
        noise_scale: float = 1.0,
    ) -> List[float]:
        """Perform a LiDAR scan from the given pose.

        Parameters
        ----------
        world : World
            World object containing obstacles and bounds.
        pose : tuple
            (x, y, yaw) of the scanner in world frame.
        noise_scale : float
            Multiplier for measurement noise standard deviation.

        Returns
        -------
        list[float]
            Distances for each ray, in meters, clipped to [0, max_range].
        """
        x, y, yaw = pose
        num_rays = self.config.num_rays
        fov_rad = math.radians(self.config.fov_deg)
        # Center FOV around rover heading
        start_angle = yaw - fov_rad / 2.0
        dtheta = fov_rad / max(num_rays - 1, 1)

        ranges: List[float] = []
        for i in range(num_rays):
            ray_angle = start_angle + i * dtheta
            r = self._cast_single_ray(world, x, y, ray_angle, self.config.max_range)
            # Add Gaussian noise
            if self.config.noise_std > 0.0:
                r += self.rng.gauss(0.0, self.config.noise_std * noise_scale)
            r = max(0.0, min(self.config.max_range, r))
            ranges.append(r)
        return ranges

    # ------------------------------------------------------------------
    # Ray casting
    # ------------------------------------------------------------------
    def _cast_single_ray(
        self,
        world: World,
        x: float,
        y: float,
        angle: float,
        max_range: float,
    ) -> float:
        """Compute distance to nearest intersection with obstacles or world bounds."""
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Intersection with world boundaries (rectangle from (0,0) to (W,H))
        t_max = self._ray_aabb_distance(x, y, dx, dy, 0.0, 0.0, world.width, world.height)
        if t_max is None:
            t_max = max_range

        distance = min(t_max, max_range)

        # Test each obstacle AABB
        for obs in world.obstacles:
            xmin, ymin, xmax, ymax = obs.bounds
            t = self._ray_aabb_distance(x, y, dx, dy, xmin, ymin, xmax, ymax)
            if t is not None and 0.0 <= t <= distance:
                distance = t

        return min(distance, max_range)

    @staticmethod
    def _ray_aabb_distance(
        ox: float,
        oy: float,
        dx: float,
        dy: float,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> float | None:
        """Ray-AABB intersection using slab method.

        Returns the distance t along the ray origin + t * direction, or None for no hit.
        """
        tmin = -math.inf
        tmax = math.inf

        # X slab
        if abs(dx) < 1e-8:
            if ox < xmin or ox > xmax:
                return None
        else:
            tx1 = (xmin - ox) / dx
            tx2 = (xmax - ox) / dx
            tmin = max(tmin, min(tx1, tx2))
            tmax = min(tmax, max(tx1, tx2))

        # Y slab
        if abs(dy) < 1e-8:
            if oy < ymin or oy > ymax:
                return None
        else:
            ty1 = (ymin - oy) / dy
            ty2 = (ymax - oy) / dy
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))

        if tmax < 0.0 or tmin > tmax:
            return None

        # We only care about the first intersection in front of origin
        if tmin >= 0.0:
            return tmin
        if tmax >= 0.0:
            return tmax
        return None

