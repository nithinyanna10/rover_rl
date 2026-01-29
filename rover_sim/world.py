from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import json
import math
import random


@dataclass
class Obstacle:
    """Axis-aligned rectangular obstacle in world coordinates.

    Coordinates are defined with origin at bottom-left of the world:
    - x increases to the right
    - y increases upward

    Attributes
    ----------
    x : float
        X coordinate of the rectangle center (meters).
    y : float
        Y coordinate of the rectangle center (meters).
    w : float
        Width of the rectangle (meters).
    h : float
        Height of the rectangle (meters).
    """

    x: float
    y: float
    w: float
    h: float

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (xmin, ymin, xmax, ymax)."""
        half_w = self.w / 2.0
        half_h = self.h / 2.0
        return (self.x - half_w, self.y - half_h, self.x + half_w, self.y + half_h)


class World:
    """2D world containing static rectangular obstacles and goal.

    Parameters
    ----------
    width : float
        World width in meters.
    height : float
        World height in meters.
    obstacles : list[Obstacle]
        Initial obstacle list.
    goal : tuple[float, float]
        Goal position (x, y) in meters.
    """

    def __init__(
        self,
        width: float,
        height: float,
        obstacles: Optional[List[Obstacle]] = None,
        goal: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.obstacles: List[Obstacle] = list(obstacles) if obstacles is not None else []
        self.goal = goal

    # ------------------------------------------------------------------
    # Map loading / generation
    # ------------------------------------------------------------------
    @classmethod
    def from_map_dict(cls, width: float, height: float, data: Dict[str, Any]) -> "World":
        """Create world from a dict describing obstacles and goal."""
        obstacles_data = data.get("obstacles", [])
        obstacles = [
            Obstacle(
                float(o["x"]),
                float(o["y"]),
                float(o["w"]),
                float(o["h"]),
            )
            for o in obstacles_data
        ]
        goal_data = data.get("goal", {"x": width * 0.8, "y": height * 0.8})
        goal = (float(goal_data["x"]), float(goal_data["y"]))
        return cls(width=width, height=height, obstacles=obstacles, goal=goal)

    @classmethod
    def from_map_file(cls, width: float, height: float, path: str) -> "World":
        """Create world from a JSON map file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_map_dict(width=width, height=height, data=data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize world description to a Python dict."""
        return {
            "width": self.width,
            "height": self.height,
            "goal": {"x": self.goal[0], "y": self.goal[1]},
            "obstacles": [
                {"x": o.x, "y": o.y, "w": o.w, "h": o.h} for o in self.obstacles
            ],
        }

    # ------------------------------------------------------------------
    # Random obstacle generation
    # ------------------------------------------------------------------
    def clear_obstacles(self) -> None:
        """Remove all obstacles."""
        self.obstacles.clear()

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add a single obstacle."""
        self.obstacles.append(obstacle)

    def generate_random_obstacles(
        self,
        num_min: int,
        num_max: int,
        min_size: Tuple[float, float],
        max_size: Tuple[float, float],
        rng: random.Random,
        margin: float = 0.5,
    ) -> None:
        """Generate random axis-aligned rectangular obstacles.

        Obstacles are kept inside the world bounds with an outer margin.
        """
        self.clear_obstacles()
        num = rng.randint(num_min, num_max)
        min_w, min_h = min_size
        max_w, max_h = max_size
        for _ in range(num):
            w = rng.uniform(min_w, max_w)
            h = rng.uniform(min_h, max_h)
            x = rng.uniform(margin + w / 2.0, self.width - margin - w / 2.0)
            y = rng.uniform(margin + h / 2.0, self.height - margin - h / 2.0)
            self.add_obstacle(Obstacle(x=x, y=y, w=w, h=h))

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------
    def is_out_of_bounds(self, x: float, y: float, radius: float) -> bool:
        """Check if a circular rover is out of the world bounds."""
        if x - radius < 0.0 or x + radius > self.width:
            return True
        if y - radius < 0.0 or y + radius > self.height:
            return True
        return False

    def check_collision(self, x: float, y: float, radius: float) -> bool:
        """Return True if a circular rover collides with any obstacle or boundary."""
        if self.is_out_of_bounds(x, y, radius):
            return True
        for obs in self.obstacles:
            if self._circle_rect_collision(x, y, radius, obs):
                return True
        return False

    @staticmethod
    def _circle_rect_collision(
        cx: float,
        cy: float,
        radius: float,
        obstacle: Obstacle,
    ) -> bool:
        """Circle-rectangle collision.

        Based on closest-point-on-rectangle test.
        """
        xmin, ymin, xmax, ymax = obstacle.bounds
        closest_x = min(max(cx, xmin), xmax)
        closest_y = min(max(cy, ymin), ymax)
        dx = cx - closest_x
        dy = cy - closest_y
        return dx * dx + dy * dy <= radius * radius

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def distance_to_goal(self, x: float, y: float) -> float:
        """Euclidean distance from (x,y) to goal."""
        gx, gy = self.goal
        return math.hypot(gx - x, gy - y)

