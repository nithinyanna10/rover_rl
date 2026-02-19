"""
Procedural map generation for rover RL training and evaluation.

Generates rectangular obstacle layouts: corridors, mazes, rooms, clutter,
and hybrid layouts. Output is compatible with World.from_map_dict().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import json
import math
import random
import os

from .world import World, Obstacle
from .geometry_utils import rect_overlap, clamp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MapGeneratorConfig:
    """Parameters for procedural map generation."""

    width: float = 20.0
    height: float = 20.0
    margin: float = 0.5
    min_obstacle_size: Tuple[float, float] = (0.5, 0.5)
    max_obstacle_size: Tuple[float, float] = (3.0, 3.0)
    corridor_width: float = 2.0
    wall_thickness: float = 0.8
    num_rooms_x: int = 3
    num_rooms_y: int = 3
    room_margin: float = 0.3
    clutter_count_min: int = 5
    clutter_count_max: int = 20
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Corridor layouts
# ---------------------------------------------------------------------------


def generate_corridor_map(
    width: float,
    height: float,
    corridor_width: float = 2.0,
    wall_thickness: float = 0.8,
    num_turns: int = 2,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Generate a map with a winding corridor from bottom-left toward top-right.
    Obstacles are axis-aligned rectangles forming walls.
    """
    rng = rng or random.Random()
    obstacles_data: List[Dict[str, float]] = []
    margin = wall_thickness + 0.2

    # Waypoints for corridor centerline (simplified: zigzag)
    waypoints: List[Tuple[float, float]] = [
        (margin + corridor_width / 2, margin + corridor_width / 2),
        (width - margin - corridor_width / 2, height - margin - corridor_width / 2),
    ]
    if num_turns >= 1:
        mid_x = (waypoints[0][0] + waypoints[1][0]) / 2
        mid_y = (waypoints[0][1] + waypoints[1][1]) / 2
        waypoints.insert(1, (mid_x, waypoints[0][1]))
        waypoints.insert(2, (mid_x, waypoints[-1][1]))

    # Build corridor by adding wall rectangles along the path
    half = corridor_width / 2 + wall_thickness
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        nx = -dy / length
        ny = dx / length
        # Left wall
        cx_left = (x0 + x1) / 2 + nx * half
        cy_left = (y0 + y1) / 2 + ny * half
        wall_w = abs(dx) + 2 * wall_thickness if abs(dx) > abs(dy) else 2 * wall_thickness
        wall_h = abs(dy) + 2 * wall_thickness if abs(dy) >= abs(dx) else 2 * wall_thickness
        obstacles_data.append({"x": cx_left, "y": cy_left, "w": wall_w, "h": wall_h})
        # Right wall
        cx_right = (x0 + x1) / 2 - nx * half
        cy_right = (y0 + y1) / 2 - ny * half
        obstacles_data.append({"x": cx_right, "y": cy_right, "w": wall_w, "h": wall_h})

    goal = (width - 2.0, height - 2.0)
    return {"obstacles": obstacles_data, "goal": {"x": goal[0], "y": goal[1]}}


def generate_u_shape(
    width: float,
    height: float,
    opening_side: str = "top",
    wall_thickness: float = 0.8,
) -> Dict[str, Any]:
    """Generate a U-shaped obstacle (three walls)."""
    obstacles_data: List[Dict[str, float]] = []
    wt = wall_thickness
    gap = min(width, height) * 0.25

    if opening_side == "top":
        # Left wall
        obstacles_data.append({"x": wt, "y": height / 2, "w": wt * 2, "h": height})
        # Bottom wall
        obstacles_data.append({"x": width / 2, "y": wt, "w": width, "h": wt * 2})
        # Right wall
        obstacles_data.append({"x": width - wt, "y": height / 2, "w": wt * 2, "h": height})
    elif opening_side == "bottom":
        obstacles_data.append({"x": wt, "y": height / 2, "w": wt * 2, "h": height})
        obstacles_data.append({"x": width / 2, "y": height - wt, "w": width, "h": wt * 2})
        obstacles_data.append({"x": width - wt, "y": height / 2, "w": wt * 2, "h": height})
    else:
        obstacles_data.append({"x": width / 2, "y": wt, "w": width, "h": wt * 2})
        obstacles_data.append({"x": wt, "y": height / 2, "w": wt * 2, "h": height})
        obstacles_data.append({"x": width - wt, "y": height / 2, "w": wt * 2, "h": height})

    goal = (width - 2.0, height - 2.0)
    return {"obstacles": obstacles_data, "goal": {"x": goal[0], "y": goal[1]}}


# ---------------------------------------------------------------------------
# Random clutter
# ---------------------------------------------------------------------------


def generate_clutter_map(
    width: float,
    height: float,
    num_obstacles_min: int = 5,
    num_obstacles_max: int = 25,
    min_size: Tuple[float, float] = (0.5, 0.5),
    max_size: Tuple[float, float] = (3.0, 3.0),
    margin: float = 0.5,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Generate a map with random axis-aligned obstacles (no overlap with start/goal zones)."""
    rng = rng or random.Random()
    obstacles_data: List[Dict[str, float]] = []
    num = rng.randint(num_obstacles_min, num_obstacles_max)
    start_zone = (margin, margin, margin + 3.0, margin + 3.0)
    goal_zone = (width - margin - 3.0, height - margin - 3.0, width - margin, height - margin)

    min_w, min_h = min_size
    max_w, max_h = max_size
    attempts = 0
    max_attempts = num * 50
    while len(obstacles_data) < num and attempts < max_attempts:
        attempts += 1
        w = rng.uniform(min_w, max_w)
        h = rng.uniform(min_h, max_h)
        x = rng.uniform(margin + w / 2, width - margin - w / 2)
        y = rng.uniform(margin + h / 2, height - margin - h / 2)
        xmin, xmax = x - w / 2, x + w / 2
        ymin, ymax = y - h / 2, y + h / 2
        if rect_overlap(xmin, ymin, xmax, ymax, *start_zone) or rect_overlap(
            xmin, ymin, xmax, ymax, *goal_zone
        ):
            continue
        obstacles_data.append({"x": x, "y": y, "w": w, "h": h})

    goal = (width - 2.0, height - 2.0)
    return {"obstacles": obstacles_data, "goal": {"x": goal[0], "y": goal[1]}}


# ---------------------------------------------------------------------------
# Grid of rooms with doors
# ---------------------------------------------------------------------------


def generate_rooms_map(
    width: float,
    height: float,
    num_rooms_x: int = 3,
    num_rooms_y: int = 3,
    wall_thickness: float = 0.6,
    door_width: float = 1.2,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Generate a grid of rooms with interior walls and door gaps.
    """
    rng = rng or random.Random()
    obstacles_data: List[Dict[str, float]] = []
    wt = wall_thickness
    room_w = width / num_rooms_x
    room_h = height / num_rooms_y

    # Horizontal walls (between rows)
    for row in range(num_rooms_y):
        y_center = (row + 0.5) * room_h
        for col in range(num_rooms_x - 1):
            x_gap_center = (col + 1) * room_w
            # Wall left of door
            x_left = x_gap_center - door_width / 2 - wt
            if x_left > col * room_w + wt:
                obstacles_data.append({
                    "x": (col * room_w + x_left) / 2 + wt / 2,
                    "y": y_center,
                    "w": x_left - col * room_w,
                    "h": wt * 2,
                })
            # Wall right of door
            x_right = x_gap_center + door_width / 2 + wt
            if x_right < (col + 2) * room_w - wt:
                obstacles_data.append({
                    "x": (x_right + (col + 2) * room_w) / 2 - wt / 2,
                    "y": y_center,
                    "w": (col + 2) * room_w - x_right,
                    "h": wt * 2,
                })

    # Vertical walls (between columns)
    for col in range(num_rooms_x):
        x_center = (col + 0.5) * room_w
        for row in range(num_rooms_y - 1):
            y_gap_center = (row + 1) * room_h
            y_bot = y_gap_center - door_width / 2 - wt
            if y_bot > row * room_h + wt:
                obstacles_data.append({
                    "x": x_center,
                    "y": (row * room_h + y_bot) / 2 + wt / 2,
                    "w": wt * 2,
                    "h": y_bot - row * room_h,
                })
            y_top = y_gap_center + door_width / 2 + wt
            if y_top < (row + 2) * room_h - wt:
                obstacles_data.append({
                    "x": x_center,
                    "y": (y_top + (row + 2) * room_h) / 2 - wt / 2,
                    "w": wt * 2,
                    "h": (row + 2) * room_h - y_top,
                })

    goal = (width - 2.0, height - 2.0)
    return {"obstacles": obstacles_data, "goal": {"x": goal[0], "y": goal[1]}}


# ---------------------------------------------------------------------------
# Simple maze (grid-based)
# ---------------------------------------------------------------------------


def generate_simple_maze(
    width: float,
    height: float,
    cell_size: float = 2.0,
    wall_thickness: float = 0.5,
    path_width_ratio: float = 0.6,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Generate a grid maze: cells are either passable or wall.
    Uses a simple random DFS-style removal of walls to ensure connectivity.
    """
    rng = rng or random.Random()
    obstacles_data: List[Dict[str, float]] = []
    wt = wall_thickness
    nc_x = max(2, int(width / cell_size))
    nc_y = max(2, int(height / cell_size))
    cell_w = width / nc_x
    cell_h = height / nc_y

    # Grid of walls: (nc_x+1) x (nc_y+1) horizontal segments, (nc_x+1) x (nc_y+1) vertical
    # We represent which edges are open (no wall)
    # 0 = wall between cells, 1 = open
    h_walls = [[1 for _ in range(nc_y)] for _ in range(nc_x + 1)]  # vertical walls
    v_walls = [[1 for _ in range(nc_y + 1)] for _ in range(nc_x)]   # horizontal walls

    # Open borders for start and goal
    h_walls[0][0] = 0
    h_walls[nc_x][nc_y - 1] = 0
    v_walls[0][0] = 0
    v_walls[nc_x - 1][nc_y] = 0

    # Randomly open some internal walls so there is a path (simplified: open ~40%)
    for i in range(nc_x + 1):
        for j in range(nc_y):
            if 0 < i < nc_x and rng.random() < path_width_ratio:
                h_walls[i][j] = 0
    for i in range(nc_x):
        for j in range(nc_y + 1):
            if 0 < j < nc_y and rng.random() < path_width_ratio:
                v_walls[i][j] = 0

    # Build obstacle rectangles for remaining walls
    for i in range(nc_x + 1):
        for j in range(nc_y):
            if h_walls[i][j] == 1:
                x = i * cell_w
                y = j * cell_h
                obstacles_data.append({"x": x + wt, "y": y + cell_h / 2, "w": wt * 2, "h": cell_h + wt * 2})
    for i in range(nc_x):
        for j in range(nc_y + 1):
            if v_walls[i][j] == 1:
                x = i * cell_w
                y = j * cell_h
                obstacles_data.append({"x": x + cell_w / 2, "y": y + wt, "w": cell_w + wt * 2, "h": wt * 2})

    goal = (width - 2.0, height - 2.0)
    return {"obstacles": obstacles_data, "goal": {"x": goal[0], "y": goal[1]}}


# ---------------------------------------------------------------------------
# Bottleneck (narrow passage)
# ---------------------------------------------------------------------------


def generate_bottleneck_map(
    width: float,
    height: float,
    passage_width: float = 1.5,
    wall_thickness: float = 0.8,
    orientation: str = "horizontal",
) -> Dict[str, Any]:
    """Single narrow passage between two large obstacle blocks."""
    obstacles_data: List[Dict[str, float]] = []
    wt = wall_thickness
    half_pass = passage_width / 2
    mid_x = width / 2
    mid_y = height / 2

    if orientation == "horizontal":
        # Left block
        obstacles_data.append({
            "x": mid_x - (mid_x + half_pass) / 2 - wt,
            "y": mid_y,
            "w": mid_x - half_pass + wt,
            "h": height - 2 * wt,
        })
        # Right block
        obstacles_data.append({
            "x": mid_x + (width - mid_x - half_pass) / 2 + wt,
            "y": mid_y,
            "w": width - mid_x - half_pass + wt,
            "h": height - 2 * wt,
        })
    else:
        obstacles_data.append({
            "x": mid_x,
            "y": mid_y - (mid_y + half_pass) / 2 - wt,
            "w": width - 2 * wt,
            "h": mid_y - half_pass + wt,
        })
        obstacles_data.append({
            "x": mid_x,
            "y": mid_y + (height - mid_y - half_pass) / 2 + wt,
            "w": width - 2 * wt,
            "h": height - mid_y - half_pass + wt,
        })

    goal = (width - 2.0, height - 2.0)
    return {"obstacles": obstacles_data, "goal": {"x": goal[0], "y": goal[1]}}


# ---------------------------------------------------------------------------
# Hybrid and presets
# ---------------------------------------------------------------------------


def generate_hybrid_map(
    width: float,
    height: float,
    preset: str = "corridor_clutter",
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Generate a hybrid map from a preset name.
    Presets: 'corridor_clutter', 'rooms_plus_clutter', 'maze_sparse', 'u_plus_clutter'.
    """
    rng = rng or random.Random()
    if preset == "corridor_clutter":
        data = generate_corridor_map(width, height, num_turns=2, rng=rng)
        clutter = generate_clutter_map(width, height, 3, 10, rng=rng)
        data["obstacles"] = data["obstacles"] + clutter["obstacles"]
        return data
    if preset == "rooms_plus_clutter":
        data = generate_rooms_map(width, height, 3, 3, rng=rng)
        clutter = generate_clutter_map(width, height, 2, 8, rng=rng)
        data["obstacles"] = data["obstacles"] + clutter["obstacles"]
        return data
    if preset == "maze_sparse":
        return generate_simple_maze(width, height, cell_size=2.5, rng=rng)
    if preset == "u_plus_clutter":
        data = generate_u_shape(width, height, "top")
        clutter = generate_clutter_map(width, height, 4, 12, rng=rng)
        data["obstacles"] = data["obstacles"] + clutter["obstacles"]
        return data
    return generate_clutter_map(width, height, 5, 15, rng=rng)


def world_from_generator(
    gen_config: MapGeneratorConfig,
    preset: str = "clutter",
) -> World:
    """
    Build a World instance from a generator preset.
    """
    rng = random.Random(gen_config.seed) if gen_config.seed is not None else random.Random()
    if preset == "corridor":
        data = generate_corridor_map(
            gen_config.width,
            gen_config.height,
            corridor_width=gen_config.corridor_width,
            wall_thickness=gen_config.wall_thickness,
            rng=rng,
        )
    elif preset == "u_shape":
        data = generate_u_shape(
            gen_config.width,
            gen_config.height,
            wall_thickness=gen_config.wall_thickness,
        )
    elif preset == "clutter":
        data = generate_clutter_map(
            gen_config.width,
            gen_config.height,
            gen_config.clutter_count_min,
            gen_config.clutter_count_max,
            gen_config.min_obstacle_size,
            gen_config.max_obstacle_size,
            gen_config.margin,
            rng=rng,
        )
    elif preset == "rooms":
        data = generate_rooms_map(
            gen_config.width,
            gen_config.height,
            gen_config.num_rooms_x,
            gen_config.num_rooms_y,
            gen_config.wall_thickness,
            rng=rng,
        )
    elif preset == "maze":
        data = generate_simple_maze(
            gen_config.width,
            gen_config.height,
            wall_thickness=gen_config.wall_thickness,
            rng=rng,
        )
    elif preset == "bottleneck":
        data = generate_bottleneck_map(
            gen_config.width,
            gen_config.height,
            wall_thickness=gen_config.wall_thickness,
        )
    else:
        data = generate_hybrid_map(
            gen_config.width,
            gen_config.height,
            preset=preset,
            rng=rng,
        )
    return World.from_map_dict(gen_config.width, gen_config.height, data)


def save_generated_map(data: Dict[str, Any], path: str) -> None:
    """Write generated map dict to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
