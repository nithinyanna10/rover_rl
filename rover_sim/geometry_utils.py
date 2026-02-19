"""
Geometry utilities for the rover RL simulation.

Provides point-line distance, segment intersection, polygon helpers,
angle normalization, and coordinate transforms used by map generation,
collision checks, and reward shaping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


# ---------------------------------------------------------------------------
# Angle and coordinate helpers
# ---------------------------------------------------------------------------


def wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi] radians."""
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference from angle a to angle b (radians)."""
    d = wrap_angle(b - a)
    return d


def normalize_angle_positive(theta: float) -> float:
    """Wrap angle to [0, 2*pi)."""
    t = theta % (2.0 * math.pi)
    if t < 0.0:
        t += 2.0 * math.pi
    return t


def point_in_rect(
    px: float,
    py: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> bool:
    """Return True if point (px, py) is inside axis-aligned rectangle [xmin,ymin]-[xmax,ymax]."""
    return xmin <= px <= xmax and ymin <= py <= ymax


def rect_overlap(
    a_xmin: float,
    a_ymin: float,
    a_xmax: float,
    a_ymax: float,
    b_xmin: float,
    b_ymin: float,
    b_xmax: float,
    b_ymax: float,
) -> bool:
    """Return True if two axis-aligned rectangles overlap."""
    if a_xmax < b_xmin or b_xmax < a_xmin:
        return False
    if a_ymax < b_ymin or b_ymax < a_ymin:
        return False
    return True


# ---------------------------------------------------------------------------
# Point-to-line and point-to-segment distance
# ---------------------------------------------------------------------------


def point_to_line_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """
    Perpendicular distance from point (px, py) to infinite line through (x1,y1)-(x2,y2).
    """
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def point_to_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Tuple[float, float, float]:
    """
    Distance from point to line segment, and closest point on segment.

    Returns
    -------
    (distance, closest_x, closest_y)
    """
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - x1, py - y1), x1, y1
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.hypot(px - cx, py - cy), cx, cy


# ---------------------------------------------------------------------------
# Segment intersection
# ---------------------------------------------------------------------------


@dataclass
class SegmentIntersection:
    """Result of segment-segment intersection test."""

    intersects: bool
    x: float
    y: float
    t_a: float  # parameter on segment A [0,1]
    t_b: float  # parameter on segment B [0,1]


def segment_intersect(
    a_x1: float,
    a_y1: float,
    a_x2: float,
    a_y2: float,
    b_x1: float,
    b_y1: float,
    b_x2: float,
    b_y2: float,
) -> Optional[SegmentIntersection]:
    """
    Find intersection of line segment A (a_x1,a_y1)-(a_x2,a_y2)
    and segment B (b_x1,b_y1)-(b_x2,b_y2).
    Returns SegmentIntersection or None if no intersection.
    """
    dx_a = a_x2 - a_x1
    dy_a = a_y2 - a_y1
    dx_b = b_x2 - b_x1
    dy_b = b_y2 - b_y1

    denom = dx_a * dy_b - dy_a * dx_b
    if abs(denom) < 1e-10:
        return None

    t_num = (b_x1 - a_x1) * dy_b - (b_y1 - a_y1) * dx_b
    s_num = (b_x1 - a_x1) * dy_a - (b_y1 - a_y1) * dx_a
    t = t_num / denom
    s = s_num / denom

    if 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0:
        x = a_x1 + t * dx_a
        y = a_y1 + t * dy_a
        return SegmentIntersection(intersects=True, x=x, y=y, t_a=t, t_b=s)
    return None


# ---------------------------------------------------------------------------
# Circle-rectangle and circle-segment
# ---------------------------------------------------------------------------


def circle_segment_intersect(
    cx: float,
    cy: float,
    radius: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> bool:
    """
    True if circle (cx, cy, radius) intersects line segment (x1,y1)-(x2,y2).
    """
    dist, _, _ = point_to_segment_distance(cx, cy, x1, y1, x2, y2)
    return dist <= radius


def circle_rect_intersect(
    cx: float,
    cy: float,
    radius: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> bool:
    """
    True if circle intersects axis-aligned rectangle.
    Uses closest-point test.
    """
    closest_x = min(max(cx, xmin), xmax)
    closest_y = min(max(cy, ymin), ymax)
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy <= radius * radius


# ---------------------------------------------------------------------------
# Polygon helpers (for future map shapes)
# ---------------------------------------------------------------------------


def polygon_centroid(vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Compute centroid of a simple polygon (list of (x,y) vertices)."""
    n = len(vertices)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return vertices[0][0], vertices[0][1]
    ax = 0.0
    ay = 0.0
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        cross = xi * yj - xj * yi
        signed_area += cross
        ax += (xi + xj) * cross
        ay += (yi + yj) * cross
    if abs(signed_area) < 1e-10:
        return vertices[0][0], vertices[0][1]
    signed_area *= 0.5
    ax /= 6.0 * signed_area
    ay /= 6.0 * signed_area
    return ax, ay


def point_in_polygon(px: float, py: float, vertices: List[Tuple[float, float]]) -> bool:
    """
    Ray-casting test: True if (px, py) is inside polygon (list of (x,y) in order).
    """
    n = len(vertices)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-10) + xi
        ):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Transform: world <-> body frame
# ---------------------------------------------------------------------------


def world_to_body(
    wx: float,
    wy: float,
    ox: float,
    oy: float,
    yaw: float,
) -> Tuple[float, float]:
    """
    Transform world point (wx, wy) to body frame with origin (ox, oy) and heading yaw.
    Body +x is forward (cos(yaw), sin(yaw)).
    """
    dx = wx - ox
    dy = wy - oy
    c = math.cos(yaw)
    s = math.sin(yaw)
    bx = c * dx + s * dy
    by = -s * dx + c * dy
    return bx, by


def body_to_world(
    bx: float,
    by: float,
    ox: float,
    oy: float,
    yaw: float,
) -> Tuple[float, float]:
    """Transform body frame (bx, by) to world with origin (ox, oy) and heading yaw."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    wx = ox + c * bx - s * by
    wy = oy + s * bx + c * by
    return wx, wy


# ---------------------------------------------------------------------------
# Sampling and clamping
# ---------------------------------------------------------------------------


def clamp(value: float, vmin: float, vmax: float) -> float:
    """Clamp value to [vmin, vmax]."""
    return max(vmin, min(vmax, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: a + t*(b - a), t typically in [0,1]."""
    return a + t * (b - a)


def smooth_step(t: float) -> float:
    """Smooth step function: 0 for t<=0, 1 for t>=1, smooth in between."""
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    return math.hypot(x2 - x1, y2 - y1)


def normalize_2d(dx: float, dy: float) -> Tuple[float, float]:
    """Return (dx, dy) normalized; if zero vector, return (0, 0)."""
    length = math.hypot(dx, dy)
    if length < 1e-10:
        return 0.0, 0.0
    return dx / length, dy / length
