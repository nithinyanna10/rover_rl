from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import math
import random


@dataclass
class RoverState:
    """State of the rover in world coordinates.

    Attributes
    ----------
    x : float
        X position (meters).
    y : float
        Y position (meters).
    yaw : float
        Heading (radians), CCW from +x.
    v : float
        Linear velocity (m/s).
    w : float
        Angular velocity (rad/s).
    """

    x: float
    y: float
    yaw: float
    v: float
    w: float


class Rover:
    """Differential-drive rover (controlled by v, w).

    The simulator uses a simple unicycle model with optional acceleration limits
    and domain randomization (slip, friction).
    """

    def __init__(
        self,
        radius: float,
        max_linear_speed: float,
        max_angular_speed: float,
        linear_accel_limit: float,
        angular_accel_limit: float,
        rng: random.Random,
    ) -> None:
        self.radius = radius
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.linear_accel_limit = linear_accel_limit
        self.angular_accel_limit = angular_accel_limit
        self.rng = rng

        self.state = RoverState(x=0.0, y=0.0, yaw=0.0, v=0.0, w=0.0)

        # Domain randomization factors
        self.friction_scale = 1.0
        self.slip_std_linear = 0.0
        self.slip_std_angular = 0.0

    # ------------------------------------------------------------------
    # State manipulation
    # ------------------------------------------------------------------
    def reset(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        friction_scale: float = 1.0,
        slip_std_linear: float = 0.0,
        slip_std_angular: float = 0.0,
    ) -> None:
        """Reset rover state and domain-randomization parameters."""
        self.state = RoverState(x=x, y=y, yaw=yaw, v=0.0, w=0.0)
        self.friction_scale = friction_scale
        self.slip_std_linear = slip_std_linear
        self.slip_std_angular = slip_std_angular

    def get_state(self) -> RoverState:
        """Return a copy of current state."""
        s = self.state
        return RoverState(x=s.x, y=s.y, yaw=s.yaw, v=s.v, w=s.w)

    # ------------------------------------------------------------------
    # Dynamics integration
    # ------------------------------------------------------------------
    def step(self, v_cmd: float, w_cmd: float, dt: float) -> None:
        """Update rover state given commanded velocities and time step.

        Acceleration limits and domain randomization (friction and slip) are applied.
        """
        # Apply acceleration limits
        v_target = self._clamp(v_cmd, -self.max_linear_speed, self.max_linear_speed)
        w_target = self._clamp(w_cmd, -self.max_angular_speed, self.max_angular_speed)

        dv = v_target - self.state.v
        max_dv = self.linear_accel_limit * dt
        if abs(dv) > max_dv:
            dv = math.copysign(max_dv, dv)

        dw = w_target - self.state.w
        max_dw = self.angular_accel_limit * dt
        if abs(dw) > max_dw:
            dw = math.copysign(max_dw, dw)

        v = self.state.v + dv
        w = self.state.w + dw

        # Apply friction scaling
        v *= self.friction_scale
        w *= self.friction_scale

        # Apply slip noise
        if self.slip_std_linear > 0.0:
            v += self.rng.gauss(0.0, self.slip_std_linear)
        if self.slip_std_angular > 0.0:
            w += self.rng.gauss(0.0, self.slip_std_angular)

        # Integrate unicycle model
        x = self.state.x + v * math.cos(self.state.yaw) * dt
        y = self.state.y + v * math.sin(self.state.yaw) * dt
        yaw = self._wrap_angle(self.state.yaw + w * dt)

        self.state = RoverState(x=x, y=y, yaw=yaw, v=v, w=w)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_angle(theta: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (theta + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _clamp(value: float, vmin: float, vmax: float) -> float:
        return max(vmin, min(vmax, value))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize current rover state to a dict for logging/telemetry."""
        s = self.state
        return {
            "x": s.x,
            "y": s.y,
            "yaw": s.yaw,
            "v": s.v,
            "w": s.w,
        }

    def goal_relative_vector(self, goal: Tuple[float, float]) -> Tuple[float, float]:
        """Goal position expressed in rover frame (dx, dy).

        In rover frame:
        - +x is forward along rover heading
        - +y is left of rover heading
        """
        gx, gy = goal
        dx_world = gx - self.state.x
        dy_world = gy - self.state.y
        cos_yaw = math.cos(self.state.yaw)
        sin_yaw = math.sin(self.state.yaw)
        # Rotate world delta into rover frame (inverse rotation)
        dx_body = cos_yaw * dx_world + sin_yaw * dy_world
        dy_body = -sin_yaw * dx_world + cos_yaw * dy_world
        return dx_body, dy_body

