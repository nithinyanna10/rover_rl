from __future__ import annotations

from typing import List, Tuple, Optional
import math

import pygame

from .world import World
from .rover import RoverState


Color = Tuple[int, int, int]


class PygameRenderer:
    """Top-down 2D visualization of the rover, obstacles, goal and LiDAR.

    Coordinates:
    - World origin (0,0) is mapped to the bottom-left of the screen.
    - Y axis is flipped so that world +y is up while screen y increases downward.
    """

    BG_COLOR: Color = (20, 20, 20)
    OBSTACLE_COLOR: Color = (120, 120, 120)
    ROVER_COLOR: Color = (0, 180, 255)
    GOAL_COLOR: Color = (0, 255, 0)
    TRAIL_COLOR: Color = (255, 255, 0)
    LIDAR_COLOR: Color = (255, 100, 100)

    def __init__(
        self,
        world: World,
        window_width: int,
        window_height: int,
        show_lidar: bool = True,
        show_trail: bool = True,
        trail_max_length: int = 500,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("RL Rover Simulation")
        self.screen = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()

        self.world = world
        self.window_width = window_width
        self.window_height = window_height
        self.show_lidar = show_lidar
        self.show_trail = show_trail
        self.trail_max_length = trail_max_length
        self.trail: List[Tuple[float, float]] = []

        # Scale from meters to pixels
        self.scale_x = window_width / world.width
        self.scale_y = window_height / world.height

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------
    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pygame screen coordinates."""
        sx = int(x * self.scale_x)
        sy = int(self.window_height - y * self.scale_y)
        return sx, sy

    def _meters_to_pixels(self, r: float) -> int:
        """Convert a length in meters to pixels (average of axes)."""
        return int(r * 0.5 * (self.scale_x + self.scale_y))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def draw(
        self,
        rover_state: RoverState,
        lidar_ranges: Optional[List[float]] = None,
        lidar_fov_deg: float = 0.0,
        dt: float = 0.0,
        fps: float = 0.0,
    ) -> None:
        """Render one frame."""
        self.screen.fill(self.BG_COLOR)

        # Draw obstacles
        for obs in self.world.obstacles:
            xmin, ymin, xmax, ymax = obs.bounds
            sx, sy = self._world_to_screen(xmin, ymin)
            sw = int((xmax - xmin) * self.scale_x)
            sh = int((ymax - ymin) * self.scale_y)
            # y already bottom-left; convert to top-left for pygame
            sy = sy - sh
            pygame.draw.rect(
                self.screen,
                self.OBSTACLE_COLOR,
                pygame.Rect(sx, sy, sw, sh),
            )

        # Draw goal
        gx, gy = self.world.goal
        goal_pos = self._world_to_screen(gx, gy)
        pygame.draw.circle(
            self.screen,
            self.GOAL_COLOR,
            goal_pos,
            self._meters_to_pixels(0.3),
            width=2,
        )

        # Trail
        if self.show_trail:
            self.trail.append((rover_state.x, rover_state.y))
            if len(self.trail) > self.trail_max_length:
                self.trail = self.trail[-self.trail_max_length :]
            if len(self.trail) >= 2:
                pts = [self._world_to_screen(p[0], p[1]) for p in self.trail]
                pygame.draw.lines(self.screen, self.TRAIL_COLOR, False, pts, 2)

        # Draw LiDAR rays
        if self.show_lidar and lidar_ranges is not None and len(lidar_ranges) > 0:
            self._draw_lidar(rover_state, lidar_ranges, lidar_fov_deg)

        # Draw rover body
        self._draw_rover(rover_state)

        # HUD text: FPS
        self._draw_hud(dt, fps)

        pygame.display.flip()

    def _draw_rover(self, state: RoverState) -> None:
        center = self._world_to_screen(state.x, state.y)
        radius_px = self._meters_to_pixels(0.4)
        pygame.draw.circle(self.screen, self.ROVER_COLOR, center, radius_px, width=2)

        # Heading arrow
        arrow_len = self._meters_to_pixels(0.8)
        hx = state.x + math.cos(state.yaw) * 0.8
        hy = state.y + math.sin(state.yaw) * 0.8
        head = self._world_to_screen(hx, hy)
        pygame.draw.line(self.screen, self.ROVER_COLOR, center, head, width=3)

    def _draw_lidar(
        self, state: RoverState, ranges: List[float], fov_deg: float
    ) -> None:
        num_rays = len(ranges)
        if num_rays == 0:
            return
        fov_rad = math.radians(fov_deg)
        start_angle = state.yaw - fov_rad / 2.0
        dtheta = fov_rad / max(num_rays - 1, 1)

        sx, sy = self._world_to_screen(state.x, state.y)
        for i, r in enumerate(ranges):
            angle = start_angle + i * dtheta
            ex = state.x + r * math.cos(angle)
            ey = state.y + r * math.sin(angle)
            ex_s, ey_s = self._world_to_screen(ex, ey)
            pygame.draw.line(
                self.screen,
                self.LIDAR_COLOR,
                (sx, sy),
                (ex_s, ey_s),
                width=1,
            )

    def _draw_hud(self, dt: float, fps: float) -> None:
        font = pygame.font.SysFont("monospace", 14)
        text = f"dt={dt:.3f}s  FPS={fps:.1f}"
        surf = font.render(text, True, (255, 255, 255))
        self.screen.blit(surf, (5, 5))

    def tick(self, target_fps: int) -> float:
        """Cap frame rate and return achieved FPS."""
        fps = self.clock.get_fps()
        self.clock.tick(target_fps)
        return fps

    def close(self) -> None:
        pygame.quit()

