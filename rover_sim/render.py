from __future__ import annotations

from typing import List, Tuple, Optional
import math

import pygame

from .world import World
from .rover import RoverState


Color = Tuple[int, int, int]

# Modern dark theme palette
THEME = {
    "bg": (18, 22, 32),
    "grid": (28, 34, 48),
    "obstacle_fill": (45, 52, 70),
    "obstacle_edge": (65, 75, 98),
    "obstacle_highlight": (85, 95, 120),
    "goal_inner": (0, 230, 180),
    "goal_outer": (0, 180, 140),
    "goal_glow": (0, 140, 110),
    "rover_fill": (100, 220, 255),
    "rover_outline": (40, 140, 200),
    "rover_arrow": (140, 240, 255),
    "trail_start": (60, 160, 200),
    "trail_end": (100, 220, 255),
    "hud_bg": (28, 34, 48),
    "hud_border": (55, 65, 88),
    "hud_text": (200, 220, 255),
    "lidar_close": (255, 90, 90),
    "lidar_mid": (255, 180, 100),
    "lidar_far": (100, 200, 255),
}


class PygameRenderer:
    """Top-down 2D visualization of the rover, obstacles, goal and LiDAR.

    Coordinates:
    - World origin (0,0) is mapped to the bottom-left of the screen.
    - Y axis is flipped so that world +y is up while screen y increases downward.
    """

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
    def _draw_grid(self) -> None:
        """Draw a subtle grid for scale and depth."""
        step_m = 2.0
        color = THEME["grid"]
        w_m, h_m = self.world.width, self.world.height
        x = 0.0
        while x <= w_m:
            start = self._world_to_screen(x, 0.0)
            end = self._world_to_screen(x, h_m)
            pygame.draw.line(self.screen, color, start, end, 1)
            x += step_m
        y = 0.0
        while y <= h_m:
            start = self._world_to_screen(0.0, y)
            end = self._world_to_screen(w_m, y)
            pygame.draw.line(self.screen, color, start, end, 1)
            y += step_m

    def draw(
        self,
        rover_state: RoverState,
        lidar_ranges: Optional[List[float]] = None,
        lidar_fov_deg: float = 0.0,
        dt: float = 0.0,
        fps: float = 0.0,
    ) -> None:
        """Render one frame."""
        self.screen.fill(THEME["bg"])
        self._draw_grid()

        # Draw obstacles (filled + edge)
        for obs in self.world.obstacles:
            xmin, ymin, xmax, ymax = obs.bounds
            sx, sy = self._world_to_screen(xmin, ymin)
            sw = int((xmax - xmin) * self.scale_x)
            sh = int((ymax - ymin) * self.scale_y)
            sy = sy - sh
            rect = pygame.Rect(sx, sy, sw, sh)
            pygame.draw.rect(self.screen, THEME["obstacle_fill"], rect)
            pygame.draw.rect(self.screen, THEME["obstacle_edge"], rect, 2)
            # subtle inner highlight (top-left)
            pygame.draw.line(self.screen, THEME["obstacle_highlight"], (sx, sy), (sx + sw, sy), 1)
            pygame.draw.line(self.screen, THEME["obstacle_highlight"], (sx, sy), (sx, sy + sh), 1)

        # Draw goal (glow rings + filled center)
        gx, gy = self.world.goal
        goal_pos = self._world_to_screen(gx, gy)
        r_inner = self._meters_to_pixels(0.25)
        r_outer = self._meters_to_pixels(0.4)
        r_glow = self._meters_to_pixels(0.55)
        pygame.draw.circle(self.screen, THEME["goal_glow"], goal_pos, r_glow, 1)
        pygame.draw.circle(self.screen, THEME["goal_outer"], goal_pos, r_outer, 2)
        pygame.draw.circle(self.screen, THEME["goal_inner"], goal_pos, r_inner)
        pygame.draw.circle(self.screen, THEME["goal_outer"], goal_pos, r_inner, 1)

        # Trail (gradient from old to new)
        if self.show_trail:
            self.trail.append((rover_state.x, rover_state.y))
            if len(self.trail) > self.trail_max_length:
                self.trail = self.trail[-self.trail_max_length :]
            if len(self.trail) >= 2:
                pts = [self._world_to_screen(p[0], p[1]) for p in self.trail]
                n = len(pts) - 1
                for i in range(n):
                    t = (i + 1) / max(n, 1)
                    r = int(THEME["trail_start"][0] + t * (THEME["trail_end"][0] - THEME["trail_start"][0]))
                    g = int(THEME["trail_start"][1] + t * (THEME["trail_end"][1] - THEME["trail_start"][1]))
                    b = int(THEME["trail_start"][2] + t * (THEME["trail_end"][2] - THEME["trail_start"][2]))
                    w = 2 if i == n - 1 else 1
                    pygame.draw.line(self.screen, (r, g, b), pts[i], pts[i + 1], w)

        # Draw LiDAR rays (behind rover, with distance-based color)
        if self.show_lidar and lidar_ranges is not None and len(lidar_ranges) > 0:
            self._draw_lidar(rover_state, lidar_ranges, lidar_fov_deg)

        # Draw rover body
        self._draw_rover(rover_state)

        self._draw_hud(dt, fps)
        pygame.display.flip()

    def _draw_rover(self, state: RoverState) -> None:
        center = self._world_to_screen(state.x, state.y)
        radius_px = max(2, self._meters_to_pixels(0.4))
        pygame.draw.circle(self.screen, THEME["rover_fill"], center, radius_px, 0)
        pygame.draw.circle(self.screen, THEME["rover_outline"], center, radius_px, 2)

        # Heading arrow (thick line + small triangle head)
        arrow_len = 0.8
        hx = state.x + math.cos(state.yaw) * arrow_len
        hy = state.y + math.sin(state.yaw) * arrow_len
        head = self._world_to_screen(hx, hy)
        pygame.draw.line(self.screen, THEME["rover_arrow"], center, head, 4)
        # Arrowhead: small triangle
        tip_angle = state.yaw
        back_angle = state.yaw + math.pi * 0.85
        wing = 0.15
        wx1 = hx + math.cos(back_angle) * wing
        wy1 = hy + math.sin(back_angle) * wing
        back_angle2 = state.yaw - math.pi * 0.85
        wx2 = hx + math.cos(back_angle2) * wing
        wy2 = hy + math.sin(back_angle2) * wing
        tri = [head, self._world_to_screen(wx1, wy1), self._world_to_screen(wx2, wy2)]
        pygame.draw.polygon(self.screen, THEME["rover_arrow"], tri)
        pygame.draw.polygon(self.screen, THEME["rover_outline"], tri, 1)

    def _draw_lidar(
        self, state: RoverState, ranges: List[float], fov_deg: float
    ) -> None:
        num_rays = len(ranges)
        if num_rays == 0:
            return
        max_range = max(ranges) if ranges else 1.0
        if max_range < 1e-6:
            max_range = 1.0
        fov_rad = math.radians(fov_deg)
        start_angle = state.yaw - fov_rad / 2.0
        dtheta = fov_rad / max(num_rays - 1, 1)

        sx, sy = self._world_to_screen(state.x, state.y)
        close, mid, far = THEME["lidar_close"], THEME["lidar_mid"], THEME["lidar_far"]
        for i, r in enumerate(ranges):
            t = min(1.0, r / max_range)  # 0 = close, 1 = far
            # Interpolate close -> mid -> far
            if t < 0.5:
                u = t * 2.0
                color = (
                    int(close[0] + u * (mid[0] - close[0])),
                    int(close[1] + u * (mid[1] - close[1])),
                    int(close[2] + u * (mid[2] - close[2])),
                )
            else:
                u = (t - 0.5) * 2.0
                color = (
                    int(mid[0] + u * (far[0] - mid[0])),
                    int(mid[1] + u * (far[1] - mid[1])),
                    int(mid[2] + u * (far[2] - mid[2])),
                )
            angle = start_angle + i * dtheta
            ex = state.x + r * math.cos(angle)
            ey = state.y + r * math.sin(angle)
            ex_s, ey_s = self._world_to_screen(ex, ey)
            pygame.draw.line(self.screen, color, (sx, sy), (ex_s, ey_s), 2)

    def _draw_hud(self, dt: float, fps: float) -> None:
        pad = 10
        font = pygame.font.SysFont("monospace", 13)
        text = f"  dt={dt:.3f}s   FPS={fps:.1f}  "
        surf = font.render(text, True, THEME["hud_text"])
        r = surf.get_rect(topleft=(pad, pad))
        panel = r.inflate(pad, pad)
        pygame.draw.rect(self.screen, THEME["hud_bg"], panel)
        pygame.draw.rect(self.screen, THEME["hud_border"], panel, 1)
        self.screen.blit(surf, (panel.x + 4, panel.y + 4))

    def tick(self, target_fps: int) -> float:
        """Cap frame rate and return achieved FPS."""
        fps = self.clock.get_fps()
        self.clock.tick(target_fps)
        return fps

    def close(self) -> None:
        pygame.quit()

