"""A four-lane obstacle-avoidance environment.

The agent occupies one of four lanes on a scrolling road and chooses a target
lane every step. Obstacles spawn at the top at a fixed interval and descend at a
fixed speed. An episode ends on contact or at ``max_frames``.

This is a discrete decision problem under a forward-looking occupancy
observation. There is no steering model and no vehicle physics.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from crashcourse.config import EnvConfig

CAR_COLORS = [
    (220, 50, 50), (50, 100, 220), (220, 180, 50), (50, 220, 100),
    (220, 100, 200), (150, 50, 150), (220, 100, 50), (180, 180, 180),
]
ROAD = (45, 45, 45)
WHITE = (255, 255, 255)


class DriveEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.cfg.lanes)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.cfg.obs_size,), dtype=np.float32
        )

        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None

        # Populated by reset(). Declared here so the object is never half built.
        self.car_lane = 0
        self.car_x = 0.0
        self.car_y = self.cfg.screen_height - 100
        self.obstacles: list[tuple[int, float, tuple[int, int]]] = []
        self.obstacle_colors: dict[tuple[int, int], tuple[int, int, int]] = {}
        self.frame_count = 0
        self._next_spawn = self.cfg.spawn_interval
        self.episode_return = 0.0

    # ------------------------------------------------------------------ core

    def _lane_to_x(self, lane: int) -> float:
        return lane * self.cfg.lane_width + (self.cfg.lane_width - self.cfg.car_size) / 2.0

    def _x_to_lane(self, x: float) -> int:
        """Lane index implied by a pixel position.

        "centered" measures from the car centre, so the index flips exactly at
        the midpoint between two lane centres. "legacy" reproduces the original
        rounding, which flipped the index while the car body was still 25px
        short of that midpoint.
        """
        if self.cfg.lane_readout == "legacy":
            idx = round(x / self.cfg.lane_width)
        else:
            centre = x + self.cfg.car_size / 2.0
            idx = round(centre / self.cfg.lane_width - 0.5)
        return int(max(0, min(self.cfg.lanes - 1, idx)))

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.car_lane = int(self.np_random.integers(0, self.cfg.lanes))
        self.car_x = self._lane_to_x(self.car_lane)
        self.car_y = self.cfg.screen_height - 100
        self.obstacles = []
        self.obstacle_colors = {}
        self.frame_count = 0
        self._next_spawn = self.cfg.spawn_interval
        self.episode_return = 0.0
        if self.render_mode == "human":
            self._ensure_display()
            self.render()
        return self._get_obs(), self._info()

    def _get_obs(self) -> np.ndarray:
        cfg = self.cfg
        grid = np.zeros((cfg.lanes, cfg.obs_depth), dtype=np.float32)
        reach = cfg.obs_depth * cfg.segment_px
        for lane, y, _ in self.obstacles:
            dist = self.car_y - (y + cfg.car_size)
            if 0.0 <= dist < reach:
                grid[lane, int(dist // cfg.segment_px)] = 1.0

        parts = [grid.reshape(-1)]
        if cfg.position_feature == "lane":
            parts.append(np.array([self.car_lane / (cfg.lanes - 1)], dtype=np.float32))
        elif cfg.position_feature == "continuous":
            span = cfg.screen_width - cfg.car_size
            parts.append(np.array([self.car_x / span], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def step(self, action: int):
        cfg = self.cfg
        action = int(action)
        prev_lane = self.car_lane

        target_x = self._lane_to_x(action)
        if cfg.lane_transition == "instant":
            self.car_x = target_x
        else:
            self.car_x += (target_x - self.car_x) * cfg.lerp_alpha
        self.car_lane = self._x_to_lane(self.car_x)

        self.obstacles = [
            (lane, y + cfg.obstacle_speed, oid) for lane, y, oid in self.obstacles
            if y + cfg.obstacle_speed < cfg.screen_height + 100
        ]

        self.frame_count += 1
        if self.frame_count >= self._next_spawn:
            self._next_spawn = self.frame_count + cfg.spawn_interval + (
                int(self.np_random.integers(0, cfg.spawn_jitter + 1))
                if cfg.spawn_jitter else 0)
            lane = int(self.np_random.integers(0, cfg.lanes))
            oid = (lane, self.frame_count)
            self.obstacles.append((lane, -100.0, oid))
            colour_idx = int(self.np_random.integers(0, len(CAR_COLORS)))
            self.obstacle_colors[oid] = CAR_COLORS[colour_idx]

        reward = cfg.reward_survive
        if 0 < self.car_lane < cfg.lanes - 1:
            reward += cfg.reward_center_bonus
        if self.car_lane != prev_lane:
            reward += cfg.reward_lane_change

        terminated = self._collides()
        if terminated:
            reward = cfg.reward_crash

        truncated = (not terminated) and self.frame_count >= cfg.max_frames
        self.episode_return += reward

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, self._info()

    def _collides(self) -> bool:
        cfg = self.cfg
        size = cfg.car_size
        tol = size * cfg.collision_x_tol
        for lane, y, _ in self.obstacles:
            if abs(self.car_x - self._lane_to_x(lane)) < tol:
                if self.car_y < y + size and self.car_y + size > y:
                    return True
        return False

    def _info(self) -> dict[str, Any]:
        return {"frames": self.frame_count, "return": self.episode_return, "lane": self.car_lane}

    # ----------------------------------------------------------------- render

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            surface = pygame.Surface((self.cfg.screen_width, self.cfg.screen_height))
            self._draw(surface, with_text=False)
            return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

        self._ensure_display()
        if self._screen is None:
            return None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None
        if self._screen is None:
            return None
        self._draw(self._screen, with_text=True)
        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(self.metadata["render_fps"])
        return None

    def _draw(self, surface: pygame.Surface, with_text: bool) -> None:
        cfg = self.cfg
        surface.fill(ROAD)
        offset = self.frame_count * cfg.obstacle_speed
        for lane in range(1, cfg.lanes):
            x = lane * cfg.lane_width
            y = offset % 30
            while y < cfg.screen_height:
                pygame.draw.line(surface, (255, 255, 200), (x, y), (x, y + 20), 2)
                y += 30

        for lane, oy, oid in self.obstacles:
            _draw_car(surface, self._lane_to_x(lane), oy, cfg.car_size,
                      self.obstacle_colors.get(oid, (200, 50, 50)))
        _draw_car(surface, self.car_x, self.car_y, cfg.car_size, WHITE)

        if with_text and self._font is not None:
            label = self._font.render(f"frames {self.frame_count}", True, WHITE)
            surface.blit(label, (10, 10))

    def _ensure_display(self) -> None:
        if self._screen is not None:
            return
        if not pygame.get_init():
            pygame.init()
        self._screen = pygame.display.set_mode((self.cfg.screen_width, self.cfg.screen_height))
        pygame.display.set_caption("CrashCourse")
        self._clock = pygame.time.Clock()
        if pygame.font.get_init():
            self._font = pygame.font.SysFont(None, 28)

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
            self._screen = None
            self._clock = None
            self._font = None


def _draw_car(surface, x, y, size, colour) -> None:
    x, y = int(x), int(y)
    pygame.draw.rect(surface, colour, (x, y, size, size))
    pygame.draw.rect(surface, (100, 150, 200), (x + 4, y + 5, size - 8, 8))
    for cx in (x + 6, x + size - 6):
        pygame.draw.circle(surface, (255, 255, 150), (cx, y + 3), 2)
        pygame.draw.circle(surface, (255, 100, 100), (cx, y + size - 3), 2)
