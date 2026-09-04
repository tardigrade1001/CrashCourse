"""Policies that share one interface, so every baseline is evaluated the same way.

A policy maps a stacked observation to an action. Simple policies read only the
most recent frame of the stack. The trained agent reads the whole stack.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from crashcourse.config import EnvConfig


class Policy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, stacked_obs: np.ndarray) -> int: ...


def latest_frame(stacked_obs: np.ndarray, obs_size: int) -> np.ndarray:
    """The newest frame of a stack. VecFrameStack appends the newest at the end."""
    return stacked_obs[-obs_size:]


def occupancy_grid(obs: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    """Reshape one frame back into (lanes, depth) occupancy."""
    return obs[: cfg.lanes * cfg.obs_depth].reshape(cfg.lanes, cfg.obs_depth)


class RandomPolicy:
    """Uniform over lanes. The floor any learned policy must clear."""

    name = "random"

    def __init__(self, cfg: EnvConfig, seed: int = 0):
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, stacked_obs: np.ndarray) -> int:
        return int(self._rng.integers(0, self.cfg.lanes))


class StayPolicy:
    """Hold the starting lane forever. Separates dodging skill from doing nothing."""

    name = "stay"

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._lane: int | None = None

    def reset(self) -> None:
        self._lane = None

    def act(self, stacked_obs: np.ndarray) -> int:
        if self._lane is None:
            frame = latest_frame(stacked_obs, self.cfg.obs_size)
            self._lane = int(round(frame[-1] * (self.cfg.lanes - 1)))
        return self._lane


class GreedySafestLane:
    """Hand-written controller: hold the current lane while it is clear.

    Clearance is the index of the nearest occupied segment in a lane, so a larger
    value means more room. The policy commits to the current lane until its
    clearance drops below ``danger``, then steps to the adjacent lane with the
    most room. Only adjacent lanes are considered because a smooth transition
    passes through the lane in between.
    """

    name = "greedy"

    def __init__(self, cfg: EnvConfig, danger: int = 4):
        self.cfg = cfg
        self.danger = danger
        self._lane = 0

    def reset(self) -> None:
        self._lane = 0

    def act(self, stacked_obs: np.ndarray) -> int:
        cfg = self.cfg
        frame = latest_frame(stacked_obs, cfg.obs_size)
        grid = occupancy_grid(frame, cfg)
        if cfg.position_feature == "lane":
            self._lane = int(round(frame[-1] * (cfg.lanes - 1)))
        elif cfg.position_feature == "continuous":
            span = cfg.screen_width - cfg.car_size
            self._lane = int(round((frame[-1] * span) / cfg.lane_width))
        self._lane = max(0, min(cfg.lanes - 1, self._lane))

        clearance = [_clearance(grid[i], cfg.obs_depth) for i in range(cfg.lanes)]
        if clearance[self._lane] >= self.danger:
            return self._lane

        candidates = [i for i in (self._lane - 1, self._lane + 1) if 0 <= i < cfg.lanes]
        candidates.append(self._lane)
        best = max(candidates, key=lambda i: (clearance[i], _is_centre(i, cfg.lanes)))
        return best


def _clearance(lane_column: np.ndarray, depth: int) -> int:
    hits = np.flatnonzero(lane_column > 0.5)
    return int(hits[0]) if hits.size else depth


def _is_centre(lane: int, lanes: int) -> int:
    return 1 if 0 < lane < lanes - 1 else 0


class TrainedPolicy:
    """Wraps a Stable-Baselines3 model."""

    name = "ppo"

    def __init__(self, model, deterministic: bool = True, name: str = "ppo"):
        self.model = model
        self.deterministic = deterministic
        self.name = name

    def reset(self) -> None:
        pass

    def act(self, stacked_obs: np.ndarray) -> int:
        action, _ = self.model.predict(stacked_obs, deterministic=self.deterministic)
        return int(action)


class HoverExploit:
    """Alternates between two adjacent lanes and never steers anywhere else.

    Under a lateral collision tolerance below 1.0 the unsafe bands around two
    adjacent lane centres do not meet, and this cycle parks the car in the gap
    between them. It survives indefinitely without reading the observation at
    all, which makes it a direct test for that class of defect. Under the
    corrected tolerance it performs like a random policy.
    """

    name = "hover"

    def __init__(self, cfg: EnvConfig, lane_a: int = 0, lane_b: int = 1):
        self.cfg = cfg
        self.lane_a = lane_a
        self.lane_b = lane_b
        self._k = 0

    def reset(self) -> None:
        self._k = 0

    def act(self, stacked_obs: np.ndarray) -> int:
        self._k += 1
        return self.lane_a if self._k % 2 else self.lane_b
