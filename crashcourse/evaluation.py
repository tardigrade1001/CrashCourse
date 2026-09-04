"""Evaluation harness.

Every policy is run on the same list of episode seeds, so comparisons are
paired. Frame stacking is reproduced here exactly as Stable-Baselines3'
VecFrameStack does it, which lets a trained agent and a hand-written baseline
share one code path.

Three quantities are reported and never conflated:

    frames  survival length of an episode, capped at EnvConfig.max_frames
    return  cumulative PPO reward, which depends on the reward configuration
    success episode reached the cap without a collision
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from crashcourse.config import EnvConfig
from crashcourse.env import DriveEnv


class FrameStack:
    """Matches VecFrameStack: zero filled on reset, newest frame at the end."""

    def __init__(self, obs_size: int, n_stack: int):
        self.obs_size = obs_size
        self.n_stack = max(1, n_stack)
        self.buffer = np.zeros(self.obs_size * self.n_stack, dtype=np.float32)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        self.buffer[:] = 0.0
        self.buffer[-self.obs_size:] = obs
        return self.buffer.copy()

    def push(self, obs: np.ndarray) -> np.ndarray:
        if self.n_stack > 1:
            self.buffer = np.roll(self.buffer, -self.obs_size)
        self.buffer[-self.obs_size:] = obs
        return self.buffer.copy()


@dataclass
class EpisodeResult:
    seed: int
    frames: int
    ret: float
    success: bool


@dataclass
class EvalReport:
    policy: str
    episodes: list[EpisodeResult]

    @property
    def frames(self) -> np.ndarray:
        return np.array([e.frames for e in self.episodes], dtype=float)

    @property
    def returns(self) -> np.ndarray:
        return np.array([e.ret for e in self.episodes], dtype=float)

    @property
    def success_rate(self) -> float:
        return float(np.mean([e.success for e in self.episodes]))

    def summary(self, n_boot: int = 10_000, seed: int = 0) -> dict:
        f, r = self.frames, self.returns
        lo, hi = bootstrap_ci(f, np.mean, n_boot=n_boot, seed=seed)
        return {
            "policy": self.policy,
            "n_episodes": len(self.episodes),
            "frames_mean": float(f.mean()),
            "frames_ci_lo": lo,
            "frames_ci_hi": hi,
            "frames_sd": float(f.std(ddof=1)) if len(f) > 1 else 0.0,
            "frames_median": float(np.median(f)),
            "frames_min": int(f.min()),
            "frames_max": int(f.max()),
            "return_mean": float(r.mean()),
            "return_sd": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
            "success_rate": self.success_rate,
        }


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    draws = stat(values[idx], axis=1)
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def episode_seeds(n_episodes: int, base: int = 10_000) -> list[int]:
    """A fixed, documented seed list. Held out from every training run."""
    return [base + i for i in range(n_episodes)]


def evaluate(policy, cfg: EnvConfig, seeds: Sequence[int], n_stack: int = 4) -> EvalReport:
    env = DriveEnv(cfg)
    stack = FrameStack(cfg.obs_size, n_stack)
    results: list[EpisodeResult] = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            stacked = stack.reset(obs)
            policy.reset()
            total, frames = 0.0, 0
            while True:
                action = policy.act(stacked)
                obs, reward, terminated, truncated, info = env.step(action)
                stacked = stack.push(obs)
                total += reward
                frames = info["frames"]
                if terminated or truncated:
                    break
            results.append(EpisodeResult(int(seed), frames, total, bool(truncated)))
    finally:
        env.close()
    return EvalReport(policy.name, results)


def evaluate_model(model, cfg: EnvConfig, seeds: Sequence[int], n_stack: int = 4,
                   n_parallel: int = 25, name: str = "ppo") -> EvalReport:
    """Same protocol as evaluate(), with one forward pass covering many episodes.

    Each parallel slot owns a seeded environment, so results are identical to
    the sequential path and only the batching differs.
    """
    obs_size = cfg.obs_size
    results: list[EpisodeResult] = []
    seeds = list(seeds)
    for start in range(0, len(seeds), n_parallel):
        chunk = seeds[start:start + n_parallel]
        envs = [DriveEnv(cfg) for _ in chunk]
        buf = np.zeros((len(chunk), obs_size * max(1, n_stack)), dtype=np.float32)
        for i, (env, seed) in enumerate(zip(envs, chunk)):
            obs, _ = env.reset(seed=int(seed))
            buf[i, -obs_size:] = obs
        live = np.ones(len(chunk), dtype=bool)
        rets = np.zeros(len(chunk))
        frames = np.zeros(len(chunk), dtype=int)
        success = np.zeros(len(chunk), dtype=bool)
        for _ in range(cfg.max_frames):
            if not live.any():
                break
            actions, _ = model.predict(buf, deterministic=True)
            for i, env in enumerate(envs):
                if not live[i]:
                    continue
                obs, reward, terminated, truncated, info = env.step(int(actions[i]))
                if n_stack > 1:
                    buf[i] = np.roll(buf[i], -obs_size)
                buf[i, -obs_size:] = obs
                rets[i] += reward
                frames[i] = info["frames"]
                if terminated or truncated:
                    live[i] = False
                    success[i] = truncated
        for env in envs:
            env.close()
        results.extend(
            EpisodeResult(int(s), int(f), float(r), bool(ok))
            for s, f, r, ok in zip(chunk, frames, rets, success)
        )
    return EvalReport(name, results)
