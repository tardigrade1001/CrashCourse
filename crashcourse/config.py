"""Configuration objects for the CrashCourse environment and training runs.

Every experiment in this repository is defined by a YAML file under ``configs/``.
Nothing that affects a result is hard coded in a script.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EnvConfig:
    """Everything that defines the task itself."""

    lanes: int = 4
    screen_width: int = 400
    screen_height: int = 600
    car_size: int = 50

    # Observation
    obs_depth: int = 20           # number of forward segments per lane
    segment_px: int = 50          # depth of one segment in pixels
    position_feature: str = "lane"  # "lane" | "continuous" | "none"

    # Dynamics
    lane_transition: str = "smooth"  # "smooth" | "instant"
    lerp_alpha: float = 0.15
    obstacle_speed: int = 10
    spawn_interval: int = 10
    # Frames of uniform jitter added to the spawn period. At 0 the period is
    # exactly spawn_interval, which lets a controller deciding at a commensurate
    # rate sample a stroboscopically stable world. See docs on decision rate.
    spawn_jitter: int = 0
    max_frames: int = 3000

    # Collision
    # Lateral overlap threshold as a fraction of car width. Two cars of equal
    # width touch when their centres are one width apart, so 1.0 is the
    # physically correct value. The original 0.8 left a 20px corridor between
    # adjacent lanes that no obstacle could occupy. See configs/ablations.
    collision_x_tol: float = 1.0

    # Reward terms
    reward_survive: float = 1.0
    reward_center_bonus: float = 0.1
    reward_lane_change: float = -0.1
    reward_crash: float = -100.0

    # Reported lane index. "centered" measures from the car centre.
    # "legacy" reproduces the pre-fix rounding that led the car body by 25px.
    lane_readout: str = "centered"

    def __post_init__(self) -> None:
        if self.lane_transition not in ("smooth", "instant"):
            raise ValueError(f"lane_transition must be smooth or instant, got {self.lane_transition!r}")
        if self.position_feature not in ("lane", "continuous", "none"):
            raise ValueError(f"position_feature invalid: {self.position_feature!r}")
        if self.lane_readout not in ("centered", "legacy"):
            raise ValueError(f"lane_readout invalid: {self.lane_readout!r}")

    @property
    def lane_width(self) -> int:
        return self.screen_width // self.lanes

    @property
    def obs_size(self) -> int:
        extra = 0 if self.position_feature == "none" else 1
        return self.lanes * self.obs_depth + extra


@dataclass
class TrainConfig:
    """Everything that defines the learning run."""

    algo: str = "PPO"
    policy: str = "MlpPolicy"
    total_timesteps: int = 1_000_000
    n_envs: int = 8
    n_stack: int = 4
    net_arch: list[int] = field(default_factory=lambda: [1024, 1024, 1024])
    n_steps: int = 16384
    batch_size: int = 2048
    learning_rate: float = 5e-5
    ent_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: str = "auto"
    eval_freq_steps: int = 100_000   # counted in environment transitions
    n_eval_episodes: int = 10


@dataclass
class RunConfig:
    name: str = "unnamed"
    description: str = ""
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        path = Path(path)
        if not path.exists():
            candidate = REPO_ROOT / "configs" / path.name
            if candidate.exists():
                path = candidate
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        env_raw = dict(raw.get("env", {}))
        train_raw = dict(raw.get("train", {}))
        _reject_unknown(EnvConfig, env_raw, "env")
        _reject_unknown(TrainConfig, train_raw, "train")
        return cls(
            name=raw.get("name", "unnamed"),
            description=raw.get("description", ""),
            env=EnvConfig(**env_raw),
            train=TrainConfig(**train_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reject_unknown(dc, raw: dict, section: str) -> None:
    known = {f.name for f in fields(dc)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown keys in '{section}': {sorted(unknown)}")
