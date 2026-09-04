"""Tests for the environment contract and for the two defects found in V5."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from crashcourse.config import EnvConfig, RunConfig  # noqa: E402
from crashcourse.env import DriveEnv  # noqa: E402
from crashcourse.evaluation import episode_seeds, evaluate  # noqa: E402
from crashcourse.policies import HoverExploit, RandomPolicy  # noqa: E402


def rollout(env: DriveEnv, actions):
    obs, _ = env.reset(seed=123)
    trace = [obs.copy()]
    for action in actions:
        obs, reward, terminated, truncated, _ = env.step(action)
        trace.append((obs.copy(), reward))
        if terminated or truncated:
            break
    return trace


def test_same_seed_reproduces_episode():
    actions = [i % 4 for i in range(400)]
    a = rollout(DriveEnv(), actions)
    b = rollout(DriveEnv(), actions)
    assert len(a) == len(b)
    for x, y in zip(a[1:], b[1:]):
        assert np.array_equal(x[0], y[0])
        assert x[1] == y[1]


def test_different_seeds_differ():
    env_a, env_b = DriveEnv(), DriveEnv()
    obs_a, _ = env_a.reset(seed=1)
    obs_b, _ = env_b.reset(seed=2)
    traces = []
    for env in (env_a, env_b):
        seq = []
        for k in range(300):
            obs, *_ = env.step(k % 4)
            seq.append(obs)
        traces.append(np.stack(seq))
    assert not np.array_equal(traces[0], traces[1])


def test_observation_always_inside_space():
    env = DriveEnv()
    obs, _ = env.reset(seed=5)
    assert env.observation_space.contains(obs)
    for k in range(1000):
        obs, _, terminated, truncated, _ = env.step(k % 4)
        assert env.observation_space.contains(obs), f"escaped at step {k}"
        if terminated or truncated:
            obs, _ = env.reset(seed=k)


def test_lane_stays_valid():
    env = DriveEnv()
    env.reset(seed=7)
    for k in range(1000):
        _, _, terminated, truncated, info = env.step(np.random.randint(0, 4))
        assert 0 <= info["lane"] < env.cfg.lanes
        assert 0 <= env.car_x <= env.cfg.screen_width - env.cfg.car_size
        if terminated or truncated:
            env.reset(seed=k)


def test_truncates_at_max_frames():
    cfg = EnvConfig(max_frames=200, spawn_interval=10_000)  # no obstacles
    env = DriveEnv(cfg)
    env.reset(seed=1)
    for _ in range(199):
        _, _, terminated, truncated, _ = env.step(0)
        assert not (terminated or truncated)
    _, _, terminated, truncated, info = env.step(0)
    assert truncated and not terminated
    assert info["frames"] == 200


def test_collision_terminates_and_penalises():
    cfg = EnvConfig()
    env = DriveEnv(cfg)
    env.reset(seed=3)
    env.obstacles = [(env.car_lane, float(env.car_y), (0, 0))]
    _, reward, terminated, truncated, _ = env.step(env.car_lane)
    assert terminated and not truncated
    assert reward == cfg.reward_crash


def test_rgb_array_render_returns_frame():
    env = DriveEnv(render_mode="rgb_array")
    env.reset(seed=1)
    frame = env.render()
    assert frame.shape == (env.cfg.screen_height, env.cfg.screen_width, 3)
    assert frame.dtype == np.uint8


def test_declared_render_modes_all_work():
    for mode in DriveEnv.metadata["render_modes"]:
        if mode == "human":
            continue  # needs a display
        env = DriveEnv(render_mode=mode)
        env.reset(seed=1)
        assert env.render() is not None


def test_rejects_unknown_render_mode():
    with pytest.raises(ValueError):
        DriveEnv(render_mode="ascii")


def test_sb3_env_checker():
    from stable_baselines3.common.env_checker import check_env
    check_env(DriveEnv(), warn=True)


# --- regression tests for the two defects found in V5 -----------------------

def test_no_safe_corridor_between_lanes():
    """A policy that ignores the observation must not survive.

    With a lateral tolerance below 1.0 the unsafe bands around adjacent lanes do
    not meet, and alternating between two lanes parks the car in the gap.
    """
    cfg = EnvConfig()
    report = evaluate(HoverExploit(cfg), cfg, episode_seeds(10), n_stack=4)
    assert report.success_rate == 0.0, "hover policy survived, the lane gap is back"


def test_permissive_tolerance_reproduces_the_exploit():
    cfg = EnvConfig(collision_x_tol=0.8)
    report = evaluate(HoverExploit(cfg), cfg, episode_seeds(5), n_stack=4)
    assert report.success_rate == 1.0


def test_lane_readout_flips_at_the_midpoint():
    """The reported lane must match where the car body actually is."""
    env = DriveEnv(EnvConfig(lane_readout="centered"))
    lane_width, size = env.cfg.lane_width, env.cfg.car_size
    midpoint_x = lane_width - size / 2.0  # centre sits exactly between lanes 0 and 1
    assert env._x_to_lane(midpoint_x - 1) == 0
    assert env._x_to_lane(midpoint_x + 1) == 1

    legacy = DriveEnv(EnvConfig(lane_readout="legacy"))
    assert legacy._x_to_lane(midpoint_x - 1) == 1, "legacy readout led the car body"


def test_random_baseline_is_weak():
    cfg = EnvConfig()
    report = evaluate(RandomPolicy(cfg, seed=0), cfg, episode_seeds(20), n_stack=4)
    assert report.success_rate == 0.0
    assert report.frames.mean() < 500


def test_every_shipped_config_loads():
    paths = list((REPO_ROOT / "configs").rglob("*.yaml"))
    assert paths
    for path in paths:
        cfg = RunConfig.from_yaml(path)
        env = DriveEnv(cfg.env)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (cfg.env.obs_size,)


def test_config_rejects_typos():
    from crashcourse.config import RunConfig
    with pytest.raises(ValueError):
        RunConfig.from_dict({"env": {"lane_transtion": "smooth"}})


def test_spawn_jitter_breaks_the_periodic_clock():
    """A fixed spawn period lets a commensurate decision rate alias with it.

    With jitter the spawn period varies, so no decision rate samples the world
    in a fixed phase. This guards the environment against the aliasing that the
    decision-rate sweep uncovered.
    """
    from crashcourse.policies import GreedySafestLane, HeldPolicy

    periodic = EnvConfig()
    jittered = EnvConfig(spawn_interval=6, spawn_jitter=8)
    seeds = episode_seeds(12)

    def at(cfg, every):
        policy = HeldPolicy(GreedySafestLane(cfg, danger=2), every)
        return evaluate(policy, cfg, seeds, n_stack=4).frames.mean()

    aliased = at(periodic, 5)
    neighbour = max(at(periodic, 4), at(periodic, 6))
    assert aliased > 4 * neighbour, "the periodic spawn clock stopped aliasing"

    spread = [at(jittered, e) for e in (4, 5, 6)]
    assert max(spread) < 4 * min(spread), "jitter left an aliasing spike behind"


def test_spawn_jitter_defaults_to_the_published_behaviour():
    cfg = EnvConfig()
    assert cfg.spawn_jitter == 0
    env = DriveEnv(cfg)
    env.reset(seed=0)
    spawns = []
    for _ in range(200):
        before = len(env.obstacles)
        env.step(0)
        if len(env.obstacles) > before:
            spawns.append(env.frame_count)
    env.close()
    gaps = {b - a for a, b in zip(spawns, spawns[1:])}
    assert gaps == {cfg.spawn_interval}, f"spawn period drifted: {gaps}"
