"""Watch a policy play, or record it to a GIF.

    python watch.py --config configs/v5_smooth.yaml --model results/v5_smooth/seed0/final_model.zip
    python watch.py --config configs/v5_smooth.yaml --policy hover --episodes 1
    python watch.py --config configs/v5_smooth.yaml --policy greedy --gif docs/greedy.gif

The environment renders itself inside step() when render_mode is "human", so
nothing here calls render() again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from crashcourse.config import RunConfig  # noqa: E402
from crashcourse.env import DriveEnv  # noqa: E402
from crashcourse.evaluation import FrameStack  # noqa: E402
from crashcourse.policies import (  # noqa: E402
    GreedySafestLane, HoverExploit, RandomPolicy, StayPolicy, TrainedPolicy,
)

SIMPLE = {"random": RandomPolicy, "stay": StayPolicy, "greedy": GreedySafestLane,
          "hover": HoverExploit}


def build_policy(name: str, model_path: str | None, cfg: RunConfig):
    if model_path:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

        from crashcourse.sb3_compat import load_ppo
        venv = DummyVecEnv([lambda: DriveEnv(cfg.env)])
        if cfg.train.n_stack > 1:
            venv = VecFrameStack(venv, n_stack=cfg.train.n_stack)
        return TrainedPolicy(load_ppo(Path(model_path), venv, device="auto"))
    if name not in SIMPLE:
        raise SystemExit(f"unknown policy {name!r}, choose from {sorted(SIMPLE)} or pass --model")
    return SIMPLE[name](cfg.env)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/v5_smooth.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--policy", default="greedy", choices=sorted(SIMPLE))
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--gif", default=None, help="write a GIF here rather than opening a window")
    p.add_argument("--gif-every", type=int, default=2, help="keep every Nth frame")
    args = p.parse_args()

    cfg = RunConfig.from_yaml(Path(args.config))
    policy = build_policy(args.policy, args.model, cfg)
    mode = "rgb_array" if args.gif else "human"
    env = DriveEnv(cfg.env, render_mode=mode)
    stack = FrameStack(cfg.env.obs_size, cfg.train.n_stack)
    frames_out: list[np.ndarray] = []

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            stacked = stack.reset(obs)
            policy.reset()
            total, step = 0.0, 0
            while True:
                obs, reward, terminated, truncated, info = env.step(policy.act(stacked))
                stacked = stack.push(obs)
                total += reward
                step += 1
                if args.gif and step % args.gif_every == 0:
                    frames_out.append(env.render())
                if terminated or truncated:
                    break
            outcome = "survived" if truncated else "crashed"
            print(f"episode {ep + 1}: {info['frames']} frames, return {total:.0f}, {outcome}")
    finally:
        env.close()

    if args.gif and frames_out:
        from PIL import Image
        out = Path(args.gif)
        out.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(f) for f in frames_out]
        images[0].save(out, save_all=True, append_images=images[1:], duration=33, loop=0)
        print(f"wrote {out} ({len(images)} frames)")


if __name__ == "__main__":
    main()
