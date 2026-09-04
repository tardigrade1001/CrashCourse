"""Render the archived V5 agent on the archived V5 environment.

    python experiments/archive_v5_gif.py --out docs/demo.gif

This targets ``ai_drive_game_env.py`` and ``models/ppo_drive_final.zip`` as they
were published, so the recording matches the run those files produce. Pygame is
driven through the dummy video backend, which keeps the surface offscreen and
lets the frames be read directly.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from crashcourse.config import EnvConfig  # noqa: E402
from crashcourse.env import DriveEnv as CurrentEnv  # noqa: E402
from crashcourse.evaluation import FrameStack  # noqa: E402
from crashcourse.sb3_compat import load_ppo  # noqa: E402

from ai_drive_game_env import DriveEnv as ArchiveEnv  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/ppo_drive_final.zip")
    p.add_argument("--out", default="docs/demo.gif")
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--frames", type=int, default=900, help="environment steps to record")
    p.add_argument("--every", type=int, default=2, help="keep every Nth frame")
    p.add_argument("--scale", type=float, default=0.75)
    args = p.parse_args()

    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    shell = VecFrameStack(DummyVecEnv([lambda: CurrentEnv(EnvConfig())]), n_stack=4)
    model = load_ppo(REPO_ROOT / args.model, shell, device="auto")

    random.seed(args.seed)
    np.random.seed(args.seed)
    env = ArchiveEnv(render_mode="human")
    stack = FrameStack(81, 4)
    obs, _ = env.reset()
    stacked = stack.reset(obs)

    frames: list[np.ndarray] = []
    for step in range(args.frames):
        action, _ = model.predict(stacked, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        stacked = stack.push(obs)
        if step % args.every == 0:
            surface = pygame.display.get_surface()
            if surface is not None:
                frames.append(np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)))
        if terminated or truncated:
            break

    print(f"recorded {len(frames)} frames over {info['score']} environment steps")
    env.close()

    from PIL import Image
    images = [Image.fromarray(f) for f in frames]
    if args.scale != 1.0:
        size = (int(images[0].width * args.scale), int(images[0].height * args.scale))
        images = [im.resize(size, Image.LANCZOS) for im in images]
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=33, loop=0, optimize=True)
    print("wrote", out, f"({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
