"""Does a controller depend on the obstacle spawn clock?

    python experiments/spawn_clock.py

The v5 environment spawns an obstacle every 10 frames exactly. That fixed period
is a signal a policy can read. This compares every controller on the periodic
environment against a jittered one whose spawn period is uniform on 6 to 14
frames, which holds the mean period at 10 and so holds obstacle density
constant. A controller that reads the clock loses ground when the clock goes
away. A controller that reads only the road keeps its performance.

Writes results/spawn_clock.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack  # noqa: E402

from crashcourse.config import EnvConfig  # noqa: E402
from crashcourse.env import DriveEnv  # noqa: E402
from crashcourse.evaluation import evaluate, episode_seeds  # noqa: E402
from crashcourse.policies import GreedySafestLane, RandomPolicy, StayPolicy, TrainedPolicy  # noqa: E402
from crashcourse.sb3_compat import load_ppo  # noqa: E402

PERIODIC = EnvConfig()
JITTERED = EnvConfig(spawn_interval=6, spawn_jitter=8)
DEFAULT_MODEL = "results_probe/v5_smooth/seed99/best_model.zip"


def best_greedy(cfg: EnvConfig, seeds) -> tuple[float, int]:
    scores = [(evaluate(GreedySafestLane(cfg, danger=d), cfg, seeds, n_stack=4)
               .frames.mean(), d) for d in range(1, 13)]
    return max(scores)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--episodes", type=int, default=24)
    args = p.parse_args()

    seeds = episode_seeds(args.episodes)
    shell = VecFrameStack(DummyVecEnv([lambda: DriveEnv(PERIODIC)]), n_stack=4)
    model = load_ppo(REPO_ROOT / args.model, shell, device="auto")

    out = {"episodes": args.episodes, "rows": {}}
    for label, cfg in [("periodic", PERIODIC), ("jittered", JITTERED)]:
        greedy, danger = best_greedy(cfg, seeds)
        out["rows"][label] = {
            "ppo": float(evaluate(TrainedPolicy(model), cfg, seeds, n_stack=4).frames.mean()),
            "greedy": float(greedy),
            "greedy_danger": danger,
            "random": float(evaluate(RandomPolicy(cfg), cfg, seeds, n_stack=4).frames.mean()),
            "stay": float(evaluate(StayPolicy(cfg), cfg, seeds, n_stack=4).frames.mean()),
        }
        print(label, {k: round(v, 1) if isinstance(v, float) else v
                      for k, v in out["rows"][label].items()}, flush=True)

    a, b = out["rows"]["periodic"], out["rows"]["jittered"]
    for key in ("ppo", "greedy", "random"):
        out.setdefault("retained", {})[key] = b[key] / a[key]
        print(f"{key:8s} retains {b[key] / a[key]:.0%} of its periodic result")

    path = REPO_ROOT / "results" / "spawn_clock.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
