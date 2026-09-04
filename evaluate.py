"""Evaluate any set of policies on one environment configuration.

    python evaluate.py --config configs/v5_smooth.yaml --episodes 200
    python evaluate.py --config configs/v5_smooth.yaml --model results/v5_smooth/seed0/final_model.zip

Every policy sees the same list of held-out episode seeds, so the comparison is
paired. Results are written as CSV and JSON under results/eval/ and printed as a
Markdown table that can be pasted into the README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from crashcourse.config import RunConfig  # noqa: E402
from crashcourse.evaluation import (  # noqa: E402
    episode_seeds, evaluate, evaluate_model,
)
from crashcourse.policies import (  # noqa: E402
    GreedySafestLane, HoverExploit, RandomPolicy, StayPolicy,
)


def collect(cfg: RunConfig, seeds, model_paths: list[Path], greedy_danger: int):
    env_cfg, n_stack = cfg.env, cfg.train.n_stack
    reports = [
        evaluate(RandomPolicy(env_cfg, seed=0), env_cfg, seeds, n_stack),
        evaluate(StayPolicy(env_cfg), env_cfg, seeds, n_stack),
        evaluate(GreedySafestLane(env_cfg, danger=greedy_danger), env_cfg, seeds, n_stack),
        evaluate(HoverExploit(env_cfg), env_cfg, seeds, n_stack),
    ]
    if model_paths:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

        from crashcourse.env import DriveEnv
        from crashcourse.sb3_compat import load_ppo

        venv = DummyVecEnv([lambda: DriveEnv(env_cfg)])
        if n_stack > 1:
            venv = VecFrameStack(venv, n_stack=n_stack)
        for path in model_paths:
            model = load_ppo(path, venv, device="auto")
            label = f"ppo:{path.parent.name}" if path.parent.name.startswith("seed") else "ppo"
            reports.append(evaluate_model(model, env_cfg, seeds, n_stack, name=label))
        venv.close()
    return reports


def sweep_greedy(cfg: RunConfig, seeds) -> int:
    """Pick the heuristic threshold fairly, by sweeping it on the same seeds."""
    best, best_score = 1, -1.0
    for danger in (1, 2, 3, 4, 6, 8, 12, 20):
        rep = evaluate(GreedySafestLane(cfg.env, danger=danger), cfg.env, seeds, cfg.train.n_stack)
        score = float(rep.frames.mean())
        if score > best_score:
            best, best_score = danger, score
    return best


def as_markdown(rows: list[dict]) -> str:
    head = ("| Policy | Frames (mean) | 95% CI | Median | Success | Return |\n"
            "|---|---|---|---|---|---|\n")
    body = "".join(
        "| {policy} | {frames_mean:.0f} | [{frames_ci_lo:.0f}, {frames_ci_hi:.0f}] | "
        "{frames_median:.0f} | {success_rate:.0%} | {return_mean:.0f} |\n".format(**r)
        for r in rows
    )
    return head + body


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/v5_smooth.yaml")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--model", action="append", default=[], help="repeatable")
    p.add_argument("--greedy-danger", type=int, default=None, help="default: swept")
    p.add_argument("--out", default="results/eval")
    args = p.parse_args()

    cfg = RunConfig.from_yaml(Path(args.config))
    seeds = episode_seeds(args.episodes)
    danger = args.greedy_danger if args.greedy_danger is not None else sweep_greedy(cfg, seeds[:50])
    models = [Path(m) for m in args.model]

    reports = collect(cfg, seeds, models, danger)
    rows = [r.summary() for r in reports]
    for row in rows:
        row["config"] = cfg.name
    rows.sort(key=lambda r: -r["frames_mean"])

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg.name}.json").write_text(
        json.dumps({"config": cfg.name, "episodes": args.episodes,
                    "greedy_danger": danger, "rows": rows}, indent=2), encoding="utf-8")
    (out_dir / f"{cfg.name}.md").write_text(as_markdown(rows), encoding="utf-8")

    print(f"\nconfig {cfg.name}   {args.episodes} held-out episodes   greedy threshold {danger}\n")
    print(as_markdown(rows))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
