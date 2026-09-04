"""Run the whole study: every configuration, every seed, then evaluate them all.

    python experiments/run_ablation.py --seeds 3
    python experiments/run_ablation.py --seeds 3 --only v5_smooth v3_instant
    python experiments/run_ablation.py --evaluate-only

Each training run is a separate process, so one crash cannot take the study with
it. Finished runs are skipped, which makes the script resumable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from crashcourse.config import RunConfig  # noqa: E402
from crashcourse.evaluation import episode_seeds, evaluate_model  # noqa: E402

PYTHON = sys.executable


def all_configs(only: list[str] | None) -> list[Path]:
    paths = [REPO_ROOT / "configs" / "v5_smooth.yaml", REPO_ROOT / "configs" / "v3_instant.yaml"]
    paths += sorted((REPO_ROOT / "configs" / "ablations").glob("*.yaml"))
    if only:
        wanted = set(only)
        paths = [p for p in paths if RunConfig.from_yaml(p).name in wanted]
    return paths


def train_one(config: Path, seed: int, out_root: Path, timesteps: int | None) -> bool:
    cfg = RunConfig.from_yaml(config)
    done = out_root / cfg.name / f"seed{seed}" / "final_model.zip"
    if done.exists():
        print(f"  skip {cfg.name} seed{seed} (already done)")
        return True
    cmd = [PYTHON, "-u", str(REPO_ROOT / "train.py"), "--config", str(config),
           "--seed", str(seed), "--out", out_root.name]
    if timesteps:
        cmd += ["--timesteps", str(timesteps)]
    started = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    ok = proc.returncode == 0 and done.exists()
    status = "ok" if ok else f"FAILED rc={proc.returncode}"
    print(f"  {cfg.name} seed{seed}: {status} in {time.time() - started:.0f}s")
    if not ok:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
    return ok


def evaluate_all(configs: list[Path], seeds: list[int], out_root: Path, episodes: int) -> list[dict]:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    from crashcourse.env import DriveEnv
    from crashcourse.sb3_compat import load_ppo

    eval_seeds = episode_seeds(episodes)
    rows: list[dict] = []
    for config in configs:
        cfg = RunConfig.from_yaml(config)
        venv = DummyVecEnv([lambda: DriveEnv(cfg.env)])
        if cfg.train.n_stack > 1:
            venv = VecFrameStack(venv, n_stack=cfg.train.n_stack)
        for seed in seeds:
            model_path = out_root / cfg.name / f"seed{seed}" / "final_model.zip"
            if not model_path.exists():
                continue
            model = load_ppo(model_path, venv, device="auto")
            rep = evaluate_model(model, cfg.env, eval_seeds, cfg.train.n_stack,
                                 name=f"{cfg.name}/seed{seed}")
            row = rep.summary()
            row.update(config=cfg.name, seed=seed, description=cfg.description.strip())
            rows.append(row)
            print("  {config:22s} seed{seed}  frames {frames_mean:7.1f}  success {success_rate:.2f}".format(**row))
        venv.close()
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--out", default="results")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--evaluate-only", action="store_true")
    args = p.parse_args()

    out_root = REPO_ROOT / args.out
    configs = all_configs(args.only)
    seeds = list(range(args.seeds))
    print(f"{len(configs)} configs x {len(seeds)} seeds")

    if not args.evaluate_only:
        for config in configs:
            print(RunConfig.from_yaml(config).name)
            for seed in seeds:
                train_one(config, seed, out_root, args.timesteps)

    print("\nevaluating")
    rows = evaluate_all(configs, seeds, out_root, args.episodes)
    out = out_root / "study.json"
    out.write_text(json.dumps({"episodes": args.episodes, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
