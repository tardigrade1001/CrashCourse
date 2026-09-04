"""Train one configuration, with one seed, into one results directory.

    python train.py --config configs/v5_smooth.yaml --seed 0

Every run writes results/<config name>/seed<k>/ containing the config actually
used, the monitor logs, the periodic evaluations and the final model. Nothing
that affects the outcome is read from anywhere else.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from crashcourse.config import RunConfig  # noqa: E402
from crashcourse.env import DriveEnv  # noqa: E402


def build_env(cfg: RunConfig, seed: int, n_envs: int, monitor_dir: Path | None):
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = make_vec_env(
        DriveEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"config": cfg.env},
        vec_env_cls=vec_cls,
        monitor_dir=str(monitor_dir) if monitor_dir else None,
    )
    return VecFrameStack(env, n_stack=cfg.train.n_stack) if cfg.train.n_stack > 1 else env


def train(config_path: Path, seed: int, out_root: Path, timesteps: int | None) -> Path:
    cfg = RunConfig.from_yaml(config_path)
    if timesteps is not None:
        cfg.train.total_timesteps = timesteps

    out_dir = out_root / cfg.name / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")

    set_random_seed(seed)
    env = build_env(cfg, seed, cfg.train.n_envs, out_dir / "monitor")
    # A different seed offset keeps evaluation episodes away from training episodes.
    eval_env = build_env(cfg, seed + 5000, 1, None)

    # EvalCallback counts callback invocations, and one invocation covers n_envs
    # transitions. Convert so eval_freq_steps means environment transitions.
    eval_freq = max(1, cfg.train.eval_freq_steps // cfg.train.n_envs)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir),
        log_path=str(out_dir),
        eval_freq=eval_freq,
        n_eval_episodes=cfg.train.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    model = PPO(
        cfg.train.policy,
        env,
        seed=seed,
        verbose=1,
        device=cfg.train.device,
        tensorboard_log=str(out_dir / "tb"),
        policy_kwargs={"net_arch": list(cfg.train.net_arch)},
        n_steps=cfg.train.n_steps,
        batch_size=cfg.train.batch_size,
        learning_rate=cfg.train.learning_rate,
        ent_coef=cfg.train.ent_coef,
        gamma=cfg.train.gamma,
        gae_lambda=cfg.train.gae_lambda,
    )

    meta = {
        "config": str(config_path),
        "config_name": cfg.name,
        "seed": seed,
        "total_timesteps": cfg.train.total_timesteps,
        "device": str(model.device),
        "policy_parameters": sum(p.numel() for p in model.policy.parameters()),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    model.learn(total_timesteps=cfg.train.total_timesteps, callback=eval_cb, progress_bar=False)
    model.save(str(out_dir / "final_model"))

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["num_timesteps"] = model.num_timesteps
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    env.close()
    eval_env.close()
    print(f"saved to {out_dir}")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/v5_smooth.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=None, help="override the config")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    train(Path(args.config), args.seed, REPO_ROOT / args.out, args.timesteps)


if __name__ == "__main__":
    main()
