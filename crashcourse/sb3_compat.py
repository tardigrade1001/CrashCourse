"""Loading Stable-Baselines3 checkpoints across NumPy major versions.

An SB3 zip stores its metadata as cloudpickled objects. A checkpoint written
under NumPy 2.x cannot be unpickled by NumPy 1.x, which is what the archived
V5 model in ``models/`` hits.

``policy.pth`` inside the same zip is plain tensors and always loads. This
module reads the architecture out of the JSON metadata, rebuilds an equivalent
PPO, and copies the weights in. Checkpoints written by this repository load
through the normal path.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import torch
from stable_baselines3 import PPO


def read_metadata(path: str | Path) -> dict:
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("data"))


def load_ppo(path: str | Path, env, device: str = "auto") -> PPO:
    path = Path(path)
    try:
        return PPO.load(path, env=env, device=device)
    except (ModuleNotFoundError, AttributeError, ImportError):
        return _load_from_weights(path, env, device)


def _load_from_weights(path: Path, env, device: str) -> PPO:
    meta = read_metadata(path)
    policy_kwargs = meta.get("policy_kwargs") or {}
    model = PPO(
        meta.get("policy_class_name", "MlpPolicy") if isinstance(
            meta.get("policy_class_name"), str) else "MlpPolicy",
        env,
        device=device,
        policy_kwargs=policy_kwargs,
        n_steps=int(meta.get("n_steps", 2048)),
        batch_size=int(meta.get("batch_size", 64)),
        gamma=float(meta.get("gamma", 0.99)),
        gae_lambda=float(meta.get("gae_lambda", 0.95)),
        ent_coef=float(meta.get("ent_coef", 0.0)),
        verbose=0,
    )
    with zipfile.ZipFile(path) as z:
        state = torch.load(io.BytesIO(z.read("policy.pth")),
                           map_location=model.device, weights_only=False)
    model.policy.load_state_dict(state)
    model.policy.eval()
    model.num_timesteps = int(meta.get("num_timesteps", 0))
    return model
