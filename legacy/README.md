# legacy

Four folders moved here on 2026-09-02 from the top level of
`F:\PC Files\Projects`. They are the earlier passes that led to this one. The
parent folder was named `car5` until the same day. Nothing was deleted. Every
file count and byte total was verified identical before and after the move.

## The published repo

`https://github.com/tardigrade1001/CrashCourse` is public and was last pushed on
2026-05-06 at commit `68ddd46`. It is the only one of these five ever published.
`car3` and `car4` both carry that repo with `origin` set and the same 12 commits.
The parent folder has no git at all, so the newest code lives outside version
control.

What GitHub currently shows is `car3`. Everything in `car4` and in the parent
folder is unpublished.

## The folders

| Folder | Approach | Git | Files | MB | Last write |
|---|---|---|---|---|---|
| `car` | Gemini drives the game | none | 18 | 0.1 | 2026-05-06 |
| `car2` | first PPO run, Stable-Baselines3 | none | 19 | 1.2 | 2026-05-06 |
| `car3` | PPO, clean at published HEAD | 12 commits | 138 | 219.4 | 2026-05-06 |
| `car4` | PPO, HEAD plus unpushed work | 12 commits | 140 | 217.8 | 2026-05-08 |

Total 315 files, 438.5 MB.

## What separates them

`car` is the odd one. It has no reinforcement learning in it. `gemini_connector.py`
calls a model to steer, and `ai_drive_log.csv` holds the run log. The RL line
starts at `car2`.

`car2` is the first PPO attempt. It kept eight 142 KB checkpoints across two
variants, `ppo_drive_*` and `ppo_ultra_*`, along with `train_fast.py`,
`train_ultra.py` and `train_ultra_visual.py`. None of those three script names
survive into `car3`.

`car3` is what the world sees. Working tree clean against `68ddd46` apart from
`.claude\settings.local.json` and a `.pyc`. It still has `demo.gif`.

`car4` is `car3` carried forward with 20 uncommitted changes that were never
pushed: `create_graphs.py`, `run_agent.py`, `train_rl_agent.py`, the README, all
five graph PNGs, both 57 MB model archives, `logs\monitor.csv`,
`logs\evaluations.npz`, three new run directories `logs\PPO_6` through
`logs\PPO_8`, and `demo.gif` deleted.

The parent folder took `car4`'s versions of `create_graphs.py`, `run_agent.py`
and `train_rl_agent.py` unchanged, then moved past it with a newer
`ai_drive_game_env.py` and a newer README. `play_random.py` is byte-identical
across `car3`, `car4` and the parent.

## Reverting

Each folder is a plain move. Moving one back to `F:\PC Files\Projects\` restores
it exactly. Renaming this parent back to `car5` restores the old name.
