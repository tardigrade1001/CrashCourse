"""Figures for the decision-rate comparison.

    python experiments/cadence_figure.py

Reads results/cadence_local.json and results/gemini_cadence.json, both written
by the sweeps, and writes docs/figures/06_decision_rate.png and
docs/figures/07_hosted_latency.png.

Every controller runs on the same held-out episode seeds and the same
environment. A controller deciding every k steps holds its action for the k-1
steps in between, which is what a hosted model does in a real loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crashcourse import figstyle as fs  # noqa: E402

RESULTS = REPO_ROOT / "results"
V1_HZ = 1.07          # measured in legacy/car/ai_drive_log.csv
RENDER_FPS = 60       # DriveEnv.metadata["render_fps"]


def load():
    local = json.loads((RESULTS / "cadence_local.json").read_text())
    hosted = json.loads((RESULTS / "gemini_cadence.json").read_text())
    gem = {int(k): v for k, v in hosted.items()}
    return local, gem


def fig_decision_rate(local, gem) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    cad = np.array(local["cadences"], dtype=float)

    ax.plot(cad, local["greedy"], "o-", color=fs.POLICY["greedy"],
            label="hand-written controller, retuned at each rate")
    ax.plot(cad, local["ppo"], "s-", color=fs.POLICY["ppo"],
            label="PPO agent, trained to act every step")

    gc = np.array(sorted(gem), dtype=float)
    gv = [gem[int(c)]["summary"]["frames_mean"] for c in gc]
    lo = [gem[int(c)]["summary"]["frames_ci_lo"] for c in gc]
    hi = [gem[int(c)]["summary"]["frames_ci_hi"] for c in gc]
    ax.plot(gc, gv, "D-", color=fs.POLICY["gemini"], label="hosted model over the network")
    ax.fill_between(gc, lo, hi, color=fs.POLICY["gemini"], alpha=fs.BAND_ALPHA, linewidth=0)

    ax.axhline(local["stay"], color=fs.GREY, linestyle=":", linewidth=1.2,
               label=f"hold one lane, {local['stay']:.0f} frames")

    anchor = RENDER_FPS / V1_HZ
    ax.axvline(anchor, color=fs.PURPLE, linestyle="--", linewidth=1.2)
    ax.annotate(f"V1 measured rate\n{V1_HZ:.2f} Hz", xy=(anchor, 0.62),
                xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", color=fs.PURPLE, fontsize=9.5, va="center")

    ax.set_xscale("log")
    ax.set_xticks(cad)
    ax.set_xticklabels([f"{int(c)}" for c in cad])
    fs.finish(ax, "Environment steps between decisions",
              "Episode length (frames)", "Survival against decision rate")
    ax.set_ylim(0, max(local["greedy"]) * 1.12)
    fs.save(fig, "06_decision_rate")
    plt.close(fig)


def fig_latency(gem) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    lat = np.concatenate([np.array(v["latencies"]) for v in gem.values()]) * 1000.0
    lat = lat[lat < 3000]
    ax.hist(lat, bins=45, color=fs.POLICY["gemini"], edgecolor=fs.INK, linewidth=0.6)
    med = float(np.median(lat))
    ax.axvline(med, color=fs.RED, linestyle="--", label=f"median {med:.0f} ms")
    ax.axvline(1000.0 / V1_HZ, color=fs.PURPLE, linestyle=":",
               label=f"V1 measured, {1000.0 / V1_HZ:.0f} ms")
    fs.finish(ax, "Decision latency (ms)", "Decisions",
              f"Hosted decision latency, {lat.size} decisions")
    fs.save(fig, "07_hosted_latency")
    plt.close(fig)


def fig_spawn_clock() -> None:
    """Every controller on the periodic environment against the jittered one."""
    data = json.loads((RESULTS / "spawn_clock.json").read_text())
    rows = data["rows"]
    names = ["ppo", "greedy", "random"]
    labels = ["PPO agent", "hand-written\ncontroller", "uniform random"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, [rows["periodic"][n] for n in names], w,
           color=fs.CONFIG["v3_instant"], edgecolor=fs.INK, linewidth=0.8,
           label="spawn every 10 frames")
    ax.bar(x + w / 2, [rows["jittered"][n] for n in names], w,
           color=fs.CONFIG["v5_smooth"], edgecolor=fs.INK, linewidth=0.8,
           label="spawn jittered, 6 to 14, same mean")
    for i, n in enumerate(names):
        kept = rows["jittered"][n] / rows["periodic"][n]
        ax.text(i, max(rows["periodic"][n], rows["jittered"][n]) + 25,
                f"keeps {kept:.0%}", ha="center", fontsize=10, color=fs.INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    fs.finish(ax, None, "Episode length (frames)",
              f"Reliance on the spawn clock, {data['episodes']} seeds", legend=False)
    ax.set_ylim(0, max(rows["periodic"][n] for n in names) * 1.52)
    ax.legend(loc="upper center", frameon=False)
    fs.save(fig, "08_spawn_clock")
    plt.close(fig)


def main() -> None:
    local, gem = load()
    fs.apply()
    fig_decision_rate(local, gem)
    fig_latency(gem)
    fig_spawn_clock()
    allv = np.concatenate([np.array(v["latencies"]) for v in gem.values()]) * 1000.0
    print(f"hosted decisions {allv.size}, median {np.median(allv):.0f} ms, "
          f"{1000.0 / np.median(allv):.2f} Hz, V1 measured {V1_HZ:.2f} Hz")


if __name__ == "__main__":
    main()
