"""Build every figure in the README from the study output.

    python make_figures.py

Reads results/study.json and the per-run evaluations.npz files. Writes PNGs to
docs/figures/. No number in a figure is typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from crashcourse.evaluation import bootstrap_ci  # noqa: E402

RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "docs" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ACCENT = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.dpi": 140,
})


def tidy(ax, xlabel="", ylabel=""):
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def load_study() -> list[dict]:
    path = RESULTS / "study.json"
    if not path.exists():
        raise SystemExit(f"missing {path}. Run experiments/run_ablation.py first.")
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def by_config(rows: list[dict]) -> dict[str, dict]:
    """Aggregate seeds. The spread reported is across seeds."""
    out: dict[str, dict] = {}
    for row in rows:
        out.setdefault(row["config"], {"means": [], "success": [], "description": row.get("description", "")})
        out[row["config"]]["means"].append(row["frames_mean"])
        out[row["config"]]["success"].append(row["success_rate"])
    for name, rec in out.items():
        means = np.array(rec["means"], dtype=float)
        rec["mean"] = float(means.mean())
        rec["lo"], rec["hi"] = bootstrap_ci(means) if len(means) > 1 else (float(means[0]), float(means[0]))
        rec["n_seeds"] = len(means)
        rec["success"] = float(np.mean(rec["success"]))
    return out


# --------------------------------------------------------------- learning curve

def figure_learning_curves(configs: list[str], filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    plotted = 0
    for colour, name in zip(SERIES, configs):
        runs = sorted((RESULTS / name).glob("seed*/evaluations.npz"))
        if not runs:
            continue
        curves, steps = [], None
        for run in runs:
            z = np.load(run)
            steps = z["timesteps"]
            curves.append(z["ep_lengths"].mean(axis=1))
        length = min(len(c) for c in curves)
        stack = np.stack([c[:length] for c in curves])
        steps = steps[:length]
        mean = stack.mean(axis=0)
        ax.plot(steps, mean, color=colour, linewidth=2.0, zorder=3, label=name)
        if stack.shape[0] > 1:
            lo, hi = stack.min(axis=0), stack.max(axis=0)
            ax.fill_between(steps, lo, hi, color=colour, alpha=0.16, linewidth=0, zorder=2)
        ax.annotate(name, (steps[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                    color=colour, fontsize=9, va="center", fontweight="medium")
        plotted += 1
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(title, loc="left", pad=12)
    tidy(ax, "environment transitions", "episode length (frames)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.14)
    if plotted >= 2:
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, bbox_inches="tight")
    plt.close(fig)
    print("wrote", filename)


# ------------------------------------------------------------------- ablations

def figure_ablations(agg: dict[str, dict], filename: str) -> None:
    baseline = "v5_smooth"
    names = [n for n in agg if n != baseline]
    names.sort(key=lambda n: agg[n]["mean"])
    order = names + [baseline] if baseline in agg else names
    values = [agg[n]["mean"] for n in order]
    errs = np.array([[agg[n]["mean"] - agg[n]["lo"] for n in order],
                     [agg[n]["hi"] - agg[n]["mean"] for n in order]])
    errs = np.clip(errs, 0, None)
    colours = [ACCENT if n == baseline else SERIES[0] for n in order]

    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(order) + 1.8))
    y = np.arange(len(order))
    ax.barh(y, values, height=0.62, color=colours, zorder=3)
    ax.errorbar(values, y, xerr=errs, fmt="none", ecolor=INK_SOFT,
                elinewidth=1.2, capsize=3, zorder=4)
    for yi, value in zip(y, values):
        ax.annotate(f"{value:,.0f}", (value, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_yticks(y, order)
    ax.set_title("Mean survival by configuration, one variable changed at a time",
                 loc="left", pad=12)
    ax.grid(axis="x", alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlabel("episode length (frames), mean over seeds, bars show 95% CI")
    ax.margins(x=0.12)
    seeds = {rec["n_seeds"] for rec in agg.values()}
    ax.annotate(f"canonical configuration in red · {min(seeds)} seeds per configuration",
                (0, 1), xycoords="axes fraction", xytext=(0, 16), textcoords="offset points",
                fontsize=9, color=INK_SOFT)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, bbox_inches="tight")
    plt.close(fig)
    print("wrote", filename)


# -------------------------------------------------------------------- baselines

def figure_baselines(filename: str) -> None:
    """Two panels: the broken collision model beside the corrected one."""
    paths = {
        "collision tolerance 0.8 (original)": RESULTS / "eval" / "permissive_collision.json",
        "collision tolerance 1.0 (corrected)": RESULTS / "eval" / "v5_smooth.json",
    }
    panels = {k: json.loads(p.read_text(encoding="utf-8"))["rows"]
              for k, p in paths.items() if p.exists()}
    if not panels:
        print("skipped baselines figure, run evaluate.py first")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 3.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (title, rows) in zip(axes, panels.items()):
        rows = sorted(rows, key=lambda r: r["frames_mean"])
        labels = [r["policy"] for r in rows]
        values = [r["frames_mean"] for r in rows]
        errs = np.array([[r["frames_mean"] - r["frames_ci_lo"] for r in rows],
                         [r["frames_ci_hi"] - r["frames_mean"] for r in rows]])
        errs = np.clip(errs, 0, None)
        colours = [ACCENT if lbl == "hover" else SERIES[0] for lbl in labels]
        y = np.arange(len(labels))
        ax.barh(y, values, height=0.6, color=colours, zorder=3)
        ax.errorbar(values, y, xerr=errs, fmt="none", ecolor=INK_SOFT,
                    elinewidth=1.2, capsize=3, zorder=4)
        for yi, value in zip(y, values):
            ax.annotate(f"{value:,.0f}", (value, yi), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, labels)
        ax.set_title(title, loc="left", pad=10, fontsize=11)
        ax.grid(axis="x", alpha=0.7)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.margins(x=0.18)
        ax.set_xlabel("episode length (frames)")
    fig.suptitle("The hover policy ignores the observation entirely", x=0.01, ha="left",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, bbox_inches="tight")
    plt.close(fig)
    print("wrote", filename)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_study()
    agg = by_config(rows)
    figure_learning_curves(["v5_smooth", "v3_instant"], "01_learning_curve.png",
                           "Learning under instant and smooth lane changes")
    figure_learning_curves(["v5_smooth", "small_net", "no_framestack", "shallow_vision"],
                           "02_ablation_curves.png",
                           "Learning under single-variable changes")
    figure_ablations(agg, "03_ablations.png")
    figure_baselines("04_baselines.png")

    table = ["| Configuration | Frames (mean over seeds) | 95% CI | Success | Seeds |",
             "|---|---|---|---|---|"]
    for name in sorted(agg, key=lambda n: -agg[n]["mean"]):
        rec = agg[name]
        table.append(f"| {name} | {rec['mean']:,.0f} | [{rec['lo']:,.0f}, {rec['hi']:,.0f}] "
                     f"| {rec['success']:.0%} | {rec['n_seeds']} |")
    out = REPO_ROOT / "results" / "study_table.md"
    out.write_text("\n".join(table) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
