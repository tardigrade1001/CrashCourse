"""Figures built from a training monitor log.

    python create_graphs.py --monitor logs/monitor.csv

The Stable-Baselines3 monitor CSV carries three columns: ``r`` episode return,
``l`` episode length in frames, and ``t`` wall-clock seconds since the run
started. Timesteps are recovered as the cumulative sum of ``l``, which is what
the x-axis of every learning figure shows.

Every visual choice lives in crashcourse/figstyle.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crashcourse import figstyle as fs  # noqa: E402

CAP = 3000  # episode length cap set by EnvConfig.max_frames


def load_monitor(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1)
    df["timesteps"] = df["l"].cumsum()
    return df


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def fig_learning_curve(df: pd.DataFrame, window: int) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(df["timesteps"], df["l"], s=6, alpha=0.22, color=fs.GREY,
               linewidths=0, label="individual episodes", zorder=2)
    ax.plot(df["timesteps"], smooth(df["l"], window), color=fs.RED,
            label=f"moving mean, window {window} episodes", zorder=3)
    ax.axhline(CAP, color=fs.CAP_LINE, linestyle="--", linewidth=1.2,
               label=f"episode length cap, {CAP:,} frames", zorder=1)
    fs.finish(ax, "Environment transitions", "Episode length (frames)",
              "Survival across training")
    ax.set_xlim(0, df["timesteps"].max())
    ax.set_ylim(0, CAP * 1.08)
    fs.save(fig, "01_learning_curve")
    plt.close(fig)


def fig_return_curve(df: pd.DataFrame, window: int) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(df["timesteps"], smooth(df["r"], window), color=fs.RED,
            label=f"moving mean, window {window} episodes")
    rolling_sd = df["r"].rolling(window=window, min_periods=1).std().fillna(0)
    mean = smooth(df["r"], window)
    ax.fill_between(df["timesteps"], mean - rolling_sd, mean + rolling_sd,
                    color=fs.RED, alpha=fs.BAND_ALPHA, linewidth=0,
                    label="±1 standard deviation")
    fs.finish(ax, "Environment transitions", "Episode return",
              "Return across training")
    ax.set_xlim(0, df["timesteps"].max())
    fs.save(fig, "02_return_curve")
    plt.close(fig)


def fig_consistency(df: pd.DataFrame, window: int) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rolling_sd = df["r"].rolling(window=window, min_periods=2).std()
    ax.plot(df["timesteps"], rolling_sd, color=fs.BLUE,
            label=f"rolling standard deviation, window {window}")
    fs.finish(ax, "Environment transitions", "Standard deviation of return",
              "Run-to-run spread across training")
    ax.set_xlim(0, df["timesteps"].max())
    ax.set_ylim(bottom=0)
    fs.save(fig, "03_consistency")
    plt.close(fig)


def fig_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(df["l"], bins=40, color=fs.BLUE, edgecolor=fs.INK, linewidth=0.6)
    ax.axvline(df["l"].mean(), color=fs.RED, linestyle="--",
               label=f"mean {df['l'].mean():,.0f} frames")
    ax.axvline(df["l"].median(), color=fs.INK, linestyle=":",
               label=f"median {df['l'].median():,.0f} frames")
    at_cap = (df["l"] >= CAP).mean()
    fs.finish(ax, "Episode length (frames)", "Episodes",
              f"Episode length distribution, {len(df):,} episodes")
    ax.annotate(f"{at_cap:.1%} of episodes reached the cap",
                xy=(0.97, 0.86), xycoords="axes fraction", ha="right",
                fontsize=10, color=fs.INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=2))
    fs.save(fig, "04_length_distribution")
    plt.close(fig)


def fig_dashboard(df: pd.DataFrame, window: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6))
    steps = df["timesteps"]

    ax = axes[0, 0]
    mean = smooth(df["r"], window)
    sd = df["r"].rolling(window=window, min_periods=1).std().fillna(0)
    ax.plot(steps, mean, color=fs.RED, label="mean return")
    ax.fill_between(steps, mean - sd, mean + sd, color=fs.RED,
                    alpha=fs.BAND_ALPHA, linewidth=0, label="±1 SD")
    fs.finish(ax, "Environment transitions", "Return", "Return")
    fs.panel_tag(ax, "(a)")

    ax = axes[0, 1]
    ax.plot(steps, smooth(df["l"], window), color=fs.BLUE, label="mean length")
    ax.axhline(CAP, color=fs.CAP_LINE, linestyle="--", linewidth=1.2, label="cap")
    fs.finish(ax, "Environment transitions", "Frames", "Episode length")
    fs.panel_tag(ax, "(b)")

    ax = axes[1, 0]
    q25 = df["r"].rolling(window=window, min_periods=1).quantile(0.25)
    q75 = df["r"].rolling(window=window, min_periods=1).quantile(0.75)
    med = df["r"].rolling(window=window, min_periods=1).median()
    ax.fill_between(steps, q25, q75, color=fs.ORANGE, alpha=0.3, linewidth=0,
                    label="25th to 75th percentile")
    ax.plot(steps, med, color=fs.ORANGE, label="median")
    fs.finish(ax, "Environment transitions", "Return", "Interquartile range")
    fs.panel_tag(ax, "(c)")

    ax = axes[1, 1]
    at_cap = (df["l"] >= CAP).rolling(window=window, min_periods=1).mean() * 100
    ax.plot(steps, at_cap, color=fs.GREEN, label="episodes reaching the cap")
    fs.finish(ax, "Environment transitions", "Percent", "Episodes at the cap")
    fs.panel_tag(ax, "(d)")

    fig.tight_layout()
    fs.save(fig, "05_dashboard")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--monitor", default="logs/monitor.csv")
    p.add_argument("--window", type=int, default=100, help="smoothing window in episodes")
    args = p.parse_args()

    path = REPO_ROOT / args.monitor
    df = load_monitor(path)
    fs.apply()

    print(f"{len(df):,} episodes, {df['l'].sum():,} environment transitions, "
          f"{df['t'].max() / 60:.1f} minutes of wall clock")

    fig_learning_curve(df, args.window)
    fig_return_curve(df, args.window)
    fig_consistency(df, args.window)
    fig_distribution(df)
    fig_dashboard(df, args.window)


if __name__ == "__main__":
    main()
