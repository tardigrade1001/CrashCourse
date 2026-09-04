"""Single source of truth for every figure in this repository.

Generator scripts carry zero literal colours, sizes, or fonts. Everything visual
lives here, so re-theming the repository means editing this file and re-running
the generators. Colours are semantic, keyed by what they mean, so a swap stays
meaningful across every figure.

The visual grammar follows the author's thesis figure system: white background,
black frame, inward ticks on all four sides, DejaVu Sans, restrained colour,
300 dpi output.

    from crashcourse import figstyle as fs
    fs.apply()
    ...plot using fs.POLICY["ppo"], fs.finish(ax), fs.save(fig, name)...
"""

from __future__ import annotations

import os

import matplotlib as mpl

# -- neutral palette ---------------------------------------------------------
RED = "#E8352B"      # primary / emphasis
BLUE = "#2E6FB0"     # secondary series
GREY = "#5A5A5A"     # baseline / control
INK = "#1A1A1A"      # near-black reference
GREEN = "#2E7D32"
PURPLE = "#8E44AD"
ORANGE = "#C97A2B"
TEAL = "#2A9D8F"
PALETTE = [RED, BLUE, GREEN, PURPLE, ORANGE, GREY]

# -- semantic policy identity, consistent across every figure ----------------
POLICY = {
    "ppo": RED,        # the trained agent, principal result
    "greedy": INK,     # hand-written controller, reference
    "random": GREY,    # uniform action baseline
    "stay": GREY,      # hold one lane, baseline
    "hover": PURPLE,   # degenerate policy that ignores the observation
}

# -- semantic configuration identity -----------------------------------------
CONFIG = {
    "v5_smooth": RED,    # canonical, interpolated lane changes
    "v3_instant": BLUE,  # published iteration, immediate lane changes
}

CAP_LINE = GREY        # the episode length cap
BAND_ALPHA = 0.18      # spread band around a mean curve

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(HERE), "docs", "figures")


def apply() -> None:
    """Global rcParams. The house look."""
    mpl.rcParams.update({
        "figure.figsize": (5.2, 4.0),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12.5,
        "axes.titlesize": 12,
        "axes.linewidth": 1.1,
        "axes.axisbelow": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.linewidth": 1.7,
        "lines.markersize": 6,
        "errorbar.capsize": 3,
    })


def frame(ax, lw: float = 1.1) -> None:
    """Black frame and inward ticks on all four sides."""
    for spine in ax.spines.values():
        spine.set_edgecolor(INK)
        spine.set_linewidth(lw)
    ax.tick_params(colors=INK, which="both", top=True, right=True, direction="in")


def finish(ax, xlabel=None, ylabel=None, title=None, legend=True) -> None:
    """Shared per-axes finishing."""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    frame(ax)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend()


def panel_tag(ax, tag: str, x: float = -0.14, y: float = 1.03) -> None:
    """Bold (a)/(b) panel label in axes coordinates."""
    ax.text(x, y, tag, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="left")


def save(fig, name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name if name.endswith(".png") else name + ".png")
    fig.savefig(path)
    print("saved:", os.path.relpath(path, os.path.dirname(HERE)))
    return path
