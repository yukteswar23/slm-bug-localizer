"""Shared publication style for every figure in this project.

All plotting scripts import from here so fonts, colors, and export settings
stay identical across experiments. Figures are saved as vector PDF (for the
paper) and 300-dpi PNG (for quick viewing) under figures/.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"  # paper-ready copies; experiment figures live in each checkpoint
RESULTS_DIR = ROOT / "results"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}

# One fixed color per model, used in every figure of the paper.
MODEL_COLORS = {
    "gemma-4-e2b": PALETTE["blue_main"],
    "qwen3.5-4b": PALETTE["teal"],
    "gemma-4-31b": PALETTE["neutral"],
}
MODEL_LABELS = {
    "gemma-4-e2b": "Gemma 4 E2B",
    "qwen3.5-4b": "Qwen3.5-4B",
    "gemma-4-31b": "Gemma 4 31B (API)",
}


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 11
    axes_linewidth: float = 1.2
    font_family: tuple = ("Arial", "DejaVu Sans", "sans-serif")


def apply_publication_style(style: FigureStyle = FigureStyle()):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(style.font_family),
        "font.size": style.font_size,
        "axes.titlesize": style.font_size + 1,
        "axes.labelsize": style.font_size,
        "xtick.labelsize": style.font_size - 1,
        "ytick.labelsize": style.font_size - 1,
        "legend.fontsize": style.font_size - 1,
        "legend.frameon": False,
        "axes.linewidth": style.axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": style.axes_linewidth,
        "ytick.major.width": style.axes_linewidth,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,  # embed TrueType so text stays editable in the PDF
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def panel_label(ax, letter: str):
    """Bold (a), (b), ... label in the top-left corner, as in journal figures."""
    ax.text(-0.14, 1.06, f"({letter})", transform=ax.transAxes,
            fontweight="bold", va="bottom", ha="left")


def finalize_figure(fig, name: str, out_dir: Path = FIG_DIR, formats=("pdf", "png"), dpi=300):
    """Save into out_dir (normally <checkpoint>/figures)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, pad_inches=0.05)
        paths.append(path)
    plt.close(fig)
    return paths
