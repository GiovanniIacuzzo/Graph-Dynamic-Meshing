"""Shared publication style and vector-PDF-only figure export for GDM."""
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "fixed": "#475569",
    "zero": "#94A3B8",
    "raw": "#D97757",
    "safe": "#167D8D",
    "target": "#7C6BA8",
    "accent": "#C28E2C",
    "success": "#34856A",
    "warning": "#C45A53",
    "grid": "#D7DCE2",
}


def set_publication_style() -> None:
    """Set consistent, restrained, English-language journal figure defaults."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        # Embed scalable fonts in PDF for publication-ready vector output.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 120,
    })


def save_figure(
    fig: mpl.figure.Figure,
    path: str | Path,
    *,
    close: bool = True,
) -> list[Path]:
    """Save exactly one vector PDF figure and optionally close the figure.

    ``path`` is treated as a file stem regardless of its suffix. Raster preview
    and SVG exports are intentionally disabled to keep experiment outputs lean.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stem = target.with_suffix("") if target.suffix.lower() in {".png", ".pdf", ".svg", ".eps"} else target
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    if close:
        plt.close(fig)
    return [pdf_path]


set_publication_style()
