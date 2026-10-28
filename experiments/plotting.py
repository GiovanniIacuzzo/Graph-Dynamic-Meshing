"""Publication-grade Matplotlib helpers for GDM experiments.

Every figure is exported as vector PDF and SVG plus a high-resolution PNG preview.
The numerical results are never transformed by this module.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_publication_style() -> None:
    """Apply a restrained, consistent style suitable for scientific papers."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.22,
        "axes.grid": True,
        "axes.axisbelow": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 120,
        "savefig.dpi": 400,
    })


def save_figure(fig, path, *, png_dpi: int = 400, dpi: int | None = None) -> list[Path]:
    """Save one figure as PDF/SVG (vector) and PNG (preview), returning paths."""
    if dpi is not None:  # Backward-compatible alias used by older experiment scripts.
        png_dpi = int(dpi)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stem = target.with_suffix("") if target.suffix.lower() in {".png", ".pdf", ".svg", ".eps"} else target
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".svg"), stem.with_suffix(".png")]
    fig.savefig(outputs[0], format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outputs[1], format="svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outputs[2], format="png", dpi=png_dpi, bbox_inches="tight", pad_inches=0.04)
    return outputs


set_publication_style()
