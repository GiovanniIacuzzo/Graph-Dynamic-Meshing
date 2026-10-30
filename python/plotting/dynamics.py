"""Reusable publication plots for transient PDE/dynamic-mesh experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from .style import PALETTE, save_figure, set_publication_style


def plot_time_series(
    time: Sequence[float], series: Mapping[str, Sequence[float]], labels: Mapping[str, str], output_path: str | Path,
    *, ylabel: str = r"Relative $L^2$ error [\%]", xlabel: str = "Time $t$", title: str = "Transient numerical error",
) -> list[Path]:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    colours = [PALETTE["fixed"], PALETTE["raw"], PALETTE["safe"], PALETTE["target"], PALETTE["accent"]]
    for i, (key, values) in enumerate(series.items()):
        ax.plot(time, values, label=labels.get(key, key), color=colours[i % len(colours)])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.grid(True, axis="y")
    fig.tight_layout()
    return save_figure(fig, output_path)


def plot_mesh_validity(
    time: Sequence[float], series: Mapping[str, Sequence[float]], labels: Mapping[str, str], epsilon: float, output_path: str | Path,
) -> list[Path]:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    colours = [PALETTE["fixed"], PALETTE["raw"], PALETTE["safe"], PALETTE["target"], PALETTE["accent"]]
    for i, (key, values) in enumerate(series.items()):
        ax.plot(time, values, label=labels.get(key, key), color=colours[i % len(colours)])
    ax.axhline(epsilon, color=PALETTE["warning"], linestyle="--", linewidth=1.0, label=rf"Admissibility floor $\epsilon={epsilon:g}$")
    ax.set_yscale("log"); ax.set_xlabel("Time $t$"); ax.set_ylabel(r"Minimum oriented Jacobian $\min_e J_e$")
    ax.set_title("Mesh admissibility over time"); ax.legend(); ax.grid(True, which="both", axis="y")
    fig.tight_layout()
    return save_figure(fig, output_path)


def plot_runtime_decomposition(
    labels: Sequence[str], components: Mapping[str, Sequence[float]], output_path: str | Path,
) -> list[Path]:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    bottoms = np.zeros(len(labels), dtype=float)
    colours = [PALETTE["fixed"], PALETTE["raw"], PALETTE["safe"], PALETTE["target"], PALETTE["accent"]]
    for i, (key, values) in enumerate(components.items()):
        vals = np.asarray(values, dtype=float)
        ax.bar(np.arange(len(labels)), vals, bottom=bottoms, label=key, color=colours[i % len(colours)], width=0.65)
        bottoms += vals
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("Wall-clock time [s]"); ax.set_title("Runtime decomposition")
    ax.legend(); ax.grid(True, axis="y"); ax.grid(False, axis="x")
    fig.tight_layout()
    return save_figure(fig, output_path)


def plot_mesh_field_snapshots(snapshots: Sequence[Any], cells: np.ndarray, output_path: str | Path) -> list[Path]:
    """Plot a sequence of (time, nodes, field) snapshots as mesh/field rows."""
    set_publication_style()
    if not snapshots:
        raise ValueError("At least one mesh/field snapshot is required.")
    fig, axes = plt.subplots(2, len(snapshots), figsize=(2.5 * len(snapshots), 4.8), squeeze=False)
    contour = None
    for col, (snap_t, snap_nodes, snap_u) in enumerate(snapshots):
        triang = mtri.Triangulation(snap_nodes[:, 0], snap_nodes[:, 1], cells)
        axes[0, col].triplot(triang, color=PALETTE["fixed"], linewidth=0.35)
        axes[0, col].set_title(f"Mesh, $t={snap_t:.3f}$")
        contour = axes[1, col].tricontourf(triang, snap_u, levels=24, cmap="viridis")
        axes[1, col].set_title(rf"$u_h$, $t={snap_t:.3f}$")
        for row in range(2):
            axes[row, col].set_aspect("equal")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    if contour is not None:
        fig.colorbar(contour, ax=list(axes[1, :]), shrink=0.8, label=r"$u_h$")
    fig.tight_layout()
    return save_figure(fig, output_path)
