"""Stage-A paper figures (safety, scaling, OOD sweep, ablation, mesh views)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Patch

from .style import PALETTE, save_figure, set_publication_style


def plot_safety_curve(k_values, alphas, raw_min_j, safe_min_j, epsilon: float, output_path) -> list[Path]:
    set_publication_style(); fig, ax1 = plt.subplots(figsize=(6.3, 3.6)); ax2 = ax1.twinx()
    p1, = ax1.plot(k_values, alphas, "o-", color=PALETTE["safe"], label=r"Safety scaling $\alpha$")
    p2, = ax2.plot(k_values, raw_min_j, "s--", color=PALETTE["warning"], label=r"Raw $\min_e J_e$")
    p3, = ax2.plot(k_values, safe_min_j, "^-", color=PALETTE["target"], label=r"Filtered $\min_e J_e$")
    ax2.axhline(epsilon, color="#111827", linestyle=":", linewidth=1.0, label=rf"Floor $\epsilon={epsilon:g}$")
    ax2.axhline(0.0, color=PALETTE["grid"], linewidth=0.7)
    ax1.set_xlabel(r"Displacement multiplier $k$"); ax1.set_ylabel(r"Safety scaling $\alpha$"); ax2.set_ylabel(r"Minimum oriented Jacobian $\min_e J_e$")
    ax1.set_ylim(-0.04, 1.04); ax1.grid(True, axis="y"); ax1.grid(False, axis="x")
    h2, l2 = ax2.get_legend_handles_labels()
    lines = [p1] + h2
    labels = [p1.get_label()] + l2
    ax1.legend(lines, labels, loc="best", fontsize=7)
    fig.tight_layout(); return save_figure(fig, output_path)


def plot_runtime_scaling(n_nodes, t_oracle, t_gnn, t_safety, output_path) -> list[Path]:
    set_publication_style(); fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.loglog(n_nodes, t_oracle, "s-", color=PALETTE["target"], label="Offline optimizer (3-iteration budget)")
    ax.loglog(n_nodes, np.asarray(t_gnn) + np.asarray(t_safety), "o-", color=PALETTE["safe"], label="GNN + safety filter")
    ax.loglog(n_nodes, t_safety, "^--", color=PALETTE["raw"], label="Safety filter")
    ax.set_xlabel("Number of mesh nodes $N$"); ax.set_ylabel("CPU time [ms, log scale]")
    ax.set_title("Stage-A runtime scaling"); ax.legend(); ax.grid(True, which="both", axis="y")
    fig.tight_layout(); return save_figure(fig, output_path)


def plot_ood_error(sigmas, e_fixed, e_gnn, e_target, output_path, *, training_range=(0.03, 0.08), ood_range=(0.018, 0.03)) -> list[Path]:
    set_publication_style(); fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.axvspan(ood_range[0], ood_range[1], color=PALETTE["warning"], alpha=0.08, label="Sharp-gradient OOD range")
    ax.axvspan(training_range[0], training_range[1], color=PALETTE["safe"], alpha=0.07, label="Training range")
    ax.plot(sigmas, np.asarray(e_fixed)*1e4, "s--", color=PALETTE["fixed"], label="Fixed mesh")
    ax.plot(sigmas, np.asarray(e_gnn)*1e4, "o-", color=PALETTE["safe"], label="GNN + safety filter")
    ax.plot(sigmas, np.asarray(e_target)*1e4, "^-", color=PALETTE["target"], label="Stored optimizer target")
    ax.set_xlabel(r"Gaussian width $\sigma$"); ax.set_ylabel(r"Integrated interpolation error ($\times 10^{-4}$)")
    ax.set_title("Interpolation error across field scales"); ax.legend(fontsize=7); ax.grid(True, axis="y")
    fig.tight_layout(); return save_figure(fig, output_path)


def plot_gap_reduction(sigmas, ratios, output_path, *, ood_range=(0.018, 0.03)) -> list[Path]:
    set_publication_style(); fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.axvspan(ood_range[0], ood_range[1], color=PALETTE["warning"], alpha=0.08, label="Sharp-gradient OOD range")
    ax.plot(sigmas, ratios, "D-", color=PALETTE["accent"], label="Gap reduction versus stored target")
    ax.axhline(0.0, color=PALETTE["fixed"], linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Gaussian width $\sigma$"); ax.set_ylabel("Relative gap reduction [%]")
    ax.set_title("Gap reduction versus stored target meshes")
    ax.legend(fontsize=7); ax.grid(True, axis="y")
    fig.tight_layout(); return save_figure(fig, output_path)


def plot_loss_ablation(variants, errors, ratios, output_path) -> list[Path]:
    set_publication_style(); fig, ax1 = plt.subplots(figsize=(6.2, 3.6)); ax2 = ax1.twinx()
    x = np.arange(len(variants)); width = 0.34
    ax1.bar(x-width/2, errors, width, color=PALETTE["raw"], alpha=0.8, label=r"Interpolation error ($\times 10^{-4}$)")
    ax2.bar(x+width/2, ratios, width, color=PALETTE["safe"], alpha=0.8, label="Gap reduction [%]")
    ax1.set_xticks(x, variants, rotation=12, ha="right"); ax1.set_ylabel(r"Interpolation error ($\times 10^{-4}$)"); ax2.set_ylabel("Gap reduction [%]")
    ax1.set_title("Loss-formulation ablation"); ax1.grid(True, axis="y"); ax1.grid(False, axis="x")
    h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax1.legend(h1+h2,l1+l2,loc="best",fontsize=7)
    fig.tight_layout(); return save_figure(fig, output_path)


def plot_mesh_comparison(meshes, cells, output_path) -> list[Path]:
    """``meshes`` is a sequence of (label, coordinates, nodal_field) tuples."""
    set_publication_style(); fig, axes = plt.subplots(1, len(meshes), figsize=(3.0*len(meshes), 3.15), squeeze=False)
    contour = None
    for ax, (label, nodes, values) in zip(axes[0], meshes):
        triang = mtri.Triangulation(nodes[:,0], nodes[:,1], cells)
        ax.triplot(triang, color="#374151", linewidth=0.35, alpha=0.55)
        contour = ax.tricontourf(triang, values, levels=16, cmap="viridis")
        ax.set_title(label); ax.set_aspect("equal"); ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    if contour is not None: fig.colorbar(contour, ax=list(axes[0]), shrink=0.82, label="Reference field")
    fig.tight_layout(); return save_figure(fig, output_path)


def plot_safety_stress(raw_nodes, safe_nodes, cells, inverted_cells, alpha: float, output_path) -> list[Path]:
    set_publication_style(); fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.2))
    triang_raw = mtri.Triangulation(raw_nodes[:,0], raw_nodes[:,1], cells)
    axes[0].triplot(triang_raw, color="#7C8798", linewidth=0.35)
    for idx in inverted_cells:
        pts = raw_nodes[cells[int(idx)]]
        axes[0].fill(pts[:,0], pts[:,1], color=PALETTE["warning"], alpha=0.55, linewidth=0)
    axes[0].set_title(f"Unfiltered deformation ({len(inverted_cells)} inverted cells)")
    tri_safe = mtri.Triangulation(safe_nodes[:,0], safe_nodes[:,1], cells)
    axes[1].triplot(tri_safe, color="#1F2937", linewidth=0.4)
    axes[1].set_title(rf"Safety-filtered mesh ($\alpha={alpha:.3f}$)")
    for ax in axes:
        ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.tight_layout(); return save_figure(fig, output_path)
