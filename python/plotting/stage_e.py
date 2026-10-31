"""Publication-ready vector PDF plots for the Stage-E1 verification campaign."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .style import PALETTE, save_figure, set_publication_style


def _save(fig: plt.Figure, path: str | Path) -> None:
    set_publication_style()
    save_figure(fig, path)


def plot_spatial_convergence(rows: Sequence[Mapping], path: str | Path) -> None:
    data = sorted(rows, key=lambda r: int(r["nx"]))
    h = np.asarray([float(r["h"]) for r in data])
    e_l2 = np.asarray([float(r["time_average_absolute_l2_error"]) for r in data])
    e_h1 = np.asarray([float(r["time_average_absolute_h1_seminorm_error"]) for r in data])
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    ax.loglog(h, e_l2, "o-", color=PALETTE["safe"], label=r"$L^2$ error")
    ax.loglog(h, e_h1, "s-", color=PALETTE["raw"], label=r"$H^1$ seminorm error")
    if len(h) >= 2:
        ax.loglog(h, e_l2[0] * (h / h[0])**2, "--", color=PALETTE["safe"], alpha=.65, label=r"$O(h^2)$ reference")
        ax.loglog(h, e_h1[0] * (h / h[0]), ":", color=PALETTE["raw"], alpha=.8, label=r"$O(h)$ reference")
    ax.set_xlabel(r"Mesh size $h$")
    ax.set_ylabel("Time-averaged absolute error")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_temporal_convergence(rows: Sequence[Mapping], path: str | Path) -> None:
    data = sorted(rows, key=lambda r: float(r["dt"]), reverse=True)
    dt = np.asarray([float(r["dt"]) for r in data])
    err = np.asarray([float(r["cauchy_l2_difference_dt_dt_half"]) for r in data])
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    ax.loglog(dt, err, "o-", color=PALETTE["target"], label=r"$\|u_{\Delta t}-u_{\Delta t/2}\|_{L^2}$")
    if len(dt) >= 2:
        ax.loglog(dt, err[0] * dt / dt[0], "--", color=PALETTE["zero"], label=r"$O(\Delta t)$ reference")
    ax.set_xlabel(r"Time step $\Delta t$")
    ax.set_ylabel("Final-time Cauchy difference")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_peclet_distribution(rows: Sequence[Mapping], path: str | Path) -> None:
    data = sorted(rows, key=lambda r: (float(r["time"]), str(r["mesh_type"])))
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for mesh_type, color, marker in (("fixed", PALETTE["fixed"], "o"), ("ale", PALETTE["safe"], "s")):
        subset = [r for r in data if r["mesh_type"] == mesh_type]
        if not subset:
            continue
        times = [float(r["time"]) for r in subset]
        median = [float(r["Pe_median"]) for r in subset]
        p95 = [float(r["Pe_p95"]) for r in subset]
        ax.plot(times, median, marker=marker, color=color, label=f"{mesh_type.upper()} median")
        ax.plot(times, p95, linestyle="--", color=color, alpha=.8, label=f"{mesh_type.upper()} 95th percentile")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Element Péclet number $Pe_K$")
    ax.legend(ncol=2)
    fig.tight_layout()
    _save(fig, path)


def plot_supg_comparison(rows: Sequence[Mapping], path: str | Path) -> None:
    order = ["fixed", "static_ale", "moving_ale"]
    modes = ["galerkin", "supg"]
    fig, ax = plt.subplots(figsize=(4.7, 2.8))
    x = np.arange(len(order), dtype=float)
    width = .34
    for j, mode in enumerate(modes):
        ys = []
        for mesh_type in order:
            match = next((r for r in rows if r["mesh_type"] == mesh_type and r["mode"] == mode), None)
            ys.append(float(match["time_average_absolute_l2_error"]) if match is not None else np.nan)
        ax.bar(x + (j-.5)*width, ys, width, label=mode, color=PALETTE["fixed"] if mode == "galerkin" else PALETTE["safe"])
    ax.set_xticks(x, ["Fixed", "Static ALE", "Moving ALE"])
    ax.set_ylabel(r"Time-averaged absolute $L^2$ error")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_work_precision(rows: Sequence[Mapping], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(3.65, 2.75))
    for mode, color, marker in (("galerkin", PALETTE["fixed"], "o"), ("supg", PALETTE["safe"], "s")):
        subset = sorted((r for r in rows if r["mode"] == mode), key=lambda r: int(r["n_dofs"]))
        if not subset:
            continue
        runtime = [float(r["total_wall_seconds_median"]) for r in subset]
        error = [float(r["time_average_absolute_l2_error"]) for r in subset]
        ax.loglog(runtime, error, marker=marker, color=color, label=mode)
        for r, x, y in zip(subset, runtime, error):
            ax.annotate(str(r["nx"]), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Median CPU wall-clock time (s)")
    ax.set_ylabel(r"Time-averaged absolute $L^2$ error")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)
