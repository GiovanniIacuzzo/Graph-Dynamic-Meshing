"""Publication-quality vector-PDF figures for the Stage-D PDE-aware benchmark.

All figures enforce English labels, consistent journal styling, and vector-only
PDF export via ``plotting.style.save_figure``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .style import PALETTE, save_figure, set_publication_style

STRATEGIES: List[tuple[str, str, str, str]] = [
    ("fixed_fem", "Fixed FEM", PALETTE["fixed"], "-"),
    ("stage_a_gnn", "Stage A GNN + safety filter (ALE)", PALETTE["raw"], "--"),
    ("stage_d_gnn", "Stage D GNN + safety filter (ALE)", PALETTE["safe"], "-."),
]

STRATEGY_SHORT_LABELS: Dict[str, str] = {
    "fixed_fem": "Fixed\nFEM",
    "stage_a_gnn": "Stage A GNN\n+ safety (ALE)",
    "stage_d_gnn": "Stage D GNN\n+ safety (ALE)",
}

SPLITS: List[tuple[str, str]] = [
    ("test_interp", "In-domain interpolation (test_interp)"),
    ("test_extrap", "Out-of-distribution extrapolation (test_extrap)"),
]


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str | None = None,
    strategy: str | None = None,
) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for row in rows:
        if split is not None and str(row.get("split")) != split:
            continue
        if strategy is not None and str(row.get("strategy")) != strategy:
            continue
        out.append(row)
    return out


def _time_grid_and_matrix(
    timestep_rows: Sequence[Mapping[str, Any]],
    split: str,
    strategy: str,
    metric: str,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    subset = _group_rows(timestep_rows, split=split, strategy=strategy)
    if not subset:
        return np.asarray([], dtype=float), np.empty((0, 0), dtype=float)
    by_case: Dict[str, List[tuple[int, float, float]]] = {}
    for row in subset:
        cid = str(row["case_id"])
        step = int(row["step"])
        t_val = float(row["time"])
        m_val = float(row[metric]) * scale
        by_case.setdefault(cid, []).append((step, t_val, m_val))
    case_ids = sorted(by_case.keys())
    first_series = sorted(by_case[case_ids[0]], key=lambda item: item[0])
    times = np.asarray([item[1] for item in first_series], dtype=float)
    matrix = np.zeros((len(case_ids), len(first_series)), dtype=float)
    for idx, cid in enumerate(case_ids):
        ordered = sorted(by_case[cid], key=lambda item: item[0])
        matrix[idx, :] = [item[2] for item in ordered]
    return times, matrix


def plot_l2_error_over_time(
    timestep_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """Figure 1: Relative L2 error over time for the three strategies."""
    set_publication_style()
    out_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=False)

    for ax, (split_key, split_title) in zip(axes, SPLITS):
        has_data = False
        for strat_key, strat_label, colour, linestyle in STRATEGIES:
            times, mat = _time_grid_and_matrix(
                timestep_rows, split_key, strat_key, "relative_l2_error", scale=100.0
            )
            if mat.size == 0:
                continue
            has_data = True
            mean_curve = np.mean(mat, axis=0)
            q25_curve = np.quantile(mat, 0.25, axis=0)
            q75_curve = np.quantile(mat, 0.75, axis=0)
            ax.plot(
                times,
                mean_curve,
                label=strat_label,
                color=colour,
                linestyle=linestyle,
                linewidth=1.55,
            )
            if mat.shape[0] > 1:
                ax.fill_between(
                    times, q25_curve, q75_curve, color=colour, alpha=0.16, linewidth=0.0
                )
        ax.set_title(split_title)
        ax.set_xlabel("Time $t$")
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        if has_data:
            ax.legend(loc="upper left", fontsize=6.6)
    axes[0].set_ylabel(r"Relative $L^2(\Omega)$ error [\%]")
    fig.suptitle(r"Transient PDE relative $L^2$ error (mean and interquartile band)")
    fig.tight_layout()
    return save_figure(fig, out_dir / "fig1_relative_l2_error_over_time")


def plot_integrated_and_final_errors(
    case_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """Figure 2: Comparison of time-integrated and final relative L2 errors."""
    set_publication_style()
    out_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=False)
    metrics_spec = [
        ("integrated_l2_error", r"Time-integrated relative $L^2$ error $\overline{E}_{L^2}$ [\%]"),
        ("final_l2_error", r"Final relative $L^2$ error $E_{L^2}(T)$ [\%]"),
    ]
    x_splits = np.arange(len(SPLITS), dtype=float)
    width = 0.24

    for ax, (metric_key, ylabel) in zip(axes, metrics_spec):
        for s_idx, (strat_key, strat_label, colour, _) in enumerate(STRATEGIES):
            offset = (s_idx - 1) * width
            means: List[float] = []
            stds: List[float] = []
            positions: List[float] = []
            for split_idx, (split_key, _) in enumerate(SPLITS):
                subset = _group_rows(case_rows, split=split_key, strategy=strat_key)
                vals = np.asarray(
                    [float(r[metric_key]) * 100.0 for r in subset if np.isfinite(float(r[metric_key]))],
                    dtype=float,
                )
                if vals.size == 0:
                    continue
                pos = float(x_splits[split_idx] + offset)
                positions.append(pos)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
                jitter = np.linspace(-0.04, 0.04, vals.size) if vals.size > 1 else np.zeros(1)
                ax.scatter(
                    pos + jitter,
                    vals,
                    color=colour,
                    edgecolor="white",
                    linewidth=0.35,
                    s=18,
                    alpha=0.82,
                    zorder=4,
                )
            if positions:
                ax.bar(
                    positions,
                    means,
                    width=width * 0.88,
                    yerr=stds,
                    capsize=2.5,
                    label=strat_label,
                    color=colour,
                    alpha=0.32,
                    edgecolor=colour,
                    linewidth=0.9,
                    error_kw={"elinewidth": 0.9, "ecolor": "#1E293B"},
                )
        ax.set_xticks(x_splits, ["In-domain\n(test_interp)", "Extrapolation\n(test_extrap)"])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.legend(loc="best", fontsize=6.5)
    axes[0].set_title(r"Time-integrated $L^2$ error")
    axes[1].set_title(r"Final $L^2$ error at $t = T$")
    fig.suptitle("Integrated and final PDE accuracy across held-out splits")
    fig.tight_layout()
    return save_figure(fig, out_dir / "fig2_integrated_and_final_l2_error")


def plot_computational_cost_comparison(
    case_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """Figure 3: Total computational cost and mutually exclusive component breakdown."""
    set_publication_style()
    out_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))

    timed_rows = [r for r in runtime_rows if not bool(r.get("is_warmup", False))]
    source_rows = timed_rows if timed_rows else list(case_rows)

    x = np.arange(len(STRATEGIES), dtype=float)
    x_labels = [STRATEGY_SHORT_LABELS[s[0]] for s in STRATEGIES]
    has_split_mesh_timers = any(
        ("safety_filter_time" in r and "coord_update_time" in r) for r in source_rows
    )
    if has_split_mesh_timers:
        component_specs = [
            ("fem_solve_time", "Eulerian FEM solve", PALETTE["fixed"]),
            ("ale_time", "Native ALE-FEM solve", PALETTE["safe"]),
            ("gnn_inference_time", "GNN inference & features", PALETTE["raw"]),
            ("safety_filter_time", "Native C++ safety filter", PALETTE["accent"]),
            ("coord_update_time", "Mesh coordinate update", PALETTE["warning"]),
            ("remapping_time", "Remapping", PALETTE["target"]),
            ("other_overhead_time", "Other runtime overhead", PALETTE["zero"]),
        ]
    else:
        component_specs = [
            ("fem_solve_time", "Eulerian FEM solve", PALETTE["fixed"]),
            ("ale_time", "Native ALE-FEM solve", PALETTE["safe"]),
            ("gnn_inference_time", "GNN inference & features", PALETTE["raw"]),
            ("mesh_update_time", "Safety filter & mesh update", PALETTE["accent"]),
            ("remapping_time", "Remapping", PALETTE["target"]),
            ("other_overhead_time", "Other runtime overhead", PALETTE["zero"]),
        ]

    bottoms = np.zeros(len(STRATEGIES), dtype=float)
    for comp_key, comp_label, comp_colour in component_specs:
        comp_means = np.zeros(len(STRATEGIES), dtype=float)
        for idx, (strat_key, _, _, _) in enumerate(STRATEGIES):
            subset = [r for r in source_rows if str(r.get("strategy")) == strat_key]
            vals = np.asarray(
                [float(r.get(comp_key, 0.0)) for r in subset if np.isfinite(float(r.get(comp_key, 0.0)))],
                dtype=float,
            )
            comp_means[idx] = float(np.mean(vals)) if vals.size else 0.0
        if np.any(comp_means > 1.0e-9):
            axes[0].bar(
                x,
                comp_means,
                bottom=bottoms,
                width=0.56,
                label=comp_label,
                color=comp_colour,
                alpha=0.88,
            )
            bottoms += comp_means
    axes[0].set_xticks(x, x_labels)
    axes[0].set_ylabel("Mean execution time per episode [s]")
    axes[0].set_title("Runtime component decomposition")
    axes[0].grid(True, axis="y")
    axes[0].grid(False, axis="x")
    axes[0].legend(loc="upper left", fontsize=6.2)

    width = 0.24
    x_splits = np.arange(len(SPLITS), dtype=float)
    for s_idx, (strat_key, strat_label, colour, _) in enumerate(STRATEGIES):
        offset = (s_idx - 1) * width
        for split_idx, (split_key, _) in enumerate(SPLITS):
            subset = [
                r for r in source_rows
                if str(r.get("split")) == split_key and str(r.get("strategy")) == strat_key
            ]
            vals = np.asarray([float(r["total_wall_time"]) for r in subset], dtype=float)
            if vals.size == 0:
                continue
            pos = float(x_splits[split_idx] + offset)
            med = float(np.median(vals))
            q25 = float(np.quantile(vals, 0.25))
            q75 = float(np.quantile(vals, 0.75))
            mean_val = float(np.mean(vals))
            axes[1].bar(
                pos,
                med,
                width=width * 0.86,
                color=colour,
                alpha=0.34,
                edgecolor=colour,
                linewidth=0.9,
                label=strat_label if split_idx == 0 else None,
            )
            axes[1].errorbar(
                pos,
                med,
                yerr=[[max(0.0, med - q25)], [max(0.0, q75 - med)]],
                fmt="o",
                color=colour,
                ecolor="#111827",
                elinewidth=1.0,
                capsize=2.6,
                markersize=3.8,
                zorder=4,
            )
            axes[1].scatter(
                [pos],
                [mean_val],
                marker="D",
                s=18,
                color="#111827",
                zorder=5,
                label="Mean" if (s_idx == 0 and split_idx == 0) else None,
            )
    axes[1].set_xticks(x_splits, ["In-domain\n(test_interp)", "Extrapolation\n(test_extrap)"])
    axes[1].set_ylabel("Total wall-clock time [s]")
    axes[1].set_title("Total wall time (median, IQR, and mean)")
    axes[1].grid(True, axis="y")
    axes[1].grid(False, axis="x")
    axes[1].legend(loc="upper left", fontsize=6.3)

    fig.suptitle("Computational cost comparison across strategies")
    fig.tight_layout()
    return save_figure(fig, out_dir / "fig3_computational_cost_comparison")


def plot_geometric_quality_evolution(
    timestep_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    epsilon: float = 1.0e-4,
) -> list[Path]:
    """Figure 4: Evolution of minimum Jacobian and element quality over time."""
    set_publication_style()
    out_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

    all_steps = list(timestep_rows)
    for strat_key, strat_label, colour, linestyle in STRATEGIES:
        subset = [r for r in all_steps if str(r.get("strategy")) == strat_key]
        if not subset:
            continue
        by_step: Dict[int, List[Mapping[str, Any]]] = {}
        for r in subset:
            by_step.setdefault(int(r["step"]), []).append(r)
        steps_sorted = sorted(by_step.keys())
        times = np.asarray([float(by_step[s][0]["time"]) for s in steps_sorted], dtype=float)
        min_j_worst = np.asarray(
            [min(float(item["min_jacobian"]) for item in by_step[s]) for s in steps_sorted],
            dtype=float,
        )
        min_j_median = np.asarray(
            [float(np.median([float(item["min_jacobian"]) for item in by_step[s]])) for s in steps_sorted],
            dtype=float,
        )
        mean_q = np.asarray(
            [float(np.mean([float(item["mean_mesh_quality"]) for item in by_step[s]])) for s in steps_sorted],
            dtype=float,
        )
        q25_q = np.asarray(
            [float(np.quantile([float(item["mean_mesh_quality"]) for item in by_step[s]], 0.25)) for s in steps_sorted],
            dtype=float,
        )
        q75_q = np.asarray(
            [float(np.quantile([float(item["mean_mesh_quality"]) for item in by_step[s]], 0.75)) for s in steps_sorted],
            dtype=float,
        )

        axes[0].plot(
            times,
            min_j_worst,
            label=f"{strat_label} (worst case)",
            color=colour,
            linestyle=linestyle,
            linewidth=1.5,
        )
        axes[0].plot(
            times,
            min_j_median,
            color=colour,
            linestyle=":",
            linewidth=1.0,
            alpha=0.75,
        )
        axes[1].plot(
            times,
            mean_q,
            label=strat_label,
            color=colour,
            linestyle=linestyle,
            linewidth=1.5,
        )
        if len(by_step[steps_sorted[0]]) > 1:
            axes[1].fill_between(times, q25_q, q75_q, color=colour, alpha=0.15, linewidth=0.0)

    axes[0].axhline(
        epsilon,
        color=PALETTE["warning"],
        linestyle="--",
        linewidth=1.0,
        label=rf"Safety floor $\epsilon={epsilon:g}$",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Time $t$")
    axes[0].set_ylabel(r"Minimum oriented Jacobian $\min_e J_e$")
    axes[0].set_title("Mesh admissibility along trajectories")
    axes[0].grid(True, which="both", axis="y")
    axes[0].grid(False, axis="x")
    axes[0].legend(loc="best", fontsize=6.2)

    axes[1].set_xlabel("Time $t$")
    axes[1].set_ylabel(r"Mean Escobar element quality $\bar{\eta}$")
    axes[1].set_title("Element quality evolution")
    axes[1].grid(True, axis="y")
    axes[1].grid(False, axis="x")
    axes[1].legend(loc="best", fontsize=6.5)

    fig.suptitle("Geometric quality and admissibility over time")
    fig.tight_layout()
    return save_figure(fig, out_dir / "fig4_geometric_quality_evolution")


def plot_safety_filter_intervention(
    timestep_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """Figure 5: Safety filter contraction alpha and intervention rate over time."""
    set_publication_style()
    out_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

    for strat_key, strat_label, colour, linestyle in STRATEGIES:
        subset = [r for r in timestep_rows if str(r.get("strategy")) == strat_key and int(r.get("step", 0)) > 0]
        if not subset:
            continue
        by_step: Dict[int, List[Mapping[str, Any]]] = {}
        for r in subset:
            by_step.setdefault(int(r["step"]), []).append(r)
        steps_sorted = sorted(by_step.keys())
        times = np.asarray([float(by_step[s][0]["time"]) for s in steps_sorted], dtype=float)
        alpha_mean = np.asarray(
            [float(np.mean([float(item["safety_alpha"]) for item in by_step[s]])) for s in steps_sorted],
            dtype=float,
        )
        alpha_min = np.asarray(
            [float(np.min([float(item["safety_alpha"]) for item in by_step[s]])) for s in steps_sorted],
            dtype=float,
        )
        intervention_rate = np.asarray(
            [100.0 * float(np.mean([float(item["safety_intervened"]) for item in by_step[s]])) for s in steps_sorted],
            dtype=float,
        )

        axes[0].plot(
            times,
            alpha_mean,
            label=f"{strat_label} (mean)",
            color=colour,
            linestyle=linestyle,
            linewidth=1.5,
        )
        if strat_key != "fixed_fem":
            axes[0].plot(
                times,
                alpha_min,
                color=colour,
                linestyle=":",
                linewidth=1.0,
                alpha=0.75,
                label=f"{strat_label} (min)",
            )
        axes[1].plot(
            times,
            intervention_rate,
            label=strat_label,
            color=colour,
            linestyle=linestyle,
            linewidth=1.5,
        )

    axes[0].set_ylim(-0.04, 1.05)
    axes[0].set_xlabel("Time $t$")
    axes[0].set_ylabel(r"Safety filter contraction factor $\alpha_{\mathrm{global}}$")
    axes[0].set_title(r"Evolution of $\alpha_{\mathrm{global}}(t)$")
    axes[0].grid(True, axis="y")
    axes[0].grid(False, axis="x")
    axes[0].legend(loc="lower left", fontsize=6.0)

    axes[1].set_ylim(-4.0, 104.0)
    axes[1].set_xlabel("Time $t$")
    axes[1].set_ylabel(r"Safety filter intervention rate [\%]")
    axes[1].set_title(r"Percentage of cases with $\alpha_{\mathrm{global}} < 1$")
    axes[1].grid(True, axis="y")
    axes[1].grid(False, axis="x")
    axes[1].legend(loc="best", fontsize=6.5)

    fig.suptitle("Safety filter intervention along the PDE trajectory")
    fig.tight_layout()
    return save_figure(fig, out_dir / "fig5_safety_filter_intervention")


def plot_interp_vs_extrap_comparison(
    case_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """Figure 6: Direct interpolation vs extrapolation comparison."""
    set_publication_style()
    out_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))

    fixed_by_case: Dict[str, Mapping[str, Any]] = {
        str(r["case_id"]): r for r in case_rows if str(r.get("strategy")) == "fixed_fem"
    }

    dynamic_strategies = [
        ("stage_a_gnn", "Stage A GNN + safety filter (ALE)", PALETTE["raw"]),
        ("stage_d_gnn", "Stage D GNN + safety filter (ALE)", PALETTE["safe"]),
    ]
    x_splits = np.arange(len(SPLITS), dtype=float)
    width = 0.30

    for s_idx, (strat_key, strat_label, colour) in enumerate(dynamic_strategies):
        offset = (s_idx - 0.5) * width
        means_delta: List[float] = []
        stds_delta: List[float] = []
        positions: List[float] = []
        for split_idx, (split_key, _) in enumerate(SPLITS):
            subset = _group_rows(case_rows, split=split_key, strategy=strat_key)
            rel_deltas: List[float] = []
            for row in subset:
                cid = str(row["case_id"])
                base = fixed_by_case.get(cid)
                if base is None:
                    continue
                e_fixed = max(float(base["integrated_l2_error"]), 1.0e-12)
                e_dyn = float(row["integrated_l2_error"])
                rel_deltas.append(100.0 * (e_dyn - e_fixed) / e_fixed)
            if not rel_deltas:
                continue
            vals = np.asarray(rel_deltas, dtype=float)
            pos = float(x_splits[split_idx] + offset)
            positions.append(pos)
            means_delta.append(float(np.mean(vals)))
            stds_delta.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
            jitter = np.linspace(-0.04, 0.04, vals.size) if vals.size > 1 else np.zeros(1)
            axes[0].scatter(
                pos + jitter,
                vals,
                color=colour,
                edgecolor="white",
                linewidth=0.35,
                s=20,
                alpha=0.85,
                zorder=4,
            )
        if positions:
            axes[0].bar(
                positions,
                means_delta,
                width=width * 0.88,
                yerr=stds_delta,
                capsize=2.6,
                color=colour,
                alpha=0.32,
                edgecolor=colour,
                linewidth=0.9,
                label=strat_label,
                error_kw={"elinewidth": 0.9, "ecolor": "#1E293B"},
            )

    axes[0].axhline(0.0, color="#111827", linestyle="--", linewidth=0.9, label="Fixed FEM baseline (0%)")
    axes[0].set_xticks(x_splits, ["In-domain\n(test_interp)", "Extrapolation\n(test_extrap)"])
    axes[0].set_ylabel(r"Relative change in $\overline{E}_{L^2}$ vs Fixed FEM [\%]")
    axes[0].set_title(r"Relative $\overline{E}_{L^2}$ change ($<0$: improvement)")
    axes[0].grid(True, axis="y")
    axes[0].grid(False, axis="x")
    axes[0].legend(loc="best", fontsize=6.4)

    markers = {"test_interp": "o", "test_extrap": "s"}
    for strat_key, strat_label, colour, _ in STRATEGIES:
        for split_key, _ in SPLITS:
            subset = _group_rows(case_rows, split=split_key, strategy=strat_key)
            if not subset:
                continue
            sigmas = np.asarray([float(r["sigma"]) for r in subset], dtype=float)
            int_l2 = np.asarray([float(r["integrated_l2_error"]) * 100.0 for r in subset], dtype=float)
            order = np.argsort(sigmas)
            short_split = "interp" if split_key == "test_interp" else "extrap"
            axes[1].scatter(
                sigmas[order],
                int_l2[order],
                color=colour,
                marker=markers[split_key],
                s=24,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.35,
                label=f"{strat_label} ({short_split})",
            )
    axes[1].axvspan(0.035, 0.075, color=PALETTE["grid"], alpha=0.28, label=r"Training $\sigma$ range")
    axes[1].set_xlabel(r"Gaussian width parameter $\sigma$")
    axes[1].set_ylabel(r"Time-integrated relative $L^2$ error $\overline{E}_{L^2}$ [\%]")
    axes[1].set_title(r"Generalization across feature scale $\sigma$")
    axes[1].grid(True, axis="y")
    axes[1].legend(loc="best", fontsize=5.6, ncol=2)

    fig.suptitle("In-domain interpolation vs out-of-distribution extrapolation")
    fig.tight_layout()
    return save_figure(fig, out_dir / "fig6_interpolation_vs_extrapolation")


def plot_paired_improvement_stage_d_vs_fixed(
    paired_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """Figure 7: Paired case-by-case improvement of Stage D + safety filter + ALE vs Fixed FEM."""
    set_publication_style()
    out_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))

    metrics_pair = [
        ("rel_diff_integrated_l2_pct", r"$\Delta \overline{E}_{L^2} / \overline{E}_{L^2}^{\mathrm{fixed}}$ [\%]"),
        ("rel_diff_final_l2_pct", r"$\Delta E_{L^2}(T) / E_{L^2}^{\mathrm{fixed}}(T)$ [\%]"),
    ]
    x_pos = np.arange(len(SPLITS), dtype=float)
    width = 0.28
    colours_m = [PALETTE["safe"], PALETTE["target"]]

    for m_idx, (m_key, m_label) in enumerate(metrics_pair):
        offset = (m_idx - 0.5) * width
        for split_idx, (split_key, _) in enumerate(SPLITS):
            sub = [r for r in paired_rows if str(r.get("split")) == split_key]
            vals = np.asarray(
                [float(r[m_key]) for r in sub if np.isfinite(float(r[m_key]))], dtype=float
            )
            if vals.size == 0:
                continue
            pos = float(x_pos[split_idx] + offset)
            mean_val = float(np.mean(vals))
            if vals.size > 1:
                rng = np.random.default_rng(1701 + m_idx * 10 + split_idx)
                boot = np.mean(vals[rng.integers(0, vals.size, size=(4000, vals.size))], axis=1)
                ci_low = float(np.quantile(boot, 0.025))
                ci_high = float(np.quantile(boot, 0.975))
            else:
                ci_low = mean_val
                ci_high = mean_val

            axes[0].bar(
                pos,
                mean_val,
                width=width * 0.86,
                color=colours_m[m_idx],
                alpha=0.34,
                edgecolor=colours_m[m_idx],
                linewidth=0.9,
                label=m_label if split_idx == 0 else None,
            )
            axes[0].errorbar(
                pos,
                mean_val,
                yerr=[[max(0.0, mean_val - ci_low)], [max(0.0, ci_high - mean_val)]],
                fmt="o",
                color=colours_m[m_idx],
                ecolor="#111827",
                elinewidth=1.0,
                capsize=2.8,
                markersize=3.8,
                zorder=5,
            )
            jitter = np.linspace(-0.045, 0.045, vals.size) if vals.size > 1 else np.zeros(1)
            axes[0].scatter(
                pos + jitter,
                vals,
                color=colours_m[m_idx],
                edgecolor="white",
                linewidth=0.35,
                s=18,
                alpha=0.82,
                zorder=4,
            )

    axes[0].axhline(0.0, color="#111827", linestyle="--", linewidth=0.9)
    axes[0].set_xticks(x_pos, ["In-domain\n(test_interp)", "Extrapolation\n(test_extrap)"])
    axes[0].set_ylabel(r"Paired relative error change vs Fixed FEM [\%]")
    axes[0].set_title("Paired accuracy gain (bars: mean & 95% bootstrap CI)")
    axes[0].grid(True, axis="y")
    axes[0].grid(False, axis="x")
    axes[0].legend(loc="best", fontsize=6.6)

    markers = {
        "test_interp": ("o", "In-domain (test_interp)", PALETTE["safe"]),
        "test_extrap": ("s", "Extrapolation (test_extrap)", PALETTE["raw"]),
    }
    for split_key, (marker, label, colour) in markers.items():
        sub = [r for r in paired_rows if str(r.get("split")) == split_key]
        if not sub:
            continue
        speed_ratios = np.asarray(
            [float(r["speed_ratio_stage_d_vs_fixed_median"]) for r in sub], dtype=float
        )
        rel_gains = np.asarray([float(r["rel_diff_integrated_l2_pct"]) for r in sub], dtype=float)
        axes[1].scatter(
            speed_ratios,
            rel_gains,
            marker=marker,
            color=colour,
            s=28,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
    axes[1].axhline(0.0, color="#111827", linestyle="--", linewidth=0.9, label="No accuracy change (0%)")
    axes[1].set_xlabel(r"Computational speed ratio $T_{\mathrm{wall}}^{\mathrm{Stage\,D}} / T_{\mathrm{wall}}^{\mathrm{Fixed}}$")
    axes[1].set_ylabel(r"Paired relative change in $\overline{E}_{L^2}$ [\%]")
    axes[1].set_title("Accuracy gain vs runtime overhead per case")
    axes[1].grid(True, which="both")
    axes[1].legend(loc="best", fontsize=6.8)

    fig.suptitle("Paired evaluation: Stage D GNN + safety filter (ALE) vs Fixed FEM")
    fig.tight_layout()
    return save_figure(fig, out_dir / "fig7_paired_stage_d_vs_fixed_fem")


def plot_stage_d_pde_benchmark(
    case_rows: Sequence[Mapping[str, Any]],
    timestep_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    epsilon: float = 1.0e-4,
    paired_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, list[Path]]:
    """Generate all required vector-PDF figures for the Stage-D PDE benchmark."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, list[Path]] = {
        "fig1_relative_l2_error_over_time": plot_l2_error_over_time(timestep_rows, out_dir),
        "fig2_integrated_and_final_l2_error": plot_integrated_and_final_errors(case_rows, out_dir),
        "fig3_computational_cost_comparison": plot_computational_cost_comparison(
            case_rows, runtime_rows, out_dir
        ),
        "fig4_geometric_quality_evolution": plot_geometric_quality_evolution(
            timestep_rows, out_dir, epsilon=epsilon
        ),
        "fig5_safety_filter_intervention": plot_safety_filter_intervention(timestep_rows, out_dir),
        "fig6_interpolation_vs_extrapolation": plot_interp_vs_extrap_comparison(case_rows, out_dir),
    }
    if paired_rows:
        outputs["fig7_paired_stage_d_vs_fixed_fem"] = plot_paired_improvement_stage_d_vs_fixed(
            paired_rows, out_dir
        )
    return outputs