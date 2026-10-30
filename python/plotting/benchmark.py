"""Publication-quality plots for Stage-A GDM benchmarks.

Figures are written as vector vector PDF only.
Summary-only plots use the available means and standard deviations; they never
pretend aggregate values are per-sample distributions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .style import PALETTE, save_figure, set_publication_style

METHODS = [
    ("B0_fixed", "Fixed mesh", PALETTE["fixed"]),
    ("B_zero", "Zero displacement", PALETTE["zero"]),
    ("B3_gnn_raw", "GNN (unfiltered)", PALETTE["raw"]),
    ("B4_gnn_safety", "GNN + safety filter", PALETTE["safe"]),
    ("B5_oracle", "Stored optimizer target", PALETTE["target"]),
]
SPLITS = [("test_interp", "In-domain test"), ("test_extrap", "Extrapolation test")]
METHOD_LABELS_SHORT = {
    "B0_fixed": "Fixed mesh",
    "B_zero": "Zero displacement",
    "B3_gnn_raw": "GNN\n(raw)",
    "B4_gnn_safety": "GNN + safety\nfilter",
    "B5_oracle": "Stored optimizer\ntarget",
}


def _finish(fig, out_dir: Path, stem: str) -> list[Path]:
    return save_figure(fig, out_dir / stem)


def _finite_array(values: Any, scale: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1) * scale
    return arr[np.isfinite(arr)]


def plot_benchmark_results(
    results: Mapping[str, Any],
    output_path: str | Path,
    *,
    epsilon: float = 1.0e-4,
) -> dict[str, list[Path]]:
    """Plot raw per-sample results into four independent publication figures.

    ``output_path`` is retained for compatibility; only its directory is used.
    """
    set_publication_style()
    output = Path(output_path)
    out_dir = output if (output.exists() and output.is_dir()) else output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, list[Path]] = {}

    # 1. Distribution of interpolation errors, preserving per-sample variation.
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.35), sharey=True)
    for ax, (split_key, split_label) in zip(axes, SPLITS):
        data_map = results.get(split_key, {})
        rows = []
        for key, label, colour in METHODS:
            vals = _finite_array(data_map.get(key, {}).get("interp_error", []), 1e4)
            if vals.size:
                rows.append((key, label, colour, vals))
        if not rows:
            ax.text(0.5, 0.5, "No valid results", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        bp = ax.boxplot(
            [r[3] for r in rows], patch_artist=True, widths=0.58, showfliers=True,
            medianprops={"color": "#111827", "linewidth": 1.15},
            whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
            flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.45},
        )
        for patch, (_, _, colour, _) in zip(bp["boxes"], rows):
            patch.set_facecolor(colour); patch.set_alpha(0.24); patch.set_edgecolor(colour)
        ax.set_xticks(np.arange(1, len(rows) + 1), [METHOD_LABELS_SHORT[r[0]] for r in rows])
        ax.set_title(split_label)
        ax.grid(True, axis="y"); ax.grid(False, axis="x")
    axes[0].set_ylabel(r"Integrated interpolation error $E_{\mathrm{interp}}$ ($\times 10^{-4}$)")
    fig.suptitle("Interpolation error across mesh-adaptation baselines")
    fig.tight_layout()
    paths["error"] = _finish(fig, out_dir, "stage_a_error_comparison")

    # 2. Geometric admissibility. Two panels avoid pooling the in-domain and OOD distributions.
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.35), sharey=True)
    for ax, (split_key, split_label) in zip(axes, SPLITS):
        data_map = results.get(split_key, {})
        rows = []
        for key, label, colour in METHODS:
            vals = _finite_array(data_map.get(key, {}).get("min_jacobian", []))
            if vals.size:
                rows.append((key, label, colour, vals))
        if rows:
            bp = ax.boxplot(
                [r[3] for r in rows], patch_artist=True, widths=0.58, showfliers=False,
                medianprops={"color": "#111827", "linewidth": 1.1},
                whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
            )
            for patch, (_, _, colour, _) in zip(bp["boxes"], rows):
                patch.set_facecolor(colour); patch.set_alpha(0.24); patch.set_edgecolor(colour)
            ax.set_xticks(np.arange(1, len(rows) + 1), [METHOD_LABELS_SHORT[r[0]] for r in rows])
        ax.axhline(epsilon, color=PALETTE["warning"], linestyle="--", linewidth=1.0,
                   label=rf"Admissibility floor $\epsilon={epsilon:g}$")
        ax.set_title(split_label); ax.grid(True, axis="y"); ax.grid(False, axis="x")
        ax.legend(loc="best", fontsize=6.8)
    axes[0].set_ylabel(r"Minimum oriented Jacobian $\min_e J_e$")
    fig.suptitle("Mesh admissibility under learned deformation")
    fig.tight_layout()
    paths["safety"] = _finish(fig, out_dir, "stage_a_mesh_admissibility")

    # 3. Element quality, again showing the split-wise sample distributions.
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.35), sharey=True)
    for ax, (split_key, split_label) in zip(axes, SPLITS):
        data_map = results.get(split_key, {})
        rows = []
        for key, label, colour in METHODS:
            vals = _finite_array(data_map.get(key, {}).get("mean_quality", []))
            if vals.size:
                rows.append((key, label, colour, vals))
        if rows:
            bp = ax.boxplot(
                [r[3] for r in rows], patch_artist=True, widths=0.58, showfliers=False,
                medianprops={"color": "#111827", "linewidth": 1.1},
                whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
            )
            for patch, (_, _, colour, _) in zip(bp["boxes"], rows):
                patch.set_facecolor(colour); patch.set_alpha(0.24); patch.set_edgecolor(colour)
            ax.set_xticks(np.arange(1, len(rows) + 1), [METHOD_LABELS_SHORT[r[0]] for r in rows])
        ax.set_title(split_label); ax.grid(True, axis="y"); ax.grid(False, axis="x")
    axes[0].set_ylabel(r"Mean element quality $\eta$")
    fig.suptitle("Element-quality distribution after adaptation")
    fig.tight_layout()
    paths["quality"] = _finish(fig, out_dir, "stage_a_element_quality")

    # 4. Timings are separated by split and use only positive measurements.
    # The evaluator warms up the model before recording timings. Fixed/zero
    # baselines intentionally have no measured inference time and are omitted.
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.35), sharey=True)
    runtime_keys = [
        ("B3_gnn_raw", "GNN forward pass", PALETTE["raw"]),
        ("B4_gnn_safety", "GNN + safety filter", PALETTE["safe"]),
        ("B5_oracle", "Stored offline optimizer", PALETTE["target"]),
    ]
    for ax, (split_key, split_label) in zip(axes, SPLITS):
        data_map = results.get(split_key, {})
        rows = []
        for key, label, colour in runtime_keys:
            vals = _finite_array(data_map.get(key, {}).get("timing_ms", []))
            vals = vals[vals > 0]
            if vals.size:
                rows.append((key, label, colour, vals))
        if rows:
            bp = ax.boxplot(
                [r[3] for r in rows], patch_artist=True, widths=0.58, showfliers=True,
                medianprops={"color": "#111827", "linewidth": 1.1},
                flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.45},
            )
            for patch, (_, _, colour, _) in zip(bp["boxes"], rows):
                patch.set_facecolor(colour); patch.set_alpha(0.24); patch.set_edgecolor(colour)
            ax.set_xticks(np.arange(1, len(rows) + 1), [r[1] for r in rows])
        ax.set_yscale("log"); ax.set_title(split_label)
        ax.grid(True, which="both", axis="y"); ax.grid(False, axis="x")
    axes[0].set_ylabel("Reported execution time per sample [ms, log scale]")
    fig.suptitle("Online inference and offline target-generation cost")
    fig.tight_layout()
    paths["runtime"] = _finish(fig, out_dir, "stage_a_runtime_comparison")
    return paths


def plot_benchmark_summary(
    summary: Mapping[str, Any],
    output_dir: str | Path,
    *,
    epsilon: float = 1.0e-4,
) -> dict[str, list[Path]]:
    """Plot an aggregate JSON summary; means/SD are not treated as distributions."""
    set_publication_style()
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    present_methods = [m for m in METHODS if any(m[0] in summary.get(s, {}) for s, _ in SPLITS)]
    if not present_methods:
        raise ValueError("Summary JSON has no recognized Stage-A benchmark methods.")
    paths: dict[str, list[Path]] = {}
    x = np.arange(len(present_methods))
    labels = [METHOD_LABELS_SHORT[m[0]] for m in present_methods]
    offsets = {"test_interp": -0.12, "test_extrap": 0.12}
    markers = {"test_interp": "o", "test_extrap": "s"}

    # Figure 1: horizontal mean +/- sample SD bars keep method labels readable.
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.8), sharey=True)
    y = np.arange(len(present_methods))
    for ax, (split_key, split_label) in zip(axes, SPLITS):
        data = summary.get(split_key, {})
        max_endpoint = 0.0
        for j, (key, label, colour) in enumerate(present_methods):
            rec = data.get(key)
            if not rec:
                continue
            mean = float(rec["mean_interp_error_x1e4"])
            sd = max(0.0, float(rec.get("std_interp_error_x1e4", 0.0)))
            if np.isfinite(mean) and np.isfinite(sd):
                ax.errorbar(mean, j, xerr=sd, fmt="o", ms=4.4, capsize=2.8,
                            color=colour, ecolor=colour, linewidth=1.0)
                max_endpoint = max(max_endpoint, mean + sd)
        ax.set_title(split_label)
        ax.set_xlabel(r"Mean $E_{\mathrm{interp}}$ ($\times 10^{-4}$; whiskers: SD)")
        ax.set_xlim(left=0.0, right=max_endpoint * 1.08 if max_endpoint > 0 else 1.0)
        ax.set_yticks(y, labels)
        ax.grid(True, axis="x"); ax.grid(False, axis="y")
    axes[0].invert_yaxis()
    fig.suptitle("Stage-A interpolation error: mean and sample spread")
    fig.tight_layout()
    paths["error"] = _finish(fig, out_dir, "stage_a_error_comparison")

    # Figure 2: minimum observed Jacobian, with log axis for interpretable safety margins.
    fig, ax = plt.subplots(figsize=(6.4, 3.45))
    for j, (key, label, colour) in enumerate(present_methods):
        for split_key, split_label in SPLITS:
            rec = summary.get(split_key, {}).get(key)
            if not rec:
                continue
            val = float(rec.get("min_jacobian", np.nan))
            if np.isfinite(val) and val > 0.0:
                ax.scatter(j + offsets[split_key], val, s=28, marker=markers[split_key],
                           color=colour, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(epsilon, color=PALETTE["warning"], linestyle="--", linewidth=1.0,
               label=rf"Admissibility floor $\epsilon={epsilon:g}$")
    ax.set_yscale("log"); ax.set_xticks(x, labels)
    ax.set_ylabel(r"Minimum observed oriented Jacobian $\min_e J_e$")
    ax.set_title("Worst-case mesh admissibility (aggregate summary)")
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", color="#111827", label="In-domain test"),
        Line2D([0], [0], marker="s", linestyle="none", color="#111827", label="Extrapolation test"),
        Line2D([0], [0], color=PALETTE["warning"], linestyle="--", label=rf"Floor $\epsilon={epsilon:g}$"),
    ]
    ax.legend(handles=legend, fontsize=7.1); ax.grid(True, which="both", axis="y"); ax.grid(False, axis="x")
    fig.tight_layout()
    paths["safety"] = _finish(fig, out_dir, "stage_a_mesh_admissibility")

    # Figure 3: absolute mean element quality. Values are means only; no spread is inferred.
    fig, ax = plt.subplots(figsize=(6.4, 3.45))
    for j, (key, label, colour) in enumerate(present_methods):
        for split_key, split_label in SPLITS:
            rec = summary.get(split_key, {}).get(key)
            if not rec:
                continue
            val = float(rec.get("mean_quality", np.nan))
            if np.isfinite(val):
                ax.scatter(j + offsets[split_key], val, s=30, marker=markers[split_key],
                           color=colour, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(x, labels); ax.set_ylabel(r"Mean element quality $\bar{\eta}$")
    ax.set_title("Mean element quality by test split")
    ax.grid(True, axis="y"); ax.grid(False, axis="x")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="none", color="#111827", label="In-domain test"),
        Line2D([0], [0], marker="s", linestyle="none", color="#111827", label="Extrapolation test"),
    ], fontsize=7.1)
    fig.tight_layout()
    paths["quality"] = _finish(fig, out_dir, "stage_a_element_quality")

    # Figure 4: means only, and only positive times. Zero entries mean “not timed”, not zero cost.
    fig, ax = plt.subplots(figsize=(6.4, 3.45))
    runtime_keys = [
        ("B3_gnn_raw", "GNN forward pass", PALETTE["raw"]),
        ("B4_gnn_safety", "GNN + safety filter", PALETTE["safe"]),
        ("B5_oracle", "Stored offline optimizer", PALETTE["target"]),
    ]
    positions, labels_rt, colours_rt = [], [], []
    xpos = 0
    tick_positions, tick_labels = [], []
    for split_key, split_label in SPLITS:
        for key, label, colour in runtime_keys:
            rec = summary.get(split_key, {}).get(key)
            if not rec:
                continue
            val = float(rec.get("mean_time_ms", np.nan))
            if not np.isfinite(val) or val <= 0.0:
                continue
            positions.append(xpos); labels_rt.append(f"{label}\n{split_label}"); colours_rt.append(colour)
            ax.scatter(xpos, val, s=34, color=colour, edgecolor="white", linewidth=0.4, zorder=3)
            xpos += 1
        tick_positions.append(xpos - max(1, len(runtime_keys)) / 2)
        tick_labels.append(split_label)
        xpos += 0.4
    ax.set_yscale("log")
    # Split labels make the two measurement groups readable without overloading the x-axis.
    ax.set_xticks(tick_positions, tick_labels)
    ax.set_xlim(-0.5, max(0.5, xpos - 0.5))
    ax.set_ylabel("Reported mean execution time per sample [ms, log scale]")
    ax.set_title("Reported online and offline runtime (summary means)")
    ax.grid(True, which="both", axis="y"); ax.grid(False, axis="x")
    legend_handles = [Line2D([0], [0], marker="o", linestyle="none", color=c, label=l)
                      for _, l, c in runtime_keys]
    ax.legend(handles=legend_handles, fontsize=7.1)
    fig.tight_layout()
    paths["runtime"] = _finish(fig, out_dir, "stage_a_runtime_comparison")
    return paths


def plot_benchmark_summary_file(
    summary_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    epsilon: float = 1.0e-4,
) -> dict[str, list[Path]]:
    """Load ``benchmark_summary.json`` and export PDF figures."""
    src = Path(summary_path)
    summary = json.loads(src.read_text(encoding="utf-8"))
    return plot_benchmark_summary(summary, output_dir or src.parent, epsilon=epsilon)


def plot_benchmark_raw_file(
    raw_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    epsilon: float = 1.0e-4,
) -> dict[str, list[Path]]:
    """Load saved per-sample JSON results and export distribution plots."""
    src = Path(raw_path)
    raw = json.loads(src.read_text(encoding="utf-8"))
    return plot_benchmark_results(raw, output_dir or src.parent, epsilon=epsilon)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export vector Stage-A benchmark figures.")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--summary", help="Path to benchmark_summary.json (means and standard deviations).")
    src_group.add_argument("--raw-results", help="Path to benchmark_raw_results.json (per-sample values).")
    parser.add_argument("--output-dir", default=None, help="Output directory (defaults to the input file's directory).")
    parser.add_argument("--epsilon", type=float, default=1.0e-4, help="Mesh admissibility threshold.")
    args = parser.parse_args()
    if args.summary:
        paths = plot_benchmark_summary_file(args.summary, args.output_dir, epsilon=args.epsilon)
    else:
        paths = plot_benchmark_raw_file(args.raw_results, args.output_dir, epsilon=args.epsilon)
    for key, outputs in paths.items():
        print(f"{key}: " + ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
