"""Forensic GDM safety-filter and moving-source tracking audit.

Runs a small, explicitly selected subset of cases using the *frozen Stage-D
checkpoint*. It records the raw GNN proposal, the native-filtered displacement,
limiting-cell diagnostics and mesh resolution around the moving Gaussian
source. It does not tune the checkpoint or any hyperparameter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
import matplotlib.tri as mtri

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LOG = logging.getLogger("gdm.stage_f.safety_tracking_audit")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return obj


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key)); seen.add(str(key))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _attach_case_metadata(
    rows: Sequence[Mapping[str, Any]], *, case_id: str, split: str, seed: int
) -> list[dict[str, Any]]:
    """Attach trajectory identity to every tracking row, including t=0.

    The initial mesh-tracking row is created before the time-stepping loop in
    ``run_pde_trajectory_episode`` and historically lacked case metadata,
    unlike subsequent rows.  Enriching at the per-case boundary guarantees
    that both initial and evolved meshes are grouped consistently.
    Existing non-matching metadata is treated as a data-integrity error.
    """
    enriched: list[dict[str, Any]] = []
    expected = {"case_id": str(case_id), "split": str(split), "seed": int(seed)}
    for index, source in enumerate(rows):
        row = dict(source)
        for key, value in expected.items():
            if key in row and str(row[key]) != str(value):
                raise ValueError(
                    f"Tracking row {index} has inconsistent {key}={row[key]!r}; "
                    f"expected {value!r} for case {case_id}."
                )
            row[key] = value
        enriched.append(row)
    return enriched


def _plot_case_snapshots(
    npz_path: Path,
    out_path: Path,
    case_id: str,
    *,
    preview_path: Path | None = None,
    dpi: int = 600,
) -> None:
    """Create a publication-quality, four-panel moving-source figure.

    The figure uses the saved mesh and nodal field snapshots only. It does not
    rerun the PDE solver, the GNN, or the native safety filter. All panels share
    one scalar-field normalization, one colourbar, and the same spatial axes.
    The current numerical field is plotted on its own deformed P1 mesh.
    """
    from matplotlib import patheffects as path_effects
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    npz_path = Path(npz_path)
    out_path = Path(out_path)
    if preview_path is None:
        preview_path = out_path.with_suffix(".png")
    else:
        preview_path = Path(preview_path)

    with np.load(npz_path) as data:
        required = {"times", "nodes", "fields", "centers", "cells", "sigma"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Snapshot archive is missing arrays: {sorted(missing)}")
        times = np.asarray(data["times"], dtype=np.float64)
        nodes = np.asarray(data["nodes"], dtype=np.float64)
        fields = np.asarray(data["fields"], dtype=np.float64)
        centers = np.asarray(data["centers"], dtype=np.float64)
        cells = np.asarray(data["cells"], dtype=np.int64)
        sigma = float(data["sigma"])

    # Strict shape and numerical checks prevent a visually plausible plot from
    # being generated from mismatched snapshots or an invalid mesh.
    if times.ndim != 1 or times.size < 1:
        raise ValueError("times must be a non-empty one-dimensional array")
    n_times = int(times.size)
    if nodes.ndim != 3 or nodes.shape[0] != n_times or nodes.shape[2] != 2:
        raise ValueError("nodes must have shape (n_times, n_nodes, 2)")
    if fields.shape != nodes.shape[:2]:
        raise ValueError("fields must have shape (n_times, n_nodes)")
    if centers.shape != (n_times, 2):
        raise ValueError("centers must have shape (n_times, 2)")
    if cells.ndim != 2 or cells.shape[1] != 3 or cells.size == 0:
        raise ValueError("cells must be a non-empty (n_cells, 3) array")
    if cells.min() < 0 or cells.max() >= nodes.shape[1]:
        raise ValueError("cells contain node indices outside the nodes array")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    for name, array in (("times", times), ("nodes", nodes), ("fields", fields), ("centers", centers)):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinite values")
    if n_times > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")

    vmin = float(np.min(fields))
    vmax = float(np.max(fields))
    if not vmax > vmin:
        raise ValueError("The saved scalar fields have no non-zero range")

    # One norm and one set of levels are shared by every time snapshot. This is
    # essential: independent per-panel normalizations make temporal amplitudes
    # visually incomparable and invalidate a shared colourbar.
    norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
    levels = np.linspace(vmin, vmax, 31)


    # Journal-oriented compact styling: restrained mesh edges, clear symbols,
    # serif typography, and a wide single-row layout suited to a two-column paper.
    style = {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.2,
        "axes.labelsize": 7.4,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.55,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }

    # Colour-blind-conscious accents that remain distinguishable over viridis.
    centre_colour = "#D55E00"       # vermilion: prescribed Gaussian centre
    peak_colour = "#202124"         # near-black: nodal maximizer of u_h
    sigma_colour = "#E69F00"        # orange: r = sigma
    two_sigma_colour = "#56B4E9"    # sky blue: r = 2 sigma
    mesh_colour = "#17212B"

    with plt.rc_context(style):
        fig = plt.figure(figsize=(7.25, 2.58), constrained_layout=False)
        grid = fig.add_gridspec(
            1, 5,
            width_ratios=(1.0, 1.0, 1.0, 1.0, 0.055),
            left=0.075, right=0.945, bottom=0.225, top=0.785,
            wspace=0.185,
        )
        axes = [fig.add_subplot(grid[0, j]) for j in range(4)]
        colour_ax = fig.add_subplot(grid[0, 4])

        mappable = None
        peak_offsets: list[float] = []
        min_signed_jacobians: list[float] = []
        panel_labels = ("(a)", "(b)", "(c)", "(d)")

        for k, (ax, time) in enumerate(zip(axes, times)):
            xy = nodes[k]
            tri = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=cells)

            # Check orientation on the actual coordinates used for this panel.
            p0 = xy[cells[:, 0]]
            p1 = xy[cells[:, 1]]
            p2 = xy[cells[:, 2]]
            jac = ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                   - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
            min_jac = float(np.min(jac))
            min_signed_jacobians.append(min_jac)
            if min_jac <= 0.0:
                raise ValueError(
                    f"Snapshot at t={time:.6g} contains a non-positive signed element Jacobian "
                    f"(minimum {min_jac:.6e}); refusing to present it as an admissible mesh."
                )

            field = fields[k]
            centre = centers[k]
            peak_index = int(np.argmax(field))
            peak_xy = xy[peak_index]
            peak_offset = float(np.linalg.norm(peak_xy - centre) / sigma)
            peak_offsets.append(peak_offset)

            # Filled contours use exactly the same norm and levels for all times.
            mappable = ax.tricontourf(
                tri, field, levels=levels, cmap="viridis", norm=norm,
                antialiased=True, zorder=1,
            )

            # Overlay the current deformed triangular mesh; thin edges preserve
            # the scalar field while making the r-adaptive geometry inspectable.
            ax.triplot(
                tri, color=mesh_colour, linewidth=0.17, alpha=0.27,
                zorder=2, solid_capstyle="round",
            )

            cx, cy = map(float, centre)
            ax.add_patch(plt.Circle(
                (cx, cy), sigma, fill=False, edgecolor=sigma_colour,
                linewidth=0.90, linestyle="-", zorder=4,
            ))
            ax.add_patch(plt.Circle(
                (cx, cy), 2.0 * sigma, fill=False, edgecolor=two_sigma_colour,
                linewidth=0.95, linestyle=(0, (3.0, 1.8)), zorder=4,
            ))

            # The nodal maximizer is an open circle; the prescribed Gaussian
            # centre is a compact plus drawn above it. The two locations are
            # close in this trajectory, so the separate peak-offset annotation
            # is retained to quantify their actual (unshifted) separation.
            ax.plot(
                float(peak_xy[0]), float(peak_xy[1]), linestyle="none",
                marker="o", markersize=5.5, markerfacecolor="white",
                markeredgecolor=peak_colour, markeredgewidth=1.15,
                zorder=7,
            )
            centre_halo = [
                path_effects.Stroke(linewidth=2.5, foreground="white"),
                path_effects.Normal(),
            ]
            ax.plot(
                cx, cy, linestyle="none", marker="+", markersize=6.8,
                markeredgewidth=1.55, color=centre_colour,
                path_effects=centre_halo, zorder=8,
            )

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks((0.0, 0.5, 1.0))
            ax.set_yticks((0.0, 0.5, 1.0))
            ax.tick_params(axis="both", direction="out", length=2.2, width=0.5, pad=1.5)
            ax.set_xlabel(r"$x$", labelpad=0.5)
            if k == 0:
                ax.set_ylabel(r"$y$", labelpad=1.0)
            else:
                ax.tick_params(axis="y", labelleft=False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("#454545")

            ax.set_title(f"{panel_labels[k]}  $t={time:.3f}$", pad=4.0, fontweight="semibold")
            ax.text(
                0.97, 0.035, rf"$d_{{\mathrm{{peak}}}}/\sigma={peak_offset:.2f}$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.1,
                color="#202124",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
                zorder=10,
            )

        if mappable is None:
            raise RuntimeError("No scalar-field panels were generated")

        # A single external colourbar describes the common scalar normalization.
        colourbar = fig.colorbar(mappable, cax=colour_ax)
        colourbar.set_label(r"Numerical solution $u_h$", rotation=90, labelpad=4.0, fontsize=7.0)
        colourbar.locator = MaxNLocator(nbins=5)
        colourbar.update_ticks()
        colourbar.ax.tick_params(labelsize=6.2, length=2.0, width=0.5, pad=1.5)
        colourbar.outline.set_linewidth(0.5)

        legend_handles = [
            Line2D([], [], color=centre_colour, marker="+", linestyle="none",
                   markersize=7.0, markeredgewidth=1.5, label="Gaussian center"),
            Line2D([], [], color=peak_colour, marker="o", markerfacecolor="white",
                   linestyle="none", markersize=5.0, markeredgewidth=1.1,
                   label=r"Nodal maximum of $u_h$"),
            Line2D([], [], color=sigma_colour, linestyle="-", linewidth=1.0,
                   label=r"$r=\sigma$"),
            Line2D([], [], color=two_sigma_colour, linestyle=(0, (3.0, 1.8)), linewidth=1.0,
                   label=r"$r=2\sigma$"),
        ]
        fig.legend(
            handles=legend_handles, loc="upper center", bbox_to_anchor=(0.50, 0.975),
            ncol=4, frameon=False, fontsize=6.7, handlelength=1.45,
            handletextpad=0.45, columnspacing=1.25, borderaxespad=0.0,
        )
        fig.text(
            0.50, 0.075,
            r"Nodal field $u_h$ on the corresponding deformed $P_1$ triangulation.",
            ha="center", va="center", fontsize=6.5, color="#404040",
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path, format="pdf", bbox_inches="tight", pad_inches=0.025,
            facecolor="white",
            metadata={
                "Title": f"Moving-source tracking on the deformed finite-element mesh, case {case_id}",
                "Subject": "Four saved numerical solution and mesh snapshots with common scalar-field normalization",
                "Creator": "GDM source-tracking publication plot",
            },
        )
        fig.savefig(
            preview_path, format="png", dpi=int(dpi), bbox_inches="tight",
            pad_inches=0.025, facecolor="white",
        )
        plt.close(fig)

    LOG.info(
        "Publication figure saved: %s (preview: %s); global u_h range=[%.6g, %.6g], "
        "peak offsets/sigma=%s, minimum signed J=%s",
        out_path, preview_path, vmin, vmax,
        [round(v, 4) for v in peak_offsets],
        [f"{v:.4e}" for v in min_signed_jacobians],
    )
def _plot_tracking(rows: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    case_ids = sorted({str(r["case_id"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for case_id in case_ids:
        subset = sorted((r for r in rows if str(r["case_id"]) == case_id), key=lambda r: float(r["time"]))
        t = np.asarray([float(r["time"]) for r in subset])
        ratio = np.asarray([float(r["annulus_h_refinement_ratio"]) for r in subset])
        axes[0].plot(t, ratio, linewidth=1.1, label=case_id)
        axes[1].plot(t, [float(r["source_center_x"]) for r in subset], linewidth=1.1, label=f"{case_id}: x")
        axes[1].plot(t, [float(r["source_center_y"]) for r in subset], linewidth=1.1, linestyle="--", label=f"{case_id}: y")
    axes[0].axhline(1.0, color="black", linestyle=":", linewidth=0.9)
    axes[0].set_xlabel("Time"); axes[0].set_ylabel("Global mean cell size / annulus mean cell size")
    axes[0].set_title("Resolution around moving source")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_xlabel("Time"); axes[1].set_ylabel("Moving Gaussian centre")
    axes[1].set_title("Prescribed source trajectory (diagnostic only)")
    axes[1].set_ylim(0.0, 1.0); axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=6.0, ncol=2)
    fig.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def _plot_safety(events: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    if not events:
        return
    alpha = np.asarray([float(r["alpha_global"]) for r in events])
    before = np.asarray([float(r["min_jacobian_before"]) for r in events])
    after = np.asarray([float(r["min_jacobian_after"]) for r in events])
    eps = np.asarray([float(r["epsilon"]) for r in events])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    bins = np.linspace(0.0, 1.0, 21)
    axes[0].hist(alpha, bins=bins, edgecolor="white")
    axes[0].set_xlabel(r"Global accepted fraction $\alpha$")
    axes[0].set_ylabel("Adaptation events")
    axes[0].set_title("Safety-filter acceptance")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].scatter(before, after, s=18, alpha=0.8)
    low = min(float(np.min(before)), float(np.min(after)), float(np.min(eps)))
    high = max(float(np.max(before)), float(np.max(after)), float(np.max(eps)))
    axes[1].plot([low, high], [low, high], color="black", linestyle=":", linewidth=0.8)
    axes[1].axhline(float(np.median(eps)), color="red", linestyle="--", linewidth=0.9, label=r"$\epsilon$")
    axes[1].set_xlabel(r"Minimum Jacobian before filtering")
    axes[1].set_ylabel(r"Minimum Jacobian after filtering")
    axes[1].set_title("Geometric margin before/after")
    axes[1].legend(); axes[1].grid(True, alpha=0.25)
    fig.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def _summarize(events: Sequence[Mapping[str, Any]], tracking: Sequence[Mapping[str, Any]], epsilon: float) -> dict[str, Any]:
    if not events:
        raise RuntimeError("No safety-filter events were captured")
    a = np.asarray([float(r["alpha_global"]) for r in events])
    raw = np.asarray([float(r["raw_displacement_l2"]) for r in events])
    safe = np.asarray([float(r["safe_displacement_l2"]) for r in events])
    slack = np.asarray([float(r["safety_slack_absolute"]) for r in events])
    return {
        "event_count": int(len(events)),
        "filter_intervention_count": int(np.count_nonzero(a < 1.0 - 1e-12)),
        "filter_intervention_fraction": float(np.mean(a < 1.0 - 1e-12)),
        "alpha_min": float(np.min(a)),
        "alpha_q05": float(np.quantile(a, 0.05)),
        "alpha_median": float(np.median(a)),
        "alpha_q95": float(np.quantile(a, 0.95)),
        "alpha_le_0_1_count": int(np.count_nonzero(a <= 0.1)),
        "alpha_le_1e_3_count": int(np.count_nonzero(a <= 1.0e-3)),
        "alpha_le_1e_6_count": int(np.count_nonzero(a <= 1.0e-6)),
        "alpha_zero_count": int(np.count_nonzero(a <= 1.0e-15)),
        "raw_displacement_l2_total": float(np.sum(raw)),
        "safe_displacement_l2_total": float(np.sum(safe)),
        "aggregate_displacement_retention_ratio": float(np.sum(safe) / np.sum(raw)) if np.sum(raw) > 0 else 1.0,
        "median_displacement_retention_ratio": float(np.median([float(r["displacement_retention_ratio"]) for r in events])),
        "max_scalar_filter_identity_residual": float(max(float(r["scalar_filter_identity_relative_residual"]) for r in events)),
        "min_safe_jacobian": float(min(float(r["min_jacobian_after"]) for r in events)),
        "min_safety_slack": float(np.min(slack)),
        "events_with_raw_inadmissible_candidate": int(sum(int(r["raw_candidate_cells_below_epsilon"]) > 0 for r in events)),
        "events_with_raw_inverted_candidate": int(sum(int(r["raw_candidate_inverted_cells"]) > 0 for r in events)),
        "mean_active_cells_at_final_safe_mesh": float(np.mean([int(r["safe_cells_within_active_tolerance"]) for r in events])),
        "epsilon": float(epsilon),
        "tracking_time_rows": int(len(tracking)),
        "tracking_annulus_h_refinement_ratio_median": float(np.nanmedian([float(r["annulus_h_refinement_ratio"]) for r in tracking])) if tracking else None,
        "tracking_annulus_density_enrichment_median": float(np.nanmedian([float(r["annulus_density_enrichment"]) for r in tracking])) if tracking else None,
        "interpretation": "Diagnostics only; no checkpoint, monitor parameter, epsilon or safety-filter decision was changed.",
    }


def run(config_path: str = "config/stage_f.yaml", split: str = "validation", max_cases: int = 4,
        case_ids: Sequence[str] | None = None, overwrite: bool = False) -> Path:
    # Imports are delayed so unit tests can run without the native extension.
    from experiments.stage_f.run import _load_case_specs, _read_yaml as read_f_yaml, _stage_d_config
    from experiments.stage_d.benchmark_pde import build_inference_engine_for_checkpoint, run_pde_trajectory_episode

    cfg_path = (ROOT / config_path).resolve()
    stage_f_cfg = read_f_yaml(cfg_path)
    stage_d_cfg = _stage_d_config(stage_f_cfg)
    if split not in {"validation", "test_interp", "test_extrap"}:
        raise ValueError("split must be validation, test_interp or test_extrap")
    cases = _load_case_specs(stage_d_cfg, split, max_cases=None)
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if str(c["case_id"]) in wanted]
        missing = wanted - {str(c["case_id"]) for c in cases}
        if missing:
            raise ValueError(f"Unknown case IDs in split {split}: {sorted(missing)}")
    else:
        # Deterministic coverage across the validation sigma range rather than
        # taking the first few manifest entries, which may cluster in parameter space.
        cases = sorted(cases, key=lambda c: (float(c["sigma"]), float(c["beta"]), str(c["case_id"])))
        count = min(len(cases), max(1, int(max_cases)))
        if count == 1:
            indices = [len(cases) // 2]
        else:
            indices = sorted(set(int(round(k * (len(cases) - 1) / (count - 1))) for k in range(count)))
        cases = [cases[i] for i in indices]
    if not cases:
        raise RuntimeError(f"No cases selected from split {split}")

    out_dir = (ROOT / stage_f_cfg["experiment"]["output_dir"] / "safety_tracking_audit" / split).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is non-empty: {out_dir}. Use --overwrite only after backing it up.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = (ROOT / stage_f_cfg["source"]["gdm_checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage-D checkpoint not found: {checkpoint}")
    engine = build_inference_engine_for_checkpoint(checkpoint, stage_d_cfg, device="cpu")
    events_all: list[dict[str, Any]] = []
    tracking_all: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []

    LOG.info("Safety/tracking audit: split=%s, cases=%d, checkpoint=%s", split, len(cases), checkpoint)
    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        LOG.info("Case %d/%d: %s", index, len(cases), case_id)
        result = run_pde_trajectory_episode(
            case, "stage_d_gnn", stage_d_cfg, engine,
            compute_diagnostics=True, capture_safety_audit=True, capture_mesh_snapshots=True,
        )
        event_rows = [dict(r) for r in result["safety_audit_events"]]
        # The initial t=0 tracking row is assembled before the native time loop
        # and therefore has no case_id/split/seed. Attach identity to *all* rows
        # here, so plotting and CSV grouping also work for the initial mesh.
        tracking_rows = _attach_case_metadata(
            result["mesh_tracking_rows"],
            case_id=case_id,
            split=split,
            seed=int(case["case_seed"]),
        )
        epsilon = float(stage_d_cfg["geometry"]["safety"]["epsilon"])
        for row in event_rows:
            row["epsilon"] = epsilon
        events_all.extend(event_rows); tracking_all.extend(tracking_rows)
        case_out = out_dir / case_id
        case_out.mkdir(parents=True, exist_ok=True)
        snaps = result["mesh_snapshots"]
        vectors = result["safety_event_vectors"]
        np.savez_compressed(
            case_out / "safety_event_vectors.npz",
            times=np.asarray([v["time"] for v in vectors], dtype=np.float64),
            nodes_before=np.stack([v["nodes_before"] for v in vectors]),
            raw_displacements=np.stack([v["raw_displacement"] for v in vectors]),
            safe_displacements=np.stack([v["safe_displacement"] for v in vectors]),
        )
        np.savez_compressed(
            case_out / "mesh_snapshots.npz",
            times=np.asarray([s["time"] for s in snaps], dtype=np.float64),
            steps=np.asarray([s["step"] for s in snaps], dtype=np.int64),
            nodes=np.stack([s["nodes"] for s in snaps]),
            fields=np.stack([s["field"] for s in snaps]),
            centers=np.stack([s["source_center"] for s in snaps]),
            cells=np.asarray(result["audit_cells"], dtype=np.int64),
            sigma=np.asarray(float(case["sigma"])),
        )
        _plot_case_snapshots(case_out / "mesh_snapshots.npz", case_out / "mesh_source_tracking.pdf", case_id)
        case_summaries.append({
            "case_id": case_id, "split": split, "seed": int(case["case_seed"]),
            "n_safety_events": len(event_rows),
            "mean_alpha": float(np.mean([r["alpha_global"] for r in event_rows])) if event_rows else 1.0,
            "min_alpha": float(np.min([r["alpha_global"] for r in event_rows])) if event_rows else 1.0,
            "integrated_l2_error": float(result["case_summary"]["integrated_l2_error"]),
            "final_l2_error": float(result["case_summary"]["final_l2_error"]),
        })

    _write_csv(out_dir / "safety_filter_events.csv", events_all)
    _write_csv(out_dir / "mesh_tracking_metrics.csv", tracking_all)
    _plot_tracking(tracking_all, out_dir / "mesh_source_tracking_summary.pdf")
    _plot_safety(events_all, out_dir / "safety_filter_diagnostics.pdf")
    epsilon = float(stage_d_cfg["geometry"]["safety"]["epsilon"])
    summary = _summarize(events_all, tracking_all, epsilon)
    summary.update({
        "split": split,
        "case_ids": [str(c["case_id"]) for c in cases],
        "case_selection": "explicit case IDs" if case_ids else "deterministic spread across validation sigma/beta ordering",
        "case_summaries": case_summaries,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "stage_f_config_sha256": _sha256(cfg_path),
        "native_extension": __import__("gdm._gdm_core", fromlist=["__file__"]).__file__,
        "native_backend_source_sha256": _sha256(ROOT / "cpp_core/src/safety_filter.cpp"),
        "safety_parameters": dict(stage_d_cfg["geometry"]["safety"]),
        "mesh_tracking_definition": "Element-centroid annulus 0.5 sigma <= r <= 2 sigma around moving Gaussian centre; diagnostic only.",
    })
    (out_dir / "safety_tracking_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "case_summaries.json").write_text(json.dumps(case_summaries, indent=2) + "\n", encoding="utf-8")
    LOG.info("Audit outputs: %s", out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/stage_f.yaml")
    parser.add_argument("--split", default="validation", choices=["validation", "test_interp", "test_extrap"])
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--overwrite", action="store_true")

    # Plot-only path: reuses a saved mesh_snapshots.npz and does not invoke the
    # GNN, the FEM solver, or the safety filter.
    parser.add_argument(
        "--plot-existing-snapshots", type=Path, default=None,
        help="Regenerate only the source-tracking figure from an existing mesh_snapshots.npz.",
    )
    parser.add_argument(
        "--output-figure", type=Path, default=None,
        help="Destination PDF for --plot-existing-snapshots (a PNG preview is also written).",
    )
    parser.add_argument(
        "--case-label", default="case_00054",
        help="Short case identifier used only in PDF metadata/logging.",
    )
    parser.add_argument(
        "--figure-dpi", type=int, default=600,
        help="Resolution of the companion PNG preview (the PDF remains vector-based).",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.plot_existing_snapshots is not None:
        if args.output_figure is None:
            parser.error("--output-figure is required when --plot-existing-snapshots is used")
        if args.figure_dpi < 150:
            parser.error("--figure-dpi must be at least 150")
        if not args.plot_existing_snapshots.is_file():
            parser.error(f"Snapshot archive does not exist: {args.plot_existing_snapshots}")
        _plot_case_snapshots(
            args.plot_existing_snapshots,
            args.output_figure,
            args.case_label,
            dpi=args.figure_dpi,
        )
        print(f"Publication figure written to: {args.output_figure}")
        print(f"PNG preview written to: {args.output_figure.with_suffix('.png')}")
        return

    if args.output_figure is not None:
        parser.error("--output-figure can only be used together with --plot-existing-snapshots")

    out = run(args.config, args.split, args.max_cases, args.case_ids, args.overwrite)
    print(f"Safety/tracking audit written to: {out}")


if __name__ == "__main__":
    main()
