"""Quality audit and diagnostic plots for oracle-labelled mesh datasets."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from .style import PALETTE, save_figure, set_publication_style

SPLIT_ORDER = ["train", "validation", "test_interp", "test_extrap"]
SPLIT_LABELS = {"train": "Train", "validation": "Validation", "test_interp": "In-domain test", "test_extrap": "Extrapolation test", "unknown": "Unknown"}


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def audit_dataset(
    manifest_path: str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    max_iter: int | None = None,
    minimum_converged_fraction: float | None = None,  # deprecated compatibility alias
    minimum_target_acceptance_rate: float = 0.95,
    minimum_objective_reduction_fraction: float = 0.50,
    minimum_interpolation_error_reduction_fraction: float = 0.50,
    minimum_first_order_stationarity_rate: float = 0.95,  # diagnostic only, not an acceptance gate
    expected_quadrature_rule: str | None = None,
    expected_quadrature_subdivision_level: int | None = None,
) -> dict[str, Any]:
    """Audit finite-budget reference targets, optimizer diagnostics, and geometry.

    First-order stationarity is reported but is not the target-acceptance criterion.
    This is read-only: no sample is modified and no target is silently removed.
    """
    manifest_path, raw_dir, output_dir = Path(manifest_path), Path(raw_dir), Path(output_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path.resolve()}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    bad: list[dict[str, str]] = []
    for split, sample_ids in manifest.items():
        for sample_id in sample_ids:
            path = raw_dir / f"{sample_id}.pkl"
            if not path.is_file():
                missing.append(str(path))
                continue
            try:
                with path.open("rb") as stream:
                    sample = pickle.load(stream)
                metadata = sample["metadata"]
                nodes = np.asarray(sample["mesh"]["nodes"], dtype=np.float64)
                displacement = np.asarray(sample["target"]["displacement"], dtype=np.float64)
                if nodes.ndim != 2 or nodes.shape[1] != 2 or displacement.shape != nodes.shape:
                    raise ValueError(f"invalid array shapes: nodes={nodes.shape}, displacement={displacement.shape}")
                if not np.isfinite(nodes).all() or not np.isfinite(displacement).all():
                    raise ValueError("non-finite coordinates/displacements")
                oracle = metadata.get("oracle_metrics", {})
                objective_metadata = metadata.get("oracle_objective", {})
                if not isinstance(objective_metadata, dict):
                    objective_metadata = {}
                params = metadata.get("parameters", {})
                if not isinstance(params, dict):
                    params = {}
                disp_norm = np.linalg.norm(displacement, axis=1)
                rows.append({
                    "sample_id": sample_id,
                    "split": split,
                    "parameters": params,
                    "mu_x": params.get("mu_x", np.nan),
                    "mu_y": params.get("mu_y", np.nan),
                    "sigma": params.get("sigma", np.nan),
                    "amplitude": params.get("amplitude", np.nan),
                    "converged": bool(metadata.get("converged", False)),
                    "target_accepted": bool(metadata.get("target_accepted", metadata.get("converged", False))),
                    "target_acceptance_reason": str(metadata.get("target_acceptance_reason", "legacy_converged_fallback")),
                    "iterations": float(metadata.get("iterations", np.nan)),
                    # SciPy L-BFGS-B status 1 can indicate an iteration or function-evaluation
                    # budget stop. Use the broader label instead of claiming every such case
                    # is specifically a max-iteration termination.
                    "budget_exhausted": bool(
                        int(oracle.get("status", -1)) == 1
                        or (max_iter is not None and int(oracle.get("iterations_last_pass", metadata.get("iterations", -1))) >= max_iter and not bool(metadata.get("converged", False)))
                    ),
                    "optimizer_status": int(oracle.get("status", -1)),
                    "optimizer_success": bool(oracle.get("success", metadata.get("converged", False))),
                    "nfev": float(oracle.get("nfev_total", oracle.get("nfev", np.nan))),
                    "njev": float(oracle.get("njev_total", oracle.get("njev", np.nan))),
                    "disp_rms_component": float(np.sqrt(np.mean(displacement**2))),
                    "disp_node_mean": float(np.mean(disp_norm)),
                    "disp_node_p95": float(np.percentile(disp_norm, 95)),
                    "disp_node_max": float(np.max(disp_norm)),
                    "min_jacobian": float(oracle.get("min_jacobian", np.nan)),
                    "mean_quality": float(oracle.get("mean_escobar_quality", np.nan)),
                    "safety_alpha": float(oracle.get("safety_alpha_applied", np.nan)),
                    "constraint_penalty": float(oracle.get("constraint_penalty", np.nan)),
                    "objective_reduction_fraction": float(oracle.get("objective_reduction_fraction", np.nan)),
                    "interpolation_error_reduction_fraction": float(oracle.get("interpolation_error_reduction_fraction", np.nan)),
                    "optimizer_message": str(oracle.get("message", "")),
                    "iterations_last_pass": float(oracle.get("iterations_last_pass", np.nan)),
                    "retry_count": int(oracle.get("retry_count", 0)),
                    "candidate_is_valid": oracle.get("candidate_is_valid", None),
                    "candidate_min_jacobian": float(oracle.get("candidate_min_jacobian", np.nan)),
                    "safety_filter_active": oracle.get(
                        "safety_filter_active",
                        (float(oracle.get("safety_alpha_applied", 1.0)) < 1.0 - 1.0e-12),
                    ),
                    "projected_gradient_inf_norm": float(oracle.get("projected_gradient_inf_norm", np.nan)),
                    "projected_gradient_l2_norm": float(oracle.get("projected_gradient_l2_norm", np.nan)),
                    "gradient_tolerance": float(oracle.get("gradient_tolerance", np.nan)),
                    "gradient_tolerance_met": oracle.get("gradient_tolerance_met", None),
                    "termination_by_function_reduction": oracle.get("termination_by_function_reduction", None),
                    "termination_by_projected_gradient": oracle.get("termination_by_projected_gradient", None),
                    "num_inverted_cells": int(oracle.get("num_inverted_cells", -1)),
                    "num_degenerate_cells": int(oracle.get("num_degenerate_cells", -1)),
                    "initial_objective": float(oracle.get("initial_objective", np.nan)),
                    "final_objective": float(oracle.get("total_objective", np.nan)),
                    "quadrature_rule": oracle.get(
                        "quadrature_rule", objective_metadata.get("quadrature_rule")
                    ),
                    "quadrature_subdivision_level": oracle.get(
                        "quadrature_subdivision_level", objective_metadata.get("quadrature_subdivision_level")
                    ),
                    "quadrature_points_per_cell": oracle.get(
                        "quadrature_points_per_cell", objective_metadata.get("quadrature_points_per_cell")
                    ),
                })
            except Exception as exc:  # record malformed inputs instead of hiding them
                bad.append({"sample_id": sample_id, "error": str(exc)})

    summary: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "raw_dir": str(raw_dir),
        "n_manifest": int(sum(len(v) for v in manifest.values())),
        "n_read": len(rows),
        "missing_files": missing,
        "malformed_samples": bad,
        "max_iter_threshold": max_iter,
        "splits": {},
    }
    observed_rules = sorted({str(r["quadrature_rule"]) for r in rows if r["quadrature_rule"] is not None})
    observed_levels = sorted({int(r["quadrature_subdivision_level"]) for r in rows if r["quadrature_subdivision_level"] is not None})
    missing_quad_metadata = [r["sample_id"] for r in rows if r["quadrature_rule"] is None or r["quadrature_subdivision_level"] is None]
    quadrature_mismatches = []
    if expected_quadrature_rule is not None or expected_quadrature_subdivision_level is not None:
        for r in rows:
            if (
                (expected_quadrature_rule is not None and r["quadrature_rule"] != expected_quadrature_rule)
                or (expected_quadrature_subdivision_level is not None and r["quadrature_subdivision_level"] != expected_quadrature_subdivision_level)
            ):
                quadrature_mismatches.append(r["sample_id"])
    quadrature_metadata_consistent = not quadrature_mismatches
    if expected_quadrature_rule is not None or expected_quadrature_subdivision_level is not None:
        quadrature_metadata_consistent = quadrature_metadata_consistent and not missing_quad_metadata and len(rows) == int(sum(len(v) for v in manifest.values()))
    summary["oracle_quadrature"] = {
        "observed_rules": observed_rules,
        "observed_subdivision_levels": observed_levels,
        "expected_rule": expected_quadrature_rule,
        "expected_subdivision_level": expected_quadrature_subdivision_level,
        "missing_metadata_sample_ids": missing_quad_metadata,
        "mismatched_sample_ids": quadrature_mismatches,
    }
    for split in SPLIT_ORDER + sorted(set(manifest) - set(SPLIT_ORDER)):
        part = [r for r in rows if r["split"] == split]
        if not part:
            continue
        iters = np.asarray([r["iterations"] for r in part], dtype=float)
        finite_iter = iters[np.isfinite(iters)]
        status_counts: dict[str, int] = {}
        message_counts: dict[str, int] = {}
        for r in part:
            status_key = str(r["optimizer_status"])
            status_counts[status_key] = status_counts.get(status_key, 0) + 1
            message_key = r["optimizer_message"] or "(empty message)"
            message_counts[message_key] = message_counts.get(message_key, 0) + 1
        split_summary: dict[str, Any] = {
            "n_samples": len(part),
            "n_converged": int(sum(r["converged"] for r in part)),
            "convergence_rate": float(np.mean([r["converged"] for r in part])),
            "n_target_accepted": int(sum(r["target_accepted"] for r in part)),
            "target_acceptance_rate": float(np.mean([r["target_accepted"] for r in part])),
            "n_optimizer_success": int(sum(r["optimizer_success"] for r in part)),
            "n_budget_exhausted": int(sum(r["budget_exhausted"] for r in part)),
            "n_first_order_stationary": int(sum(r["gradient_tolerance_met"] is True for r in part)),
            "n_gradient_diagnostics": int(sum(r["gradient_tolerance_met"] is not None for r in part)),
            "first_order_stationarity_rate": (
                float(np.mean([r["gradient_tolerance_met"] is True for r in part if r["gradient_tolerance_met"] is not None]))
                if any(r["gradient_tolerance_met"] is not None for r in part) else None
            ),
            "n_safety_filter_active": int(sum(r["safety_filter_active"] is True for r in part)),
            "n_accepted_candidate_invalid": int(sum(r["target_accepted"] and r["candidate_is_valid"] is False for r in part)),
            "n_accepted_targets_safety_repaired": int(sum(r["target_accepted"] and r["safety_filter_active"] is True for r in part)),
            "optimizer_status_counts": status_counts,
            "optimizer_message_counts": message_counts,
            "median_iterations": float(np.median(finite_iter)) if finite_iter.size else None,
            "p95_iterations": float(np.percentile(finite_iter, 95)) if finite_iter.size else None,
        }
        for count_key in ("nfev", "njev"):
            arr = np.asarray([r[count_key] for r in part], dtype=float)
            arr = arr[np.isfinite(arr)]
            split_summary[f"median_{count_key}"] = float(np.median(arr)) if arr.size else None
            split_summary[f"p95_{count_key}"] = float(np.percentile(arr, 95)) if arr.size else None
        for key in (
            "disp_rms_component", "disp_node_mean", "disp_node_p95", "disp_node_max",
            "min_jacobian", "mean_quality", "safety_alpha", "constraint_penalty",
            "objective_reduction_fraction", "interpolation_error_reduction_fraction",
        ):
            arr = np.asarray([r[key] for r in part], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                split_summary[key] = {
                    "median": float(np.median(arr)),
                    "p95": float(np.percentile(arr, 95)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "positive_fraction": float(np.mean(arr > 0.0)),
                }
            else:
                split_summary[key] = None

        # Keep failed optimisation attempts visible, but do not mix their
        # objective-reduction statistics with the subset of accepted targets.
        split_summary["target_quality_by_acceptance"] = {}
        for accepted, group_name in ((True, "accepted_targets"), (False, "rejected_targets")):
            group = [r for r in part if r["target_accepted"] is accepted]
            group_metrics: dict[str, Any] = {"n_samples": len(group)}
            for key in ("objective_reduction_fraction", "interpolation_error_reduction_fraction", "disp_node_mean", "min_jacobian", "constraint_penalty"):
                values = np.asarray([r[key] for r in group], dtype=float)
                values = values[np.isfinite(values)]
                group_metrics[key] = {
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                } if values.size else None
            split_summary["target_quality_by_acceptance"][group_name] = group_metrics
        summary["splits"][split] = split_summary

    if not rows:
        raise RuntimeError(f"No readable samples were found in {raw_dir.resolve()}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dataset_quality_audit.json"

    # Per-sample diagnostics are the primary tool for investigating outliers.
    # Nested parameter dictionaries are retained in JSON and flattened in CSV.
    diagnostics_json = output_dir / "sample_diagnostics.json"
    diagnostics_json.write_text(
        json.dumps(_json_safe(rows), indent=2, allow_nan=False), encoding="utf-8"
    )
    csv_fields = [
        "sample_id", "split", "mu_x", "mu_y", "sigma", "amplitude", "converged",
        "target_accepted", "target_acceptance_reason",
        "optimizer_success", "optimizer_status", "optimizer_message", "iterations",
        "iterations_last_pass", "retry_count", "nfev", "njev", "budget_exhausted",
        "projected_gradient_inf_norm", "projected_gradient_l2_norm", "gradient_tolerance",
        "gradient_tolerance_met", "termination_by_function_reduction",
        "termination_by_projected_gradient", "candidate_is_valid", "candidate_min_jacobian",
        "safety_filter_active", "safety_alpha", "min_jacobian", "mean_quality",
        "num_inverted_cells", "num_degenerate_cells", "constraint_penalty",
        "objective_reduction_fraction", "interpolation_error_reduction_fraction",
        "quadrature_rule", "quadrature_subdivision_level", "quadrature_points_per_cell",
    ]
    with (output_dir / "sample_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    set_publication_style()
    split_keys = [s for s in SPLIT_ORDER if any(r["split"] == s for r in rows)]
    split_labels = [SPLIT_LABELS.get(s, s.replace("_", " ")) for s in split_keys]

    # Figure 1: reference-target acceptance and iteration-budget saturation.
    fig, ax = plt.subplots(figsize=(5.7, 3.2))
    xpos = np.arange(len(split_keys)); width = 0.36
    rates = [summary["splits"][s]["target_acceptance_rate"] for s in split_keys]
    nhit = [summary["splits"][s]["n_budget_exhausted"] / summary["splits"][s]["n_samples"] for s in split_keys]
    ax.bar(xpos - width/2, rates, width, color=PALETTE["safe"], label="Accepted reference-target rate")
    ax.bar(xpos + width/2, nhit, width, color=PALETTE["warning"], label="Optimizer-budget exhaustion")
    ax.set_xticks(xpos, split_labels)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Fraction of samples")
    ax.set_title("Reference-target acceptance and optimizer budget")
    ax.legend(loc="best"); ax.grid(True, axis="y"); ax.grid(False, axis="x")
    fig.tight_layout()
    save_figure(fig, output_dir / "dataset_oracle_convergence")

    # Figure 2: displacement scale. The target is an optimised field, so its scale is important.
    fig, ax = plt.subplots(figsize=(5.7, 3.2))
    datasets = [np.asarray([r["disp_node_mean"] for r in rows if r["split"] == s], dtype=float) for s in split_keys]
    bp = ax.boxplot(datasets, patch_artist=True, widths=0.55, showfliers=True, medianprops={"color": "#111827", "linewidth": 1.1}, flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.5})
    for patch, colour in zip(bp["boxes"], [PALETTE["fixed"], PALETTE["safe"], PALETTE["raw"], PALETTE["target"]][:len(datasets)]):
        patch.set_facecolor(colour); patch.set_alpha(0.25); patch.set_edgecolor(colour)
    ax.set_xticks(np.arange(1, len(split_keys)+1), split_labels)
    ax.set_ylabel(r"Mean nodal displacement $\|\Delta X\|_2$")
    ax.set_title("Oracle target displacement magnitude")
    ax.grid(True, axis="y"); ax.grid(False, axis="x")
    fig.tight_layout()
    save_figure(fig, output_dir / "dataset_target_displacement")

    # Figure 3: Jacobian safety margin and optimizer iteration count.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    axj, axi = axes
    jsets = [np.asarray([r["min_jacobian"] for r in rows if r["split"] == s], dtype=float) for s in split_keys]
    jsets = [a[np.isfinite(a)] for a in jsets]
    if any(a.size for a in jsets):
        bp = axj.boxplot(jsets, patch_artist=True, widths=0.55, showfliers=False, medianprops={"color": "#111827", "linewidth": 1.1})
        for patch, colour in zip(bp["boxes"], [PALETTE["fixed"], PALETTE["safe"], PALETTE["raw"], PALETTE["target"]][:len(jsets)]):
            patch.set_facecolor(colour); patch.set_alpha(0.25); patch.set_edgecolor(colour)
    axj.axhline(1.0e-4, color=PALETTE["warning"], linestyle="--", linewidth=1.0, label=r"Admissibility floor $\epsilon=10^{-4}$")
    axj.set_xticks(np.arange(1, len(split_keys)+1), split_labels)
    axj.set_ylabel(r"Minimum oriented Jacobian $\min_e J_e$")
    axj.set_title("Final target mesh validity"); axj.legend(fontsize=6.8)
    iterations = [np.asarray([r["iterations"] for r in rows if r["split"] == s], dtype=float) for s in split_keys]
    axi.boxplot(iterations, patch_artist=True, widths=0.55, showfliers=True, medianprops={"color": "#111827", "linewidth": 1.1})
    axi.set_xticks(np.arange(1, len(split_keys)+1), split_labels)
    axi.set_ylabel("L-BFGS-B iterations")
    axi.set_title("Optimizer effort")
    for ax in axes:
        ax.grid(True, axis="y"); ax.grid(False, axis="x")
    fig.tight_layout()
    save_figure(fig, output_dir / "dataset_oracle_geometry_iterations")

    # Figure 4: newly generated datasets carry objective and interpolation-error
    # reduction diagnostics from the original mesh to the final stored target.
    has_reduction = any(
        np.isfinite(r["objective_reduction_fraction"])
        or np.isfinite(r["interpolation_error_reduction_fraction"])
        for r in rows
    )
    if has_reduction:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), sharey=False)
        reduction_specs = [
            ("objective_reduction_fraction", "Total-objective reduction [%]"),
            ("interpolation_error_reduction_fraction", "Interpolation-error reduction [%]"),
        ]
        split_colours = [PALETTE["fixed"], PALETTE["safe"], PALETTE["raw"], PALETTE["target"]]
        for ax, (key, label) in zip(axes, reduction_specs):
            values = []
            labels = []
            colours = []
            for split, split_label, colour in zip(split_keys, split_labels, split_colours):
                arr = np.asarray([100.0 * r[key] for r in rows if r["split"] == split], dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    values.append(arr); labels.append(split_label); colours.append(colour)
            if values:
                bp = ax.boxplot(values, patch_artist=True, widths=0.55, showfliers=True,
                                medianprops={"color": "#111827", "linewidth": 1.1},
                                flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.45})
                for patch, colour in zip(bp["boxes"], colours):
                    patch.set_facecolor(colour); patch.set_alpha(0.25); patch.set_edgecolor(colour)
                ax.axhline(0.0, color=PALETTE["warning"], linestyle="--", linewidth=0.9)
                ax.set_xticks(np.arange(1, len(labels) + 1), labels)
            else:
                ax.text(0.5, 0.5, "Not available in this dataset", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            ax.set_ylabel(label)
            ax.grid(True, axis="y"); ax.grid(False, axis="x")
        fig.suptitle("Optimization target improvement diagnostics")
        fig.tight_layout()
        save_figure(fig, output_dir / "dataset_target_improvement")

    missing_improvement_diagnostics = any(
        v.get("objective_reduction_fraction") is None
        or v.get("interpolation_error_reduction_fraction") is None
        for v in summary["splits"].values()
    )
    nonpositive_objective_reduction = any(
        v.get("objective_reduction_fraction") is not None
        and v["objective_reduction_fraction"]["median"] <= 0.0
        for v in summary["splits"].values()
    )
    nonpositive_interpolation_reduction = any(
        v.get("interpolation_error_reduction_fraction") is not None
        and v["interpolation_error_reduction_fraction"]["median"] <= 0.0
        for v in summary["splits"].values()
    )
    if minimum_converged_fraction is not None:
        # Backward-compatible alias; the acceptance contract applies to target_accepted.
        minimum_target_acceptance_rate = float(minimum_converged_fraction)
    minimum_target_acceptance_rate = float(minimum_target_acceptance_rate)
    if not 0.0 < minimum_target_acceptance_rate <= 1.0:
        raise ValueError("minimum_target_acceptance_rate must be in (0, 1].")
    minimum_objective_reduction_fraction = float(minimum_objective_reduction_fraction)
    minimum_interpolation_error_reduction_fraction = float(minimum_interpolation_error_reduction_fraction)
    if not 0.0 <= minimum_objective_reduction_fraction < 1.0:
        raise ValueError("minimum_objective_reduction_fraction must be in [0, 1).")
    if not 0.0 <= minimum_interpolation_error_reduction_fraction < 1.0:
        raise ValueError("minimum_interpolation_error_reduction_fraction must be in [0, 1).")
    minimum_first_order_stationarity_rate = float(minimum_first_order_stationarity_rate)
    if not 0.0 <= minimum_first_order_stationarity_rate <= 1.0:
        raise ValueError("minimum_first_order_stationarity_rate must be in [0, 1].")
    missing_gradient_diagnostics = any(
        v["n_gradient_diagnostics"] < v["n_samples"] for v in summary["splits"].values()
    )
    # Stationarity is reported, but is not used to gate a finite-budget target.
    insufficient_first_order_stationarity = any(
        v["first_order_stationarity_rate"] is None
        or v["first_order_stationarity_rate"] < minimum_first_order_stationarity_rate
        for v in summary["splits"].values()
    )
    invalid_accepted_candidates = [
        r["sample_id"] for r in rows
        if r["target_accepted"] and (
            r["candidate_is_valid"] is not True
            or r["num_inverted_cells"] != 0
            or r["num_degenerate_cells"] != 0
        )
    ]
    repaired_accepted_targets = [
        r["sample_id"] for r in rows
        if r["target_accepted"] and r["safety_filter_active"] is True
    ]
    accepted_target_quality_violations = [
        r["sample_id"] for r in rows
        if r["target_accepted"] and (
            not np.isfinite(r["objective_reduction_fraction"])
            or r["objective_reduction_fraction"] < minimum_objective_reduction_fraction
            or not np.isfinite(r["interpolation_error_reduction_fraction"])
            or r["interpolation_error_reduction_fraction"] < minimum_interpolation_error_reduction_fraction
        )
    ]
    summary["target_acceptance_contract"] = {
        "definition": "finite-budget reference target; not a claim of first-order stationarity",
        "minimum_target_acceptance_rate": minimum_target_acceptance_rate,
        "minimum_objective_reduction_fraction": minimum_objective_reduction_fraction,
        "minimum_interpolation_error_reduction_fraction": minimum_interpolation_error_reduction_fraction,
    }
    summary["audit_checks"] = {
        "missing_improvement_diagnostics": bool(missing_improvement_diagnostics),
        "nonpositive_median_objective_reduction": bool(nonpositive_objective_reduction),
        "nonpositive_median_interpolation_error_reduction": bool(nonpositive_interpolation_reduction),
        "accepted_candidate_geometry_violations": invalid_accepted_candidates,
        "accepted_targets_changed_by_safety_filter": repaired_accepted_targets,
        "accepted_target_quality_violations": accepted_target_quality_violations,
        "required_target_acceptance_rate": minimum_target_acceptance_rate,
        "required_objective_reduction_fraction": minimum_objective_reduction_fraction,
        "required_interpolation_error_reduction_fraction": minimum_interpolation_error_reduction_fraction,
        "missing_first_order_gradient_diagnostics": bool(missing_gradient_diagnostics),
        "first_order_stationarity_is_diagnostic_only": True,
        "insufficient_first_order_stationarity": bool(insufficient_first_order_stationarity),
        "diagnostic_stationarity_reference_rate": minimum_first_order_stationarity_rate,
        "quadrature_metadata_consistent": bool(quadrature_metadata_consistent),
        "quadrature_metadata_missing_sample_ids": missing_quad_metadata if expected_quadrature_rule is not None or expected_quadrature_subdivision_level is not None else [],
        "quadrature_mismatched_sample_ids": quadrature_mismatches,
    }
    summary["audit_status"] = "WARN" if (
        missing or bad or missing_improvement_diagnostics or not quadrature_metadata_consistent
        or any(v["target_acceptance_rate"] < minimum_target_acceptance_rate for v in summary["splits"].values())
        or nonpositive_objective_reduction or nonpositive_interpolation_reduction
        or invalid_accepted_candidates or repaired_accepted_targets or accepted_target_quality_violations
        or missing_gradient_diagnostics
    ) else "PASS"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Dataset audit: {summary['audit_status']} — {len(rows)}/{summary['n_manifest']} samples read")
    for split, rec in summary["splits"].items():
        stationarity = rec["first_order_stationarity_rate"]
        stationarity_msg = f"; projected-gradient criterion met={stationarity:.1%}" if stationarity is not None else ""
        print(f"  {split}: {rec['n_target_accepted']}/{rec['n_samples']} reference targets accepted; "
              f"{rec['n_converged']} first-order stationary; {rec['n_budget_exhausted']} budget stops{stationarity_msg}")
    print(f"  Per-sample diagnostics: {output_dir / 'sample_diagnostics.csv'}")
    if summary["audit_status"] == "WARN":
        print("WARNING: reference-target acceptance checks failed; inspect sample_diagnostics.csv/json before training.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest_oracle_v4.json")
    parser.add_argument("--raw-dir", default="data/raw_oracle_v4")
    parser.add_argument("--output-dir", default="results/stage_a_oracle_v4/figures/dataset_audit")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--minimum-target-acceptance-rate", type=float, default=0.95)
    parser.add_argument("--minimum-objective-reduction-fraction", type=float, default=0.50)
    parser.add_argument("--minimum-interpolation-error-reduction-fraction", type=float, default=0.50)
    parser.add_argument("--expected-quadrature-rule", default=None)
    parser.add_argument("--expected-quadrature-subdivision-level", type=int, default=None)
    args = parser.parse_args()
    audit_dataset(
        args.manifest, args.raw_dir, args.output_dir,
        max_iter=args.max_iter,
        minimum_target_acceptance_rate=args.minimum_target_acceptance_rate,
        minimum_objective_reduction_fraction=args.minimum_objective_reduction_fraction,
        minimum_interpolation_error_reduction_fraction=args.minimum_interpolation_error_reduction_fraction,
        expected_quadrature_rule=args.expected_quadrature_rule,
        expected_quadrature_subdivision_level=args.expected_quadrature_subdivision_level,
    )


if __name__ == "__main__":
    main()
