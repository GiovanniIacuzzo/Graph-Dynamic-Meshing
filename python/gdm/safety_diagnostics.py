"""Post-hoc geometric diagnostics for GDM safety-filter events and source tracking.

These helpers do not alter the controller or the native safety-filter decision.
They analyse the raw proposal, the accepted displacement and the mesh/feature
alignment after the fact.
"""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np


def signed_triangle_jacobians(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Return signed 2-D triangle Jacobians using the native filter convention."""
    x = np.asarray(nodes, dtype=np.float64)
    tri = np.asarray(cells, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("nodes must have shape (N, 2)")
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("cells must have shape (M, 3)")
    if tri.size and (tri.min() < 0 or tri.max() >= x.shape[0]):
        raise ValueError("cells contain out-of-range node indices")
    p0, p1, p2 = x[tri[:, 0]], x[tri[:, 1]], x[tri[:, 2]]
    return ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))


def safety_event_diagnostics(
    *,
    nodes: np.ndarray,
    cells: np.ndarray,
    boundary_nodes: np.ndarray,
    raw_displacement: np.ndarray,
    filter_result: Mapping[str, Any],
    epsilon: float,
    event_time: float,
    source_center: tuple[float, float],
) -> dict[str, Any]:
    """Build an event-level audit row without changing the filter output.

    The accepted displacement should equal alpha_global times the raw
    displacement after applying the same boundary locking convention. The
    residual of that identity is recorded as an implementation invariant.
    """
    x = np.asarray(nodes, dtype=np.float64)
    cells_arr = np.asarray(cells, dtype=np.int64)
    raw = np.asarray(raw_displacement, dtype=np.float64)
    bnd = np.asarray(boundary_nodes, dtype=np.int64)
    eps = float(epsilon)
    if raw.shape != x.shape:
        raise ValueError("raw_displacement must have the same shape as nodes")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(x)):
        raise ValueError("nodes and raw_displacement must be finite")
    if eps <= 0.0 or not np.isfinite(eps):
        raise ValueError("epsilon must be finite and positive")

    locked_raw = raw.copy()
    boundary_raw = raw[bnd].copy() if bnd.size else np.empty((0, 2), dtype=float)
    if bnd.size:
        locked_raw[bnd] = 0.0

    alpha = float(filter_result["alpha_global"])
    safe_nodes = np.asarray(filter_result["safe_nodes"], dtype=np.float64)
    if safe_nodes.shape != x.shape or not np.all(np.isfinite(safe_nodes)):
        raise ValueError("safe_nodes has invalid shape or non-finite values")
    safe_disp = safe_nodes - x
    raw_candidate_nodes = x + locked_raw

    raw_norm = float(np.linalg.norm(locked_raw))
    safe_norm = float(np.linalg.norm(safe_disp))
    expected_safe = alpha * locked_raw
    scalar_identity_residual = float(
        np.linalg.norm(safe_disp - expected_safe) / max(raw_norm, np.finfo(float).tiny)
    )
    retention = safe_norm / raw_norm if raw_norm > 0.0 else (1.0 if safe_norm == 0.0 else np.nan)

    j_before = signed_triangle_jacobians(x, cells_arr)
    j_raw = signed_triangle_jacobians(raw_candidate_nodes, cells_arr)
    j_safe = signed_triangle_jacobians(safe_nodes, cells_arr)
    limiting = int(filter_result.get("limiting_cell_index", -1))
    if 0 <= limiting < cells_arr.shape[0]:
        limiter_centroid = np.mean(x[cells_arr[limiting]], axis=0)
        limiter_distance = float(np.linalg.norm(limiter_centroid - np.asarray(source_center)))
    else:
        limiter_centroid = np.asarray([np.nan, np.nan])
        limiter_distance = float("nan")

    # Tolerance is intentionally used only to classify near-active constraints;
    # it does not relax the native admissibility condition.
    roundoff_tol = max(1.0e-12, 1.0e-9 * abs(eps))
    active_tol = max(10.0 * roundoff_tol, 1.0e-8 * eps)
    active_count = int(np.count_nonzero(j_safe <= eps + active_tol))
    raw_bad_count = int(np.count_nonzero(j_raw < eps - roundoff_tol))

    return {
        "event_time": float(event_time),
        "source_center_x": float(source_center[0]),
        "source_center_y": float(source_center[1]),
        "alpha_global": alpha,
        "filter_activated": bool(filter_result.get("filter_activated", alpha < 1.0 - 1e-12)),
        "is_safe": bool(filter_result.get("is_safe", False)),
        "raw_displacement_l2": raw_norm,
        "safe_displacement_l2": safe_norm,
        "removed_displacement_l2": float(max(0.0, raw_norm - safe_norm)),
        "displacement_retention_ratio": float(retention),
        "displacement_removed_fraction": float(1.0 - retention) if np.isfinite(retention) else np.nan,
        "raw_displacement_linf": float(np.max(np.linalg.norm(locked_raw, axis=1))),
        "safe_displacement_linf": float(np.max(np.linalg.norm(safe_disp, axis=1))),
        "boundary_raw_displacement_l2": float(np.linalg.norm(boundary_raw)),
        "scalar_filter_identity_relative_residual": scalar_identity_residual,
        "min_jacobian_before": float(np.min(j_before)),
        "min_jacobian_raw_candidate": float(np.min(j_raw)),
        "min_jacobian_after": float(np.min(j_safe)),
        "raw_candidate_cells_below_epsilon": raw_bad_count,
        "raw_candidate_inverted_cells": int(np.count_nonzero(j_raw <= 0.0)),
        "safe_cells_within_active_tolerance": active_count,
        "safety_slack_absolute": float(np.min(j_safe) - eps),
        "safety_slack_relative_to_epsilon": float((np.min(j_safe) - eps) / eps),
        "limiting_cell_index": limiting,
        "limiting_cell_centroid_x": float(limiter_centroid[0]),
        "limiting_cell_centroid_y": float(limiter_centroid[1]),
        "limiting_cell_distance_to_source": limiter_distance,
        "native_roundoff_tolerance": roundoff_tol,
        "active_constraint_classification_tolerance": active_tol,
    }


def mesh_source_tracking_metrics(
    *,
    nodes: np.ndarray,
    cells: np.ndarray,
    source_center: tuple[float, float],
    sigma: float,
    domain: Mapping[str, float],
    time: float,
    step: int,
) -> dict[str, Any]:
    """Quantify mesh resolution around the moving Gaussian's gradient annulus.

    The annulus [0.5 sigma, 2 sigma] is a diagnostic region around the moving
    source, not a prescribed optimal mesh. For a Gaussian envelope its steepest
    radial variation is O(sigma) away from the centre; the exact PDE field also
    contains a smooth sin(pi*x) sin(pi*y) factor. Metrics must therefore be read
    as tracking/resolution diagnostics, not as an oracle target.
    """
    x = np.asarray(nodes, dtype=np.float64)
    tri = np.asarray(cells, dtype=np.int64)
    cx, cy = map(float, source_center)
    sig = float(sigma)
    if sig <= 0.0 or not np.isfinite(sig):
        raise ValueError("sigma must be finite and positive")
    jac = signed_triangle_jacobians(x, tri)
    areas = 0.5 * jac
    if np.any(areas <= 0.0):
        raise ValueError("tracking metrics require positively oriented non-degenerate cells")
    centroids = np.mean(x[tri], axis=1)
    radii = np.linalg.norm(centroids - np.asarray([cx, cy]), axis=1)
    r_inner, r_outer = 0.5 * sig, 2.0 * sig
    in_annulus = (radii >= r_inner) & (radii <= r_outer)

    global_mean_area = float(np.mean(areas))
    local_mean_area = float(np.mean(areas[in_annulus])) if np.any(in_annulus) else float("nan")
    local_median_area = float(np.median(areas[in_annulus])) if np.any(in_annulus) else float("nan")
    domain_area = float((domain["x_max"] - domain["x_min"]) * (domain["y_max"] - domain["y_min"]))
    annulus_area = float(np.pi * (r_outer**2 - r_inner**2))
    # The sampled cases keep the diagnostic annulus inside the square domain.
    # If a future case clips it, avoid presenting the un-clipped normalization.
    wholly_inside = (
        cx - r_outer >= float(domain["x_min"]) and cx + r_outer <= float(domain["x_max"])
        and cy - r_outer >= float(domain["y_min"]) and cy + r_outer <= float(domain["y_max"])
    )
    if wholly_inside and domain_area > 0.0:
        geometric_fraction = annulus_area / domain_area
        density_enrichment = (float(np.mean(in_annulus)) / geometric_fraction) if geometric_fraction > 0.0 else np.nan
    else:
        geometric_fraction = float("nan")
        density_enrichment = float("nan")

    edge_lengths = np.linalg.norm(x[tri[:, [1, 2, 0]]] - x[tri[:, [0, 1, 2]]], axis=2)
    cell_h = np.sqrt(2.0 * areas)
    global_h = float(np.mean(cell_h))
    local_h = float(np.mean(cell_h[in_annulus])) if np.any(in_annulus) else float("nan")
    nearest_node_distance = float(np.min(np.linalg.norm(x - np.asarray([cx, cy]), axis=1)))
    if np.any(in_annulus):
        # Inverse-area weighting estimates where the densest part of the annular
        # mesh distribution sits; it is not forced to coincide with the source centre.
        weights = 1.0 / areas[in_annulus]
        concentration_centroid = np.average(centroids[in_annulus], axis=0, weights=weights)
        concentration_centroid_distance = float(
            np.linalg.norm(concentration_centroid - np.asarray([cx, cy]))
        )
    else:
        concentration_centroid = np.asarray([np.nan, np.nan])
        concentration_centroid_distance = float("nan")

    return {
        "time": float(time),
        "step": int(step),
        "source_center_x": cx,
        "source_center_y": cy,
        "sigma": sig,
        "annulus_inner_radius": r_inner,
        "annulus_outer_radius": r_outer,
        "annulus_element_count": int(np.count_nonzero(in_annulus)),
        "annulus_element_fraction": float(np.mean(in_annulus)),
        "annulus_geometric_area_fraction": geometric_fraction,
        "annulus_density_enrichment": float(density_enrichment),
        "global_mean_element_area": global_mean_area,
        "annulus_mean_element_area": local_mean_area,
        "annulus_median_element_area": local_median_area,
        "annulus_resolution_ratio": float(global_mean_area / local_mean_area) if np.isfinite(local_mean_area) and local_mean_area > 0 else np.nan,
        "global_mean_cell_h": global_h,
        "annulus_mean_cell_h": local_h,
        "annulus_h_refinement_ratio": float(global_h / local_h) if np.isfinite(local_h) and local_h > 0 else np.nan,
        "nearest_node_distance_to_source": nearest_node_distance,
        "annulus_concentration_centroid_x": float(concentration_centroid[0]),
        "annulus_concentration_centroid_y": float(concentration_centroid[1]),
        "annulus_concentration_centroid_distance": concentration_centroid_distance,
        "annulus_edge_length_mean": float(np.mean(edge_lengths[in_annulus])) if np.any(in_annulus) else np.nan,
    }
