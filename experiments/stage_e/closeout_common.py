"""Streaming and output helpers for the incremental Stage-E1 close-out."""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .common import (
    ROOT,
    directional_element_metrics,
    configure_cpu_threads,
    get_native_solver_configs,
    load_config,
    l2_p1_difference,
    make_domain,
    make_pde,
    maximum_element_diameter,
    native_modules,
    number_of_steps,
    solution_extrema_and_oscillations,
)


def closeout_root(cfg: Mapping[str, Any]) -> Path:
    relative = cfg.get("closeout", {}).get("output_dir", "results/stage_e/closeout")
    path = Path(str(relative))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def ensure_closeout_tree(cfg: Mapping[str, Any]) -> Path:
    root = closeout_root(cfg)
    for part in ("solver_verification", "work_precision", "supg", "tables", "figures", "audit"):
        (root / part).mkdir(parents=True, exist_ok=True)
    return root


def atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_or_replace_csv_row(path: Path, row: Mapping[str, Any], key_fields: tuple[str, ...]) -> None:
    """Upsert one configuration row atomically; supports interruption/resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    for key in row:
        if key not in fields:
            fields.append(key)
    key = tuple(str(row.get(k, "")) for k in key_fields)
    replaced = False
    for index, previous in enumerate(rows):
        previous_key = tuple(str(previous.get(k, "")) for k in key_fields)
        if previous_key == key:
            rows[index] = {field: str(row.get(field, "")) for field in fields}
            replaced = True
            break
    if not replaced:
        rows.append({field: str(row.get(field, "")) for field in fields})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _integral_add(accumulator: float, previous: float, current: float, step: float) -> float:
    return accumulator + 0.5 * step * (previous + current)


def run_paired_fixed_streaming(
    cfg: Mapping[str, Any], nx: int, fine_dt: float, mode: str,
) -> dict[str, Any]:
    """Run dt=2*fine_dt and dt=fine_dt in lockstep, without storing trajectories.

    The fine trajectory supplies the time-averaged exact-error metrics. The
    coarse/fine difference is measured at aligned times and integrated with the
    trapezoidal rule. Memory is O(number of vertices), not O(steps*vertices).
    """
    if mode not in {"galerkin", "supg"}:
        raise ValueError(f"Unsupported stabilization mode: {mode}")
    NativeMeshAdapter, _, create_fem_solver, _, compute_error_details = native_modules()
    T = float(cfg["pde"]["T"])
    n_fine = number_of_steps(T, fine_dt)
    if n_fine % 2:
        raise ValueError("fine_dt must give an even number of steps so dt_coarse=2*fine_dt ends at T.")
    coarse_dt = 2.0 * fine_dt
    solver_cfg, stabilization_cfg = get_native_solver_configs(cfg)
    stab_coeffs = {k: float(stabilization_cfg[k]) for k in ("c_time", "c_advection", "c_diffusion")}

    setup_start = time.perf_counter()
    mesh = NativeMeshAdapter.from_domain(make_domain(cfg, nx), nx=nx, ny=nx)
    pde = make_pde(cfg)
    fine_solver = create_fem_solver(mesh, pde, mode, stab_coeffs, solver_cfg)
    coarse_solver = create_fem_solver(mesh, pde, mode, stab_coeffs, solver_cfg)
    setup_seconds = time.perf_counter() - setup_start

    u0 = np.asarray(pde.evaluate_solution(mesh.nodes, 0.0), dtype=np.float64)
    u_fine = u0.copy()
    u_coarse = u0.copy()
    initial = compute_error_details(mesh, u_fine, pde, 0.0)
    prev_l2 = float(initial["abs_l2_error"])
    prev_h1 = float(initial["abs_h1_seminorm_error"])
    prev_rel_l2 = float(initial["rel_l2_error"])
    prev_rel_h1 = float(initial["rel_h1_seminorm_error"])
    prev_diff = l2_p1_difference(mesh, u_fine, u_coarse)
    l2_integral = h1_integral = rel_l2_integral = rel_h1_integral = 0.0
    diff_integral = 0.0
    fine_assembly = fine_solve = coarse_assembly = coarse_solve = 0.0
    fine_wall = coarse_wall = 0.0
    fine_residual_max = fine_tolerance_max = 0.0
    coarse_residual_max = coarse_tolerance_max = 0.0

    for step in range(1, n_fine + 1):
        t_next = step * fine_dt
        tick = time.perf_counter()
        result = fine_solver.step_backward_euler_with_diagnostics(u_fine, fine_dt, t_next)
        fine_wall += time.perf_counter() - tick
        u_fine = np.asarray(result["u_next"], dtype=np.float64)
        fine_assembly += float(result["assembly_seconds"])
        fine_solve += float(result["solve_seconds"])
        fine_residual_max = max(fine_residual_max, float(result["linear_solver_residual"]))
        fine_tolerance_max = max(fine_tolerance_max, float(result["linear_solver_tolerance"]))

        details = compute_error_details(mesh, u_fine, pde, t_next)
        cur_l2 = float(details["abs_l2_error"])
        cur_h1 = float(details["abs_h1_seminorm_error"])
        cur_rel_l2 = float(details["rel_l2_error"])
        cur_rel_h1 = float(details["rel_h1_seminorm_error"])
        l2_integral = _integral_add(l2_integral, prev_l2, cur_l2, fine_dt)
        h1_integral = _integral_add(h1_integral, prev_h1, cur_h1, fine_dt)
        rel_l2_integral = _integral_add(rel_l2_integral, prev_rel_l2, cur_rel_l2, fine_dt)
        rel_h1_integral = _integral_add(rel_h1_integral, prev_rel_h1, cur_rel_h1, fine_dt)
        prev_l2, prev_h1 = cur_l2, cur_h1
        prev_rel_l2, prev_rel_h1 = cur_rel_l2, cur_rel_h1

        if step % 2 == 0:
            tick = time.perf_counter()
            coarse = coarse_solver.step_backward_euler_with_diagnostics(u_coarse, coarse_dt, t_next)
            coarse_wall += time.perf_counter() - tick
            u_coarse = np.asarray(coarse["u_next"], dtype=np.float64)
            coarse_assembly += float(coarse["assembly_seconds"])
            coarse_solve += float(coarse["solve_seconds"])
            coarse_residual_max = max(coarse_residual_max, float(coarse["linear_solver_residual"]))
            coarse_tolerance_max = max(coarse_tolerance_max, float(coarse["linear_solver_tolerance"]))
            current_diff = l2_p1_difference(mesh, u_fine, u_coarse)
            diff_integral = _integral_add(diff_integral, prev_diff, current_diff, coarse_dt)
            prev_diff = current_diff

    extrema = solution_extrema_and_oscillations(mesh, pde, u_fine, T)
    avg_l2 = l2_integral / T
    avg_h1 = h1_integral / T
    avg_rel_l2 = rel_l2_integral / T
    avg_rel_h1 = rel_h1_integral / T
    avg_diff = diff_integral / T
    ratio = avg_diff / max(avg_l2, np.finfo(float).tiny)
    return {
        "mode": mode, "nx": int(nx), "ny": int(nx), "h": maximum_element_diameter(mesh),
        "n_nodes": int(mesh.num_nodes), "n_elements": int(mesh.num_cells), "n_dofs": int(mesh.num_nodes),
        "dt_temporal_comparison_coarse": coarse_dt, "dt_used": fine_dt,
        "time_average_absolute_l2_error": avg_l2,
        "time_average_absolute_h1_seminorm_error": avg_h1,
        "time_average_relative_l2_error": avg_rel_l2,
        "time_average_relative_h1_seminorm_error": avg_rel_h1,
        "absolute_l2_error_final": float(details["abs_l2_error"]),
        "absolute_h1_seminorm_error_final": float(details["abs_h1_seminorm_error"]),
        "relative_l2_error_final": float(details["rel_l2_error"]),
        "relative_h1_seminorm_error_final": float(details["rel_h1_seminorm_error"]),
        "temporal_difference_time_average_l2": avg_diff,
        "temporal_contamination_ratio": ratio,
        "temporal_separation_pass": bool(ratio <= float(cfg["closeout"]["temporal_contamination_limit"])),
        "temporal_contamination_limit": float(cfg.get("closeout", {}).get("temporal_contamination_limit", 0.10)),
        "setup_seconds": setup_seconds,
        "fine_solver_wall_seconds": fine_wall,
        "fine_assembly_seconds": fine_assembly,
        "fine_solve_seconds": fine_solve,
        "coarse_solver_wall_seconds": coarse_wall,
        "coarse_assembly_seconds": coarse_assembly,
        "coarse_solve_seconds": coarse_solve,
        "fine_max_linear_solver_residual": fine_residual_max,
        "fine_max_linear_solver_tolerance": fine_tolerance_max,
        "coarse_max_linear_solver_residual": coarse_residual_max,
        "coarse_max_linear_solver_tolerance": coarse_tolerance_max,
        "solution_min": extrema["solution_min"], "solution_max": extrema["solution_max"],
        "overshoot": extrema["overshoot"], "undershoot": extrema["undershoot"],
        "error_residual_local_extrema_count": extrema["error_residual_local_extrema_count"],
        "cells_with_local_error_extrema": extrema["cells_with_local_error_extrema"],
        "fraction_cells_with_local_error_extrema": extrema["fraction_cells_with_local_error_extrema"],
        "high_gradient_node_fraction": extrema["high_gradient_node_fraction"],
        "high_gradient_mean_absolute_nodal_error": extrema["high_gradient_mean_absolute_nodal_error"],
        "high_gradient_maximum_absolute_nodal_error": extrema["high_gradient_maximum_absolute_nodal_error"],
    }


def tau_decomposition(
    beta_speed: np.ndarray, h_beta: np.ndarray, dt: float, nu: float,
    c_time: float, c_advection: float, c_diffusion: float,
) -> dict[str, np.ndarray]:
    """Return tau and the fractional squared-frequency contribution of each term."""
    temporal = np.full_like(h_beta, c_time / dt, dtype=np.float64)
    advective = c_advection * beta_speed / h_beta
    diffusive = c_diffusion * nu / (h_beta * h_beta)
    denom2 = temporal**2 + advective**2 + diffusive**2
    tau = 1.0 / np.sqrt(denom2)
    return {
        "tau": tau,
        "temporal_fraction": temporal**2 / denom2,
        "advective_fraction": advective**2 / denom2,
        "diffusive_fraction": diffusive**2 / denom2,
        "temporal_frequency": temporal,
        "advective_frequency": advective,
        "diffusive_frequency": diffusive,
    }


def native_fixed_mesh_metrics(cfg: Mapping[str, Any], nx: int) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    NativeMeshAdapter, _, _, _, _ = native_modules()
    mesh = NativeMeshAdapter.from_domain(make_domain(cfg, nx), nx=nx, ny=nx)
    p = cfg["pde"]
    beta = np.tile(np.array([float(p["bx"]), float(p["by"])], dtype=np.float64), (mesh.num_cells, 1))
    h_beta, pe = directional_element_metrics(
        np.asarray(mesh.nodes, dtype=np.float64), np.asarray(mesh.cells, dtype=np.int64), beta, float(p["nu"])
    )
    speed = np.linalg.norm(beta, axis=1)
    return mesh, None, beta, h_beta, pe


def closeout_metadata() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "configured_cpu_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "timing_policy": "CPU only; one configured numerical thread; warm-up excluded; measured repetitions retained individually",
    }
