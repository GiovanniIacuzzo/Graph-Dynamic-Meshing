"""Shared, auditable utilities for the GDM Stage-E1 numerical campaign."""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "stage_e.yaml"


def configure_cpu_threads(count: int = 1) -> None:
    """Set common CPU thread limits before native numerical work starts."""
    value = str(max(1, int(count)))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configurazione Stage E non valida: {config_path}")
    cfg["_config_path"] = str(config_path.resolve())
    return cfg


def stage_e_root(cfg: Mapping[str, Any]) -> Path:
    return (ROOT / str(cfg["experiment"].get("output_dir", "results/stage_e"))).resolve()


def prepare_output_tree(cfg: Mapping[str, Any]) -> Path:
    root = stage_e_root(cfg)
    for name in ("solver_verification", "work_precision", "tables", "figures", "diagnostics"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def configure_logging(cfg: Mapping[str, Any], filename: str = "experiment.log") -> logging.Logger:
    root = prepare_output_tree(cfg)
    logger = logging.getLogger("gdm.stage_e")
    logger.setLevel(logging.INFO)
    for old_handler in tuple(logger.handlers):
        old_handler.close()
        logger.removeHandler(old_handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(root / filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def atomic_write_json(path: str | Path, payload: Mapping[str, Any], *, merge: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    if merge and target.exists():
        with target.open("r", encoding="utf-8") as stream:
            previous = json.load(stream)
        if isinstance(previous, dict):
            previous.update(data)
            data = previous
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(target)


def merge_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str]) -> None:
    """Append new configuration rows without replacing rows already completed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    fields: list[str] = []
    if target.exists():
        with target.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or [])
            existing = list(reader)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        return

    def key(row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(row.get(field, "")) for field in key_fields)

    seen = {key(row) for row in existing}
    for row in rows:
        row_key = key(row)
        if row_key in seen:
            continue
        existing.append({field: row.get(field, "") for field in fields})
        seen.add(row_key)

    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
    temporary.replace(target)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def csv_contains_keys(path: str | Path, key_fields: Sequence[str], required: Sequence[Sequence[Any]]) -> bool:
    rows = read_csv_rows(path)
    existing = {tuple(str(row.get(field, "")) for field in key_fields) for row in rows}
    wanted = {tuple(str(value) for value in values) for values in required}
    return bool(wanted) and wanted.issubset(existing)


def write_latex_table(path: str | Path, caption: str, label: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    colspec = "l" + "r" * (len(columns) - 1)
    row_end = " \\\\"  # LaTeX row terminator: two backslashes.
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        " & ".join(columns) + row_end,
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(str(value) for value in row) + row_end)
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def native_modules():
    """Lazy import keeps pure-Python diagnostics inspectable without a local deal.II build."""
    python_root = ROOT / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    from gdm.native_api import (
        NativeMeshAdapter,
        create_pde_model,
        create_fem_solver,
        create_ale_solver,
        compute_native_continuous_error_details,
    )
    return NativeMeshAdapter, create_pde_model, create_fem_solver, create_ale_solver, compute_native_continuous_error_details


def make_domain(cfg: Mapping[str, Any], nx: int, ny: int | None = None) -> dict[str, float]:
    domain = cfg["domain"]
    return {
        "x_min": float(domain["x_min"]),
        "x_max": float(domain["x_max"]),
        "y_min": float(domain["y_min"]),
        "y_max": float(domain["y_max"]),
        "nx": int(nx),
        "ny": int(nx if ny is None else ny),
    }


def make_pde(cfg: Mapping[str, Any]):
    _, create_pde_model, _, _, _ = native_modules()
    p = cfg["pde"]
    return create_pde_model(
        bx=float(p["bx"]), by=float(p["by"]), nu=float(p["nu"]),
        sigma=float(p["sigma"]), beta=float(p["beta"]), T=float(p["T"]),
    )


def get_native_solver_configs(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return dict(cfg.get("linear_solver", {})), dict(cfg.get("stabilization", {}))


def number_of_steps(t_final: float, dt: float) -> int:
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt deve essere finito e positivo.")
    steps = int(round(t_final / dt))
    if steps < 1 or abs(steps * dt - t_final) > 1.0e-10 * max(1.0, t_final):
        raise ValueError(f"T={t_final} non è multiplo intero di dt={dt}.")
    return steps


def trapezoid_average(times: Sequence[float], values: Sequence[float], t_final: float) -> float:
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if t.size < 2 or v.shape != t.shape:
        raise ValueError("Serie temporale non valida per l'integrazione trapezoidale.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("I tempi devono essere strettamente crescenti.")
    return float(np.sum(0.5 * np.diff(t) * (v[:-1] + v[1:])) / t_final)


def l2_p1_difference(mesh: Any, u_a: np.ndarray, u_b: np.ndarray) -> float:
    """Exact L2 norm of the difference of two P1 fields on the same triangles."""
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    d = np.asarray(u_a, dtype=np.float64) - np.asarray(u_b, dtype=np.float64)
    if d.shape != (nodes.shape[0],):
        raise ValueError("Dimensione delle soluzioni non coerente con la mesh.")
    integral = 0.0
    for tri in cells:
        p0, p1, p2 = nodes[tri]
        det = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        area = 0.5 * abs(float(det))
        d0, d1, d2 = d[tri]
        quadratic = 2.0 * (d0*d0 + d1*d1 + d2*d2) + 2.0 * (d0*d1 + d1*d2 + d2*d0)
        integral += (area / 12.0) * quadratic
    return float(math.sqrt(max(integral, 0.0)))


def exact_mesh_motion(nodes: np.ndarray, t: float, period: float, amplitude: float) -> np.ndarray:
    """The same prescribed harmonic nodal motion used by cpp_core/src/mms.cpp."""
    points = np.asarray(nodes, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    spatial = np.sin(np.pi * x) * np.sin(np.pi * y)
    omega = 2.0 * np.pi / period
    moved = points.copy()
    moved[:, 0] += amplitude * spatial * np.sin(omega * t)
    moved[:, 1] += 0.5 * amplitude * spatial * (np.cos(omega * t) - 1.0)
    return moved


def build_run_errors(mesh: Any, pde: Any, snapshots: Sequence[np.ndarray], times: Sequence[float]) -> dict[str, Any]:
    _, _, _, _, compute_error_details = native_modules()
    rel_l2: list[float] = []
    rel_h1: list[float] = []
    abs_l2: list[float] = []
    abs_h1: list[float] = []
    for field, t in zip(snapshots, times):
        detail = compute_error_details(mesh, np.asarray(field, dtype=np.float64), pde, float(t))
        rel_l2.append(float(detail["rel_l2_error"]))
        rel_h1.append(float(detail["rel_h1_seminorm_error"]))
        abs_l2.append(float(detail["abs_l2_error"]))
        abs_h1.append(float(detail["abs_h1_seminorm_error"]))
    t_final = float(times[-1])
    return {
        "relative_l2_error_final": rel_l2[-1],
        "relative_h1_seminorm_error_final": rel_h1[-1],
        "absolute_l2_error_final": abs_l2[-1],
        "absolute_h1_seminorm_error_final": abs_h1[-1],
        "time_average_relative_l2_error": trapezoid_average(times, rel_l2, t_final),
        "time_average_relative_h1_seminorm_error": trapezoid_average(times, rel_h1, t_final),
        "time_average_absolute_l2_error": trapezoid_average(times, abs_l2, t_final),
        "time_average_absolute_h1_seminorm_error": trapezoid_average(times, abs_h1, t_final),
        "history": {
            "time": [float(v) for v in times],
            "relative_l2_error": rel_l2,
            "relative_h1_seminorm_error": rel_h1,
            "absolute_l2_error": abs_l2,
            "absolute_h1_seminorm_error": abs_h1,
        },
    }


def run_fixed_case(
    cfg: Mapping[str, Any], nx: int, dt: float, mode: str = "galerkin", *,
    retain_snapshots: bool = True,
) -> dict[str, Any]:
    """Run the existing native FEM to T and collect solve/setup timings.

    Accuracy post-processing is performed after the measured wall-clock interval.
    The error metrics are continuous norms evaluated by the deal.II diagnostics.
    """
    NativeMeshAdapter, _, create_fem_solver, _, _ = native_modules()
    t_final = float(cfg["pde"]["T"])
    steps = number_of_steps(t_final, dt)
    solver_cfg, stab_cfg = get_native_solver_configs(cfg)
    stab_coeffs = {k: float(stab_cfg[k]) for k in ("c_time", "c_advection", "c_diffusion") if k in stab_cfg}

    wall_start = time.perf_counter()
    setup_start = time.perf_counter()
    mesh = NativeMeshAdapter.from_domain(make_domain(cfg, nx), nx=nx, ny=nx)
    pde = make_pde(cfg)
    solver = create_fem_solver(
        mesh, pde, stabilization_mode=mode,
        stabilization_config=stab_coeffs, linear_solver_config=solver_cfg,
    )
    setup_seconds = time.perf_counter() - setup_start
    u = np.asarray(pde.evaluate_solution(mesh.nodes, 0.0), dtype=np.float64)
    snapshots: list[np.ndarray] = [u.copy()]
    times = [0.0]
    assembly_seconds = 0.0
    solve_seconds = 0.0
    residuals: list[float] = []
    solver_tolerances: list[float] = []

    for step in range(1, steps + 1):
        t_next = step * dt
        result = solver.step_backward_euler_with_diagnostics(u, float(dt), float(t_next))
        u = np.asarray(result["u_next"], dtype=np.float64)
        assembly_seconds += float(result["assembly_seconds"])
        solve_seconds += float(result["solve_seconds"])
        residuals.append(float(result["linear_solver_residual"]))
        solver_tolerances.append(float(result["linear_solver_tolerance"]))
        if retain_snapshots:
            snapshots.append(u.copy())
            times.append(float(t_next))

    total_wall_seconds = time.perf_counter() - wall_start
    if not retain_snapshots:
        snapshots = [np.asarray(pde.evaluate_solution(mesh.nodes, 0.0), dtype=np.float64), u.copy()]
        times = [0.0, t_final]
    errors = build_run_errors(mesh, pde, snapshots, times)
    extrema = solution_extrema_and_oscillations(mesh, pde, u, t_final)
    return {
        "nx": int(nx), "ny": int(nx), "dt": float(dt), "mode": str(mode),
        "n_nodes": int(mesh.num_nodes), "n_elements": int(mesh.num_cells),
        "n_dofs": int(mesh.num_nodes),
        "setup_seconds": float(setup_seconds),
        "assembly_seconds": float(assembly_seconds),
        "solve_seconds": float(solve_seconds),
        "total_wall_seconds": float(total_wall_seconds),
        "other_seconds": float(max(0.0, total_wall_seconds - setup_seconds - assembly_seconds - solve_seconds)),
        "max_linear_solver_residual": float(max(residuals, default=0.0)),
        "max_linear_solver_tolerance": float(max(solver_tolerances, default=0.0)),
        "max_residual_to_tolerance_ratio": float(max((r / max(t, np.finfo(float).tiny) for r, t in zip(residuals, solver_tolerances)), default=0.0)),
        "mean_linear_solver_residual": float(np.mean(residuals)) if residuals else 0.0,
        **{k: v for k, v in errors.items() if k != "history"},
        **extrema,
        "history": errors["history"],
        "solution_history": snapshots,
        "final_solution": u,
        "mesh": mesh,
        "pde": pde,
    }


def solution_extrema_and_oscillations(mesh: Any, pde: Any, u: np.ndarray, t: float) -> dict[str, Any]:
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    exact_nodes = np.asarray(pde.evaluate_solution(nodes, float(t)), dtype=np.float64)
    # The manufactured solution is assessed on a deterministic dense grid only
    # to establish its range; this data never enters the numerical solver.
    axis = np.linspace(0.0, 1.0, 257)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    sample_points = np.column_stack((xx.ravel(), yy.ravel()))
    exact_dense = np.asarray(pde.evaluate_solution(sample_points, float(t)), dtype=np.float64)

    neighbours: list[set[int]] = [set() for _ in range(nodes.shape[0])]
    for tri in cells:
        for i in range(3):
            for j in range(3):
                if i != j:
                    neighbours[int(tri[i])].add(int(tri[j]))
    is_boundary = np.zeros(nodes.shape[0], dtype=bool)
    is_boundary[np.asarray(mesh.boundary_nodes, dtype=np.int64)] = True
    error_nodes = np.asarray(u, dtype=np.float64) - exact_nodes
    tolerance = 1.0e-8 * max(1.0, float(np.max(np.abs(exact_nodes))))
    local_error_extrema_nodes: set[int] = set()
    for i, adjacent in enumerate(neighbours):
        if is_boundary[i] or not adjacent:
            continue
        vals = error_nodes[list(adjacent)]
        if error_nodes[i] > float(np.max(vals)) + tolerance or error_nodes[i] < float(np.min(vals)) - tolerance:
            local_error_extrema_nodes.add(i)
    oscillatory_cells = sum(1 for tri in cells if any(int(v) in local_error_extrema_nodes for v in tri))

    exact = pde.evaluate_exact(nodes[:, 0], nodes[:, 1], float(t))
    exact_ux = np.asarray(exact[1], dtype=np.float64)
    exact_uy = np.asarray(exact[2], dtype=np.float64)
    exact_grad = np.hypot(exact_ux, exact_uy)
    grad_cutoff = float(np.quantile(exact_grad, 0.90))
    high_gradient = exact_grad >= grad_cutoff
    nodal_abs_error = np.abs(error_nodes)
    high_gradient_mean_error = float(np.mean(nodal_abs_error[high_gradient])) if np.any(high_gradient) else 0.0
    high_gradient_max_error = float(np.max(nodal_abs_error[high_gradient])) if np.any(high_gradient) else 0.0

    u_min = float(np.min(u))
    u_max = float(np.max(u))
    exact_min = float(np.min(exact_dense))
    exact_max = float(np.max(exact_dense))
    return {
        "solution_min": u_min,
        "solution_max": u_max,
        "exact_sampled_min": exact_min,
        "exact_sampled_max": exact_max,
        "overshoot": float(max(0.0, u_max - exact_max)),
        "undershoot": float(max(0.0, exact_min - u_min)),
        "error_residual_local_extrema_count": int(len(local_error_extrema_nodes)),
        "cells_with_local_error_extrema": int(oscillatory_cells),
        "fraction_cells_with_local_error_extrema": float(oscillatory_cells / max(1, cells.shape[0])),
        "error_residual_local_extrema_definition": "interior node whose nodal error is a strict 1-ring maximum/minimum by more than 1e-8*max(1,||u_exact||_infinity)",
        "high_gradient_node_fraction": float(np.mean(high_gradient)),
        "high_gradient_mean_absolute_nodal_error": high_gradient_mean_error,
        "high_gradient_maximum_absolute_nodal_error": high_gradient_max_error,
    }


def maximum_element_diameter(mesh: Any) -> float:
    """Maximum physical triangle diameter, used as the spatial mesh size h."""
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    triangles = nodes[cells]
    lengths = np.stack((
        np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
        np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
        np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
    ), axis=1)
    if lengths.size == 0 or not np.all(np.isfinite(lengths)):
        raise ValueError("Impossibile calcolare il diametro massimo della mesh.")
    return float(np.max(lengths))


def directional_element_metrics(
    nodes: np.ndarray, cells: np.ndarray, beta_cells: np.ndarray, nu: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return P1 directional h_beta and local Pe_K for all triangles."""
    h_values = np.empty(cells.shape[0], dtype=np.float64)
    pe_values = np.empty(cells.shape[0], dtype=np.float64)
    for e, tri in enumerate(cells):
        p0, p1, p2 = nodes[tri]
        det = float((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
        if not math.isfinite(det) or det <= 1.0e-15:
            raise ValueError(f"Elemento invertito/degenere durante la diagnostica Péclet: {e}.")
        grad = np.asarray([
            [(p1[1] - p2[1]) / det, (p2[0] - p1[0]) / det],
            [(p2[1] - p0[1]) / det, (p0[0] - p2[0]) / det],
            [(p0[1] - p1[1]) / det, (p1[0] - p0[0]) / det],
        ], dtype=np.float64)
        beta = beta_cells[e]
        speed = float(np.linalg.norm(beta))
        if speed <= 1.0e-14:
            edges = [np.linalg.norm(p1-p0), np.linalg.norm(p2-p1), np.linalg.norm(p0-p2)]
            h_beta = float(max(edges))
            pe = 0.0
        else:
            denominator = float(np.sum(np.abs(grad @ beta)))
            if denominator <= 1.0e-14 * speed:
                raise ValueError(f"h_beta non definito nell'elemento {e}.")
            h_beta = 2.0 * speed / denominator
            pe = math.inf if nu <= 0.0 else speed * h_beta / (2.0 * nu)
        h_values[e] = h_beta
        pe_values[e] = pe
    return h_values, pe_values


def distribution_row(values: np.ndarray, *, mesh_type: str, time_value: float) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mesh_type": mesh_type,
        "time": float(time_value),
        "n_elements": int(arr.size),
        "minimum": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "maximum": float(np.max(arr)),
    }


def observed_order(coarse_error: float, fine_error: float, coarse_h: float, fine_h: float) -> float | None:
    if min(coarse_error, fine_error, coarse_h, fine_h) <= 0.0:
        return None
    return float(math.log(coarse_error / fine_error) / math.log(coarse_h / fine_h))


def format_float(value: Any, digits: int = 6) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}g}"
