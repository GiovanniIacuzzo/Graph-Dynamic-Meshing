"""E1-D independent single-triangle and assembled ALE D-GCL verification."""
from __future__ import annotations

import argparse
import json

import numpy as np

from .common import configure_cpu_threads, configure_logging, exact_mesh_motion, load_config, make_domain, native_modules, prepare_output_tree


def _single_triangle_mesh(NativeMeshAdapter):
    nodes = np.array([[0.2, 0.2], [0.7, 0.2], [0.2, 0.65]], dtype=float)
    cells = np.array([[0, 1, 2]], dtype=np.int64)
    return NativeMeshAdapter.from_arrays(nodes, cells, np.array([0, 1, 2], dtype=np.int64))


def _area(points: np.ndarray, tri: np.ndarray) -> float:
    a, b, c = points[tri]
    return 0.5 * float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))


def _independent_gcl_residual(nodes_n: np.ndarray, nodes_p: np.ndarray, tri: np.ndarray, dt: float) -> tuple[float, float]:
    mid = 0.5 * (nodes_n + nodes_p)
    area_n, area_p, area_m = _area(nodes_n, tri), _area(nodes_p, tri), _area(mid, tri)
    p0, p1, p2 = mid[tri]
    det = 2.0 * area_m
    grads = np.array([
        [(p1[1]-p2[1])/det, (p2[0]-p1[0])/det],
        [(p2[1]-p0[1])/det, (p0[0]-p2[0])/det],
        [(p0[1]-p1[1])/det, (p1[0]-p0[0])/det],
    ])
    w = (nodes_p[tri] - nodes_n[tri]) / dt
    div_w = float(np.sum(w[:, 0] * grads[:, 0] + w[:, 1] * grads[:, 1]))
    lhs = (area_p-area_n)/dt
    rhs = area_m*div_w
    absolute = abs(lhs-rhs)
    relative = absolute / max(abs(lhs), abs(rhs), 1.0e-30)
    return absolute, relative


def _run_one(ale, mesh0, amp: float, period: float, dt: float) -> dict:
    nodes0 = np.asarray(mesh0.nodes, dtype=float)
    nodes1 = exact_mesh_motion(nodes0, dt, period, amp)
    mesh1 = mesh0.update_coordinates(nodes1, enforce_fixed_boundary=False)
    ones = np.ones(nodes0.shape[0], dtype=float)
    u1, diag = ale.step_ale(mesh0.core, mesh1.core, ones, dt, dt, True, True)
    u1 = np.asarray(u1, dtype=float)
    py_residuals = [_independent_gcl_residual(nodes0, nodes1, tri, dt) for tri in np.asarray(mesh0.cells, dtype=np.int64)]
    return {
        "n_nodes": int(mesh0.num_nodes), "n_elements": int(mesh0.num_cells),
        "max_independent_element_gcl_residual_absolute": float(max((r[0] for r in py_residuals), default=0.0)),
        "max_independent_element_gcl_residual_relative": float(max((r[1] for r in py_residuals), default=0.0)),
        "max_native_element_gcl_residual": float(diag.max_elem_gcl_residual),
        "max_native_algebraic_gcl_residual_linf": float(diag.algebraic_gcl_residual_linf),
        "max_dual_c_mesh_column_sum_linf": float(diag.dual_c_mesh_col_sum_linf),
        "free_stream_linf_error": float(np.max(np.abs(u1-1.0))),
        "linear_solver_residual": float(diag.linear_solver_residual),
        "linear_solver_rhs_norm": float(diag.linear_solver_rhs_norm),
        "linear_solver_tolerance": float(diag.linear_solver_tolerance),
        "linear_solver_residual_to_tolerance_ratio": float(diag.linear_solver_residual / max(diag.linear_solver_tolerance, np.finfo(float).tiny)),
        "element_residuals_native": list(map(float, diag.element_gcl_residuals)),
    }


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    configure_cpu_threads(int(cfg["experiment"].get("cpu_threads", 1)))
    logger = configure_logging(cfg)
    out = prepare_output_tree(cfg)
    saved = out / "solver_verification" / "dgcl_verification.json"
    if saved.exists():
        previous = json.loads(saved.read_text(encoding="utf-8"))
        if previous.get("pass") is True:
            logger.info("E1 D-GCL: test già superato; riuso del report (resume).")
            return previous
    NativeMeshAdapter, _, _, create_ale_solver, _ = native_modules()
    spec = cfg["peclet"]
    amp, period = float(spec["ale_motion_amplitude"]), float(spec["motion_period"])
    # Use the same finite-step mesh velocity convention as the SUPG/ALE
    # comparison and the elementwise Péclet campaign.
    dt = float(cfg["supg_comparison"]["dt"])
    # Pure GCL does not use physical coefficients or forcing.
    solver_cfg = cfg.get("linear_solver", {})
    ale = create_ale_solver(pde=None, nu=float(cfg["pde"]["nu"]), bx=0.0, by=0.0,
                            linear_solver_config=solver_cfg)
    single = _single_triangle_mesh(NativeMeshAdapter)
    single_result = _run_one(ale, single, amp, period, dt)
    grid = NativeMeshAdapter.from_domain(make_domain(cfg, 16), nx=16, ny=16)
    grid_result = _run_one(ale, grid, amp, period, dt)
    report = {
        "geometric_identity": "For affine nodal motion X(s)=X_n+s(X_{n+1}-X_n), det(F(s)) is quadratic in s; its derivative is linear. The midpoint rule exactly integrates the derivative, hence (A_{n+1}-A_n)/dt=A_mid div(w)_mid for positive orientation.",
        "algebraic_identity": "((M_{n+1}-M_n)/dt + C_mesh) 1 = 0, with C_mesh=-C_adv^w-M_div^w.",
        "roundoff_and_solver_tolerances": "Residuals are reported without loosening tolerances; linear solver residuals are checked against 10 times the effective GMRES tolerance recorded by the native core (abs_tol vs rel_tol*||rhs||_2). Algebraic and free-stream residuals include floating-point and linear-solver contributions.",
        "single_triangle": single_result,
        "structured_16x16": grid_result,
        "acceptance_thresholds": {
            "element_geometry_relative": 1.0e-10,
            "element_geometry_absolute": 1.0e-10,
            "native_element_absolute": 1.0e-10,
            "algebraic_absolute": 1.0e-10,
            "free_stream_absolute": 1.0e-9,
            "linear_solver_tolerance_factor": 10.0,
        },
        "pass": bool(
            max(single_result["max_independent_element_gcl_residual_relative"], grid_result["max_independent_element_gcl_residual_relative"]) < 1.0e-10 and
            max(single_result["max_independent_element_gcl_residual_absolute"], grid_result["max_independent_element_gcl_residual_absolute"]) < 1.0e-10 and
            max(single_result["max_native_element_gcl_residual"], grid_result["max_native_element_gcl_residual"]) < 1.0e-10 and
            max(single_result["max_native_algebraic_gcl_residual_linf"], grid_result["max_native_algebraic_gcl_residual_linf"]) < 1.0e-10 and
            max(single_result["free_stream_linf_error"], grid_result["free_stream_linf_error"]) < 1.0e-9 and
            max(single_result["linear_solver_residual_to_tolerance_ratio"], grid_result["linear_solver_residual_to_tolerance_ratio"]) <= 10.0
        ),
    }
    from .common import atomic_write_json
    atomic_write_json(out / "solver_verification" / "dgcl_verification.json", report, merge=False)
    logger.info("D-GCL check: %s", "PASS" if report["pass"] else "FAIL")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=None)
    run(parser.parse_args().config)
