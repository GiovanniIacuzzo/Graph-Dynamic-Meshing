"""E1-C compare Galerkin/SUPG on fixed, static-ALE and moving-ALE meshes."""
from __future__ import annotations

import argparse
import time

import numpy as np

from .common import (
    configure_cpu_threads, configure_logging, exact_mesh_motion, get_native_solver_configs,
    load_config, make_domain, make_pde, merge_csv, prepare_output_tree, csv_contains_keys,
    l2_p1_difference,
)


def run(config_path: str | None = None) -> list[dict]:
    cfg = load_config(config_path) if config_path else load_config()
    configure_cpu_threads(int(cfg["experiment"].get("cpu_threads", 1)))
    logger = configure_logging(cfg)
    out = prepare_output_tree(cfg)
    from .common import native_modules, build_run_errors
    NativeMeshAdapter, _, create_fem_solver, create_ale_solver, compute_error_details = native_modules()
    spec = cfg["supg_comparison"]
    nx, dt = int(spec["resolution"]), float(spec["dt"])
    mesh_types = ["fixed", "static_ale"] + (["moving_ale"] if bool(spec.get("compare_moving_ale", True)) else [])
    supg_csv = out / "solver_verification" / "supg_comparison.csv"
    if csv_contains_keys(supg_csv, ["mesh_type", "mode", "nx", "dt"],
                         [[kind, mode, nx, dt] for kind in mesh_types for mode in ("galerkin", "supg")]):
        logger.info("E1 SUPG: confronto già completo; riuso del CSV (resume).")
        import csv
        with supg_csv.open("r", newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    T = float(cfg["pde"]["T"])
    steps = int(round(T / dt))
    if abs(steps * dt - T) > 1e-10:
        raise ValueError("dt non divide T nella configurazione SUPG.")
    solver_cfg, stab_cfg = get_native_solver_configs(cfg)
    stab_coeffs = {k: float(stab_cfg[k]) for k in ("c_time", "c_advection", "c_diffusion")}
    mesh0 = NativeMeshAdapter.from_domain(make_domain(cfg, nx), nx=nx, ny=nx)
    pde = make_pde(cfg)
    results: list[dict] = []
    final_fields: dict[tuple[str, str], np.ndarray] = {}

    for mode in ("galerkin", "supg"):
        logger.info("SUPG comparison: fixed mesh mode=%s", mode)
        solver = create_fem_solver(mesh0, pde, mode, stab_coeffs, solver_cfg)
        u = np.asarray(pde.evaluate_solution(mesh0.nodes, 0.0), dtype=float)
        snapshots, times = [u.copy()], [0.0]
        residual_max = tolerance_max = 0.0
        tic = time.perf_counter()
        for n in range(1, steps + 1):
            tnext = n * dt
            ret = solver.step_backward_euler_with_diagnostics(u, dt, tnext)
            u = np.asarray(ret["u_next"], dtype=float)
            residual_max = max(residual_max, float(ret["linear_solver_residual"]))
            tolerance_max = max(tolerance_max, float(ret["linear_solver_tolerance"]))
            snapshots.append(u.copy()); times.append(tnext)
        wall = time.perf_counter() - tic
        errors = build_run_errors(mesh0, pde, snapshots, times)
        final_fields[("fixed", mode)] = u.copy()
        results.append({"mesh_type": "fixed", "mode": mode, "nx": nx, "dt": dt,
                        "time_average_absolute_l2_error": errors["time_average_absolute_l2_error"],
                        "time_average_absolute_h1_seminorm_error": errors["time_average_absolute_h1_seminorm_error"],
                        "relative_l2_error_final": errors["relative_l2_error_final"],
                        "relative_h1_seminorm_error_final": errors["relative_h1_seminorm_error_final"],
                        "wall_seconds": wall, "max_linear_solver_residual": residual_max,
                        "max_linear_solver_tolerance": tolerance_max,
                        "max_linear_solver_residual_to_tolerance_ratio": residual_max / max(tolerance_max, np.finfo(float).tiny),
                        "max_element_gcl_residual": 0.0,
                        "max_algebraic_gcl_residual_linf": 0.0})

    # Static ALE is a stringent equivalence check against fixed-grid Galerkin/SUPG.
    for mode in ("galerkin", "supg"):
        logger.info("SUPG equivalence: static ALE mode=%s", mode)
        ale = create_ale_solver(pde=pde, stabilization_mode=mode,
                                stabilization_config=stab_coeffs, linear_solver_config=solver_cfg)
        u = np.asarray(pde.evaluate_solution(mesh0.nodes, 0.0), dtype=float)
        snapshots, times = [u.copy()], [0.0]
        residual_max = tolerance_max = element_gcl_max = algebraic_gcl_max = 0.0
        tic = time.perf_counter()
        for n in range(1, steps + 1):
            tnext = n * dt
            u, diag = ale.step_ale(mesh0.core, mesh0.core, u, dt, tnext, False, True)
            u = np.asarray(u, dtype=float)
            residual_max = max(residual_max, float(diag.linear_solver_residual))
            tolerance_max = max(tolerance_max, float(diag.linear_solver_tolerance))
            element_gcl_max = max(element_gcl_max, float(diag.max_elem_gcl_residual))
            algebraic_gcl_max = max(algebraic_gcl_max, float(diag.algebraic_gcl_residual_linf))
            snapshots.append(u.copy()); times.append(tnext)
        wall = time.perf_counter() - tic
        errors = build_run_errors(mesh0, pde, snapshots, times)
        final_fields[("static_ale", mode)] = u.copy()
        results.append({"mesh_type": "static_ale", "mode": mode, "nx": nx, "dt": dt,
                        "time_average_absolute_l2_error": errors["time_average_absolute_l2_error"],
                        "time_average_absolute_h1_seminorm_error": errors["time_average_absolute_h1_seminorm_error"],
                        "relative_l2_error_final": errors["relative_l2_error_final"],
                        "relative_h1_seminorm_error_final": errors["relative_h1_seminorm_error_final"],
                        "wall_seconds": wall, "max_linear_solver_residual": residual_max,
                        "max_linear_solver_tolerance": tolerance_max,
                        "max_linear_solver_residual_to_tolerance_ratio": residual_max / max(tolerance_max, np.finfo(float).tiny),
                        "max_element_gcl_residual": element_gcl_max,
                        "max_algebraic_gcl_residual_linf": algebraic_gcl_max})

    if bool(spec.get("compare_moving_ale", True)):
        amp = float(spec["ale_motion_amplitude"])
        for mode in ("galerkin", "supg"):
            logger.info("SUPG comparison: moving ALE mode=%s", mode)
            ale = create_ale_solver(pde=pde, stabilization_mode=mode,
                                    stabilization_config=stab_coeffs, linear_solver_config=solver_cfg)
            mesh_n = mesh0
            u = np.asarray(pde.evaluate_solution(mesh_n.nodes, 0.0), dtype=float)
            snapshots, times, mesh_history = [u.copy()], [0.0], [mesh_n]
            residual_max = tolerance_max = element_gcl_max = algebraic_gcl_max = 0.0
            tic = time.perf_counter()
            for n in range(1, steps + 1):
                tnext = n * dt
                nodes_next = exact_mesh_motion(mesh0.nodes, tnext, T, amp)
                mesh_next = mesh0.update_coordinates(nodes_next, enforce_fixed_boundary=True)
                u, diag = ale.step_ale(mesh_n.core, mesh_next.core, u, dt, tnext, False, True)
                u = np.asarray(u, dtype=float)
                residual_max = max(residual_max, float(diag.linear_solver_residual))
                tolerance_max = max(tolerance_max, float(diag.linear_solver_tolerance))
                element_gcl_max = max(element_gcl_max, float(diag.max_elem_gcl_residual))
                algebraic_gcl_max = max(algebraic_gcl_max, float(diag.algebraic_gcl_residual_linf))
                snapshots.append(u.copy()); times.append(tnext); mesh_history.append(mesh_next)
                mesh_n = mesh_next
            wall = time.perf_counter() - tic
            rel_l2, rel_h1, abs_l2, abs_h1 = [], [], [], []
            for mesh, field, t in zip(mesh_history, snapshots, times):
                d = compute_error_details(mesh, field, pde, t)
                rel_l2.append(d["rel_l2_error"]); rel_h1.append(d["rel_h1_seminorm_error"])
                abs_l2.append(d["abs_l2_error"]); abs_h1.append(d["abs_h1_seminorm_error"])
            from .common import trapezoid_average
            errors = {
                "time_average_absolute_l2_error": trapezoid_average(times, abs_l2, T),
                "time_average_absolute_h1_seminorm_error": trapezoid_average(times, abs_h1, T),
                "relative_l2_error_final": rel_l2[-1], "relative_h1_seminorm_error_final": rel_h1[-1],
            }
            final_fields[("moving_ale", mode)] = u.copy()
            results.append({"mesh_type": "moving_ale", "mode": mode, "nx": nx, "dt": dt,
                            **errors, "wall_seconds": wall, "max_linear_solver_residual": residual_max,
                            "max_linear_solver_tolerance": tolerance_max,
                            "max_linear_solver_residual_to_tolerance_ratio": residual_max / max(tolerance_max, np.finfo(float).tiny),
                            "max_element_gcl_residual": element_gcl_max,
                            "max_algebraic_gcl_residual_linf": algebraic_gcl_max})

    # Compare final coefficient vectors with the exact P1 L2 norm on the common static mesh.
    zero = np.zeros(mesh0.num_nodes, dtype=float)
    for row in results:
        if row["mesh_type"] == "static_ale":
            fixed_u = final_fields[("fixed", row["mode"])]
            ale_u = final_fields[("static_ale", row["mode"])]
            difference = l2_p1_difference(mesh0, fixed_u, ale_u)
            reference_norm = l2_p1_difference(mesh0, fixed_u, zero)
            row["fixed_static_ale_final_l2_difference"] = difference
            row["fixed_static_ale_final_relative_l2_difference"] = difference / max(
                reference_norm, np.finfo(float).tiny
            )

    target = out / "solver_verification"
    merge_csv(target / "supg_comparison.csv", results, ["mesh_type", "mode", "nx", "dt"])
    from python.plotting.stage_e import plot_supg_comparison
    plot_supg_comparison(results, out / "figures" / "supg_comparison.pdf")
    logger.info("Confronto Galerkin/SUPG completato.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=None)
    run(parser.parse_args().config)
