"""Small Stage-E smoke test; does not launch 64x64 or the full campaign."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from .common import (
    atomic_write_json, configure_cpu_threads, configure_logging, exact_mesh_motion,
    load_config, make_domain, make_pde, prepare_output_tree, run_fixed_case,
)


def run(config_path: str | None = None, run_native_tests: bool | None = None) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    configure_cpu_threads(int(cfg["experiment"].get("cpu_threads", 1)))
    logger = configure_logging(cfg)
    out = prepare_output_tree(cfg)
    spec = cfg["smoke_test"]
    nx, dt = int(spec["resolution"]), float(spec["dt"])
    logger.info("Stage E smoke test: nx=%d dt=%g (nessuna griglia 64x64)", nx, dt)
    fixed_results = []
    for mode in ("galerkin", "supg"):
        result = run_fixed_case(cfg, nx, dt, mode, retain_snapshots=True)
        finite = all(np.isfinite(float(result[k])) for k in (
            "time_average_absolute_l2_error", "time_average_absolute_h1_seminorm_error",
            "assembly_seconds", "solve_seconds", "total_wall_seconds"))
        if not finite:
            raise RuntimeError(f"Smoke test {mode}: metrica non finita.")
        if float(result["max_residual_to_tolerance_ratio"]) > 10.0:
            raise RuntimeError(f"Smoke test {mode}: residuo lineare oltre 10 volte la tolleranza GMRES configurata.")
        fixed_results.append({
            "mode": mode, "n_dofs": result["n_dofs"], "n_elements": result["n_elements"],
            "time_average_absolute_l2_error": result["time_average_absolute_l2_error"],
            "time_average_absolute_h1_seminorm_error": result["time_average_absolute_h1_seminorm_error"],
            "total_wall_seconds": result["total_wall_seconds"],
            "max_linear_solver_residual": result["max_linear_solver_residual"],
            "max_linear_solver_tolerance": result["max_linear_solver_tolerance"],
            "max_residual_to_tolerance_ratio": result["max_residual_to_tolerance_ratio"],
        })

    # A moving-mesh pure-GCL step verifies the API and the basic free-stream invariant.
    from .common import native_modules
    NativeMeshAdapter, _, _, create_ale_solver, _ = native_modules()
    mesh0 = NativeMeshAdapter.from_domain(make_domain(cfg, nx), nx=nx, ny=nx)
    pde = make_pde(cfg)
    nodes1 = exact_mesh_motion(mesh0.nodes, dt, float(cfg["pde"]["T"]),
                               float(cfg["peclet"]["ale_motion_amplitude"]))
    mesh1 = mesh0.update_coordinates(nodes1, enforce_fixed_boundary=True)
    solver_cfg = cfg.get("linear_solver", {})
    ale = create_ale_solver(pde=pde, linear_solver_config=solver_cfg)
    u1, diag = ale.step_ale(mesh0.core, mesh1.core, np.ones(mesh0.num_nodes), dt, dt, True, True)
    u1 = np.asarray(u1, dtype=float)
    free_stream_error = float(np.max(np.abs(u1 - 1.0)))
    if not np.isfinite(free_stream_error):
        raise RuntimeError("Smoke test ALE-GCL: free-stream error non finito.")
    if free_stream_error >= 1.0e-8:
        raise RuntimeError(f"Smoke test ALE-GCL: free-stream error {free_stream_error:g} oltre 1e-8.")
    if float(diag.max_elem_gcl_residual) >= 1.0e-10:
        raise RuntimeError("Smoke test ALE-GCL: residuo geometrico elementare oltre 1e-10.")
    if float(diag.algebraic_gcl_residual_linf) >= 1.0e-9:
        raise RuntimeError("Smoke test ALE-GCL: residuo algebrico oltre 1e-9.")
    if float(diag.linear_solver_residual) > 10.0 * float(diag.linear_solver_tolerance):
        raise RuntimeError("Smoke test ALE-GCL: residuo lineare oltre 10 volte la tolleranza GMRES configurata.")

    do_native_tests = bool(spec.get("run_native_tests", False)) if run_native_tests is None else bool(run_native_tests)
    native_test_result = {"requested": do_native_tests, "executed": False, "returncode": None}
    if do_native_tests:
        build_dir = Path(__file__).resolve().parents[2] / "build"
        if not (build_dir / "CTestTestfile.cmake").exists():
            raise FileNotFoundError("Build directory non configurata: eseguire prima CMake e compilare il core.")
        completed = subprocess.run(["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
                                    check=False, text=True, capture_output=True)
        native_test_result = {
            "requested": True, "executed": True, "returncode": int(completed.returncode),
            "stdout": completed.stdout, "stderr": completed.stderr,
        }
        if completed.returncode != 0:
            raise RuntimeError(f"CTest ha fallito (returncode={completed.returncode}).\n{completed.stdout}\n{completed.stderr}")

    report = {
        "status": "passed", "resolution": nx, "dt": dt,
        "fixed_solver_runs": fixed_results,
        "moving_mesh_free_stream_error": free_stream_error,
        "native_gcl_element_residual": float(diag.max_elem_gcl_residual),
        "native_gcl_algebraic_residual": float(diag.algebraic_gcl_residual_linf),
        "native_linear_solver_residual": float(diag.linear_solver_residual),
        "native_linear_solver_tolerance": float(diag.linear_solver_tolerance),
        "native_ctest": native_test_result,
        "note": "Questo smoke test verifica il percorso operativo a bassa risoluzione; non sostituisce i test C++/Python completi né certifica convergenza o accuratezza scientifica.",
    }
    atomic_write_json(out / "smoke_test.json", report, merge=False)
    logger.info("Smoke test superato. Report: %s", out / "smoke_test.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--native-tests", action="store_true")
    args = parser.parse_args()
    run(args.config, run_native_tests=args.native_tests)
