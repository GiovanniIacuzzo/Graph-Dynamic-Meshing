"""Re-run one saved oracle sample without modifying its pickle or manifest.

Example:
    PYTHONPATH=python python tools/diagnostics/oracle_sample_probe.py \
        data/raw_oracle_v3_pilot/sample_00009_test_extrap.pkl \
        --config config/stage_a.yaml \
        --max-iter 2000 --passes 2
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gdm.native_api import NativeMeshAdapter  # noqa: E402
from oracle.field import GaussianBumpField  # noqa: E402
from oracle.optimizer import OfflineMeshOptimizer  # noqa: E402
from utils.utils import load_config  # noqa: E402


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="Path to the saved sample_*.pkl")
    parser.add_argument("--config", type=Path, default=Path("config/stage_a.yaml"))
    parser.add_argument("--max-iter", type=int, default=2000, help="Iteration budget per L-BFGS-B pass")
    parser.add_argument("--passes", type=int, default=2, help="Maximum number of sequential optimization passes")
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage_a_oracle_v4/diagnostics"))
    args = parser.parse_args()

    if not args.sample.is_file():
        raise FileNotFoundError(f"Sample not found: {args.sample.resolve()}")
    if args.max_iter < 1 or args.passes < 1:
        raise ValueError("--max-iter and --passes must both be positive.")

    config = load_config(args.config)
    with args.sample.open("rb") as stream:
        sample = pickle.load(stream)
    metadata = sample["metadata"]
    params = metadata["parameters"]
    initial_nodes = np.ascontiguousarray(sample["mesh"]["nodes"], dtype=np.float64)
    cells = np.ascontiguousarray(sample["mesh"]["cells"], dtype=np.int64)

    # Re-create the native reference mesh only to recover the exact boundary DoFs.
    # Refuse to continue if its numbering/topology do not match the saved sample.
    geom = config["geometry"]
    native_mesh = NativeMeshAdapter.from_domain(
        geom["domain"],
        nx=int(geom["initial_grid"]["nx"]),
        ny=int(geom["initial_grid"]["ny"]),
    )
    if native_mesh.nodes.shape != initial_nodes.shape or not np.allclose(
        native_mesh.nodes, initial_nodes, rtol=0.0, atol=1.0e-14
    ):
        raise RuntimeError("Recreated native mesh nodes do not match the saved sample; refusing to diagnose with different topology.")
    if native_mesh.cells.shape != cells.shape or not np.array_equal(native_mesh.cells, cells):
        raise RuntimeError("Recreated native mesh connectivity does not match the saved sample; refusing to diagnose with different topology.")

    field = GaussianBumpField(
        mu_x=float(params["mu_x"]),
        mu_y=float(params["mu_y"]),
        sigma=float(params["sigma"]),
        amplitude=float(params["amplitude"]),
    )
    oracle_cfg = config["oracle"]
    safe_cfg = config["geometry"]["safety"]
    optimizer = OfflineMeshOptimizer(
        reference_field=field,
        initial_nodes=initial_nodes,
        cells=cells,
        boundary_nodes=np.asarray(native_mesh.boundary_nodes, dtype=np.int64),
        lambda_D=float(oracle_cfg["objective"]["lambda_D"]),
        lambda_Q=float(oracle_cfg["objective"]["lambda_Q"]),
        epsilon=float(safe_cfg["epsilon"]),
        max_iter=int(oracle_cfg["optimizer"]["max_iter"]),
        ftol=float(oracle_cfg["optimizer"]["ftol"]),
        gtol=float(oracle_cfg["optimizer"]["gtol"]),
    )

    passes = []
    guess = None
    original_objective = float(optimizer.objective.evaluate(initial_nodes)["total_objective"])
    started = time.perf_counter()
    for pass_index in range(args.passes):
        result = optimizer.optimize(
            initial_guess_nodes=guess,
            max_iter_override=args.max_iter,
        )
        metrics = result.get("final_metrics", {})
        passes.append({
            "pass": pass_index + 1,
            "success": bool(result["success"]),
            "converged": bool(result["converged"]),
            "status": int(result["status"]),
            "message": str(result["message"]),
            "iterations": int(result["iterations"]),
            "nfev": int(result.get("nfev", 0)),
            "njev": int(result.get("njev", 0)),
            "starting_objective": float(result.get("starting_objective", np.nan)),
            "final_objective": float(result.get("final_objective", np.nan)),
            "objective_reduction_from_original": float(
                (original_objective - result["final_objective"])
                / max(abs(original_objective), 1.0e-15)
            ),
            "projected_gradient_inf_norm": metrics.get("projected_gradient_inf_norm"),
            "projected_gradient_l2_norm": metrics.get("projected_gradient_l2_norm"),
            "gradient_tolerance": metrics.get("gradient_tolerance"),
            "gradient_tolerance_met": metrics.get("gradient_tolerance_met"),
            "termination_by_function_reduction": metrics.get("termination_by_function_reduction"),
            "termination_by_projected_gradient": metrics.get("termination_by_projected_gradient"),
            "candidate_is_valid": metrics.get("candidate_is_valid"),
            "candidate_min_jacobian": metrics.get("candidate_min_jacobian"),
            "min_jacobian_after_safety": metrics.get("min_jacobian"),
            "safety_alpha": metrics.get("safety_alpha_applied"),
            "safety_filter_active": metrics.get("safety_filter_active"),
            "constraint_penalty": metrics.get("constraint_penalty"),
            "interpolation_error": metrics.get("interpolation_error"),
            "objective_reduction_from_pass_start": float(
                (result.get("starting_objective", np.nan) - result.get("final_objective", np.nan))
                / max(abs(result.get("starting_objective", np.nan)), 1.0e-15)
            ),
        })
        if result["converged"]:
            break
        guess = np.ascontiguousarray(result["candidate_nodes"], dtype=np.float64)

    report = {
        "sample_path": str(args.sample.resolve()),
        "sample_id": metadata.get("sample_id", args.sample.stem),
        "original_converged": bool(metadata.get("converged", False)),
        "parameters": params,
        "mesh_nodes": int(initial_nodes.shape[0]),
        "mesh_cells": int(cells.shape[0]),
        "configured_max_iter_per_pass": int(args.max_iter),
        "passes_requested": int(args.passes),
        "passes_run": len(passes),
        "converged_on_probe": bool(passes[-1]["converged"]) if passes else False,
        "wall_seconds": float(time.perf_counter() - started),
        "passes": passes,
        "warning": "This probe does not modify the original pickle or manifest. It repeats the configured objective and quadrature; it does not establish quadrature accuracy.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{report['sample_id']}_optimization_probe.json"
    output_path.write_text(json.dumps(_to_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Sample: {report['sample_id']}")
    print(f"Parameters: {params}")
    for row in passes:
        print(
            f"Pass {row['pass']}: status={row['status']}, success={row['success']}, "
            f"converged={row['converged']}, iterations={row['iterations']}, "
            f"projected_gradient_inf={row['projected_gradient_inf_norm']}, "
            f"message={row['message']}"
        )
    print(f"Report: {output_path.resolve()}")


if __name__ == "__main__":
    main()
