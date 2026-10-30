"""Generate Stage-D supervision labels for trajectory-aware GDM-ALE.

Each sample contains the current exact-MMS nodal state (u and an FEM-recovered
nodal gradient) and a displacement target optimized over the next adaptation
window, rather than a static Gaussian-bump interpolation objective.
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "python"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from gdm import NativeMeshAdapter, create_pde_model, compute_native_field_gradients
from oracle.trajectory_optimizer import TrajectoryMeshOptimizer


def _sample_range(rng: np.random.Generator, bounds: List[float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configurazione non valida: {path}")
    return cfg


def generate_dataset(
    config: Dict[str, Any],
    output_dir: Path,
    seed: int,
    limit_per_split: int | None = None,
    max_iter_override: int | None = None,
    overwrite: bool = False,
) -> Dict[str, List[str]]:
    stage_cfg = config["stage_d"]
    data_dir = output_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = list(data_dir.glob("stage_d_*.pkl"))
    manifest_path = data_dir / "manifest.json"
    if (existing or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            f"La directory {data_dir} contiene già campioni Stage D. "
            "Scegli un --output-dir nuovo oppure passa --overwrite esplicitamente."
        )
    if overwrite:
        for p in existing:
            p.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    dom = config["geometry"]["domain"]
    grid = config["geometry"]["initial_grid"]
    mesh = NativeMeshAdapter.from_domain(dom, nx=int(grid["nx"]), ny=int(grid["ny"]))
    geom_cfg = config["geometry"]["safety"]
    obj_cfg = config["oracle"]["objective"]
    opt_cfg = config["oracle"]["optimizer"]
    pde_cfg = config["stage_d"]["pde"]
    dt = float(stage_cfg["dt"])
    adapt_interval = int(stage_cfg["adapt_interval"])
    horizon = dt * adapt_interval
    T = float(pde_cfg["T"])
    if horizon <= 0.0 or horizon > T:
        raise ValueError("L'orizzonte di adattamento deve essere in (0,T].")
    valid_start_times = np.arange(0.0, T - horizon + 0.5 * dt, horizon, dtype=np.float64)
    valid_start_times = valid_start_times[valid_start_times + horizon <= T + 1e-12]
    if valid_start_times.size == 0:
        valid_start_times = np.array([0.0], dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    manifest: Dict[str, List[str]] = {k: [] for k in ("train", "validation", "test_interp", "test_extrap")}
    quality_by_split: Dict[str, List[Dict[str, Any]]] = {k: [] for k in manifest}
    sample_index = 0

    for split in manifest:
        spec = stage_cfg["splits"][split]
        count = int(spec["count"])
        if limit_per_split is not None:
            count = min(count, max(0, int(limit_per_split)))
        for local_idx in range(count):
            if split == "test_extrap":
                # Alternate between narrow and broad fields to avoid reporting
                # only one direction of parameter extrapolation.
                sigma_key = "sigma_extrap_low" if local_idx % 2 == 0 else "sigma_extrap_high"
                sigma = _sample_range(rng, spec[sigma_key])
            else:
                sigma = _sample_range(rng, spec["sigma"])
            beta = _sample_range(rng, spec["beta"])
            t_start = float(rng.choice(valid_start_times))
            pde = create_pde_model(
                bx=float(pde_cfg["bx"]),
                by=float(pde_cfg["by"]),
                nu=float(pde_cfg["nu"]),
                sigma=sigma,
                beta=beta,
                T=T,
            )
            u0 = np.asarray(pde.evaluate_solution(mesh.nodes, t_start), dtype=np.float64)
            grad0 = compute_native_field_gradients(mesh, u0, pde, t_start, exact_gradient=False)
            optimizer = TrajectoryMeshOptimizer(
                pde=pde,
                initial_nodes=mesh.nodes,
                cells=mesh.cells,
                boundary_nodes=mesh.boundary_nodes,
                t_start=t_start,
                horizon=horizon,
                lambda_D=float(obj_cfg["lambda_D"]),
                lambda_Q=float(obj_cfg["lambda_Q"]),
                epsilon=float(geom_cfg["epsilon"]),
                max_iter=int(max_iter_override or opt_cfg["max_iter"]),
                ftol=float(opt_cfg["ftol"]),
                gtol=float(opt_cfg["gtol"]),
                time_quadrature_order=int(stage_cfg["time_quadrature_order"]),
            )
            result = optimizer.optimize()
            metrics = result["final_metrics"]
            quality_by_split[split].append({
                "success": bool(result["converged"]),
                "relative_objective_reduction_raw": float(metrics.get("relative_objective_reduction_raw", 0.0)),
                "safety_alpha": float(metrics.get("safety_alpha_applied", 1.0)),
                "min_jacobian": float(metrics["min_jacobian"]),
                "iterations": int(result["iterations"]),
            })
            if metrics["min_jacobian"] < float(geom_cfg["epsilon"]):
                raise RuntimeError(
                    f"Target non ammissibile in {split} (min_J={metrics['min_jacobian']:.4e})."
                )
            sample_id = f"stage_d_{sample_index:05d}_{split}"
            sample_index += 1
            metadata = {
                "sample_id": sample_id,
                "split": split,
                "method": "trajectory_aware_mms_oracle",
                "parameters": {
                    "bx": float(pde.bx), "by": float(pde.by), "nu": float(pde.nu),
                    "sigma": float(sigma), "beta": float(beta), "T": float(T),
                    "t_start": float(t_start), "adapt_horizon": float(horizon),
                    "dt": dt, "adapt_interval": adapt_interval,
                    "time_quadrature_order": int(stage_cfg["time_quadrature_order"]),
                },
                "oracle_metrics": metrics,
                "optimizer_success": bool(result["converged"]),
                "optimizer_iterations": int(result["iterations"]),
            }
            sample = {
                "mesh": {"nodes": mesh.nodes.copy(), "cells": mesh.cells.copy()},
                "state": {"u": np.asarray(u0, dtype=np.float64), "grad_u": np.asarray(grad0, dtype=np.float64)},
                "target": {
                    "optimal_nodes": np.asarray(result["optimal_nodes"], dtype=np.float64),
                    "displacement": np.asarray(result["target_displacement"], dtype=np.float64),
                },
                "metadata": metadata,
            }
            with (data_dir / f"{sample_id}.pkl").open("wb") as f:
                pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
            manifest[split].append(sample_id)
            print(
                f"[{split:12s}] {local_idx + 1:03d}/{count:03d} "
                f"sigma={sigma:.4f} beta={beta:.3f} t0={t_start:.3f} "
                f"Etraj={result['final_objective']:.4e} "
                f"minJ={metrics['min_jacobian']:.3e} "
                f"alpha={metrics['safety_alpha_applied']:.5f} "
                f"dJraw={metrics.get('relative_objective_reduction_raw', 0.0):+.2%} "
                f"nit={result['iterations']} success={result['converged']}"
            )

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    summary = {
        "dataset": "Stage D trajectory-aware MMS",
        "seed": int(seed),
        "output_dir": str(data_dir),
        "horizon": horizon,
        "sample_counts": {k: len(v) for k, v in manifest.items()},
        "optimizer_summary_by_split": {
            split: {
                "n": len(rows),
                "success_count": int(sum(row["success"] for row in rows)),
                "success_fraction": float(np.mean([row["success"] for row in rows])) if rows else None,
                "median_relative_objective_reduction_raw": float(np.median([row["relative_objective_reduction_raw"] for row in rows])) if rows else None,
                "median_safety_alpha": float(np.median([row["safety_alpha"] for row in rows])) if rows else None,
                "minimum_jacobian": float(min(row["min_jacobian"] for row in rows)) if rows else None,
                "median_iterations": float(np.median([row["iterations"] for row in rows])) if rows else None,
            } for split, rows in quality_by_split.items()
        },
        "manifest": str(manifest_path),
        "mesh_nodes": int(mesh.num_nodes),
        "mesh_cells": int(mesh.num_cells),
    }
    with (data_dir / "generation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate trajectory-aware Stage-D GDM labels")
    parser.add_argument("--config", default="config/stage_d.yaml")
    parser.add_argument("--output-dir", default=None, help="Override dataset raw directory")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit-per-split", type=int, default=None, help="Small pilot: cap each split")
    parser.add_argument("--max-iter", type=int, default=None, help="Override oracle max iterations")
    parser.add_argument("--overwrite", action="store_true", help="Remove only existing stage_d_*.pkl and manifest in output dir")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = _load_config(config_path)
    output_dir = Path(args.output_dir or config["dataset"]["raw_dir"])
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    generate_dataset(
        config=config,
        output_dir=output_dir,
        seed=seed,
        limit_per_split=args.limit_per_split,
        max_iter_override=args.max_iter,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
