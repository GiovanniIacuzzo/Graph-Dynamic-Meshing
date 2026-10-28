"""Evaluate a Stage-D checkpoint on held-out trajectory-aware MMS samples.

Reports oracle-target displacement accuracy, trajectory-objective reduction,
raw/safe mesh validity, safety-filter activation, and online prediction/filter
costs separately for interpolation and extrapolation splits. This is a label-
and objective-level diagnostic, not a replacement for the full Stage-C PDE run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "python"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gdm import (
    NativeMeshAdapter,
    create_pde_model,
    compute_native_mesh_metrics,
    compute_native_mesh_metrics_arrays,
    apply_native_safety_filter,
)
from inference import MeshMoverInferenceEngine
from oracle.trajectory_objective import TrajectoryOracleObjective


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid sample file: {path}")
    return obj


def _stats(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_samples": len(rows)}
    for key in keys:
        vals = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
        if not vals:
            out[key] = {"n": 0, "mean": None, "median": None, "q25": None, "q75": None}
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[key] = {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
        }
    out["safety_intervention_fraction"] = (
        float(np.mean([bool(r["safety_intervened"]) for r in rows])) if rows else None
    )
    out["oracle_optimizer_success_fraction"] = (
        float(np.mean([bool(r["target_optimizer_success"]) for r in rows])) if rows else None
    )
    return out


def evaluate_split(
    split: str,
    sample_ids: List[str],
    raw_dir: Path,
    engine: MeshMoverInferenceEngine,
    cfg: Dict[str, Any],
    max_samples: int | None = None,
) -> Dict[str, Any]:
    if max_samples is not None:
        sample_ids = sample_ids[:max(0, int(max_samples))]
    if not sample_ids:
        raise ValueError(f"Split '{split}' has no samples in {raw_dir}")

    geom_cfg = cfg["geometry"]["safety"]
    obj_cfg = cfg["oracle"]["objective"]
    epsilon = float(geom_cfg["epsilon"])
    stage_cfg = cfg["stage_d"]
    rows: List[Dict[str, Any]] = []

    for sample_idx, sample_id in enumerate(sample_ids, start=1):
        sample_path = raw_dir / f"{sample_id}.pkl"
        sample = _load_pickle(sample_path)
        params = sample["metadata"]["parameters"]
        nodes = np.ascontiguousarray(sample["mesh"]["nodes"], dtype=np.float64)
        cells = np.ascontiguousarray(sample["mesh"]["cells"], dtype=np.int64)
        u = np.ascontiguousarray(sample["state"]["u"], dtype=np.float64)
        grad_u = np.ascontiguousarray(sample["state"]["grad_u"], dtype=np.float64)
        target_disp = np.ascontiguousarray(sample["target"]["displacement"], dtype=np.float64)
        oracle_nodes = np.ascontiguousarray(sample["target"]["optimal_nodes"], dtype=np.float64)

        # Numerical state and geometry are handled by the native backend. The
        # inference engine performs only the neural forward pass; filtering is
        # explicitly executed by the C++ safety filter below.
        mesh = NativeMeshAdapter.from_arrays(nodes, cells)
        prediction = engine.predict_raw(mesh, u, grad_u)
        raw_disp = np.asarray(prediction["raw_displacement"], dtype=np.float64)
        raw_nodes = nodes + raw_disp
        raw_geom = compute_native_mesh_metrics_arrays(raw_nodes, cells)

        filter_t0 = time.perf_counter()
        filtered = apply_native_safety_filter(
            mesh,
            raw_disp,
            epsilon=epsilon,
            safety_margin=float(geom_cfg.get("safety_margin", 1e-6)),
            tol_deg=float(geom_cfg.get("tol_deg", 1e-12)),
            lock_boundary_nodes=True,
        )
        filter_seconds = time.perf_counter() - filter_t0
        safe_nodes = np.asarray(filtered["safe_nodes"], dtype=np.float64)
        safe_disp = safe_nodes - nodes
        safe_mesh = NativeMeshAdapter.from_arrays(safe_nodes, cells, mesh.boundary_nodes)
        safe_geom = compute_native_mesh_metrics(safe_mesh)

        target_norm = max(float(np.linalg.norm(target_disp)), 1e-12)
        raw_rel_disp = float(np.linalg.norm(raw_disp - target_disp) / target_norm)
        safe_rel_disp = float(np.linalg.norm(safe_disp - target_disp) / target_norm)

        pde = create_pde_model(
            bx=float(params["bx"]),
            by=float(params["by"]),
            nu=float(params["nu"]),
            sigma=float(params["sigma"]),
            beta=float(params["beta"]),
            T=float(params["T"]),
        )
        trajectory_obj = TrajectoryOracleObjective(
            pde=pde,
            initial_nodes=nodes,
            cells=cells,
            boundary_nodes=mesh.boundary_nodes,
            t_start=float(params["t_start"]),
            horizon=float(params["adapt_horizon"]),
            lambda_D=float(obj_cfg["lambda_D"]),
            lambda_Q=float(obj_cfg["lambda_Q"]),
            epsilon=epsilon,
            time_quadrature_order=int(params.get("time_quadrature_order", stage_cfg["time_quadrature_order"])),
        )
        fixed_obj = float(trajectory_obj.evaluate(nodes)["total_objective"])
        safe_obj = float(trajectory_obj.evaluate(safe_nodes)["total_objective"])
        oracle_obj = float(trajectory_obj.evaluate(oracle_nodes)["total_objective"])
        denom = max(abs(fixed_obj), 1e-12)
        fixed_to_safe_reduction = (fixed_obj - safe_obj) / denom
        fixed_to_oracle_reduction = (fixed_obj - oracle_obj) / denom
        alpha = float(filtered.get("alpha_global", 1.0))
        gnn_seconds = float(prediction.get("inference_time", 0.0))
        row = {
            "sample_id": str(sample_id),
            "split": split,
            "sigma": float(params["sigma"]),
            "beta": float(params["beta"]),
            "t_start": float(params["t_start"]),
            "horizon": float(params["adapt_horizon"]),
            "target_optimizer_success": bool(sample["metadata"].get("optimizer_success", False)),
            "raw_displacement_relative_error": raw_rel_disp,
            "safe_displacement_relative_error": safe_rel_disp,
            "target_displacement_norm": target_norm,
            "fixed_trajectory_objective": fixed_obj,
            "safe_gnn_trajectory_objective": safe_obj,
            "target_trajectory_objective": oracle_obj,
            "fixed_to_safe_objective_reduction_fraction": float(fixed_to_safe_reduction),
            "fixed_to_target_objective_reduction_fraction": float(fixed_to_oracle_reduction),
            "safe_objective_gap_to_target": float(safe_obj - oracle_obj),
            "safety_alpha": alpha,
            "safety_intervened": bool(alpha < 1.0 - 1e-12),
            "raw_min_jacobian": float(np.min(raw_geom["jacobians"])),
            "raw_inverted_cells": int(np.sum(raw_geom["inverted"])),
            "safe_min_jacobian": float(np.min(safe_geom["jacobians"])),
            "safe_min_normalized_jacobian": float(np.min(safe_geom["normalized_jacobians"])),
            "safe_inverted_cells": int(np.sum(safe_geom["inverted"])),
            "safe_degenerate_cells": int(np.sum(safe_geom["degenerate"])),
            "gnn_inference_seconds": gnn_seconds,
            "safety_filter_seconds": float(filter_seconds),
            "online_total_seconds": float(gnn_seconds + filter_seconds),
        }
        if row["safe_min_jacobian"] < epsilon - 1e-10 or row["safe_inverted_cells"]:
            raise RuntimeError(
                f"Safety invariant failed for {sample_id}: minJ={row['safe_min_jacobian']:.6e}, "
                f"inverted={row['safe_inverted_cells']}"
            )
        if not np.all(np.isfinite([row["safe_gnn_trajectory_objective"], row["safe_displacement_relative_error"]])):
            raise RuntimeError(f"Non-finite evaluation metrics for sample {sample_id}")
        rows.append(row)
        print(
            f"[{split:12s}] {sample_idx:03d}/{len(sample_ids):03d} {sample_id}: "
            f"disp_rel={safe_rel_disp:.3f}, obj_red={fixed_to_safe_reduction:+.2%}, "
            f"alpha={alpha:.5f}, minJ={row['safe_min_jacobian']:.3e}"
        )

    keys = [
        "raw_displacement_relative_error", "safe_displacement_relative_error",
        "fixed_to_safe_objective_reduction_fraction", "fixed_to_target_objective_reduction_fraction",
        "safe_objective_gap_to_target", "safety_alpha", "raw_min_jacobian", "raw_inverted_cells",
        "safe_min_jacobian", "safe_min_normalized_jacobian", "gnn_inference_seconds",
        "safety_filter_seconds", "online_total_seconds",
    ]
    return {"summary": _stats(rows, keys), "samples": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage-D GNN on held-out trajectory-aware MMS samples")
    parser.add_argument("--config", default="config/config_stage_d.yaml")
    parser.add_argument("--checkpoint", default="results/stage_d_trajectory_aware/checkpoints/best_model.pt")
    parser.add_argument("--output", default="results/stage_d_trajectory_aware/evaluation/stage_d_holdout_metrics.json")
    parser.add_argument("--splits", nargs="+", default=["test_interp", "test_extrap"])
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="Optional smoke-test cap")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(cfg["dataset"]["raw_dir"]).resolve()
    manifest_path = Path(cfg["dataset"]["manifest_path"]).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    engine = MeshMoverInferenceEngine(checkpoint_path=checkpoint, config=cfg, device=args.device)

    results: Dict[str, Any] = {
        "experiment": "stage_d_trajectory_aware_holdout_evaluation",
        "scope": "Per-sample oracle-label and trajectory-objective evaluation; no transient PDE solve. Use Stage-C campaign for PDE time-integration metrics.",
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "manifest_path": str(manifest_path),
        "raw_dir": str(raw_dir),
        "device": args.device,
        "safety_epsilon": float(cfg["geometry"]["safety"]["epsilon"]),
        "splits": {},
    }
    for split in args.splits:
        if split not in ("train", "validation", "test_interp", "test_extrap"):
            raise ValueError(f"Unsupported split: {split}")
        ids = list(manifest.get(split, []))
        results["splits"][split] = evaluate_split(
            split=split,
            sample_ids=ids,
            raw_dir=raw_dir,
            engine=engine,
            cfg=cfg,
            max_samples=args.max_samples_per_split,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(f"Risultati salvati in: {output}")


if __name__ == "__main__":
    main()
