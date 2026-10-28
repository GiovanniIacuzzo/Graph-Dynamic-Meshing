"""Generate Stage-D self-supervised MMS samples without oracle optimization."""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "python"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from gdm import (
    NativeMeshAdapter, create_pde_model, compute_native_mesh_metrics_arrays,
    compute_native_field_gradients,
)


def _sample_mesh(base_mesh, rng, perturb_fraction: float, epsilon: float):
    nodes0 = np.asarray(base_mesh.nodes, dtype=np.float64)
    cells = np.asarray(base_mesh.cells, dtype=np.int64)
    bnd = np.asarray(base_mesh.boundary_nodes, dtype=np.int64)
    mask = np.ones(nodes0.shape[0], dtype=bool)
    mask[bnd] = False
    h = min(
        (np.max(nodes0[:, 0]) - np.min(nodes0[:, 0])) / 32.0,
        (np.max(nodes0[:, 1]) - np.min(nodes0[:, 1])) / 32.0,
    )
    amplitude = float(perturb_fraction) * h
    for attempt in range(12):
        nodes = nodes0.copy()
        if amplitude > 0.0:
            nodes[mask] += rng.uniform(-amplitude, amplitude, size=(int(mask.sum()), 2))
        nodes[bnd] = nodes0[bnd]
        metrics = compute_native_mesh_metrics_arrays(nodes, cells)
        if np.all(metrics["jacobians"] >= epsilon):
            return nodes, cells, bnd, float(amplitude)
        amplitude *= 0.5
    # Deterministic conservative fallback: the unperturbed mesh is known valid.
    return nodes0.copy(), cells.copy(), bnd.copy(), 0.0


def _draw_parameters(split: str, idx: int, rng: np.random.Generator, cfg: Dict[str, Any]):
    ranges = cfg["field"]["sampling_ranges"]
    if split == "test_extrap":
        sigma_rng = ranges["sigma_extrap_low"] if idx % 2 == 0 else ranges["sigma_extrap_high"]
        sigma = float(rng.uniform(*sigma_rng))
        beta = float(rng.uniform(*ranges["beta_extrap"]))
    elif split == "test_interp":
        sigma = float(rng.uniform(0.04, 0.07))
        beta = float(rng.uniform(0.35, 0.65))
    else:
        sigma = float(rng.uniform(ranges["sigma_min"], ranges["sigma_max"]))
        beta = float(rng.uniform(ranges["beta_min"], ranges["beta_max"]))
    return sigma, beta


def generate(config: Dict[str, Any], output_dir: Path, seed: int, overwrite: bool = False):
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("stage_ds_*.pkl"))
    manifest_path = output_dir / "manifest.json"
    if (existing or manifest_path.exists()) and not overwrite:
        raise FileExistsError(f"{output_dir} contiene già campioni Stage-D self-supervised; usa un output-dir nuovo o --overwrite.")
    if overwrite:
        for path in existing:
            path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    geom_cfg = config["geometry"]
    grid = geom_cfg["initial_grid"]
    base_mesh = NativeMeshAdapter.from_domain(geom_cfg["domain"], nx=int(grid["nx"]), ny=int(grid["ny"]))
    epsilon = float(geom_cfg["safety"]["epsilon"])
    dt = float(config["stage_d"]["dt"])
    horizon = dt * int(config["stage_d"]["adapt_interval"])
    pde_cfg = config["stage_d"]["pde"]
    T = float(pde_cfg["T"])
    start_grid = np.arange(0.0, T - horizon + 0.5 * horizon, horizon, dtype=np.float64)
    start_grid = start_grid[start_grid + horizon <= T + 1e-12]
    if len(start_grid) == 0:
        raise ValueError("Nessun tempo iniziale valido per la finestra di adattamento.")

    rng = np.random.default_rng(int(seed))
    manifest: Dict[str, list[str]] = {s: [] for s in ("train", "validation", "test_interp", "test_extrap")}
    split_counts = config["dataset"]["splits"]
    sample_index = 0
    summaries = {}
    for split, count_value in split_counts.items():
        count = int(count_value)
        split_rng = np.random.default_rng(int(seed) + 1009 * (list(split_counts).index(split) + 1))
        split_rows = []
        for idx in range(count):
            sigma, beta = _draw_parameters(split, idx, split_rng, config)
            t_start = float(split_rng.choice(start_grid))
            nodes, cells, bnd_nodes, used_perturb = _sample_mesh(
                base_mesh, split_rng, float(config["dataset"].get("mesh_perturb_fraction", 0.0)), epsilon
            )
            pde = create_pde_model(
                bx=float(pde_cfg["bx"]), by=float(pde_cfg["by"]), nu=float(pde_cfg["nu"]),
                sigma=sigma, beta=beta, T=T,
            )
            native_mesh = NativeMeshAdapter.from_arrays(nodes, cells, bnd_nodes)
            u = np.asarray(pde.evaluate_solution(native_mesh.nodes, t_start), dtype=np.float64)
            # Recovered nodal gradients are assembled by the native FEM core.
            grad_u = compute_native_field_gradients(native_mesh, u, pde, t_start, exact_gradient=False)
            sample_id = f"stage_ds_{sample_index:05d}_{split}"
            sample_index += 1
            params = {
                "bx": float(pde.bx), "by": float(pde.by), "nu": float(pde.nu),
                "sigma": sigma, "beta": beta, "T": T, "t_start": t_start,
                "adapt_horizon": horizon, "dt": dt,
                "adapt_interval": int(config["stage_d"]["adapt_interval"]),
                "time_quadrature_order": int(config["training"]["objective"]["time_quadrature_order"]),
            }
            # The target is intentionally a placeholder: Stage-D.1 training minimizes
            # the differentiable space-time objective and never treats this as a label.
            sample = {
                "mesh": {"nodes": nodes.copy(), "cells": cells.copy()},
                "state": {"u": np.asarray(u, dtype=np.float64), "grad_u": np.asarray(grad_u, dtype=np.float64)},
                "target": {"optimal_nodes": nodes.copy(), "displacement": np.zeros_like(nodes)},
                "metadata": {
                    "sample_id": sample_id, "split": split,
                    "method": "stage_d_selfsupervised_mms",
                    "target_type": "not_used_self_supervised",
                    "feature_normalization": "max_abs_u",
                    "parameters": params,
                    "mesh_perturb_amplitude": float(used_perturb),
                },
            }
            with (output_dir / f"{sample_id}.pkl").open("wb") as f:
                pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
            manifest[split].append(sample_id)
            split_rows.append({"sigma": sigma, "beta": beta, "t_start": t_start, "mesh_perturb_amplitude": used_perturb})
            print(f"[{split:12s}] {idx+1:03d}/{count:03d} sigma={sigma:.4f} beta={beta:.3f} t0={t_start:.3f}")
        summaries[split] = {"n": len(split_rows), "sigma_min": min(r["sigma"] for r in split_rows), "sigma_max": max(r["sigma"] for r in split_rows), "beta_min": min(r["beta"] for r in split_rows), "beta_max": max(r["beta"] for r in split_rows)}

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "experiment": "stage_d_selfsupervised_mms_dataset",
        "seed": int(seed), "output_dir": str(output_dir),
        "sample_counts": {k: len(v) for k, v in manifest.items()},
        "horizon": horizon, "mesh_nodes": int(base_mesh.num_nodes), "mesh_cells": int(base_mesh.num_cells),
        "target_type": "none; training objective is differentiable space-time MMS interpolation error",
        "feature_normalization": "divide u and grad_u by max(abs(u))",
        "split_summary": summaries,
        "manifest": str(manifest_path),
    }
    (output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config_stage_d_selfsupervised.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit-per-split", type=int, default=None, help="Cap counts for a format/gradients smoke test")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.limit_per_split is not None:
        for split in cfg["dataset"]["splits"]:
            cfg["dataset"]["splits"][split] = min(int(cfg["dataset"]["splits"][split]), max(0, int(args.limit_per_split)))
    output = Path(args.output_dir or cfg["dataset"]["raw_dir"])
    generate(cfg, output, int(args.seed if args.seed is not None else cfg["experiment"]["seed"]), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
