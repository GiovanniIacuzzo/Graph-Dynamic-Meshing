"""Native operator benchmark for the C++/deal.II P1 and L2 remappers.

The source state is the manufactured exact solution sampled on the fixed source
mesh. Each saved target mesh is validated and both native remapping operators
are measured on precisely the same field and geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.utils import load_config, ensure_dir
from gdm import (
    NativeMeshAdapter, create_pde_model, compute_native_mesh_metrics,
    compute_native_continuous_errors, remap_p1_native, remap_l2_native,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median_s": float(np.median(arr)),
        "q25_s": float(np.quantile(arr, 0.25)),
        "q75_s": float(np.quantile(arr, 0.75)),
        "iqr_s": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
        "raw_s": arr.tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    target_path = Path(args.targets).expanduser().resolve()
    if not target_path.is_file():
        raise FileNotFoundError(f"Archivio target non trovato: {target_path}")
    if args.repeats < 1:
        raise ValueError("--repeats deve essere >= 1")
    cfg = load_config(str(Path(args.config).resolve()))
    with np.load(target_path, allow_pickle=False) as z:
        required = {"target_nodes", "adaptation_steps", "adaptation_times", "cells", "dt", "T", "adapt_interval"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"Campi mancanti nell'archivio target: {sorted(missing)}")
        targets = np.asarray(z["target_nodes"], dtype=np.float64)
        steps = np.asarray(z["adaptation_steps"], dtype=np.int64)
        times = np.asarray(z["adaptation_times"], dtype=np.float64)
        cells = np.asarray(z["cells"], dtype=np.int64)
        initial_nodes = np.asarray(z["initial_nodes"], dtype=np.float64) if "initial_nodes" in z.files else None
        dt = float(z["dt"])
        T = float(z["T"])
        adapt_interval = int(z["adapt_interval"])
        nx = int(z["nx"]) if "nx" in z.files else int(round(np.sqrt(targets.shape[1]) - 1))
        ny = int(z["ny"]) if "ny" in z.files else nx

    if targets.ndim != 3 or targets.shape[2] != 2:
        raise ValueError(f"target_nodes deve avere shape (K,N,2), ricevuta {targets.shape}")
    if steps.shape != (targets.shape[0],) or times.shape != (targets.shape[0],):
        raise ValueError("Lunghezze incompatibili fra target e metadati temporali.")
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(times)):
        raise ValueError("Target non finiti.")
    if np.any(cells < 0) or np.any(cells >= targets.shape[1]):
        raise ValueError("La connettività contiene indici fuori intervallo.")

    domain = cfg["geometry"]["domain"]
    mesh0 = NativeMeshAdapter.from_domain(domain, nx=nx, ny=ny)
    if initial_nodes is not None and initial_nodes.shape == mesh0.nodes.shape:
        if not np.allclose(initial_nodes, mesh0.nodes, rtol=0.0, atol=1e-12):
            mesh0 = NativeMeshAdapter.from_arrays(initial_nodes, cells, mesh0.boundary_nodes)
    if not np.array_equal(mesh0.cells, cells):
        raise ValueError("La connettività salvata non coincide con la mesh nativa di riferimento.")
    pde = create_pde_model(bx=0.4, by=0.2, nu=1.0e-3, sigma=0.06, beta=0.5, T=T)
    n_windows = targets.shape[0] if args.limit_windows <= 0 else min(targets.shape[0], args.limit_windows)
    if n_windows < 1:
        raise ValueError("Nessuna finestra selezionata.")

    rows: list[dict[str, Any]] = []
    for k in range(n_windows):
        t = float(times[k])
        source_u = np.asarray(pde.evaluate_solution(mesh0.nodes, t), dtype=np.float64)
        target_mesh = mesh0.update_coordinates(targets[k])
        geom = compute_native_mesh_metrics(target_mesh)
        if int(geom["num_inverted"]) or float(geom["min_jacobian"]) < float(cfg["geometry"]["safety"]["epsilon"]) - 1e-10:
            raise RuntimeError(f"Target mesh {k} non ammissibile: minJ={geom['min_jacobian']:.6e}.")

        for method, remapper in (("p1", remap_p1_native), ("l2", remap_l2_native)):
            # Warm-up is excluded from timing statistics.
            remapper(mesh0, source_u, target_mesh)
            elapsed_s: list[float] = []
            output = None
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                output = remapper(mesh0, source_u, target_mesh)
                elapsed_s.append(time.perf_counter() - t0)
            assert output is not None
            transferred = np.asarray(output["remapped_field"], dtype=np.float64)
            err_l2, err_h1 = compute_native_continuous_errors(target_mesh, transferred, pde, t)
            row = {
                "target_index": k,
                "adaptation_step": int(steps[k]),
                "time": t,
                "method": method,
                "num_nodes": target_mesh.num_nodes,
                "num_cells": target_mesh.num_cells,
                "min_jacobian": float(geom["min_jacobian"]),
                "num_inverted": int(geom["num_inverted"]),
                "relative_l2_error_after_transfer": float(err_l2),
                "relative_h1_error_after_transfer": float(err_h1),
                "mass_before": float(output["mass_before"]),
                "mass_after": float(output["mass_after"]),
                "relative_mass_drift": float(output["relative_mass_drift"]),
                "native_elapsed_ms": float(output["elapsed_ms"]),
                "wall_timing": _summary(elapsed_s),
            }
            rows.append(row)
        print(f"[{k + 1:03d}/{n_windows:03d}] t={t:.3f}: P1/L2 native remap completati")

    by_method: dict[str, Any] = {}
    for method in ("p1", "l2"):
        subset = [r for r in rows if r["method"] == method]
        by_method[method] = {
            "n_targets": len(subset),
            "median_wall_median_s": float(np.median([r["wall_timing"]["median_s"] for r in subset])),
            "median_native_elapsed_ms": float(np.median([r["native_elapsed_ms"] for r in subset])),
            "median_relative_l2_error": float(np.median([r["relative_l2_error_after_transfer"] for r in subset])),
            "median_relative_mass_drift": float(np.median([r["relative_mass_drift"] for r in subset])),
        }
    result = {
        "experiment": "stage_c_native_transfer_operator_benchmark",
        "backend": "C++20/deal.II via pybind11 for mesh, metrics, remapping and error norms",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "target_archive": str(target_path),
        "target_archive_sha256": _sha256(target_path),
        "parameters": {"T": T, "dt": dt, "adapt_interval": adapt_interval, "nx": nx, "ny": ny, "n_targets_used": n_windows, "repeats": args.repeats},
        "summary": by_method,
        "rows": rows,
    }
    out_dir = ensure_dir(args.output_dir)
    (out_dir / "stage_c_transfer_operator_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "stage_c_transfer_operator_summary.csv").write_text(
        "target_index,step,time,method,relative_l2_error,relative_h1_error,relative_mass_drift,native_elapsed_ms,wall_median_s,wall_iqr_s\n" +
        "".join(
            f"{r['target_index']},{r['adaptation_step']},{r['time']:.9g},{r['method']},{r['relative_l2_error_after_transfer']:.9g},{r['relative_h1_error_after_transfer']:.9g},{r['relative_mass_drift']:.9g},{r['native_elapsed_ms']:.9g},{r['wall_timing']['median_s']:.9g},{r['wall_timing']['iqr_s']:.9g}\n"
            for r in rows
        ), encoding="utf-8"
    )
    print(json.dumps(by_method, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--targets", default="results/stage_c/raw_metrics/stage_c_replay_targets.npz")
    parser.add_argument("--output-dir", default="results/stage_c/transfer_benchmark")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--limit-windows", type=int, default=0, help="0 uses all stored windows")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
