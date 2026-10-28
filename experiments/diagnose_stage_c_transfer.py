"""Diagnose transfer error with the native C++/deal.II remappers.

This script consumes the reproducible target archive created by
``stage_c_campaign.py`` and reports one-step and repeated-transfer errors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.utils import ensure_dir, load_config
from gdm import NativeMeshAdapter, create_pde_model, compute_native_continuous_errors, remap_p1_native, remap_l2_native, compute_native_mesh_metrics


def run_diagnostic(config_path: str = "config/config.yaml", targets_path: str = "results/stage_c/raw_metrics/stage_c_replay_targets.npz", output_dir: str = "results/stage_c/transfer_diagnostic") -> dict:
    target_path = Path(targets_path).resolve()
    if not target_path.is_file():
        raise FileNotFoundError(f"Target archive not found: {target_path}; run experiments/stage_c_campaign.py first.")
    cfg = load_config(str(Path(config_path).resolve()))
    with np.load(target_path, allow_pickle=False) as z:
        targets = np.asarray(z["target_nodes"], dtype=np.float64)
        cells = np.asarray(z["cells"], dtype=np.int64)
        init_nodes = np.asarray(z["initial_nodes"], dtype=np.float64)
        times = np.asarray(z["adaptation_times"], dtype=np.float64)
        T = float(z["T"])
        nx = int(z["nx"]); ny = int(z["ny"])
    base = NativeMeshAdapter.from_arrays(init_nodes, cells)
    pde = create_pde_model(bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=T)
    rows = []
    p1_state = None
    l2_state = None
    for k, nodes in enumerate(targets):
        t = float(times[k])
        target = base.update_coordinates(nodes, enforce_fixed_boundary=True)
        geom = compute_native_mesh_metrics(target)
        source_u = np.asarray(pde.evaluate_solution(base.nodes, t), dtype=np.float64)
        p1 = remap_p1_native(base, source_u, target)
        l2 = remap_l2_native(base, source_u, target)
        e0, _ = compute_native_continuous_errors(base, source_u, pde, t)
        ep1, hp1 = compute_native_continuous_errors(target, p1["remapped_field"], pde, t)
        el2, hl2 = compute_native_continuous_errors(target, l2["remapped_field"], pde, t)
        if p1_state is None:
            p1_state = source_u.copy(); l2_state = source_u.copy()
        p1_state = remap_p1_native(base, p1_state, target)["remapped_field"]
        p1_state = remap_p1_native(target, p1_state, base)["remapped_field"]
        l2_state = remap_l2_native(base, l2_state, target)["remapped_field"]
        l2_state = remap_l2_native(target, l2_state, base)["remapped_field"]
        rows.append({
            "index": k, "time": t, "min_jacobian": float(geom["min_jacobian"]),
            "initial_l2": float(e0), "p1_l2": float(ep1), "p1_h1": float(hp1),
            "l2_projection_l2": float(el2), "l2_projection_h1": float(hl2),
            "p1_mass_drift": float(p1["relative_mass_drift"]),
            "l2_mass_drift": float(l2["relative_mass_drift"]),
        })
    ep1_loop, _ = compute_native_continuous_errors(base, p1_state, pde, float(times[-1]))
    el2_loop, _ = compute_native_continuous_errors(base, l2_state, pde, float(times[-1]))
    payload = {
        "experiment": "native_stage_c_transfer_diagnostic",
        "backend": "C++20/deal.II through pybind11",
        "target_archive": str(target_path), "nx": nx, "ny": ny, "rows": rows,
        "repeated_round_trip_final_l2": {"p1": float(ep1_loop), "l2": float(el2_loop)},
    }
    out = ensure_dir(output_dir)
    (out / "native_transfer_diagnostic.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["repeated_round_trip_final_l2"], indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--targets", default="results/stage_c/raw_metrics/stage_c_replay_targets.npz")
    parser.add_argument("--output-dir", default="results/stage_c/transfer_diagnostic")
    args = parser.parse_args()
    run_diagnostic(args.config, args.targets, args.output_dir)


if __name__ == "__main__":
    main()
