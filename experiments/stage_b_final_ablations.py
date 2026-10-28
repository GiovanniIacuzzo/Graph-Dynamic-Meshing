"""Native Stage-B ablations: gradient source and adaptation interval.

The C++/deal.II backend computes the PDE field, gradients, safety diagnostics,
FEM steps and remapping. Python is used only for GNN inference and reporting.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.utils import ensure_dir, load_config
from gdm import NativeDynamicRuntime
from inference import MeshMoverInferenceEngine


def _metrics(result: dict, elapsed: float) -> dict:
    return {
        "final_l2": float(result["final_l2"]),
        "time_int_l2": float(result["time_int_l2"]),
        "final_h1": float(result["final_h1"]),
        "min_jacobian": float(result["min_jacobian"]),
        "fem_time_s": float(result["fem_time_s"]),
        "remap_time_s": float(result["remap_time_s"]),
        "adapt_time_s": float(result["adapt_time_s"]),
        "safety_time_s": float(result["safety_time_s"]),
        "total_wall_time_s": float(elapsed),
    }


def run_final_ablations() -> dict:
    cfg = load_config("config/config.yaml")
    out_dir = ensure_dir("results/stage_b/ablations")
    checkpoint = Path("results/stage_a/checkpoints/best_model.pt").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint GNN non trovato: {checkpoint}")
    engine = MeshMoverInferenceEngine(checkpoint, cfg, device="cpu")
    domain = cfg["geometry"]["domain"]
    base = dict(domain_cfg=domain, nx=32, ny=32, bx=0.4, by=0.2, nu=1e-3,
                sigma=0.06, beta=0.5, T=0.5, dt=0.005,
                inference_engine=engine)
    gradient_results = {}
    for exact, label in ((False, "fem_gradient"), (True, "exact_gradient")):
        runtime = NativeDynamicRuntime(**base, adapt_interval=10, exact_gradient=exact)
        t0 = time.perf_counter()
        result = runtime.run_episode("gdm_l2", compute_diagnostics=True)
        gradient_results[label] = _metrics(result, time.perf_counter() - t0)
        print(f"{label}: final L2={100*result['final_l2']:.3f}%")

    frequency_results = {}
    for interval in (5, 10, 20, 50):
        runtime = NativeDynamicRuntime(**base, adapt_interval=interval)
        t0 = time.perf_counter()
        result = runtime.run_episode("gdm_l2", compute_diagnostics=True)
        frequency_results[f"interval_{interval}"] = {
            "interval": interval,
            "adaptation_windows": len(result["adaptation_steps"]),
            **_metrics(result, time.perf_counter() - t0),
        }
        print(f"interval={interval}: final L2={100*result['final_l2']:.3f}%")

    payload = {
        "backend": "C++20/deal.II for all mesh/PDE/FEM/ALE/remapping/metrics kernels; Python for GNN inference",
        "gradient_ablation": gradient_results,
        "adaptation_frequency_sweep": frequency_results,
    }
    (out_dir / "stage_b_final_ablations_native.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    run_final_ablations()
