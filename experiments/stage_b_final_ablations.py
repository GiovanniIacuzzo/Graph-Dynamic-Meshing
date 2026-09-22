"""
experiments/stage_b_final_ablations.py
Esperimenti conclusivi per la chiusura dello Stage B:
  1. Gradient Source Ablation: GDM + L2 con gradiente Esatto vs gradiente FEM
  2. Adaptation Frequency Sweep: GDM + L2 con adapt_interval in {5, 10, 20, 50}
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import load_config, ensure_dir
from generate_dataset import create_initial_mesh_2d
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.dynamic_solver import DynamicPDESimulator
from inference import MeshMoverInferenceEngine


def run_final_ablations():
    cfg = load_config("config/config.yaml")
    out_dir = ensure_dir("results/stage_b/ablations")
    checkpoint = Path("results/stage_a/checkpoints/best_model.pt")
    
    pde = ManufacturedConvectionDiffusion2D(
        bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=0.5
    )
    domain = cfg["geometry"]["domain"]
    engine = MeshMoverInferenceEngine(checkpoint, cfg, device="cpu")
    eps = float(cfg["geometry"]["safety"]["epsilon"])
    
    T = 0.5
    dt = 0.005

    print("\n" + "=" * 80)
    print("STAGE B — ESPERIMENTO 1: GRADIENT SOURCE ABLATION (GDM + L2-PROJECTION)")
    print("=" * 80)

    grad_experiments = [
        ("GDM + L2 (FEM Grad)", "gnn_fem"),
        ("GDM + L2 (Exact Grad)", "gnn_exact")
    ]
    grad_results = {}

    for name, mode in grad_experiments:
        mesh = create_initial_mesh_2d(domain, nx=32, ny=32)
        sim = DynamicPDESimulator(mesh, pde, cfg, engine, eps)
        
        t0 = time.perf_counter()
        res = sim.run_simulation(mode=mode, remap_method="l2", T=T, dt=dt, adapt_interval=10)
        tot_time = time.perf_counter() - t0

        t_span = res["time"][-1] - res["time"][0]
        time_int_l2 = float(np.trapz(res["rel_l2_error"], res["time"]) / t_span)
        final_l2 = float(res["rel_l2_error"][-1])
        min_j = float(min(res["min_jacobian"]))
        tot_remap_s = sum(res["timing_remap_ms"]) / 1000.0

        grad_results[name] = {
            "final_l2": final_l2,
            "time_int_l2": time_int_l2,
            "min_jacobian": min_j,
            "total_time_s": tot_time,
            "remap_time_s": tot_remap_s
        }

        print(f"\n---> {name}")
        print(f"     L2 Finale       : {final_l2 * 100:.3f}%")
        print(f"     Time-Int L2     : {time_int_l2 * 100:.3f}%")
        print(f"     Min Jacobiano   : {min_j:.4e}")
        print(f"     Tempo Remap Tot : {tot_remap_s:.2f} s")
        print(f"     Tempo Totale    : {tot_time:.2f} s")

    print("\n" + "=" * 80)
    print("STAGE B — ESPERIMENTO 2: ADAPTATION FREQUENCY SWEEP (GDM + L2-PROJECTION)")
    print("=" * 80)

    intervals = [5, 10, 20, 50]
    freq_results = {}

    print(f"{'Interval':<10} {'N Adapt':<10} {'Final L2 [%]':<14} {'Time-Int L2 [%]':<16} {'Min J':<12} {'Remap [s]':<12} {'Total [s]':<10}")
    print("-" * 84)

    for interval in intervals:
        mesh = create_initial_mesh_2d(domain, nx=32, ny=32)
        sim = DynamicPDESimulator(mesh, pde, cfg, engine, eps)
        
        t0 = time.perf_counter()
        res = sim.run_simulation(mode="gnn_fem", remap_method="l2", T=T, dt=dt, adapt_interval=interval)
        tot_time = time.perf_counter() - t0

        t_span = res["time"][-1] - res["time"][0]
        time_int_l2 = float(np.trapz(res["rel_l2_error"], res["time"]) / t_span)
        final_l2 = float(res["rel_l2_error"][-1])
        min_j = float(min(res["min_jacobian"]))
        tot_remap_s = sum(res["timing_remap_ms"]) / 1000.0
        n_adapt = 100 // interval

        freq_results[f"interval_{interval}"] = {
            "interval": interval,
            "n_adaptations": n_adapt,
            "final_l2": final_l2,
            "time_int_l2": time_int_l2,
            "min_jacobian": min_j,
            "total_time_s": tot_time,
            "remap_time_s": tot_remap_s,
            "safety_activations": res["safety_activations"]
        }

        print(f"{interval:<10d} {n_adapt:<10d} {final_l2*100:<14.3f} {time_int_l2*100:<16.3f} {min_j:<12.3e} {tot_remap_s:<12.2f} {tot_time:<10.2f}")

    # Esportazione JSON
    summary_data = {
        "gradient_ablation": grad_results,
        "frequency_sweep": freq_results
    }
    with open(out_dir / "stage_b_final_ablations.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[INFO] Dati esportati con successo in {out_dir / 'stage_b_final_ablations.json'}")


if __name__ == "__main__":
    run_final_ablations()