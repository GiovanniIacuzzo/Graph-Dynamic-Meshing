"""
tests/test_ale_replay_stage_b.py
Test C6: Replay della medesima traiettoria GNN dello Stage B:
Confronto diretto a parità di cinematica tra L2-Projection Remap e Formulazione ALE Continua.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import load_config, ensure_dir
from generate_dataset import create_initial_mesh_2d
from geometry.mesh import TriangularMesh2D
from geometry.safety import apply_safety_filter
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.fem import ConvectionDiffusionFEM2D, compute_field_gradients
from numerics.remap_l2 import remap_l2_projection
from numerics.dynamic_solver import compute_continuous_errors
from numerics.ale import ALESolver2D
from inference import MeshMoverInferenceEngine


def run_c6_replay_benchmark():
    cfg = load_config("config/config.yaml")
    out_dir = ensure_dir("results/stage_c/diagnostics")
    pde = ManufacturedConvectionDiffusion2D(bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=0.5)
    domain = cfg["geometry"]["domain"]
    checkpoint = Path("results/stage_a/checkpoints/best_model.pt")
    engine = MeshMoverInferenceEngine(checkpoint, cfg, device="cpu")
    eps = float(cfg["geometry"]["safety"]["epsilon"])

    T = 0.5
    dt = 0.005
    n_steps = int(round(T / dt))  # 100 step
    adapt_interval = 10           # 10 adattamenti

    print("\n" + "=" * 75)
    print("TEST C6 — REPLAY TRAIETTORIA STAGE B: L2-REMAP vs ALE FORMULATION")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # FASE 1: Registrazione della sequenza di mesh target generata dalla GNN
    # -------------------------------------------------------------------------
    print("---> Fase 1: Registrazione sequenza di geometrie GNN (N_a = 10)...")
    init_mesh = create_initial_mesh_2d(domain, nx=32, ny=32)
    m_curr = init_mesh
    u_curr = pde.evaluate_solution(m_curr.nodes[:, 0], m_curr.nodes[:, 1], 0.0)

    target_meshes = [init_mesh]
    current_time = 0.0

    for step in range(1, n_steps + 1):
        t_next = step * dt
        if step % adapt_interval == 0:
            grad_u = compute_field_gradients(m_curr, u_curr, pde, current_time, gradient_mode="fem")
            pred = engine.predict(m_curr, u_curr, grad_u)
            m_curr = pred["adapted_mesh"]
            target_meshes.append(m_curr)
            # Avanzamento nominale dello stato per generare le feature successive
            u_curr = pde.evaluate_solution(m_curr.nodes[:, 0], m_curr.nodes[:, 1], t_next)
        current_time = t_next

    # -------------------------------------------------------------------------
    # FASE 2: Esecuzione Pipeline Stage B (L2-Remap a gradino ogni N_a step)
    # -------------------------------------------------------------------------
    print("---> Fase 2: Esecuzione Stage B (Mesh motion discreto + L2-Projection Remap)...")
    m_b = init_mesh
    fem_b = ConvectionDiffusionFEM2D(m_b, pde)
    u_b = pde.evaluate_solution(m_b.nodes[:, 0], m_b.nodes[:, 1], 0.0)

    t0_b = time.perf_counter()
    remap_time_b = 0.0
    errs_b = [compute_continuous_errors(m_b, u_b, pde, 0.0)[0]]
    adapt_idx = 1

    for step in range(1, n_steps + 1):
        t_curr = (step - 1) * dt
        t_next = step * dt

        if step % adapt_interval == 0:
            m_target = target_meshes[adapt_idx]
            adapt_idx += 1

            # Costo e proiezione L2 esplicita
            t_r0 = time.perf_counter()
            u_b = remap_l2_projection(m_b.nodes, m_b.cells, u_b, m_target.nodes, m_target.cells)
            remap_time_b += (time.perf_counter() - t_r0)

            m_b = m_target
            fem_b = ConvectionDiffusionFEM2D(m_b, pde)

        u_b = fem_b.step_backward_euler(u_b, dt=dt, t_next=t_next)
        err_l2, _ = compute_continuous_errors(m_b, u_b, pde, t_next)
        errs_b.append(err_l2)

    tot_time_b = time.perf_counter() - t0_b

    # -------------------------------------------------------------------------
    # FASE 3: Esecuzione Pipeline Stage C (ALE continuo con velocità w)
    # -------------------------------------------------------------------------
    print("---> Fase 3: Esecuzione Stage C (Mesh velocity continua w + ALE D-GCL)...")
    m_c_curr = init_mesh
    u_c = pde.evaluate_solution(m_c_curr.nodes[:, 0], m_c_curr.nodes[:, 1], 0.0)
    ale_solver = ALESolver2D(pde, nu=1e-3, bx=0.4, by=0.2)

    t0_c = time.perf_counter()
    errs_c = [compute_continuous_errors(m_c_curr, u_c, pde, 0.0)[0]]

    # Nodi di partenza e arrivo per ciascuna finestra di adattamento
    current_base_nodes = init_mesh.nodes.copy()
    current_target_nodes = target_meshes[1].nodes.copy()
    adapt_idx = 1

    for step in range(1, n_steps + 1):
        t_next = step * dt

        # Identificazione della finestra temporale corrente
        k_in_window = (step - 1) % adapt_interval + 1  # da 1 a 10

        # Posizione lineare continua: X(t) = X_base + (k / N_a) * (X_target - X_base)
        nodes_next = current_base_nodes + (k_in_window / adapt_interval) * (current_target_nodes - current_base_nodes)
        m_c_next = init_mesh.update_coordinates(nodes_next)

        # Passo FEM ALE con convezione relativa (b - w)
        u_c, _ = ale_solver.step_ale(m_c_curr, m_c_next, u_c, dt=dt, t_next=t_next, pure_gcl=False)
        err_l2, _ = compute_continuous_errors(m_c_next, u_c, pde, t_next)
        errs_c.append(err_l2)

        m_c_curr = m_c_next

        # Al termine della finestra si aggiorna il blocco per la finestra successiva
        if step % adapt_interval == 0 and adapt_idx < len(target_meshes) - 1:
            adapt_idx += 1
            current_base_nodes = current_target_nodes.copy()
            current_target_nodes = target_meshes[adapt_idx].nodes.copy()

    tot_time_c = time.perf_counter() - t0_c

    # -------------------------------------------------------------------------
    # Metriche riassuntive del confronto
    # -------------------------------------------------------------------------
    time_arr = np.linspace(0.0, T, n_steps + 1)
    time_int_b = float(np.trapz(errs_b, time_arr) / T)
    time_int_c = float(np.trapz(errs_c, time_arr) / T)

    print("\n" + "=" * 75)
    print("TEST C6 — CONFRONTO A PARI TARGET GNN: L2-REMAP vs FORMULAZIONE ALE CONTINUA")
    print("=" * 75)
    print(f"{'Metrica':<30} {'Stage B (L2-Remap)':<22} {'Stage C (ALE)':<20}")
    print("-" * 75)
    print(f"{'Errore L2(T) Finale':<30} {errs_b[-1]*100:<21.3f}% {errs_c[-1]*100:<19.3f}%")
    print(f"{'Errore L2 Time-Integrated':<30} {time_int_b*100:<21.3f}% {time_int_c*100:<19.3f}%")
    print(f"{'Tempo Remapping Esplicito [s]':<30} {remap_time_b:<21.2f} {'0.00 (Assente)':<20}")
    print(f"{'Tempo Totale Simulazione [s]':<30} {tot_time_b:<21.2f} {tot_time_c:<19.2f}")
    print(f"{'Speedup Computazionale':<30} {'1.00x (Baseline)':<21} {tot_time_b / tot_time_c:<19.2f}x")
    print("=" * 75)

    results = {
        "final_l2_b": float(errs_b[-1]),
        "final_l2_c": float(errs_c[-1]),
        "time_int_l2_b": time_int_b,
        "time_int_l2_c": time_int_c,
        "remap_time_b_s": float(remap_time_b),
        "total_time_b_s": float(tot_time_b),
        "total_time_c_s": float(tot_time_c),
        "speedup_c_vs_b": float(tot_time_b / tot_time_c)
    }

    with open(out_dir / "c6_replay_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_c6_replay_benchmark()


"""
(gdm_env) giovanni02@MacBook-Air-del-Professore Graph-Dynamic-Meshing % pytest tests/test_ale_gcl.py tests/test_ale_convergence.py -s -v
=========================================================== test session starts ============================================================
platform darwin -- Python 3.10.21, pytest-9.1.1, pluggy-1.6.0 -- /opt/miniconda3/envs/gdm_env/bin/python3.10
cachedir: .pytest_cache
rootdir: /Users/giovanni02/Desktop/Progetti/Graph-Dynamic-Meshing
configfile: pyproject.toml
plugins: cov-7.1.0
collected 5 items                                                                                                                          

tests/test_ale_gcl.py::test_c1_mesh_motion_verification 
=================================================================
TEST C1 — MESH MOTION VERIFICATION
=================================================================
Errore Relativo L2 Velocita' : 1.6449e-04
Errore Max Linf Velocita'     : 3.9698e-05
PASSED
tests/test_ale_gcl.py::test_c2_discrete_gcl 
=================================================================
TEST C2 — DISCRETE GCL & FREE-STREAM PRESERVATION
=================================================================
Max Element-wise GCL Residual  : 4.3368e-17
Max Algebraic GCL Identity Res : 5.4210e-17
Max Free-Stream Error (u = 1)  : 2.8089e-14
PASSED
tests/test_ale_convergence.py::test_c3_ale_one_step_manufactured 
=================================================================
TEST C3 — ALE ONE-STEP MANUFACTURED SOLUTION VERIFICATION
=================================================================
Errore L2 Passo Fixed Mesh : 4.0011%
Errore L2 Passo ALE Mesh   : 4.1505%
PASSED
tests/test_ale_convergence.py::test_c4_ale_spatial_convergence 
======================================================================
TEST C4 — ALE SPATIAL CONVERGENCE (h-refinement on deformed state)
======================================================================
Mesh       h          L2 Error       p(L2)    H1 Error       p(H1)   
----------------------------------------------------------------------
8x8        0.1250     3.7062e-01     -        7.1340e-01     -       
16x16      0.0625     1.4345e-01     1.37     4.5983e-01     0.63    
32x32      0.0312     3.8812e-02     1.89     2.4297e-01     0.92    
64x64      0.0156     9.9228e-03     1.97     1.2323e-01     0.98    
PASSED
tests/test_ale_convergence.py::test_c5_ale_temporal_convergence 
=================================================================
TEST C5 — ALE TEMPORAL CONVERGENCE (Cauchy Self-Convergence)
=================================================================
dt           Cauchy L2 Error      Observed Order 
-----------------------------------------------------------------
0.0100       6.4654e-03           -              
0.0050       3.2591e-03           0.99           
0.0025       1.6482e-03           0.98           
PASSED

============================================================ 5 passed in 5.08s =============================================================
(gdm_env) giovanni02@MacBook-Air-del-Professore Graph-Dynamic-Meshing % python tests/test_ale_replay_stage_b.py

===========================================================================
TEST C6 — REPLAY TRAIETTORIA STAGE B: L2-REMAP vs ALE FORMULATION
===========================================================================
---> Fase 1: Registrazione sequenza di geometrie GNN (N_a = 10)...
---> Fase 2: Esecuzione Stage B (Mesh motion discreto + L2-Projection Remap)...
---> Fase 3: Esecuzione Stage C (Mesh velocity continua w + ALE D-GCL)...

===========================================================================
TEST C6 — CONFRONTO A PARI TARGET GNN: L2-REMAP vs FORMULAZIONE ALE CONTINUA
===========================================================================
Metrica                        Stage B (L2-Remap)     Stage C (ALE)       
---------------------------------------------------------------------------
Errore L2(T) Finale            8.917                % 9.472              %
Errore L2 Time-Integrated      7.868                % 8.268              %
Tempo Remapping Esplicito [s]  3.50                  0.00 (Assente)      
Tempo Totale Simulazione [s]   4.46                  1.97               
Speedup Computazionale         1.00x (Baseline)      2.26               x
===========================================================================
(gdm_env) giovanni02@MacBook-Air-del-Professore Graph-Dynamic-Meshing % 
"""