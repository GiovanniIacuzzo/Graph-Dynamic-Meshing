"""
experiments/diagnose_stage_c_transfer.py
Script di isolamento per diagnosticare se la perdita di accuratezza in Stage C
deriva dal remapping P1, dalla frequenza di adattamento o dalla dinamica dei nodi.
"""

from pathlib import Path
import numpy as np

from utils.utils import load_config
from generate_dataset import create_initial_mesh_2d
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.fem import ConvectionDiffusionFEM2D, compute_field_gradients
from numerics.remap import remap_p1_solution
from numerics.remap_l2 import remap_l2_projection
from numerics.dynamic_solver import compute_continuous_errors
from inference import MeshMoverInferenceEngine


def run_diagnostic():
    config = load_config("config/config.yaml")
    domain = config["geometry"]["domain"]
    checkpoint = Path("results/stage_a/checkpoints/best_model.pt")
    engine = MeshMoverInferenceEngine(checkpoint, config, device="cpu")

    pde = ManufacturedConvectionDiffusion2D(
        bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=0.5
    )

    mesh_init = create_initial_mesh_2d(domain, nx=32, ny=32)
    u_exact_0 = pde.evaluate_solution(mesh_init.nodes[:, 0], mesh_init.nodes[:, 1], 0.0)

    # -------------------------------------------------------------------------
    # TEST 1: Crollo di accuratezza per P1 vs L2 su 10 adattamenti STATICI a t=0
    # (Senza alcuna evoluzione PDE, solo proiezioni ripetute avanti/indietro)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("DIAGNOSTICA 1: ACCUMULO ERRORE DI PURO REMAPPING (10 cicli a t = 0)")
    print("=" * 75)

    grad_0 = compute_field_gradients(mesh_init, u_exact_0, pde, 0.0, gradient_mode="fem")
    pred_0 = engine.predict(mesh_init, u_exact_0, grad_0)
    mesh_adapted = pred_0["adapted_mesh"]

    err_init_l2, _ = compute_continuous_errors(mesh_init, u_exact_0, pde, 0.0)
    print(f"Errore L2 continuo iniziale su griglia 32x32 fissa : {err_init_l2*100:.4f}%")

    # A) 1 singolo step di Remap verso la mesh adattata
    u_p1_1 = remap_p1_solution(mesh_init.nodes, mesh_init.cells, u_exact_0, mesh_adapted.nodes)
    u_l2_1 = remap_l2_projection(
        mesh_init.nodes, mesh_init.cells, u_exact_0, mesh_adapted.nodes, mesh_adapted.cells
    )
    e_p1_1, _ = compute_continuous_errors(mesh_adapted, u_p1_1, pde, 0.0)
    e_l2_1, _ = compute_continuous_errors(mesh_adapted, u_l2_1, pde, 0.0)
    print(f"Dopo 1 Remap  -> Errore P1: {e_p1_1*100:.4f}% | Errore L2: {e_l2_1*100:.4f}%")

    # B) 10 cicli alternati (mesh_init <-> mesh_adapted) come avviene in 10 adattamenti
    u_p1_loop = u_exact_0.copy()
    u_l2_loop = u_exact_0.copy()
    for _ in range(5):
        u_p1_loop = remap_p1_solution(
            mesh_init.nodes, mesh_init.cells, u_p1_loop, mesh_adapted.nodes
        )
        u_p1_loop = remap_p1_solution(
            mesh_adapted.nodes, mesh_adapted.cells, u_p1_loop, mesh_init.nodes
        )

        u_l2_loop = remap_l2_projection(
            mesh_init.nodes, mesh_init.cells, u_l2_loop, mesh_adapted.nodes, mesh_adapted.cells
        )
        u_l2_loop = remap_l2_projection(
            mesh_adapted.nodes, mesh_adapted.cells, u_l2_loop, mesh_init.nodes, mesh_init.cells
        )

    e_p1_10, _ = compute_continuous_errors(mesh_init, u_p1_loop, pde, 0.0)
    e_l2_10, _ = compute_continuous_errors(mesh_init, u_l2_loop, pde, 0.0)
    print(f"Dopo 10 Remap -> Errore P1: {e_p1_10*100:.4f}% | Errore L2: {e_l2_10*100:.4f}%")
    print(f"Picco massimo soluzione -> Esatto: {u_exact_0.max():.4f} | P1: {u_p1_loop.max():.4f} | L2: {u_l2_loop.max():.4f}")

    # -------------------------------------------------------------------------
    # TEST 2: Cosa succede se adattiamo a t=0 con SOLUZIONE ESATTA su nuova mesh
    # e poi evolviamo con FEM? (Isola il guadagno geometrico della mesh GNN)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("DIAGNOSTICA 2: EVOLUZIONE DINAMICA (dt=0.005, 100 step) CON DIVERSI TRASFERIMENTI")
    print("=" * 75)

    dt = 0.005
    n_steps = 100

    for mode_label in ["FIXED", "GDM_P1_REMAP", "GDM_L2_REMAP", "GDM_EXACT_REINIT"]:
        curr_mesh = create_initial_mesh_2d(domain, nx=32, ny=32)
        u_curr = pde.evaluate_solution(curr_mesh.nodes[:, 0], curr_mesh.nodes[:, 1], 0.0)
        solver = ConvectionDiffusionFEM2D(curr_mesh, pde)

        for step in range(1, n_steps + 1):
            t_curr = (step - 1) * dt
            t_next = step * dt

            if mode_label != "FIXED" and step % 10 == 0:
                grad_u = compute_field_gradients(curr_mesh, u_curr, pde, t_curr, gradient_mode="fem")
                pred = engine.predict(curr_mesh, u_curr, grad_u)
                new_mesh = pred["adapted_mesh"]

                if mode_label == "GDM_P1_REMAP":
                    u_curr = remap_p1_solution(
                        curr_mesh.nodes, curr_mesh.cells, u_curr, new_mesh.nodes
                    )
                elif mode_label == "GDM_L2_REMAP":
                    u_curr = remap_l2_projection(
                        curr_mesh.nodes, curr_mesh.cells, u_curr, new_mesh.nodes, new_mesh.cells
                    )
                elif mode_label == "GDM_EXACT_REINIT":
                    # Caso ideale senza errore di trasferimento
                    u_curr = pde.evaluate_solution(new_mesh.nodes[:, 0], new_mesh.nodes[:, 1], t_curr)

                b_nodes = np.asarray(new_mesh.boundary_nodes, dtype=np.int64)
                u_curr[b_nodes] = pde.evaluate_solution(
                    new_mesh.nodes[b_nodes, 0], new_mesh.nodes[b_nodes, 1], t_curr
                )
                curr_mesh = new_mesh
                solver = ConvectionDiffusionFEM2D(curr_mesh, pde)

            u_curr = solver.step_backward_euler(u_curr, dt=dt, t_next=t_next)

        final_l2, final_h1 = compute_continuous_errors(curr_mesh, u_curr, pde, n_steps * dt)
        print(f"Configurazione {mode_label:<18} -> Final L2: {final_l2*100:.2f}% | Final H1: {final_h1*100:.2f}%")


if __name__ == "__main__":
    run_diagnostic()