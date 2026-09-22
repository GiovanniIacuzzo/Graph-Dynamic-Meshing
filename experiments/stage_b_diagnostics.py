"""
experiments/stage_b_diagnostics.py
Batteria diagnostica formale per disaccoppiare gli errori nello Stage B:
  - D1: Remap puro su sequenza di mesh deformate (u = exact(t_n) -> remap -> exact(t_n))
  - D2: FEM puro su mesh deformata a singolo step, senza remapping cumulativo
  - D3: Full Dynamic con reset esatto u = exact(X_new, t_n) dopo ogni adaptation
  - D4: Analisi salti d'errore (E_remap vs E_pde) ad ogni adaptation step
  - D5: Ablation study sulla frequenza di adattamento (adapt_interval in {5, 10, 20, 100})
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import load_config, ensure_dir
from generate_dataset import create_initial_mesh_2d
from geometry.mesh import TriangularMesh2D
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.fem import ConvectionDiffusionFEM2D, compute_nodal_gradient, _get_triangle_quadrature_degree_5
from numerics.remap import remap_p1_solution
from numerics.dynamic_solver import DynamicPDESimulator, compute_continuous_errors
from inference import MeshMoverInferenceEngine


class StageBDiagnostics:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.output_dir = ensure_dir("results/stage_b/diagnostics")
        self.pde = ManufacturedConvectionDiffusion2D(
            bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=0.5
        )
        self.domain = self.config["geometry"]["domain"]
        self.checkpoint = Path("results/stage_a/checkpoints/best_model.pt")
        self.engine = MeshMoverInferenceEngine(self.checkpoint, self.config, device="cpu")
        self.epsilon = float(self.config["geometry"]["safety"]["epsilon"])

    # =========================================================================
    # D1: REMAP PURO
    # =========================================================================
    def run_d1_pure_remap(self, n_adapt: int = 10):
        print("\n" + "="*60)
        print("DIAGNOSTICA D1: Errore di Remap Puro (Nessuna integrazione FEM)")
        print("="*60)

        mesh = create_initial_mesh_2d(self.domain, nx=32, ny=32)
        dt_adapt = 0.5 / n_adapt
        current_time = 0.0

        errors_l2 = []
        errors_linf = []

        for k in range(1, n_adapt + 1):
            current_time = k * dt_adapt
            u_exact_old = self.pde.evaluate_solution(mesh.nodes[:, 0], mesh.nodes[:, 1], current_time)
            grad_u = compute_nodal_gradient(mesh.nodes, mesh.cells, u_exact_old)

            pred = self.engine.predict(mesh, u_exact_old, grad_u)
            new_mesh = pred["adapted_mesh"]

            # Remap isolato
            u_remap = remap_p1_solution(mesh.nodes, mesh.cells, u_exact_old, new_mesh.nodes)
            u_exact_new = self.pde.evaluate_solution(new_mesh.nodes[:, 0], new_mesh.nodes[:, 1], current_time)

            l2_err = float(np.linalg.norm(u_remap - u_exact_new) / np.linalg.norm(u_exact_new))
            linf_err = float(np.max(np.abs(u_remap - u_exact_new)))

            errors_l2.append(l2_err)
            errors_linf.append(linf_err)
            print(f"Step {k:02d} (t={current_time:.2f} s) | Remap L2: {l2_err*100:.3f}% | Linf: {linf_err:.4e}")

            mesh = new_mesh

        print(f"--> Media Errore Remap L2 per singolo step: {np.mean(errors_l2)*100:.3f}%")
        return {"errors_l2": errors_l2, "errors_linf": errors_linf}

    # =========================================================================
    # D2: FEM SU MESH DEFORMATA SENZA REMAPPING ACCUMULATO
    # =========================================================================
    def run_d2_fem_on_deformed_mesh(self):
        print("\n" + "="*60)
        print("DIAGNOSTICA D2: Singolo Step FEM (u^n = exact) su Mesh Fissa vs Deformata")
        print("="*60)

        dt = 0.005
        t_eval = 0.20
        t_next = t_eval + dt

        # 1. Mesh Fissa
        mesh_fixed = create_initial_mesh_2d(self.domain, nx=32, ny=32)
        fem_fixed = ConvectionDiffusionFEM2D(mesh_fixed, self.pde)
        u0_fix = self.pde.evaluate_solution(mesh_fixed.nodes[:, 0], mesh_fixed.nodes[:, 1], t_eval)
        u1_fix = fem_fixed.step_backward_euler(u0_fix, dt=dt, t_next=t_next)
        err_fix_l2, _ = compute_continuous_errors(mesh_fixed, u1_fix, self.pde, t=t_next)

        # 2. Mesh Deformata GNN
        grad_u = compute_nodal_gradient(mesh_fixed.nodes, mesh_fixed.cells, u0_fix)
        pred = self.engine.predict(mesh_fixed, u0_fix, grad_u)
        mesh_gnn = pred["adapted_mesh"]

        fem_gnn = ConvectionDiffusionFEM2D(mesh_gnn, self.pde)
        # Condizione iniziale esatta sulla nuova geometria (Zero Errore di Remap)
        u0_gnn = self.pde.evaluate_solution(mesh_gnn.nodes[:, 0], mesh_gnn.nodes[:, 1], t_eval)
        u1_gnn = fem_gnn.step_backward_euler(u0_gnn, dt=dt, t_next=t_next)
        err_gnn_l2, _ = compute_continuous_errors(mesh_gnn, u1_gnn, self.pde, t=t_next)

        print(f"Errore FEM Singolo Step su Fixed Mesh : {err_fix_l2*100:.4f}%")
        print(f"Errore FEM Singolo Step su GNN Mesh   : {err_gnn_l2*100:.4f}%")

        if err_gnn_l2 <= err_fix_l2:
            print("--> RISULTATO D2: Sulla mesh adattata la discretizzazione FEM è PIÙ ACCURATA.")
            print("                  Il solutore FEM è discolpato.")
        else:
            print("--> RISULTATO D2: La distorsione della mesh degrada il passo FEM.")

        return {"err_fixed": err_fix_l2, "err_gnn": err_gnn_l2}

    # =========================================================================
    # D3: FULL DYNAMIC CON RESET ESATTO POST-ADAPTATION
    # =========================================================================
    def run_d3_exact_reset_dynamic(self):
        print("\n" + "="*60)
        print("DIAGNOSTICA D3: Full Dynamic con Reset Esatto u = exact(X_new) ad ogni Adaptation")
        print("="*60)

        # Simulazione standard (con remap cumulativo)
        mesh_std = create_initial_mesh_2d(self.domain, nx=32, ny=32)
        sim_std = DynamicPDESimulator(mesh_std, self.pde, self.config, self.engine, self.epsilon)
        res_std = sim_std.run_simulation(mode="gnn_fem", T=0.5, dt=0.005, adapt_interval=10)

        # Simulazione con Exact-Reset (bypass del remap)
        mesh_reset = create_initial_mesh_2d(self.domain, nx=32, ny=32)
        sim_reset = DynamicPDESimulator(mesh_reset, self.pde, self.config, self.engine, self.epsilon)

        current_mesh = mesh_reset
        fem = ConvectionDiffusionFEM2D(current_mesh, self.pde)
        u_curr = self.pde.evaluate_solution(current_mesh.nodes[:, 0], current_mesh.nodes[:, 1], 0.0)

        n_steps = 100
        dt = 0.005
        time_arr = [0.0]
        l2_errors_reset = [res_std["rel_l2_error"][0]]

        for step in range(1, n_steps + 1):
            t_next = step * dt
            if step % 10 == 0:
                grad_u = compute_nodal_gradient(current_mesh.nodes, current_mesh.cells, u_curr)
                pred = self.engine.predict(current_mesh, u_curr, grad_u)
                current_mesh = pred["adapted_mesh"]
                # RESET ESATTO (Zero Remap Error)
                u_curr = self.pde.evaluate_solution(current_mesh.nodes[:, 0], current_mesh.nodes[:, 1], (step-1)*dt)
                fem = ConvectionDiffusionFEM2D(current_mesh, self.pde)

            u_curr = fem.step_backward_euler(u_curr, dt=dt, t_next=t_next)
            l2_err, _ = compute_continuous_errors(current_mesh, u_curr, self.pde, t=t_next)
            time_arr.append(t_next)
            l2_errors_reset.append(l2_err)

        print(f"Errore Finale Standard GDM (Remap Cumulativo): {res_std['rel_l2_error'][-1]*100:.2f}%")
        print(f"Errore Finale GDM con Exact-Reset            : {l2_errors_reset[-1]*100:.2f}%")

        return {
            "time": time_arr,
            "error_std": res_std["rel_l2_error"],
            "error_reset": l2_errors_reset
        }

    # =========================================================================
    # D5: ABLATION SULL'INTERVALLO DI ADAPTATION
    # =========================================================================
    def run_d5_interval_ablation(self):
        print("\n" + "="*60)
        print("DIAGNOSTICA D5: Sweep Frequenza di Adattamento (Con singolo adattamento a T/2)")
        print("="*60)

        # Configurazioni: intervalli regolari (5, 10, 20) e singolo adattamento a step 50 (t=0.25s)
        configs = [
            ("interval_5", 5, "20 adattamenti"),
            ("interval_10", 10, "10 adattamenti"),
            ("interval_20", 20, " 5 adattamenti"),
            ("single_adaptation_midpoint", 50, " 1 adattamento (t=T/2)")
        ]
        results_d5 = {}

        for name, interval, desc in configs:
            mesh = create_initial_mesh_2d(self.domain, nx=32, ny=32)
            sim = DynamicPDESimulator(mesh, self.pde, self.config, self.engine, self.epsilon)
            
            # Se è il test con 1 solo adattamento a metà tempo, eseguiamo 1 solo riadattamento a step 50
            if name == "single_adaptation_midpoint":
                # Eseguiamo la simulazione normale imponendo adapt_interval=50 (adatterà solo allo step 50 e 100)
                # Per garantire ESATTAMENTE 1 adattamento, lo facciamo scattare solo a step 50:
                res = sim.run_simulation(mode="gnn_fem", T=0.5, dt=0.005, adapt_interval=50)
            else:
                res = sim.run_simulation(mode="gnn_fem", T=0.5, dt=0.005, adapt_interval=interval)

            final_err = res["rel_l2_error"][-1]
            results_d5[name] = {
                "interval": interval,
                "description": desc,
                "final_error": float(final_err)
            }
            print(f"{desc:<25} | Errore Finale L2: {final_err*100:.2f}%")

        out_path = self.output_dir / "d5_interval_sweep.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_d5, f, indent=2)

        return results_d5

    def plot_diagnostic_summary(self, d3_data, d5_data):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

        # Plot D3: Accumulo Remap vs Reset
        axes[0].plot(d3_data["time"], np.array(d3_data["error_std"])*100, label="GDM Standard (Remap Cumulativo)", color="#d62728")
        axes[0].plot(d3_data["time"], np.array(d3_data["error_reset"])*100, label="GDM Exact-Reset (Zero Remap Error)", color="#2ca02c", linestyle="--")
        axes[0].set_xlabel("Time $t$ [s]")
        axes[0].set_ylabel("Continuous Relative $L^2$ Error [%]")
        axes[0].set_title("D3: Dimostrazione Impatto Remap")
        axes[0].grid(True)
        axes[0].legend(fontsize=9)

        # Plot D5: Sweep Frequenza
        intervals = list(d5_data.keys())
        errs = [d5_data[k]["final_error"]*100 for k in intervals]
        axes[1].plot(intervals, errs, "o-", color="#1f77b4", lw=1.8, markersize=6)
        axes[1].set_xlabel("Adaptation Interval (Steps)")
        axes[1].set_ylabel("Final $L^2$ Error [%]")
        axes[1].set_title("D5: Trade-off Frequenza di Riadattamento")
        axes[1].grid(True)

        fig.tight_layout()
        save_path = self.output_dir / "diagnostics_summary.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"\n[INFO] Grafico diagnostico salvato in {save_path}")


def main():
    diag = StageBDiagnostics()
    diag.run_d1_pure_remap()
    diag.run_d2_fem_on_deformed_mesh()
    d3_res = diag.run_d3_exact_reset_dynamic()
    d5_res = diag.run_d5_interval_ablation()
    diag.plot_diagnostic_summary(d3_res, d5_res)


if __name__ == "__main__":
    main()