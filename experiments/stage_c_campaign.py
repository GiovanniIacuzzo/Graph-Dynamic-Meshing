"""
experiments/stage_c_campaign.py
Campagna finale dello Stage C:
Fixed (C0) vs GDM + L2-Remap (C1) vs GDM + ALE (C2) vs Oracle + ALE (C3).
Genera figure per il paper e la tabella riassuntiva LaTeX.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import load_config, ensure_dir
from generate_dataset import create_initial_mesh_2d
from geometry.mesh import TriangularMesh2D
from geometry.geometry import compute_mesh_metrics
from geometry.safety import apply_safety_filter
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.fem import ConvectionDiffusionFEM2D, compute_field_gradients
from numerics.remap_l2 import remap_l2_projection
from numerics.dynamic_solver import compute_continuous_errors
from numerics.ale import ALESolver2D
from oracle.field import GaussianBumpField
from oracle.optimizer import OfflineMeshOptimizer
from inference import MeshMoverInferenceEngine


class StageCCampaign:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.output_dir = ensure_dir("results/stage_c")
        self.figures_dir = ensure_dir(self.output_dir / "figures")
        self.tables_dir = ensure_dir(self.output_dir / "tables")
        self.raw_dir = ensure_dir(self.output_dir / "raw_metrics")

        self.pde = ManufacturedConvectionDiffusion2D(
            bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=0.5
        )
        self.domain = self.config["geometry"]["domain"]
        self.checkpoint_path = Path("results/stage_a/checkpoints/best_model.pt")

        self.T = 0.5
        self.dt = 0.005
        self.adapt_interval = 10
        self.nx = 32
        self.ny = 32
        self.n_steps = int(round(self.T / self.dt))

        self.engine = MeshMoverInferenceEngine(
            checkpoint_path=self.checkpoint_path,
            config=self.config,
            device="cpu"
        )
        self.eps = float(self.config["geometry"]["safety"]["epsilon"])

        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "lines.linewidth": 1.5,
            "grid.alpha": 0.3
        })

    def run_benchmark(self):
        print("\n" + "=" * 80)
        print("AVVIO BENCHMARK DINAMICO CONCLUSIVO STAGE C: ALE vs L2-REMAP vs FIXED")
        print("=" * 80)

        results = {}

        # -------------------------------------------------------------
        # C0: Fixed Mesh
        # -------------------------------------------------------------
        print("\n---> Esecuzione C0: Fixed Eulerian FEM...")
        mesh_c0 = create_initial_mesh_2d(self.domain, nx=self.nx, ny=self.ny)
        fem_c0 = ConvectionDiffusionFEM2D(mesh_c0, self.pde)
        u_c0 = self.pde.evaluate_solution(mesh_c0.nodes[:, 0], mesh_c0.nodes[:, 1], 0.0)

        t0 = time.perf_counter()
        time_arr = [0.0]
        l2_c0 = [compute_continuous_errors(mesh_c0, u_c0, self.pde, 0.0)[0]]
        fem_time_c0 = 0.0

        for step in range(1, self.n_steps + 1):
            t_next = step * self.dt
            t_f0 = time.perf_counter()
            u_c0 = fem_c0.step_backward_euler(u_c0, dt=self.dt, t_next=t_next)
            fem_time_c0 += (time.perf_counter() - t_f0)

            err_l2, _ = compute_continuous_errors(mesh_c0, u_c0, self.pde, t_next)
            time_arr.append(t_next)
            l2_c0.append(err_l2)

        tot_time_c0 = time.perf_counter() - t0
        results["C0_fixed"] = {
            "time": time_arr,
            "rel_l2_error": l2_c0,
            "final_l2": l2_c0[-1],
            "time_int_l2": float(np.trapz(l2_c0, time_arr) / self.T),
            "fem_time_s": fem_time_c0,
            "remap_time_s": 0.0,
            "adapt_time_s": 0.0,
            "total_time_s": tot_time_c0,
            "min_jacobian": min(compute_mesh_metrics(mesh_c0.nodes, mesh_c0.cells)["jacobians"])
        }

        # -------------------------------------------------------------
        # C1: GDM + L2-Remap (Stage B)
        # -------------------------------------------------------------
        print("\n---> Esecuzione C1: GDM + L2-Projection Remap (Stage B)...")
        mesh_c1 = create_initial_mesh_2d(self.domain, nx=self.nx, ny=self.ny)
        fem_c1 = ConvectionDiffusionFEM2D(mesh_c1, self.pde)
        u_c1 = self.pde.evaluate_solution(mesh_c1.nodes[:, 0], mesh_c1.nodes[:, 1], 0.0)

        t0 = time.perf_counter()
        l2_c1 = [compute_continuous_errors(mesh_c1, u_c1, self.pde, 0.0)[0]]
        fem_time_c1 = 0.0
        remap_time_c1 = 0.0
        adapt_time_c1 = 0.0
        min_j_c1 = float("inf")

        for step in range(1, self.n_steps + 1):
            t_curr = (step - 1) * self.dt
            t_next = step * self.dt

            if step % self.adapt_interval == 0:
                t_ad0 = time.perf_counter()
                grad_u = compute_field_gradients(mesh_c1, u_c1, self.pde, t_curr, gradient_mode="fem")
                pred = self.engine.predict(mesh_c1, u_c1, grad_u)
                new_mesh = pred["adapted_mesh"]
                adapt_time_c1 += (time.perf_counter() - t_ad0)

                t_rm0 = time.perf_counter()
                u_c1 = remap_l2_projection(mesh_c1.nodes, mesh_c1.cells, u_c1, new_mesh.nodes, new_mesh.cells)
                remap_time_c1 += (time.perf_counter() - t_rm0)

                mesh_c1 = new_mesh
                fem_c1 = ConvectionDiffusionFEM2D(mesh_c1, self.pde)

            t_f0 = time.perf_counter()
            u_c1 = fem_c1.step_backward_euler(u_c1, dt=self.dt, t_next=t_next)
            fem_time_c1 += (time.perf_counter() - t_f0)

            err_l2, _ = compute_continuous_errors(mesh_c1, u_c1, self.pde, t_next)
            l2_c1.append(err_l2)
            min_j_c1 = min(min_j_c1, float(min(compute_mesh_metrics(mesh_c1.nodes, mesh_c1.cells)["jacobians"])))

        tot_time_c1 = time.perf_counter() - t0
        results["C1_gdm_l2"] = {
            "time": time_arr,
            "rel_l2_error": l2_c1,
            "final_l2": l2_c1[-1],
            "time_int_l2": float(np.trapz(l2_c1, time_arr) / self.T),
            "fem_time_s": fem_time_c1,
            "remap_time_s": remap_time_c1,
            "adapt_time_s": adapt_time_c1,
            "total_time_s": tot_time_c1,
            "min_jacobian": min_j_c1
        }

        # -------------------------------------------------------------
        # C2: GDM + ALE (Metodo proposto)
        # -------------------------------------------------------------
        print("\n---> Esecuzione C2: GDM + ALE D-GCL (Metodo proposto)...")
        init_m_c2 = create_initial_mesh_2d(self.domain, nx=self.nx, ny=self.ny)
        curr_m_c2 = init_m_c2
        u_c2 = self.pde.evaluate_solution(curr_m_c2.nodes[:, 0], curr_m_c2.nodes[:, 1], 0.0)
        ale_solver = ALESolver2D(self.pde, nu=1e-3, bx=0.4, by=0.2)

        t0 = time.perf_counter()
        l2_c2 = [compute_continuous_errors(curr_m_c2, u_c2, self.pde, 0.0)[0]]
        fem_time_c2 = 0.0
        adapt_time_c2 = 0.0
        min_j_c2 = float("inf")
        snapshots_c2 = [(0.0, curr_m_c2.nodes.copy(), u_c2.copy())]

        base_nodes = curr_m_c2.nodes.copy()
        target_nodes = curr_m_c2.nodes.copy()

        for step in range(1, self.n_steps + 1):
            t_curr = (step - 1) * self.dt
            t_next = step * self.dt

            # Ricalcolo del target di adattamento ogni adapt_interval time-step
            if (step - 1) % self.adapt_interval == 0:
                t_ad0 = time.perf_counter()
                grad_u = compute_field_gradients(curr_m_c2, u_c2, self.pde, t_curr, gradient_mode="fem")
                pred = self.engine.predict(curr_m_c2, u_c2, grad_u)
                target_nodes = pred["adapted_mesh"].nodes.copy()
                base_nodes = curr_m_c2.nodes.copy()
                adapt_time_c2 += (time.perf_counter() - t_ad0)

            # Deformazione continua lineare nel sotto-intervallo: w = Delta X / (N_a * dt)
            k_step = (step - 1) % self.adapt_interval + 1
            interp_nodes = base_nodes + (k_step / self.adapt_interval) * (target_nodes - base_nodes)
            next_m_c2 = init_m_c2.update_coordinates(interp_nodes)

            t_f0 = time.perf_counter()
            u_c2, _ = ale_solver.step_ale(curr_m_c2, next_m_c2, u_c2, dt=self.dt, t_next=t_next, pure_gcl=False)
            fem_time_c2 += (time.perf_counter() - t_f0)

            err_l2, _ = compute_continuous_errors(next_m_c2, u_c2, self.pde, t_next)
            l2_c2.append(err_l2)
            min_j_c2 = min(min_j_c2, float(min(compute_mesh_metrics(next_m_c2.nodes, next_m_c2.cells)["jacobians"])))

            curr_m_c2 = next_m_c2

            if step in [int(round(self.n_steps / 3)), int(round(2 * self.n_steps / 3)), self.n_steps]:
                snapshots_c2.append((t_next, curr_m_c2.nodes.copy(), u_c2.copy()))

        tot_time_c2 = time.perf_counter() - t0
        results["C2_gdm_ale"] = {
            "time": time_arr,
            "rel_l2_error": l2_c2,
            "final_l2": l2_c2[-1],
            "time_int_l2": float(np.trapz(l2_c2, time_arr) / self.T),
            "fem_time_s": fem_time_c2,
            "remap_time_s": 0.0,
            "adapt_time_s": adapt_time_c2,
            "total_time_s": tot_time_c2,
            "min_jacobian": min_j_c2
        }

        # -------------------------------------------------------------
        # C3: Oracle + ALE
        # -------------------------------------------------------------
        print("\n---> Esecuzione C3: Oracle + ALE D-GCL...")
        init_m_c3 = create_initial_mesh_2d(self.domain, nx=self.nx, ny=self.ny)
        curr_m_c3 = init_m_c3
        u_c3 = self.pde.evaluate_solution(curr_m_c3.nodes[:, 0], curr_m_c3.nodes[:, 1], 0.0)

        t0 = time.perf_counter()
        l2_c3 = [compute_continuous_errors(curr_m_c3, u_c3, self.pde, 0.0)[0]]
        fem_time_c3 = 0.0
        adapt_time_c3 = 0.0
        min_j_c3 = float("inf")

        base_nodes_orc = curr_m_c3.nodes.copy()
        target_nodes_orc = curr_m_c3.nodes.copy()

        for step in range(1, self.n_steps + 1):
            t_curr = (step - 1) * self.dt
            t_next = step * self.dt

            if (step - 1) % self.adapt_interval == 0:
                t_ad0 = time.perf_counter()
                cx, cy = self.pde.get_center(t_curr)
                field_oracle = GaussianBumpField(mu_x=cx, mu_y=cy, sigma=self.pde.sigma, amplitude=np.exp(-self.pde.beta * t_curr))
                opt = OfflineMeshOptimizer(
                    field_oracle, curr_m_c3.nodes, curr_m_c3.cells, curr_m_c3.boundary_nodes,
                    lambda_D=self.config["oracle"]["objective"]["lambda_D"],
                    lambda_Q=self.config["oracle"]["objective"]["lambda_Q"],
                    epsilon=self.eps, max_iter=3
                )
                opt_res = opt.optimize()
                target_nodes_orc = opt_res["optimal_nodes"].copy()
                base_nodes_orc = curr_m_c3.nodes.copy()
                adapt_time_c3 += (time.perf_counter() - t_ad0)

            k_step = (step - 1) % self.adapt_interval + 1
            interp_nodes = base_nodes_orc + (k_step / self.adapt_interval) * (target_nodes_orc - base_nodes_orc)
            next_m_c3 = init_m_c3.update_coordinates(interp_nodes)

            t_f0 = time.perf_counter()
            u_c3, _ = ale_solver.step_ale(curr_m_c3, next_m_c3, u_c3, dt=self.dt, t_next=t_next, pure_gcl=False)
            fem_time_c3 += (time.perf_counter() - t_f0)

            err_l2, _ = compute_continuous_errors(next_m_c3, u_c3, self.pde, t_next)
            l2_c3.append(err_l2)
            min_j_c3 = min(min_j_c3, float(min(compute_mesh_metrics(next_m_c3.nodes, next_m_c3.cells)["jacobians"])))
            curr_m_c3 = next_m_c3

        tot_time_c3 = time.perf_counter() - t0
        results["C3_oracle_ale"] = {
            "time": time_arr,
            "rel_l2_error": l2_c3,
            "final_l2": l2_c3[-1],
            "time_int_l2": float(np.trapz(l2_c3, time_arr) / self.T),
            "fem_time_s": fem_time_c3,
            "remap_time_s": 0.0,
            "adapt_time_s": adapt_time_c3,
            "total_time_s": tot_time_c3,
            "min_jacobian": min_j_c3
        }

        self.results = results
        self.snapshots_c2 = snapshots_c2
        self.final_mesh_c2 = curr_m_c2
        self.final_u_c2 = u_c2

        self._export_results_and_table()
        self._generate_stage_c_figures()

    def _export_results_and_table(self):
        with open(self.raw_dir / "stage_c_benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump({k: {m: v for m, v in val.items() if m != "time"} for k, val in self.results.items()}, f, indent=2)

        table_path = self.tables_dir / "table_stage_c_summary.tex"
        rows = [
            ("Fixed Eulerian (C0)", "C0_fixed"),
            ("GDM w/ $L^2$-Remap (C1)", "C1_gdm_l2"),
            ("GDM w/ ALE (C2)", "C2_gdm_ale"),
            ("Oracle w/ ALE (C3)", "C3_oracle_ale")
        ]

        latex_rows = []
        for name, k in rows:
            r = self.results[k]
            speedup = self.results["C3_oracle_ale"]["total_time_s"] / r["total_time_s"]
            latex_rows.append(
                f"{name} & {r['final_l2']*100:.2f}\\% & {r['time_int_l2']*100:.2f}\\% & "
                f"{r['min_jacobian']:.2e} & {r['fem_time_s']:.2f} & {r['remap_time_s']:.2f} & "
                f"{r['total_time_s']:.2f} & {speedup:.1f}$\\times$ \\\\"
            )

        latex_table = (
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\small\n"
            "\\caption{Stage C Benchmark Summary ($T=0.5, \\Delta t=0.005, 32\\times 32$ Mesh).}\n"
            "\\label{tab:stage_c_summary}\n"
            "\\begin{tabular}{lccccccc}\n"
            "\\hline\n"
            "Configuration & Final $L^2$ & Time-Int $L^2$ & $\\min_{t,e} J_e$ & FEM Time [s] & Remap Time [s] & Total [s] & Speedup \\\\\n"
            "\\hline\n" +
            "\n".join(latex_rows) + "\n"
            "\\hline\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"\n[INFO] Tabella salvata in {table_path}")

    def _generate_stage_c_figures(self):
        print("\n---> Generazione Figure Scientifiche dello Stage C...")
        t_arr = self.results["C0_fixed"]["time"]

        # -------------------------------------------------------------
        # FIGURA C3: Confronto Dinamico Transiente dell'Errore
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(t_arr, np.array(self.results["C0_fixed"]["rel_l2_error"]) * 100, label="Fixed Mesh (C0)", color="#7f7f7f", linestyle="--")
        ax.plot(t_arr, np.array(self.results["C1_gdm_l2"]["rel_l2_error"]) * 100, label=r"GDM w/ $L^2$-Remap (C1)", color="#ff7f0e")
        ax.plot(t_arr, np.array(self.results["C2_gdm_ale"]["rel_l2_error"]) * 100, label="GDM w/ ALE (C2, Proposto)", color="#1f77b4")
        ax.plot(t_arr, np.array(self.results["C3_oracle_ale"]["rel_l2_error"]) * 100, label="Oracle w/ ALE (C3)", color="#2ca02c", linestyle=":")

        ax.set_xlabel("Time $t$ [s]")
        ax.set_ylabel(r"Continuous Relative $L^2$ Error [%]")
        ax.set_title("Fig. C3: Transient PDE Error: Remap vs ALE Formulation")
        ax.grid(True)
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig_c3_dynamic_error.png", dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # FIGURA C5: Decomposizione Cronometrica del Runtime
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        cfgs = ["GDM w/ L2-Remap", "GDM w/ ALE (Ours)", "Oracle w/ ALE"]
        fem_times = [self.results["C1_gdm_l2"]["fem_time_s"], self.results["C2_gdm_ale"]["fem_time_s"], self.results["C3_oracle_ale"]["fem_time_s"]]
        remap_times = [self.results["C1_gdm_l2"]["remap_time_s"], 0.0, 0.0]
        adapt_times = [self.results["C1_gdm_l2"]["adapt_time_s"], self.results["C2_gdm_ale"]["adapt_time_s"], self.results["C3_oracle_ale"]["adapt_time_s"]]

        x = np.arange(len(cfgs))
        width = 0.5
        ax.bar(x, fem_times, width, label="FEM Assembly & Solve", color="#1f77b4")
        ax.bar(x, remap_times, width, bottom=fem_times, label=r"$L^2$-Projection Remapping", color="#d62728")
        ax.bar(x, adapt_times, width, bottom=np.array(fem_times) + np.array(remap_times), label="GNN/Oracle Adaptation", color="#2ca02c")

        ax.set_ylabel("Execution Time [s]")
        ax.set_title("Fig. C5: Algorithmic Runtime Decomposition (100 Steps)")
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs)
        ax.grid(True, axis="y")
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig_c5_runtime_decomposition.png", dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # FIGURA C6: Snapshot Qualitativi e Tracciamento della Soluzione
        # -------------------------------------------------------------
        fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.0))
        for col, (snap_t, snap_nodes, snap_u) in enumerate(self.snapshots_c2):
            tri = mtri.Triangulation(snap_nodes[:, 0], snap_nodes[:, 1], self.final_mesh_c2.cells)
            axes[0, col].triplot(tri, "b-", lw=0.4)
            axes[0, col].set_title(f"Mesh: $t = {snap_t:.2f}$ s")

            cf = axes[1, col].tricontourf(tri, snap_u, levels=25, cmap="viridis")
            axes[1, col].set_title(f"ALE $u_h(t={snap_t:.2f})$")

            for row in range(2):
                axes[row, col].set_aspect("equal")
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

        fig.suptitle("Fig. C6: Dynamic GNN-ALE Mesh Deformation Tracking the Moving Pulse", fontsize=13)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig_c6_qualitative_tracking.png", dpi=300)
        plt.close(fig)

        print("[INFO] Figure generate con successo in results/stage_c/figures/")


def main():
    campaign = StageCCampaign()
    campaign.run_benchmark()


if __name__ == "__main__":
    main()