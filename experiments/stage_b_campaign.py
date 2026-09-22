"""
experiments/stage_b_campaign.py
Campagna dinamica definitiva Stage B: Fixed (B0) vs GDM+P1 (B1) vs GDM+L2 (B2) vs Oracle+L2 (B3).
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
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.dynamic_solver import DynamicPDESimulator
from inference import MeshMoverInferenceEngine


class StageBCampaign:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.output_dir = ensure_dir("results/stage_b")
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

        self.engine = MeshMoverInferenceEngine(
            checkpoint_path=self.checkpoint_path,
            config=self.config,
            device="cpu"
        )

        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "figure.titlesize": 13,
            "lines.linewidth": 1.5,
            "grid.alpha": 0.3
        })

    def run_benchmark(self):
        print("\n" + "=" * 75)
        print("AVVIO BENCHMARK COMPARATIVO STAGE B: P1 vs L2-PROJECTION REMAP")
        print("=" * 75)

        configs = [
            ("B0_fixed", "fixed", "p1", "Fixed Mesh (B0)"),
            ("B1_gnn_p1", "gnn_fem", "p1", "GDM w/ P1 Pointwise (B1)"),
            ("B2_gnn_l2", "gnn_fem", "l2", "GDM w/ L2-Projection (B2)"),
            ("B3_interp_oracle_l2", "oracle", "l2", "Oracle w/ L2-Projection (B3)")
        ]

        self.results = {}

        for label, mode, remap_m, desc in configs:
            print(f"\n---> Esecuzione: {label} [{desc}]")
            t0 = time.perf_counter()
            initial_mesh = create_initial_mesh_2d(self.domain, nx=self.nx, ny=self.ny)
            sim = DynamicPDESimulator(
                initial_mesh=initial_mesh,
                pde=self.pde,
                config=self.config,
                inference_engine=self.engine,
                epsilon=float(self.config["geometry"]["safety"]["epsilon"])
            )

            res = sim.run_simulation(
                mode=mode,
                remap_method=remap_m,
                T=self.T,
                dt=self.dt,
                adapt_interval=self.adapt_interval
            )
            elapsed = time.perf_counter() - t0

            # Time-integrated L2 calcolato con la regola del trapezio
            t_span = res["time"][-1] - res["time"][0]
            time_int_l2 = float(np.trapz(res["rel_l2_error"], res["time"]) / t_span)

            print(f"     Completato in {elapsed:.2f} s | L2 Finale: {res['rel_l2_error'][-1]*100:.2f}% | "
                  f"Time-Int L2: {time_int_l2*100:.2f}% | Min Jacobiano: {min(res['min_jacobian']):.4e}")

            clean_res = {k: v for k, v in res.items() if k not in ["mesh_snapshots", "final_mesh", "final_solution"]}
            clean_res["time_integrated_l2"] = time_int_l2
            self.results[label] = res

            with open(self.raw_dir / f"{label}_metrics.json", "w", encoding="utf-8") as f:
                json.dump(clean_res, f, indent=2)

        self._export_latex_summary_table()
        self._generate_figures()

    def _export_latex_summary_table(self):
        table_path = self.tables_dir / "table_stage_b_summary.tex"

        rows = []
        labels_info = [
            ("B0_fixed", "Fixed Mesh (B0)"),
            ("B1_gnn_p1", "GDM w/ P1 Remap (B1)"),
            ("B2_gnn_l2", "GDM w/ L2-Proj (B2)"),
            ("B3_interp_oracle_l2", "Oracle w/ L2-Proj (B3)")
        ]

        for label, name in labels_info:
            r = self.results[label]
            final_l2 = r["rel_l2_error"][-1]
            t_span = r["time"][-1] - r["time"][0]
            time_int_l2 = float(np.trapz(r["rel_l2_error"], r["time"]) / t_span)
            min_j = min(r["min_jacobian"])
            min_q = min(r["mean_quality"])
            tot_fem_time = sum(r["timing_fem_ms"]) / 1000.0
            tot_remap_time = sum(r["timing_remap_ms"]) / 1000.0
            tot_adapt_time = (sum(r["timing_gnn_ms"]) + sum(r["timing_safety_ms"]) + sum(r["timing_remap_ms"])) / 1000.0
            total_time = r["total_elapsed_simulation_s"]

            row_str = (
                f"{name} & {final_l2 * 100:.2f}\\% & {time_int_l2 * 100:.2f}\\% & "
                f"{min_j:.2e} & {min_q:.3f} & {tot_fem_time:.2f} & {tot_remap_time:.2f} & {total_time:.2f} \\\\"
            )
            rows.append(row_str)

        latex_content = (
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\small\n"
            "\\caption{Stage B Dynamic Benchmark Summary ($T=0.5, \\Delta t=0.005, 32\\times 32$ Mesh).}\n"
            "\\label{tab:stage_b_summary}\n"
            "\\begin{tabular}{lccccccc}\n"
            "\\hline\n"
            "Configuration & Final $L^2$ Err & Time-Int $L^2$ & $\\min_{t,e} J_e$ & Min Qual $\\eta$ & FEM Time [s] & Remap Time [s] & Total Time [s] \\\\\n"
            "\\hline\n" +
            "\n".join(rows) + "\n"
            "\\hline\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )

        with open(table_path, "w", encoding="utf-8") as f:
            f.write(latex_content)
        print(f"\n[INFO] Tabella LaTeX esportata in {table_path}")

    def _generate_figures(self):
        print("\n---> Generazione Figure Scientifiche Aggiornate...")

        conv_dir = Path("results/stage_b/convergence")
        with open(conv_dir / "spatial_convergence.json", "r") as f:
            spat_data = json.load(f)
        with open(conv_dir / "temporal_convergence.json", "r") as f:
            temp_data = json.load(f)

        # -------------------------------------------------------------
        # FIGURA 1: Spatial Convergence (h-refinement)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        h = np.array(spat_data["h_values"])
        e_l2 = np.array(spat_data.get("errors_l2", []))
        e_h1 = np.array(spat_data.get("errors_h1", []))
        p_l2 = np.mean(spat_data["rates_l2"][-2:])
        p_h1 = np.mean(spat_data["rates_h1"][-2:])

        ax.loglog(h, e_l2, "o-", color="#1f77b4", label=rf"$L^2$ error (rate $\approx {p_l2:.2f}$)")
        ax.loglog(h, e_h1, "s--", color="#d62728", label=rf"$H^1$ seminorm (rate $\approx {p_h1:.2f}$)")
        ax.loglog(h, e_l2[0] * (h / h[0])**2, ":", color="gray", label=r"Ideal $\mathcal{O}(h^2)$")
        ax.loglog(h, e_h1[0] * (h / h[0]), "-.", color="gray", label=r"Ideal $\mathcal{O}(h)$")
        ax.set_xlabel("Element Size $h$")
        ax.set_ylabel("Continuous Relative Error")
        ax.set_title("Fig. 1: Spatial Mesh Convergence ($P_1$ FEM)")
        ax.grid(True, which="both")
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig1_spatial_convergence.png", dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # FIGURA 2: Temporal Convergence (dt-refinement)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        dt = np.array(temp_data["dt_list"])
        e_dt = np.array(temp_data.get("errors_l2_dt", temp_data.get("errors_dt")))
        p_dt = np.mean(temp_data["rates_dt"][-2:])

        ax.loglog(dt, e_dt, "o-", color="#2ca02c", label=rf"BDF1 error (rate $\approx {p_dt:.2f}$)")
        ax.loglog(dt, e_dt[0] * (dt / dt[0]), ":", color="gray", label=r"Ideal $\mathcal{O}(\Delta t)$")
        ax.set_xlabel(r"Time Step $\Delta t$")
        ax.set_ylabel(r"$L^2$ Self-Convergence Error")
        ax.set_title("Fig. 2: Temporal Convergence (Backward Euler)")
        ax.grid(True, which="both")
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig2_temporal_convergence.png", dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # FIGURA 3: Dynamic PDE Error over Time (4 CURVE)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        time_arr = self.results["B0_fixed"]["time"]
        colors = {
            "B0_fixed": "#7f7f7f",
            "B1_gnn_p1": "#ff7f0e",
            "B2_gnn_l2": "#1f77b4",
            "B3_interp_oracle_l2": "#2ca02c"
        }
        labels = {
            "B0_fixed": "Fixed Mesh (B0)",
            "B1_gnn_p1": "GDM w/ P1 Remap (B1)",
            "B2_gnn_l2": "GDM w/ L2-Projection (B2)",
            "B3_interp_oracle_l2": "Oracle w/ L2-Projection (B3)"
        }

        for k in ["B0_fixed", "B1_gnn_p1", "B2_gnn_l2", "B3_interp_oracle_l2"]:
            ax.plot(time_arr, np.array(self.results[k]["rel_l2_error"]) * 100, label=labels[k], color=colors[k])

        ax.set_xlabel("Time $t$ [s]")
        ax.set_ylabel("Continuous Relative $L^2$ Error [%]")
        ax.set_title("Fig. 3: Transient PDE Error Evolution: P1 vs L2-Projection")
        ax.grid(True)
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig3_dynamic_pde_error.png", dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # FIGURA 4: Mesh Safety over Time
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        eps = float(self.config["geometry"]["safety"]["epsilon"])
        ax.plot(time_arr, self.results["B2_gnn_l2"]["min_jacobian"], color="#1f77b4", label=r"GDM ($L^2$) Safe $\min_e J_e(t)$")
        ax.plot(time_arr, self.results["B3_interp_oracle_l2"]["min_jacobian"], color="#2ca02c", linestyle="--", label=r"Oracle ($L^2$) $\min_e J_e(t)$")
        ax.axhline(eps, color="red", linestyle=":", label=rf"Safety Threshold $\epsilon = {eps:.1e}$")
        ax.set_xlabel("Time $t$ [s]")
        ax.set_ylabel(r"Minimum Jacobian $\min_e J_e$")
        ax.set_title(r"Fig. 4: Mesh Validity Guarantee ($\min_{t,e} J_e \geq \epsilon$)")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig4_mesh_safety_over_time.png", dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # FIGURA 5: Safety Activation Alpha over Time
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(time_arr, self.results["B2_gnn_l2"]["alpha_global"], "o-", markersize=3, color="#1f77b4", label=r"$\alpha_{\mathrm{global}}(t)$ (GDM $L^2$)")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
        ax.set_xlabel("Time $t$ [s]")
        ax.set_ylabel(r"Contraction Factor $\alpha_{\mathrm{global}}$")
        ax.set_title(r"Fig. 5: Safety Filter Scaling Dynamics")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True)
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig5_safety_activation.png", dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # FIGURA 6: Runtime Decomposition (B2 GDM L2 vs B3 Oracle L2)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        r_gnn = self.results["B2_gnn_l2"]
        r_orc = self.results["B3_interp_oracle_l2"]

        comp_labels = ["FEM Solve", "Adapt Logic", "Safety Filter", "Remap $L^2$"]
        times_gdm = [
            np.mean(r_gnn["timing_fem_ms"]),
            np.mean([t for t in r_gnn["timing_gnn_ms"] if t > 0]),
            np.mean([t for t in r_gnn["timing_safety_ms"] if t > 0]),
            np.mean([t for t in r_gnn["timing_remap_ms"] if t > 0])
        ]
        times_orc = [
            np.mean(r_orc["timing_fem_ms"]),
            np.mean([t for t in r_orc["timing_gnn_ms"] if t > 0]),
            0.0,
            np.mean([t for t in r_orc["timing_remap_ms"] if t > 0])
        ]

        x_pos = np.arange(len(comp_labels))
        width = 0.35
        ax.bar(x_pos - width/2, times_gdm, width, label="GDM w/ L2-Proj", color="#1f77b4")
        ax.bar(x_pos + width/2, times_orc, width, label="Oracle w/ L2-Proj", color="#2ca02c")
        ax.set_ylabel("Execution Time per Adaptation [ms]")
        ax.set_title("Fig. 6: Algorithmic Runtime Decomposition")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(comp_labels)
        ax.set_yscale("log")
        ax.grid(True, axis="y")
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig6_runtime_decomposition.png", dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # FIGURA 7: Qualitative Snapshots
        # -------------------------------------------------------------
        fig, axes = plt.subplots(3, 4, figsize=(12.0, 8.5))
        init_m = create_initial_mesh_2d(self.domain, nx=self.nx, ny=self.ny)

        snaps_fixed = self.results["B0_fixed"]["mesh_snapshots"]
        snaps_gnn = self.results["B2_gnn_l2"]["mesh_snapshots"]
        snaps_orc = self.results["B3_interp_oracle_l2"]["mesh_snapshots"]

        for col, idx in enumerate(range(len(snaps_gnn))):
            t_curr, nodes_fix, _ = snaps_fixed[idx]
            _, nodes_gnn, _ = snaps_gnn[idx]
            _, nodes_orc, _ = snaps_orc[idx]

            tri_f = mtri.Triangulation(nodes_fix[:, 0], nodes_fix[:, 1], init_m.cells)
            tri_g = mtri.Triangulation(nodes_gnn[:, 0], nodes_gnn[:, 1], init_m.cells)
            tri_o = mtri.Triangulation(nodes_orc[:, 0], nodes_orc[:, 1], init_m.cells)

            axes[0, col].triplot(tri_f, "k-", lw=0.4)
            axes[0, col].set_title(f"Fixed: $t={t_curr:.2f}$ s")

            axes[1, col].triplot(tri_g, "b-", lw=0.4)
            axes[1, col].set_title(f"GDM: $t={t_curr:.2f}$ s")

            axes[2, col].triplot(tri_o, "g-", lw=0.4)
            axes[2, col].set_title(f"Oracle: $t={t_curr:.2f}$ s")

            for r in range(3):
                axes[r, col].set_aspect("equal")
                axes[r, col].set_xticks([])
                axes[r, col].set_yticks([])

        fig.suptitle("Fig. 7: Dynamic Mesh Tracking of Moving Front", fontsize=14)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig7_qualitative_dynamic_mesh.png", dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # FIGURA 8: PDE Error Field Distribution at t = T
        # -------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
        final_mesh_gnn = self.results["B2_gnn_l2"]["final_mesh"]
        u_fem_final = self.results["B2_gnn_l2"]["final_solution"]
        u_exact_final = self.pde.evaluate_solution(final_mesh_gnn.nodes[:, 0], final_mesh_gnn.nodes[:, 1], self.T)
        abs_err = np.abs(u_fem_final - u_exact_final)

        tri_final = mtri.Triangulation(final_mesh_gnn.nodes[:, 0], final_mesh_gnn.nodes[:, 1], final_mesh_gnn.cells)

        c1 = axes[0].tricontourf(tri_final, u_exact_final, levels=30, cmap="viridis")
        axes[0].set_title(r"Exact Solution $u_{\mathrm{exact}}(X, T)$")
        fig.colorbar(c1, ax=axes[0])

        c2 = axes[1].tricontourf(tri_final, u_fem_final, levels=30, cmap="viridis")
        axes[1].set_title(r"GDM Numerical $u_h^N(X)$")
        fig.colorbar(c2, ax=axes[1])

        c3 = axes[2].tricontourf(tri_final, abs_err, levels=30, cmap="magma")
        axes[2].set_title(r"Absolute Error $|u_h^N - u_{\mathrm{exact}}|$")
        fig.colorbar(c3, ax=axes[2])

        for ax in axes:
            ax.set_aspect("equal")
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([0.0, 0.5, 1.0])

        fig.suptitle("Fig. 8: Solution Field and Error Distribution at Final Time $T=0.5$", fontsize=13)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig8_pde_error_field.png", dpi=300)
        plt.close(fig)

        print("[INFO] Tutte le figure scientifiche aggiornate salvate in results/stage_b/figures/")


def main():
    campaign = StageBCampaign()
    campaign.run_benchmark()


if __name__ == "__main__":
    main()