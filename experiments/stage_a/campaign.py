"""
experiments/stage_a/campaign.py
Campagna Sperimentale Completa Stage A orientata alla pubblicazione scientifica.

Esegue i 5 protocolli di test formali:
    T1: Benchmark principale multi-baseline (Fixed, Zero, GNN-Raw, GNN-Safety, stored optimizer target)
    T2: Safety Stress Test multi-scala (k in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    T3: Runtime Scaling rigoroso (CPU vs CPU: mediana, p95 e scaling con DoF N)
    T4: Ablation Study della formulazione della loss (Target, Warmup, Full)
    T5: Robustezza e degradazione Out-Of-Distribution (OOD sweep su sigma)

Genera le 5 figure scientifiche e le 2 visualizzazioni qualitative per il paper:
    Fig 1: E_interp vs Parametro OOD (sigma)
    Fig 2: Gap reduction versus quality-accepted finite-budget reference targets, where available
    Fig 3: Safety Curve (alpha_global e min(J) vs moltiplicatore k)
    Fig 4: Runtime Scaling (N nodi vs Tempo [ms] in scala log-log)
    Fig 5: Ablation Plot delle varianti di loss
    Vis 1: Mappe qualitative Mesh Deformata + Mappa Errore (Fixed vs GNN vs Oracle)
    Vis 2: Stress Visivo con celle invertite evidenziate in rosso (Raw k=5 vs Safety Filter)
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Inclusione radice progetto nel path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "python") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "python"))

from utils.utils import load_config, ensure_dir, setup_logger, Timer, load_pickle
from gdm import NativeMeshAdapter, compute_native_mesh_metrics_arrays, apply_native_safety_filter
from oracle.field import GaussianBumpField
from oracle.objective import OracleObjective
from oracle.optimizer import OfflineMeshOptimizer
from inference import MeshMoverInferenceEngine
from plotting.stage_a import (
    plot_safety_curve, plot_runtime_scaling, plot_ood_error,
    plot_gap_reduction, plot_loss_ablation, plot_mesh_comparison,
    plot_safety_stress,
)


class StageAPaperCampaign:
    """Orchestratore della campagna sperimentale dello Stage A."""

    def __init__(self, config_path: str = "config/stage_a.yaml"):
        self.config = load_config(ROOT_DIR / config_path)
        self.output_dir = ensure_dir(ROOT_DIR / self.config["experiment"]["output_dir"] / "paper_figures")
        self.logger = setup_logger("StageA_Campaign", log_file=self.output_dir / "campaign.log")
        
        checkpoint_path = ROOT_DIR / self.config["experiment"]["output_dir"] / "checkpoints" / "best_model.pt"
        assert checkpoint_path.exists(), f"Checkpoint non trovato in {checkpoint_path}. Eseguire prima training/train.py!"
        
        # Motore di inferenza CPU per comparazione equa CPU vs CPU con SciPy
        self.engine = MeshMoverInferenceEngine(
            checkpoint_path=checkpoint_path,
            config=self.config,
            device="cpu"
        )
        
        self.epsilon = float(self.config["geometry"]["safety"]["epsilon"])
        self.base_mesh = NativeMeshAdapter.from_domain(
            self.config["geometry"]["domain"],
            nx=int(self.config["geometry"]["initial_grid"]["nx"]),
            ny=int(self.config["geometry"]["initial_grid"]["ny"]),
        )
        self.logger.info("Setup della campagna sperimentale completato con successo.")

    # =========================================================================
    # T1: BENCHMARK PRINCIPALE
    # =========================================================================
    def run_t1_baseline_benchmark(self) -> Dict[str, Any]:
        self.logger.info("\n--- Esecuzione T1: Benchmark Principale Multi-Baseline ---")
        manifest_path = ROOT_DIR / self.config["dataset"]["manifest_path"]
        raw_dir = ROOT_DIR / self.config["dataset"]["raw_dir"]
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        results = {}
        for split in ["test_interp", "test_extrap"]:
            results[split] = {
                "B0_fixed": [], "B_zero": [], "B3_raw": [], "B4_safe": [], "B5_oracle": [],
                "target_accepted_flags": [],
                "B0_accepted_target_subset": [], "B3_accepted_target_subset": [],
                "B4_accepted_target_subset": [], "B5_accepted_target_subset": [],
            }
            for sid in manifest[split]:
                sample = load_pickle(raw_dir / f"{sid}.pkl")
                nodes_0 = sample["mesh"]["nodes"]
                cells = sample["mesh"]["cells"]
                u_0 = sample["state"]["u"]
                grad_u_0 = sample["state"]["grad_u"]
                p = sample["metadata"]["parameters"]
                opt_nodes = sample["target"].get("reference_nodes", sample["target"].get("optimal_nodes"))

                field = GaussianBumpField(p["mu_x"], p["mu_y"], p["sigma"], p["amplitude"])
                evaluator = OracleObjective(
                    reference_field=field,
                    initial_nodes=nodes_0,
                    cells=cells,
                    boundary_nodes=self.base_mesh.boundary_nodes,
                    lambda_D=self.config["oracle"]["objective"]["lambda_D"],
                    lambda_Q=self.config["oracle"]["objective"]["lambda_Q"],
                    epsilon=self.epsilon
                )

                # B0 / B_zero
                res_b0 = evaluator.evaluate(nodes_0)
                e_fixed = res_b0["interpolation_error"]
                results[split]["B0_fixed"].append(e_fixed)
                results[split]["B_zero"].append(e_fixed)

                # B5 Oracle
                res_b5 = evaluator.evaluate(opt_nodes)
                e_oracle = res_b5["interpolation_error"]
                results[split]["B5_oracle"].append(e_oracle)
                target_accepted = bool(sample["metadata"].get("target_accepted", sample["metadata"].get("converged", False)))
                results[split]["target_accepted_flags"].append(target_accepted)
                if target_accepted:
                    results[split]["B0_accepted_target_subset"].append(e_fixed)
                    results[split]["B5_accepted_target_subset"].append(e_oracle)

                # GNN
                pred = self.engine.predict(self.base_mesh, u_0, grad_u_0)
                raw_nodes = nodes_0 + pred["predicted_displacement"]
                res_raw = evaluator.evaluate(raw_nodes)
                res_safe = evaluator.evaluate(pred["new_nodes"])

                results[split]["B3_raw"].append(res_raw["interpolation_error"])
                results[split]["B4_safe"].append(res_safe["interpolation_error"])
                if target_accepted:
                    results[split]["B3_accepted_target_subset"].append(res_raw["interpolation_error"])
                    results[split]["B4_accepted_target_subset"].append(res_safe["interpolation_error"])

        # Report only on the common subset whose stored finite-budget reference targets
        # satisfy the explicit quality-acceptance contract. This is not a certified optimum.
        table_path = self.output_dir / "table_T1_baseline.tex"
        split_labels = {"test_interp": "In-domain", "test_extrap": "Extrapolation"}
        split_summary = {}
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(r"\begin{table}[t]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\small" + "\n")
            f.write(r"\caption{Stage-A interpolation error on the common subset of quality-accepted finite-budget reference targets. These targets are not certified stationary or global optima.}" + "\n")
            f.write(r"\label{tab:stage_a_accepted_target_subset}" + "\n")
            f.write(r"\begin{tabular}{llccc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Test split & Baseline & $n$ & Mean $E_{\mathrm{interp}}$ ($\times 10^{-4}$) & Target-gap reduction [\%] " + r"\\" + "\n")
            f.write(r"\midrule" + "\n")

            for split, split_label in split_labels.items():
                base = results[split]
                total = len(base["target_accepted_flags"])
                n_accepted = int(np.sum(base["target_accepted_flags"]))
                means = {}
                subset_keys = [
                    ("Fixed mesh ($B_0$)", "B0_accepted_target_subset"),
                    ("GNN, unfiltered ($B_3$)", "B3_accepted_target_subset"),
                    ("GNN + safety filter ($B_4$)", "B4_accepted_target_subset"),
                    ("Stored optimizer target ($B_5$)", "B5_accepted_target_subset"),
                ]
                if n_accepted == 0:
                    split_summary[split] = {"n_total": total, "n_accepted_targets": 0, "means_x1e4": {}, "target_gap_reduction_pct": None}
                    continue
                for _, key in subset_keys:
                    means[key] = float(np.mean(base[key])) if base[key] else float("nan")
                fixed, safe, target = means["B0_accepted_target_subset"], means["B4_accepted_target_subset"], means["B5_accepted_target_subset"]
                denom = fixed - target
                gap = (fixed - safe) / denom * 100.0 if denom > 1e-12 else float("nan")
                split_summary[split] = {
                    "n_total": total, "n_accepted_targets": n_accepted,
                    "means_x1e4": {key: value * 1e4 for key, value in means.items()},
                    "target_gap_reduction_pct": gap if np.isfinite(gap) else None,
                }
                for label, key in subset_keys:
                    error = means[key] * 1e4
                    gap_cell = f"{gap:.1f}" if label.startswith("GNN + safety") and np.isfinite(gap) else "--"
                    f.write(f"{split_label} & {label} & {n_accepted}/{total} & {error:.3f} & {gap_cell} " + r"\\" + "\n")
                f.write(r"\midrule" + "\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table}" + "\n")

        (self.output_dir / "t1_accepted_target_summary.json").write_text(
            json.dumps(split_summary, indent=2), encoding="utf-8"
        )

        self.logger.info(f"T1 completed. LaTeX table saved to {table_path}")
        return results

    # =========================================================================
    # T2 & FIGURA 3: SAFETY STRESS TEST & SAFETY CURVE
    # =========================================================================
    def run_t2_safety_stress_test(self) -> Dict[str, Any]:
        self.logger.info("\n--- Esecuzione T2: Safety Stress Test Multi-Scala ---")
        field = GaussianBumpField(mu_x=0.5, mu_y=0.5, sigma=0.04, amplitude=1.0)
        u, grad_u, _, _ = field.evaluate(self.base_mesh.nodes[:, 0], self.base_mesh.nodes[:, 1])
        pred = self.engine.predict(self.base_mesh, u, grad_u)
        base_disp = pred["predicted_displacement"]

        k_values = np.linspace(1.0, 10.0, 25)
        alphas = []
        raw_min_j = []
        safe_min_j = []
        raw_inverted = []

        for k in k_values:
            disp_k = k * base_disp
            raw_nodes = self.base_mesh.nodes + disp_k
            m_raw = compute_native_mesh_metrics_arrays(raw_nodes, self.base_mesh.cells)
            raw_min_j.append(np.min(m_raw["jacobians"]))
            raw_inverted.append(np.sum(m_raw["inverted"]))

            filtered = apply_native_safety_filter(
                self.base_mesh, disp_k, epsilon=self.epsilon, lock_boundary_nodes=True
            )
            alphas.append(float(filtered["alpha_global"]))
            safe_min_j.append(float(filtered["min_jacobian_after"]))

        fig_path = self.output_dir / "fig3_safety_curve.pdf"
        outputs = plot_safety_curve(k_values, alphas, raw_min_j, safe_min_j, self.epsilon, fig_path)
        self.logger.info("Safety curve exported: %s", [str(p) for p in outputs])

        return {"k": k_values, "alpha": alphas, "raw_min_j": raw_min_j, "safe_min_j": safe_min_j}

    # =========================================================================
    # T3 & FIGURA 4: RUNTIME SCALING CPU vs CPU
    # =========================================================================
    def run_t3_runtime_scaling(self) -> None:
        self.logger.info("\n--- Esecuzione T3: Runtime Scaling Rigoroso CPU vs CPU ---")
        resolutions = [16, 24, 32, 48, 64]
        n_nodes_list = []
        t_gnn_med, t_gnn_p95 = [], []
        t_safety_med, t_safety_p95 = [], []
        t_oracle_list = []

        field = GaussianBumpField(mu_x=0.5, mu_y=0.5, sigma=0.05, amplitude=1.0)

        for nx in resolutions:
            mesh = NativeMeshAdapter.from_domain(self.config["geometry"]["domain"], nx=nx, ny=nx)
            N = mesh.num_nodes
            n_nodes_list.append(N)
            u, grad_u, _, _ = field.evaluate(mesh.nodes[:, 0], mesh.nodes[:, 1])

            # 1. Benchmark GNN + safety on CPU (5 warm-up, 30 timed runs)
            for _ in range(5):
                self.engine.predict(mesh, u, grad_u)

            timings_gnn, timings_safety = [], []
            for _ in range(30):
                res = self.engine.predict(mesh, u, grad_u)
                timings_gnn.append(res["timing"]["gnn_time"] * 1000.0)
                timings_safety.append(res["timing"]["safety_time"] * 1000.0)

            t_gnn_med.append(np.median(timings_gnn))
            t_gnn_p95.append(np.percentile(timings_gnn, 95))
            t_safety_med.append(np.median(timings_safety))
            t_safety_p95.append(np.percentile(timings_safety, 95))

            # 2. Benchmark Oracle Offline su CPU (misura diretta di L-BFGS-B)
            opt = OfflineMeshOptimizer(
                reference_field=field,
                initial_nodes=mesh.nodes,
                cells=mesh.cells,
                boundary_nodes=mesh.boundary_nodes,
                lambda_D=self.config["oracle"]["objective"]["lambda_D"],
                lambda_Q=self.config["oracle"]["objective"]["lambda_Q"],
                epsilon=self.epsilon,
                max_iter=3
            )
            with Timer() as t_opt:
                opt.optimize()
            t_oracle_list.append(t_opt.elapsed_ms)
            self.logger.info(f"N={N:4d} | GNN (med): {t_gnn_med[-1]:.2f} ms | Safety: {t_safety_med[-1]:.2f} ms | Oracle: {t_oracle_list[-1]:.2f} ms")

        fig_path = self.output_dir / "fig4_runtime_scaling.pdf"
        outputs = plot_runtime_scaling(n_nodes_list, t_oracle_list, t_gnn_med, t_safety_med, fig_path)
        self.logger.info("Runtime-scaling figures exported: %s", [str(p) for p in outputs])

    # =========================================================================
    # T5 & FIGURE 1, 2: ROBUSTEZZA OUT-OF-DISTRIBUTION (OOD SWEEP)
    # =========================================================================
    def run_t5_ood_sweep(self) -> None:
        self.logger.info("\n--- Esecuzione T5: OOD Robustness Sweep su Sigma ---")
        sigmas = np.linspace(0.018, 0.09, 15)
        e_fixed_list, e_gnn_list, e_oracle_list = [], [], []
        r_oracle_list = []

        for s in sigmas:
            field = GaussianBumpField(mu_x=0.5, mu_y=0.5, sigma=s, amplitude=1.0)
            u, grad_u, _, _ = field.evaluate(self.base_mesh.nodes[:, 0], self.base_mesh.nodes[:, 1])

            evaluator = OracleObjective(
                reference_field=field,
                initial_nodes=self.base_mesh.nodes,
                cells=self.base_mesh.cells,
                boundary_nodes=self.base_mesh.boundary_nodes,
                lambda_D=self.config["oracle"]["objective"]["lambda_D"],
                lambda_Q=self.config["oracle"]["objective"]["lambda_Q"],
                epsilon=self.epsilon
            )

            # Fixed
            e_f = evaluator.evaluate(self.base_mesh.nodes)["interpolation_error"]
            e_fixed_list.append(e_f)

            # GNN + Safety
            pred = self.engine.predict(self.base_mesh, u, grad_u)
            e_g = evaluator.evaluate(pred["new_nodes"])["interpolation_error"]
            e_gnn_list.append(e_g)

            # Oracle
            opt = OfflineMeshOptimizer(
                reference_field=field,
                initial_nodes=self.base_mesh.nodes,
                cells=self.base_mesh.cells,
                boundary_nodes=self.base_mesh.boundary_nodes,
                lambda_D=self.config["oracle"]["objective"]["lambda_D"],
                lambda_Q=self.config["oracle"]["objective"]["lambda_Q"],
                epsilon=self.epsilon,
                max_iter=3
            )
            opt_res = opt.optimize()
            e_o = opt_res["final_metrics"]["interpolation_error"]
            e_oracle_list.append(e_o)

            if bool(opt_res.get("converged", False)) and (e_f - e_o) > 1e-12:
                r_rec = (e_f - e_g) / (e_f - e_o) * 100.0
            else:
                r_rec = float("nan")
            r_oracle_list.append(r_rec)

        fig1_path = self.output_dir / "fig1_error_vs_ood.pdf"
        outputs1 = plot_ood_error(
            sigmas, e_fixed_list, e_gnn_list, e_oracle_list, fig1_path,
            training_range=(0.03, 0.08), ood_range=(0.018, 0.03),
        )
        self.logger.info("OOD error figures exported: %s", [str(p) for p in outputs1])

        # The 3-iteration optimizer used in this diagnostic is only a low-budget
        # reference proposal, not an oracle upper bound. Report a gap only when
        # the strict first-order stationarity diagnostic is actually satisfied.
        valid_ratios = np.asarray(r_oracle_list, dtype=float)
        valid_ratios[~np.isfinite(valid_ratios)] = np.nan
        if np.isfinite(valid_ratios).any():
            fig2_path = self.output_dir / "fig2_target_gap_reduction.pdf"
            outputs2 = plot_gap_reduction(sigmas, valid_ratios, fig2_path, ood_range=(0.018, 0.03))
            self.logger.info("Target-gap figures exported: %s", [str(p) for p in outputs2])
        else:
            self.logger.warning(
                "Target-gap figure skipped: no first-order stationary target is available in the low-budget OOD sweep."
            )

    # =========================================================================
    # T4 & FIGURA 5: ABLATION DELLA LOSS FORMULATION
    # =========================================================================
    def generate_fig5_ablation(self) -> None:
        """Plot an ablation only when a machine-readable result file exists.

        Previously the figure used hard-coded historical values, which cannot
        be tied to this run. Do not regenerate that as if it were fresh evidence.
        """
        result_path = self.output_dir.parent / "ablation_summary.json"
        if not result_path.is_file():
            self.logger.warning(
                "Ablation figure skipped: %s is missing. Export results from a reproducible ablation run first.",
                result_path,
            )
            return
        import json
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        required = {"variants", "errors_in_domain", "gap_reduction_pct"}
        if not required.issubset(payload):
            raise ValueError(f"Ablation summary must contain keys {sorted(required)}: {result_path}")
        fig_path = self.output_dir / "fig5_ablation_loss.pdf"
        outputs = plot_loss_ablation(
            payload["variants"], payload["errors_in_domain"],
            payload["gap_reduction_pct"], fig_path,
        )
        self.logger.info("Ablation figures exported: %s", [str(p) for p in outputs])

    # =========================================================================
    # VISUALIZZAZIONI QUALITATIVE 1 & 2
    # =========================================================================
    def generate_qualitative_visualizations(self) -> None:
        self.logger.info("\n--- Generazione Visualizzazioni Qualitative (Vis 1 & Vis 2) ---")
        field = GaussianBumpField(mu_x=0.55, mu_y=0.55, sigma=0.045, amplitude=1.0)
        u, grad_u, _, _ = field.evaluate(self.base_mesh.nodes[:, 0], self.base_mesh.nodes[:, 1])

        pred = self.engine.predict(self.base_mesh, u, grad_u)
        gnn_nodes = pred["new_nodes"]

        opt = OfflineMeshOptimizer(
            reference_field=field,
            initial_nodes=self.base_mesh.nodes,
            cells=self.base_mesh.cells,
            boundary_nodes=self.base_mesh.boundary_nodes,
            lambda_D=self.config["oracle"]["objective"]["lambda_D"],
            lambda_Q=self.config["oracle"]["objective"]["lambda_Q"],
            epsilon=self.epsilon,
            max_iter=3
        )
        oracle_nodes = opt.optimize()["optimal_nodes"]

        # Qualitative mesh comparison is delegated to the plotting package.
        meshes = [
            ("Fixed mesh", self.base_mesh.nodes, u),
            ("GNN + safety filter", gnn_nodes, field.evaluate(gnn_nodes[:, 0], gnn_nodes[:, 1])[0]),
            ("Stored optimizer target (3-iteration budget)", oracle_nodes, field.evaluate(oracle_nodes[:, 0], oracle_nodes[:, 1])[0]),
        ]
        vis1_path = self.output_dir / "vis1_mesh_comparison.pdf"
        outputs1 = plot_mesh_comparison(meshes, self.base_mesh.cells, vis1_path)

        # VISUALIZATION 2: stress test; report inversion count explicitly.
        disp_stress = 6.0 * pred["predicted_displacement"]
        raw_stress_nodes = self.base_mesh.nodes + disp_stress
        m_stress_raw = compute_native_mesh_metrics_arrays(raw_stress_nodes, self.base_mesh.cells)
        inv_idx = np.where(m_stress_raw["inverted"])[0]
        stress_filter = apply_native_safety_filter(
            self.base_mesh, disp_stress, epsilon=self.epsilon, lock_boundary_nodes=True
        )
        alpha_s = float(stress_filter["alpha_global"])
        safe_stress_nodes = np.asarray(stress_filter["safe_nodes"], dtype=np.float64)
        vis2_path = self.output_dir / "vis2_safety_stress_visual.pdf"
        outputs2 = plot_safety_stress(
            raw_stress_nodes, safe_stress_nodes, self.base_mesh.cells, inv_idx, alpha_s, vis2_path
        )
        self.logger.info("Qualitative figures exported: %s; %s", [str(p) for p in outputs1], [str(p) for p in outputs2])


def main():
    campaign = StageAPaperCampaign()
    campaign.run_t1_baseline_benchmark()
    campaign.run_t2_safety_stress_test()
    campaign.run_t3_runtime_scaling()
    campaign.run_t5_ood_sweep()
    campaign.generate_fig5_ablation()
    campaign.generate_qualitative_visualizations()
    print("\n" + "=" * 80)
    print("CAMPAGNA SPERIMENTALE STAGE A COMPLETATA CON SUCCESSO!")
    print(f"Tutte le figure e le tabelle LaTeX sono in: {campaign.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()