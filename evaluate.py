"""
evaluate.py
Entry point per la validazione e il benchmark scientifico di Graph-Dynamic Meshing (GDM).

Risponde formalmente alla domanda:
    "La predizione della GNN ha valore numerico, geometrico e scientifico?"

Confronta sistematicamente le seguenti configurazioni di baseline (Stage A):
    - B0: Fixed Mesh (Griglia uniforme iniziale, nessun adattamento)
    - B3: GNN Mesh Mover Raw (Spostamento predetto senza filtro di sicurezza)
    - B4: GNN + Geometric Safety Filter (Nostro framework completo protetto O(M))
    - B5: Oracle Reference (Configurazione ottima variazionale calcolata offline)

Genera:
    - Tabelle comparative a terminale e formattate in LaTeX per il paper
    - Dati grezzi aggregati in formato JSON (results/benchmark_summary.json)
    - Grafici di confronto scientifico (results/benchmark_plots.png)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import time
import matplotlib.pyplot as plt

# Inclusione della cartella root nel path di sistema
sys.path.append(str(Path(__file__).resolve().parent))

from geometry.mesh import TriangularMesh2D
from geometry.geometry import compute_mesh_metrics, compute_mesh_quality_summary
from geometry.safety import apply_safety_filter
from oracle.field import GaussianBumpField
from oracle.objective import OracleObjective
from evaluation.metrics import (
    compute_displacement_metrics,
    compute_mesh_geometric_metrics,
    format_computational_timings
)
from inference import MeshMoverInferenceEngine
from utils.utils import load_config, ensure_dir, setup_logger, Timer, load_pickle


class BenchmarkSuite:
    """
    Suite di validazione scientifica per il confronto multi-baseline su test set.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: str | Path,
        output_dir: str | Path = "results"
    ):
        self.config = config
        self.output_dir = ensure_dir(output_dir)
        self.logger = setup_logger("BenchmarkSuite", log_file=self.output_dir / "benchmark.log")

        # Inizializzazione motore inferenza GNN
        self.engine = MeshMoverInferenceEngine(
            checkpoint_path=checkpoint_path,
            config=self.config
        )

        self.geom_cfg = self.config["geometry"]["safety"]
        self.epsilon = float(self.geom_cfg["epsilon"])
        self.oracle_cfg = self.config["oracle"]

    def _evaluate_configuration(
        self,
        mesh: TriangularMesh2D,
        field: GaussianBumpField,
        base_nodes: np.ndarray,
        boundary_nodes: np.ndarray,
        baseline_name: str
    ) -> Dict[str, Any]:
        """
        Calcola l'errore di interpolazione integrato e le metriche geometriche per una specifica mesh.
        """
        # Valutatore variazionale dell'errore funzionale integrato L2
        objective_evaluator = OracleObjective(
            reference_field=field,
            initial_nodes=base_nodes,
            cells=mesh.cells,
            boundary_nodes=boundary_nodes,
            lambda_D=self.oracle_cfg["objective"]["lambda_D"],
            lambda_Q=self.oracle_cfg["objective"]["lambda_Q"],
            epsilon=self.epsilon
        )

        eval_res = objective_evaluator.evaluate(mesh.nodes)
        geom_metrics = compute_mesh_geometric_metrics(mesh)

        return {
            "baseline": baseline_name,
            "interpolation_error": float(eval_res["interpolation_error"]),
            "total_objective": float(eval_res["total_objective"]),
            "is_valid": bool(eval_res["is_valid"]),
            "min_jacobian": float(geom_metrics["min_jacobian"]),
            "mean_jacobian": float(geom_metrics["mean_jacobian"]),
            "inverted_cells": int(geom_metrics["inverted_cells"]),
            "mean_mesh_quality": float(geom_metrics["mean_mesh_quality"]),
            "max_aspect_ratio": float(geom_metrics["max_aspect_ratio"])
        }

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Esegue la campagna sperimentale su tutti i campioni di test (Interpolation & Extrapolation).
        """
        manifest_path = Path(self.config["dataset"]["manifest_path"])
        raw_dir = Path(self.config["dataset"]["raw_dir"])
        assert manifest_path.exists(), f"Manifest non trovato: {manifest_path.resolve()}"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        splits_to_test = ["test_interp", "test_extrap"]
        aggregated_results: Dict[str, Dict[str, List[float]]] = {}

        baselines = ["B0_fixed", "B_zero", "B3_gnn_raw", "B4_gnn_safety", "B5_oracle"]
        for s in splits_to_test:
            aggregated_results[s] = {
                b: {
                    "interp_error": [],
                    "min_jacobian": [],
                    "mean_quality": [],
                    "inverted_cells": [],
                    "disp_l2_error": [],
                    "disp_rel_error": [],
                    "timing_ms": [],
                    "oracle_recovered_ratio": []
                }
                for b in baselines
            }

        total_tested = 0

        for split_name in splits_to_test:
            sample_ids = manifest.get(split_name, [])
            self.logger.info(f"Avvio test su split '{split_name}' ({len(sample_ids)} campioni)...")

            for sid in sample_ids:
                sample_file = raw_dir / f"{sid}.pkl"
                sample = load_pickle(sample_file)

                # Dati del campione
                nodes_0 = sample["mesh"]["nodes"]
                cells = sample["mesh"]["cells"]
                u_0 = sample["state"]["u"]
                grad_u_0 = sample["state"]["grad_u"]
                opt_nodes = sample["target"]["optimal_nodes"]
                target_disp = sample["target"]["displacement"]
                params = sample["metadata"]["parameters"]

                mesh_0 = TriangularMesh2D(nodes_0, cells)
                field = GaussianBumpField(
                    mu_x=params["mu_x"],
                    mu_y=params["mu_y"],
                    sigma=params["sigma"],
                    amplitude=params["amplitude"]
                )

                # -------------------------------------------------------------
                # Baseline B0: Fixed Mesh
                # -------------------------------------------------------------
                res_b0 = self._evaluate_configuration(mesh_0, field, nodes_0, mesh_0.boundary_nodes, "B0_fixed")
                e_fixed = res_b0["interpolation_error"]
                aggregated_results[split_name]["B0_fixed"]["interp_error"].append(e_fixed)
                aggregated_results[split_name]["B0_fixed"]["min_jacobian"].append(res_b0["min_jacobian"])
                aggregated_results[split_name]["B0_fixed"]["mean_quality"].append(res_b0["mean_mesh_quality"])
                aggregated_results[split_name]["B0_fixed"]["inverted_cells"].append(res_b0["inverted_cells"])
                aggregated_results[split_name]["B0_fixed"]["timing_ms"].append(0.0)

                # -------------------------------------------------------------
                # Baseline B_zero: Zero Displacement Predictor (Delta X = 0)
                # -------------------------------------------------------------
                disp_zero = np.zeros_like(target_disp)
                disp_m_zero = compute_displacement_metrics(disp_zero, target_disp)
                aggregated_results[split_name]["B_zero"]["interp_error"].append(e_fixed)
                aggregated_results[split_name]["B_zero"]["min_jacobian"].append(res_b0["min_jacobian"])
                aggregated_results[split_name]["B_zero"]["mean_quality"].append(res_b0["mean_mesh_quality"])
                aggregated_results[split_name]["B_zero"]["inverted_cells"].append(res_b0["inverted_cells"])
                aggregated_results[split_name]["B_zero"]["disp_l2_error"].append(disp_m_zero["displacement_l2_error"])
                aggregated_results[split_name]["B_zero"]["disp_rel_error"].append(disp_m_zero["displacement_relative_error"])
                aggregated_results[split_name]["B_zero"]["timing_ms"].append(0.0)

                # -------------------------------------------------------------
                # Baseline B5: Oracle Reference
                # -------------------------------------------------------------
                mesh_oracle = mesh_0.update_coordinates(opt_nodes)
                res_b5 = self._evaluate_configuration(mesh_oracle, field, nodes_0, mesh_0.boundary_nodes, "B5_oracle")
                e_oracle = res_b5["interpolation_error"]
                
                # Lettura del timing salvato nei metadati; se assente calcolo offline reale
                oracle_t_raw = sample["metadata"]["oracle_metrics"].get("oracle_time_seconds", None)
                if oracle_t_raw is None:
                    oracle_t_raw = sample["metadata"]["oracle_metrics"].get("time_seconds", None)
                
                if oracle_t_raw is None:
                    # Stima diretta del tempo reale su un campione per split se non registrato in precedenza
                    if total_tested == 0:
                        from oracle.optimizer import OfflineMeshOptimizer
                        opt_probe = OfflineMeshOptimizer(
                            reference_field=field,
                            initial_nodes=nodes_0,
                            cells=cells,
                            boundary_nodes=mesh_0.boundary_nodes,
                            lambda_D=self.oracle_cfg["objective"]["lambda_D"],
                            lambda_Q=self.oracle_cfg["objective"]["lambda_Q"],
                            epsilon=self.epsilon,
                            max_iter=3
                        )
                        with Timer() as t_or_probe:
                            opt_probe.optimize()
                        oracle_t_ms = t_or_probe.elapsed_seconds * 1000.0
                        self._cached_oracle_time = oracle_t_ms
                    else:
                        oracle_t_ms = getattr(self, "_cached_oracle_time", 7500.0)
                else:
                    oracle_t_ms = float(oracle_t_raw) * 1000.0

                aggregated_results[split_name]["B5_oracle"]["interp_error"].append(e_oracle)
                aggregated_results[split_name]["B5_oracle"]["min_jacobian"].append(res_b5["min_jacobian"])
                aggregated_results[split_name]["B5_oracle"]["mean_quality"].append(res_b5["mean_mesh_quality"])
                aggregated_results[split_name]["B5_oracle"]["inverted_cells"].append(res_b5["inverted_cells"])
                aggregated_results[split_name]["B5_oracle"]["timing_ms"].append(oracle_t_ms)

                # -------------------------------------------------------------
                # Predizione GNN (Misurazione Completa T_online)
                # -------------------------------------------------------------
                t_online_start = time.perf_counter()
                pred_res = self.engine.predict(mesh_0, u_0, grad_u_0)
                t_online_ms = (time.perf_counter() - t_online_start) * 1000.0

                raw_pred_disp = pred_res["predicted_displacement"]
                safe_nodes = pred_res["new_nodes"]
                safe_disp = safe_nodes - nodes_0

                # Metriche distinte di displacement per raw e safe
                disp_m_raw = compute_displacement_metrics(raw_pred_disp, target_disp)
                disp_m_safe = compute_displacement_metrics(safe_disp, target_disp)

                # -------------------------------------------------------------
                # Baseline B3: GNN Raw
                # -------------------------------------------------------------
                nodes_b3 = nodes_0 + raw_pred_disp
                mesh_b3 = mesh_0.update_coordinates(nodes_b3)
                res_b3 = self._evaluate_configuration(mesh_b3, field, nodes_0, mesh_0.boundary_nodes, "B3_gnn_raw")

                aggregated_results[split_name]["B3_gnn_raw"]["interp_error"].append(res_b3["interpolation_error"])
                aggregated_results[split_name]["B3_gnn_raw"]["min_jacobian"].append(res_b3["min_jacobian"])
                aggregated_results[split_name]["B3_gnn_raw"]["mean_quality"].append(res_b3["mean_mesh_quality"])
                aggregated_results[split_name]["B3_gnn_raw"]["inverted_cells"].append(res_b3["inverted_cells"])
                aggregated_results[split_name]["B3_gnn_raw"]["disp_l2_error"].append(disp_m_raw["displacement_l2_error"])
                aggregated_results[split_name]["B3_gnn_raw"]["disp_rel_error"].append(disp_m_raw["displacement_relative_error"])
                aggregated_results[split_name]["B3_gnn_raw"]["timing_ms"].append(pred_res["timing"]["gnn_time"] * 1000.0)

                # -------------------------------------------------------------
                # Baseline B4: GNN + Safety Filter (Framework Proposto)
                # -------------------------------------------------------------
                mesh_b4 = pred_res["adapted_mesh"]
                res_b4 = self._evaluate_configuration(mesh_b4, field, nodes_0, mesh_0.boundary_nodes, "B4_gnn_safety")
                e_gnn = res_b4["interpolation_error"]

                gap_den = max(e_fixed - e_oracle, 1e-12)
                r_oracle = (e_fixed - e_gnn) / gap_den

                aggregated_results[split_name]["B4_gnn_safety"]["interp_error"].append(e_gnn)
                aggregated_results[split_name]["B4_gnn_safety"]["min_jacobian"].append(res_b4["min_jacobian"])
                aggregated_results[split_name]["B4_gnn_safety"]["mean_quality"].append(res_b4["mean_mesh_quality"])
                aggregated_results[split_name]["B4_gnn_safety"]["inverted_cells"].append(res_b4["inverted_cells"])
                aggregated_results[split_name]["B4_gnn_safety"]["disp_l2_error"].append(disp_m_safe["displacement_l2_error"])
                aggregated_results[split_name]["B4_gnn_safety"]["disp_rel_error"].append(disp_m_safe["displacement_relative_error"])
                aggregated_results[split_name]["B4_gnn_safety"]["timing_ms"].append(t_online_ms)
                aggregated_results[split_name]["B4_gnn_safety"]["oracle_recovered_ratio"].append(r_oracle)

                total_tested += 1

        self.logger.info(f"Campagna completata su {total_tested} istanze complessive.")
        return aggregated_results


# =============================================================================
# EXPORT TABELLE E REPORT SCIENTIFICO
# =============================================================================

def generate_summary_tables(
    results: Dict[str, Any],
    output_dir: Path,
    logger: Any
) -> Dict[str, Any]:
    """
    Formatta i risultati in tabelle scalari aggregate e genera il codice sorgente LaTeX.
    """
    summary: Dict[str, Any] = {}

    table_header = (
        f"{'Baseline':<16} | {'Interp Err (x1e-4)':<18} | {'R_oracle (%)':<14} | "
        f"{'Min Jacobian':<14} | {'Mean Quality':<12} | {'Time (ms)':<10}"
    )
    separator = "-" * len(table_header)

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Confronto Benchmark Multi-Baseline per Graph-Dynamic Meshing (Stage A).}",
        "\\label{tab:stage_a_benchmark}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "\\textbf{Metodo} & \\textbf{$E_{\\text{interp}} \\; (\\times 10^{-4})$} & \\textbf{$\\min J_e$} & \\textbf{Qualità $\\eta$} & \\textbf{Celle Inv.} & \\textbf{Tempo [ms]} \\\\",
        "\\hline"
    ]

    for split_name, baselines in results.items():
        logger.info(f"\n================ TABELLA: SPLIT {split_name.upper()} ================")
        logger.info(table_header)
        logger.info(separator)

        summary[split_name] = {}
        latex_lines.append(f"\\multicolumn{{6}}{{c}}{{\\textit{{{split_name.replace('_', ' ').capitalize()}}}}} \\\\")
        latex_lines.append("\\hline")

        for b_name, metrics in baselines.items():
            mean_err = np.mean(metrics["interp_error"]) * 1e4
            std_err = np.std(metrics["interp_error"]) * 1e4
            min_j = np.min(metrics["min_jacobian"])
            mean_q = np.mean(metrics["mean_quality"])
            mean_t = np.nanmean(metrics["timing_ms"]) if len(metrics["timing_ms"]) > 0 else np.nan
            inv_c = int(np.sum(metrics["inverted_cells"]))
            mean_t = np.mean(metrics["timing_ms"])

            summary[split_name][b_name] = {
                "mean_interp_error_x1e4": float(mean_err),
                "std_interp_error_x1e4": float(std_err),
                "min_jacobian": float(min_j),
                "mean_quality": float(mean_q),
                "total_inverted_cells": inv_c,
                "mean_time_ms": float(mean_t)
            }

            r_rec_list = metrics.get("oracle_recovered_ratio", [])
            r_rec_str = f"{np.mean(r_rec_list)*100:.1f}%" if len(r_rec_list) > 0 else "-"

            row_str = (
                f"{b_name:<16} | {mean_err:.2f} +/- {std_err:.2f}{'':<5} | {r_rec_str:<14} | "
                f"{min_j:+.4e}{'':<2} | {mean_q:.4f}{'':<6} | {mean_t:.2f}"
            )
            logger.info(row_str)

            # Generazione riga LaTeX
            b_label = b_name.replace("_", "\\_")
            latex_lines.append(
                f"{b_label} & {mean_err:.2f} $\\pm$ {std_err:.2f} & {min_j:.2e} & {mean_q:.3f} & {inv_c} & {mean_t:.2f} \\\\"
            )

        latex_lines.append("\\hline")

    latex_lines.extend([
        "\\end{tabular}",
        "\\end{table}"
    ])

    # Salvataggio tabella LaTeX
    latex_path = output_dir / "table_benchmark.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    logger.info(f"\nTabella LaTeX registrata con successo in: {latex_path.resolve()}")

    return summary


# =============================================================================
# GENERAZIONE PLOT SCIENTIFICI
# =============================================================================

def plot_benchmark_results(results: Dict[str, Any], output_path: Path):
    """
    Produce grafici diagnostici per evidenziare l'abbattimento dell'errore e l'affidabilità geometrica.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Boxplot Errore di Interpolazione su test_interp
    ax0 = axes[0]
    data_interp = [
        results["test_interp"]["B0_fixed"]["interp_error"],
        results["test_interp"]["B3_gnn_raw"]["interp_error"],
        results["test_interp"]["B4_gnn_safety"]["interp_error"],
        results["test_interp"]["B5_oracle"]["interp_error"]
    ]
    ax0.boxplot(data_interp, tick_labels=["B0 (Fix)", "B3 (Raw)", "B4 (Ours)", "B5 (Opt)"])
    ax0.set_title("Errore di Interpolazione $L^2$ (Test Interp)")
    ax0.set_ylabel("Errore $E_{interp}$")
    ax0.grid(True, linestyle="--", alpha=0.6)

    # 2. Minimo Jacobiano su test_extrap (Dimostrazione di robustezza)
    ax1 = axes[1]
    data_j = [
        results["test_extrap"]["B0_fixed"]["min_jacobian"],
        results["test_extrap"]["B3_gnn_raw"]["min_jacobian"],
        results["test_extrap"]["B4_gnn_safety"]["min_jacobian"],
        results["test_extrap"]["B5_oracle"]["min_jacobian"]
    ]
    ax1.boxplot(data_j, tick_labels=["B0 (Fix)", "B3 (Raw)", "B4 (Ours)", "B5 (Opt)"])
    ax1.axhline(1e-4, color="r", linestyle="--", label="Soglia $\\epsilon$")
    ax1.set_title("Minimo Jacobiano $\\min J_e$ (Test Extrap OOD)")
    ax1.set_ylabel("Determinante Jacobiano")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 3. Trade-off Accuratezza vs Tempo di Adattamento
    ax2 = axes[2]
    colors = {"B0_fixed": "gray", "B3_gnn_raw": "orange", "B4_gnn_safety": "blue", "B5_oracle": "green"}
    for b_name in ["B0_fixed", "B4_gnn_safety", "B5_oracle"]:
        mean_t = np.mean(results["test_interp"][b_name]["timing_ms"])
        mean_e = np.mean(results["test_interp"][b_name]["interp_error"])
        ax2.scatter(mean_t, mean_e, color=colors[b_name], s=120, label=b_name)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("Pareto: Errore vs Tempo di Adattamento")
    ax2.set_xlabel("Tempo [ms] (Scala Log)")
    ax2.set_ylabel("Errore $E_{interp}$ (Scala Log)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Campagna di Benchmark Scientifico per Graph-Dynamic Meshing")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Percorso del file YAML")
    parser.add_argument("--checkpoint", type=str, default="results/stage_a/checkpoints/best_model.pt", help="Checkpoint del modello GNN")
    parser.add_argument("--output_dir", type=str, default="results/stage_a/benchmark", help="Directory di output per i report")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)

    suite = BenchmarkSuite(
        config=config,
        checkpoint_path=args.checkpoint,
        output_dir=output_dir
    )

    # Esecuzione campagna sperimentale
    raw_results = suite.run_benchmark()

    # Formattazione tabelle, esportazione LaTeX e serializzazione JSON
    summary = generate_summary_tables(raw_results, output_dir, suite.logger)

    json_path = output_dir / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generazione grafici
    plot_path = output_dir / "benchmark_plots.png"
    plot_benchmark_results(raw_results, plot_path)
    suite.logger.info(f"Grafici di benchmark esportati con successo in: {plot_path.resolve()}")
    suite.logger.info("=== Campagna di Valutazione Scientifica Completata con Successo ===")


if __name__ == "__main__":
    main()
