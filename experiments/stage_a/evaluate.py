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
    - Dati grezzi aggregati in formato JSON (benchmark_summary.json)
    - Grafici di confronto scientifico (separate vector PDF figures)
"""

# Direct-execution bootstrap: make the repository and Python source tree importable.
import sys as _gdm_sys
from pathlib import Path as _GDMPath
_GDM_ROOT = _GDMPath(__file__).resolve().parents[2]
for _gdm_path in (_GDM_ROOT, _GDM_ROOT / "python"):
    if str(_gdm_path) not in _gdm_sys.path:
        _gdm_sys.path.insert(0, str(_gdm_path))


import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import time
import math

# Inclusione della cartella root nel path di sistema
sys.path.append(str(Path(__file__).resolve().parent))

from gdm import NativeMeshAdapter, compute_native_mesh_metrics_arrays
from oracle.field import GaussianBumpField
from oracle.objective import OracleObjective
from evaluation.metrics import (
    compute_displacement_metrics,
    format_computational_timings
)
from inference import MeshMoverInferenceEngine
from utils.utils import load_config, ensure_dir, setup_logger, Timer, load_pickle
from plotting.benchmark import plot_benchmark_results as plot_benchmark_figures


def _json_safe(value):
    """Convert NumPy scalars and non-finite floats to strict JSON-compatible data."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
        nodes: np.ndarray,
        cells: np.ndarray,
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
            cells=cells,
            boundary_nodes=boundary_nodes,
            lambda_D=self.oracle_cfg["objective"]["lambda_D"],
            lambda_Q=self.oracle_cfg["objective"]["lambda_Q"],
            epsilon=self.epsilon
        )

        eval_res = objective_evaluator.evaluate(nodes)
        geom_metrics = compute_native_mesh_metrics_arrays(nodes, cells)
        jacobians = np.asarray(geom_metrics["jacobians"], dtype=np.float64)
        quality = np.asarray(geom_metrics["escobar_quality"], dtype=np.float64)
        aspect = np.asarray(geom_metrics["aspect_ratio"], dtype=np.float64)

        return {
            "baseline": baseline_name,
            "interpolation_error": float(eval_res["interpolation_error"]),
            "total_objective": float(eval_res["total_objective"]),
            "is_valid": bool(eval_res["is_valid"]),
            "min_jacobian": float(np.min(jacobians)),
            "mean_jacobian": float(np.mean(jacobians)),
            "inverted_cells": int(np.sum(geom_metrics["inverted"])),
            "mean_mesh_quality": float(np.mean(quality)),
            "max_aspect_ratio": float(np.max(aspect)),
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
        # Warm-up the exact inference + native safety path before collecting any
        # per-sample latency. Otherwise the first split includes framework/device
        # initialization and gives an incomparable timing estimate.
        warmup_ids = manifest.get("test_interp", []) or manifest.get("test_extrap", [])
        if warmup_ids:
            warmup_sample = load_pickle(raw_dir / f"{warmup_ids[0]}.pkl")
            warm_nodes = warmup_sample["mesh"]["nodes"]
            warm_cells = warmup_sample["mesh"]["cells"]
            warm_mesh = NativeMeshAdapter.from_arrays(warm_nodes, warm_cells)
            warm_params = warmup_sample["metadata"]["parameters"]
            warm_field = GaussianBumpField(
                mu_x=warm_params["mu_x"], mu_y=warm_params["mu_y"],
                sigma=warm_params["sigma"], amplitude=warm_params["amplitude"]
            )
            warm_u = warmup_sample["state"]["u"]
            warm_grad = warmup_sample["state"]["grad_u"]
            for _ in range(3):
                self.engine.predict(warm_mesh, warm_u, warm_grad)
            self.logger.info("Inference warm-up completed (3 untimed runs).")

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
                    "end_to_end_wall_ms": [],
                    "oracle_recovered_ratio": [],
                    "target_accepted": [],
                    "sample_ids": []
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
                opt_nodes = sample["target"].get("reference_nodes", sample["target"].get("optimal_nodes"))
                target_disp = sample["target"]["displacement"]
                params = sample["metadata"]["parameters"]
                for baseline_key in baselines:
                    aggregated_results[split_name][baseline_key]["sample_ids"].append(str(sid))

                mesh_0 = NativeMeshAdapter.from_arrays(nodes_0, cells)
                field = GaussianBumpField(
                    mu_x=params["mu_x"],
                    mu_y=params["mu_y"],
                    sigma=params["sigma"],
                    amplitude=params["amplitude"]
                )

                # -------------------------------------------------------------
                # Baseline B0: Fixed Mesh
                # -------------------------------------------------------------
                res_b0 = self._evaluate_configuration(nodes_0, cells, field, nodes_0, mesh_0.boundary_nodes, "B0_fixed")
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
                res_b5 = self._evaluate_configuration(opt_nodes, cells, field, nodes_0, mesh_0.boundary_nodes, "B5_oracle")
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
                target_accepted = bool(sample["metadata"].get("target_accepted", sample["metadata"].get("converged", False)))
                aggregated_results[split_name]["B5_oracle"]["target_accepted"].append(target_accepted)

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
                res_b3 = self._evaluate_configuration(nodes_b3, cells, field, nodes_0, mesh_0.boundary_nodes, "B3_gnn_raw")

                aggregated_results[split_name]["B3_gnn_raw"]["interp_error"].append(res_b3["interpolation_error"])
                aggregated_results[split_name]["B3_gnn_raw"]["min_jacobian"].append(res_b3["min_jacobian"])
                aggregated_results[split_name]["B3_gnn_raw"]["mean_quality"].append(res_b3["mean_mesh_quality"])
                aggregated_results[split_name]["B3_gnn_raw"]["inverted_cells"].append(res_b3["inverted_cells"])
                aggregated_results[split_name]["B3_gnn_raw"]["disp_l2_error"].append(disp_m_raw["displacement_l2_error"])
                aggregated_results[split_name]["B3_gnn_raw"]["disp_rel_error"].append(disp_m_raw["displacement_relative_error"])
                # Comparable component timing: graph preparation + neural forward
                # pass, measured by the inference engine itself.
                aggregated_results[split_name]["B3_gnn_raw"]["timing_ms"].append(
                    float(pred_res["inference_time"]) * 1000.0
                )

                # -------------------------------------------------------------
                # Baseline B4: GNN + Safety Filter (Framework Proposto)
                # -------------------------------------------------------------
                safe_nodes = np.asarray(pred_res["new_nodes"], dtype=np.float64)
                res_b4 = self._evaluate_configuration(safe_nodes, cells, field, nodes_0, mesh_0.boundary_nodes, "B4_gnn_safety")
                e_gnn = res_b4["interpolation_error"]

                # This is a gap reduction against a quality-checked finite-budget
                # reference target, not a certified optimum-gap guarantee.
                gap = e_fixed - e_oracle
                # A target-gap metric is reported only for quality-accepted reference
                # targets that actually improve on the fixed mesh.
                r_oracle = (e_fixed - e_gnn) / gap if (target_accepted and gap > 1.0e-12) else float("nan")

                aggregated_results[split_name]["B4_gnn_safety"]["interp_error"].append(e_gnn)
                aggregated_results[split_name]["B4_gnn_safety"]["min_jacobian"].append(res_b4["min_jacobian"])
                aggregated_results[split_name]["B4_gnn_safety"]["mean_quality"].append(res_b4["mean_mesh_quality"])
                aggregated_results[split_name]["B4_gnn_safety"]["inverted_cells"].append(res_b4["inverted_cells"])
                aggregated_results[split_name]["B4_gnn_safety"]["disp_l2_error"].append(disp_m_safe["displacement_l2_error"])
                aggregated_results[split_name]["B4_gnn_safety"]["disp_rel_error"].append(disp_m_safe["displacement_relative_error"])
                # Component timings are comparable across B3/B4: the B4 value
                # adds the native safety-filter time to the same inference timer.
                aggregated_results[split_name]["B4_gnn_safety"]["timing_ms"].append(
                    float(pred_res["inference_time"] + pred_res["filter_time"]) * 1000.0
                )
                # Also preserve external wall time for the complete predict() API,
                # including Python/native wrapper overhead, as a separate metric.
                aggregated_results[split_name]["B4_gnn_safety"]["end_to_end_wall_ms"].append(
                    float(t_online_ms)
                )
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
    """Aggregate per-sample measurements and emit an English-language LaTeX table."""
    summary: Dict[str, Any] = {}
    labels = {
        "B0_fixed": r"Fixed mesh ($B_0$)",
        "B_zero": r"Zero displacement ($B_{\mathrm{zero}}$)",
        "B3_gnn_raw": r"GNN, unfiltered ($B_3$)",
        "B4_gnn_safety": r"GNN + safety filter ($B_4$)",
        "B5_oracle": r"Budgeted reference target ($B_5$)",
    }
    table_header = (
        f"{'Baseline':<32} | {'Mean error (x1e-4) +/- SD':<27} | {'Min J':<12} | "
        f"{'Mean quality':<12} | {'Inverted cells':<14} | {'Time/sample [ms]':<17}"
    )
    logger.info(table_header)
    logger.info("-" * len(table_header))

    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Stage-A interpolation benchmark. The stored finite-budget reference target is not assumed to be a stationary or global optimum; target-acceptance counts are reported separately for each test split. Timing is not applicable to the fixed and zero-displacement baselines.}",
        r"\label{tab:stage_a_benchmark}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Baseline & $\bar E_{\mathrm{interp}}$ ($\times 10^{-4}$; mean $\pm$ SD) & $\min_e J_e$ & $\bar\eta$ & Inverted cells & Time [ms] " + r"\\",
        r"\midrule",
    ]

    for split_name, baselines in results.items():
        summary[split_name] = {}
        logger.info("\nSplit: %s", split_name)
        b5_target_acceptance = np.asarray(baselines.get("B5_oracle", {}).get("target_accepted", []), dtype=bool)
        if b5_target_acceptance.size:
            n_accepted = int(b5_target_acceptance.sum())
            acceptance_rate = float(b5_target_acceptance.mean())
            logger.info("Stored reference-target acceptance: %d/%d (%.1f%%)", n_accepted, b5_target_acceptance.size, 100 * acceptance_rate)
            latex_lines.append(
                rf"\multicolumn{{6}}{{l}}{{\textit{{{split_name.replace('_', ' ').title()}: reference target accepted for {n_accepted}/{b5_target_acceptance.size} samples ({100 * acceptance_rate:.1f}\%).}}}} \\"
            )
            latex_lines.append(r"\addlinespace[2pt]")
        else:
            n_accepted = 0

        for b_name, metrics in baselines.items():
            errors = np.asarray(metrics.get("interp_error", []), dtype=float)
            errors = errors[np.isfinite(errors)]
            jac = np.asarray(metrics.get("min_jacobian", []), dtype=float)
            jac = jac[np.isfinite(jac)]
            quality = np.asarray(metrics.get("mean_quality", []), dtype=float)
            quality = quality[np.isfinite(quality)]
            timings = np.asarray(metrics.get("timing_ms", []), dtype=float)
            timings = timings[np.isfinite(timings)]
            inverted = np.asarray(metrics.get("inverted_cells", []), dtype=float)
            inverted = inverted[np.isfinite(inverted)]

            mean_err = float(np.mean(errors) * 1e4) if errors.size else float("nan")
            std_err = float(np.std(errors, ddof=1) * 1e4) if errors.size > 1 else (0.0 if errors.size == 1 else float("nan"))
            min_j = float(np.min(jac)) if jac.size else float("nan")
            mean_q = float(np.mean(quality)) if quality.size else float("nan")
            inv_c = int(np.sum(inverted)) if inverted.size else 0
            mean_t = float(np.mean(timings)) if timings.size else float("nan")
            n_samples = int(len(metrics.get("interp_error", [])))

            summary[split_name][b_name] = {
                "n_samples": n_samples,
                "mean_interp_error_x1e4": mean_err,
                "std_interp_error_x1e4": std_err,
                "min_jacobian": min_j,
                "mean_quality": mean_q,
                "total_inverted_cells": inv_c,
                "mean_time_ms": mean_t,
            }
            wall_timings = np.asarray(metrics.get("end_to_end_wall_ms", []), dtype=float)
            wall_timings = wall_timings[np.isfinite(wall_timings) & (wall_timings > 0)]
            if wall_timings.size:
                summary[split_name][b_name]["mean_end_to_end_wall_ms"] = float(np.mean(wall_timings))
                summary[split_name][b_name]["std_end_to_end_wall_ms"] = float(np.std(wall_timings, ddof=1)) if wall_timings.size > 1 else 0.0
            convergence = np.asarray(metrics.get("target_accepted", []), dtype=bool)
            if convergence.size:
                summary[split_name][b_name]["n_accepted_targets"] = int(np.sum(convergence))
                summary[split_name][b_name]["target_acceptance_rate"] = float(np.mean(convergence))

            ratios = np.asarray(metrics.get("oracle_recovered_ratio", []), dtype=float)
            ratios = ratios[np.isfinite(ratios)]
            if b_name == "B4_gnn_safety":
                summary[split_name][b_name]["n_accepted_targets_for_gap_reduction"] = int(ratios.size)
                summary[split_name][b_name]["mean_target_gap_reduction_pct_accepted_only"] = (
                    float(np.mean(ratios) * 100.0) if ratios.size else None
                )

            # Fixed/zero baselines are not timed; their 0.0 values are sentinels.
            time_str = "--" if b_name in {"B0_fixed", "B_zero"} else (f"{mean_t:.2f}" if np.isfinite(mean_t) else "--")
            row_str = (
                f"{labels.get(b_name, b_name):<32} | {mean_err:.3f} +/- {std_err:.3f} | "
                f"{min_j:.3e} | {mean_q:.4f} | {inv_c:d} | {time_str}"
            )
            logger.info(row_str)
            label_tex = labels.get(b_name, b_name.replace("_", r"\_"))
            latex_lines.append(
                f"{label_tex} & {mean_err:.3f} $\\pm$ {std_err:.3f} & {min_j:.3e} & "
                f"{mean_q:.4f} & {inv_c:d} & {time_str} " + r"\\"
            )
        latex_lines.append(r"\midrule")

    latex_lines[-1] = r"\bottomrule"
    latex_lines.extend([r"\end{tabular}", r"\end{table}"])
    output_dir.mkdir(parents=True, exist_ok=True)
    latex_path = output_dir / "table_benchmark.tex"
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    logger.info("LaTeX table written to %s", latex_path.resolve())

    # For any claim relative to a stored reference target, all baseline errors
    # must be recomputed on the identical subset of targets that passed the
    # finite-budget acceptance contract. Do not mix all-sample means with a
    # target-gap ratio that exists only for accepted targets.
    accepted_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Stage-A interpolation errors on the common subset of quality-accepted finite-budget reference targets. These references are not certified stationary or global optima.}",
        r"\label{tab:stage_a_accepted_reference_subset}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Test split & Baseline & $n/N$ & Mean $E_{\mathrm{interp}}$ ($\times 10^{-4}$; mean $\pm$ SD) & Aggregate target-gap reduction [\%] " + r"\\",
        r"\midrule",
    ]
    for split_name, baselines in results.items():
        accepted_mask = np.asarray(baselines.get("B5_oracle", {}).get("target_accepted", []), dtype=bool)
        total = int(accepted_mask.size)
        n_accepted = int(accepted_mask.sum())
        split_label = split_name.replace("_", " ").title()
        if total == 0 or n_accepted == 0:
            accepted_lines.append(
                rf"\multicolumn{{5}}{{l}}{{\textit{{{split_label}: no quality-accepted reference targets ({n_accepted}/{total}).}}}} \\"
            )
            accepted_lines.append(r"\midrule")
            summary.setdefault(split_name, {})["accepted_reference_subset"] = {
                "n_total": total, "n_accepted_targets": n_accepted,
                "mean_interp_error_x1e4": {}, "aggregate_target_gap_reduction_pct": None,
            }
            continue

        accepted_stats = {}
        accepted_means = {}
        for b_name in ("B0_fixed", "B_zero", "B3_gnn_raw", "B4_gnn_safety", "B5_oracle"):
            metrics = baselines.get(b_name, {})
            all_errors = np.asarray(metrics.get("interp_error", []), dtype=float)
            if all_errors.size != total:
                raise ValueError(
                    f"Accepted-subset benchmark mismatch for split={split_name}, baseline={b_name}: "
                    f"{all_errors.size} errors versus {total} target flags."
                )
            subset_errors = all_errors[accepted_mask]
            subset_errors = subset_errors[np.isfinite(subset_errors)]
            mean_err = float(np.mean(subset_errors) * 1e4) if subset_errors.size else float("nan")
            sd_err = float(np.std(subset_errors, ddof=1) * 1e4) if subset_errors.size > 1 else (0.0 if subset_errors.size == 1 else float("nan"))
            accepted_stats[b_name] = (mean_err, sd_err)
            accepted_means[b_name] = mean_err / 1e4 if np.isfinite(mean_err) else float("nan")

        # This is the ratio of mean errors on the common accepted subset (same
        # estimand as the Stage-A T1 table), not the mean of per-sample ratios.
        gap_den = accepted_means.get("B0_fixed", float("nan")) - accepted_means.get("B5_oracle", float("nan"))
        gap_num = accepted_means.get("B0_fixed", float("nan")) - accepted_means.get("B4_gnn_safety", float("nan"))
        aggregate_gap_pct = (
            100.0 * gap_num / gap_den
            if np.isfinite(gap_num) and np.isfinite(gap_den) and gap_den > 1.0e-12 else None
        )
        summary.setdefault(split_name, {})["accepted_reference_subset"] = {
            "n_total": total,
            "n_accepted_targets": n_accepted,
            "mean_interp_error_x1e4": {
                name: (values[0] if np.isfinite(values[0]) else None)
                for name, values in accepted_stats.items()
            },
            "sd_interp_error_x1e4": {
                name: (values[1] if np.isfinite(values[1]) else None)
                for name, values in accepted_stats.items()
            },
            "aggregate_target_gap_reduction_pct": aggregate_gap_pct,
            "target_gap_definition": "ratio of accepted-subset mean errors: (mean E_fixed - mean E_GNN_safe)/(mean E_fixed - mean E_reference)",
        }

        for b_name in ("B0_fixed", "B_zero", "B3_gnn_raw", "B4_gnn_safety", "B5_oracle"):
            mean_err, sd_err = accepted_stats[b_name]
            err_cell = rf"{mean_err:.3f} $\pm$ {sd_err:.3f}" if np.isfinite(mean_err) and np.isfinite(sd_err) else "--"
            gap_cell = f"{aggregate_gap_pct:.2f}" if b_name == "B4_gnn_safety" and aggregate_gap_pct is not None else "--"
            accepted_lines.append(
                f"{split_label} & {labels.get(b_name, b_name)} & {n_accepted}/{total} & {err_cell} & {gap_cell} " + r"\\"
            )
        accepted_lines.append(r"\midrule")
    accepted_lines[-1] = r"\bottomrule"
    accepted_lines.extend([r"\end{tabular}", r"\end{table}"])
    accepted_table_path = output_dir / "table_benchmark_accepted_reference_subset.tex"
    accepted_table_path.write_text("\n".join(accepted_lines) + "\n", encoding="utf-8")
    logger.info("Accepted-reference subset LaTeX table written to %s", accepted_table_path.resolve())
    return summary


# =============================================================================
# GENERAZIONE PLOT SCIENTIFICI
# =============================================================================

def plot_benchmark_results(results: Dict[str, Any], output_path: Path, epsilon: float = 1.0e-4):
    """Delegate all paper figure construction to the reusable plotting package."""
    return plot_benchmark_figures(results, output_path, epsilon=epsilon)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Campagna di Benchmark Scientifico per Graph-Dynamic Meshing")
    parser.add_argument("--config", type=str, default="config/stage_a.yaml", help="Percorso del file YAML")
    parser.add_argument("--checkpoint", type=str, default="results/stage_a_oracle_v4/checkpoints/best_model.pt", help="Checkpoint del modello GNN")
    parser.add_argument("--output_dir", type=str, default="results/stage_a_oracle_v4/benchmark", help="Directory di output per i report")
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

    # Keep per-sample metrics so the figures can be regenerated without re-running inference.
    raw_path = output_dir / "benchmark_raw_results.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(raw_results), f, indent=2, allow_nan=False)

    # Aggregate summary and the English LaTeX table.
    summary = generate_summary_tables(raw_results, output_dir, suite.logger)
    summary["_metadata"] = {
        "protocol": "Stage-A field-interpolation benchmark",
        "inference_warmup_runs": 3,
        "timing_B3": "Inference-engine timing for graph construction plus GNN forward pass",
        "timing_B4": "Comparable component timing: B3 inference timer plus native safety-filter timer; the complete predict() wall-clock time is stored separately in end_to_end_wall_ms",
        "stored_target_note": "B5 is a quality-checked finite-budget optimizer reference, not a certified optimum; only accepted targets are used for target-gap reduction statistics, and first-order stationarity is recorded separately.",
        "epsilon": float(config["geometry"]["safety"]["epsilon"]),
    }
    json_path = output_dir / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)

    # Paper figures are generated by python/plotting/benchmark.py and exported
    # as vector vector PDF only, one figure per scientific question.
    figure_paths = plot_benchmark_results(
        raw_results, output_dir,
        epsilon=float(config["geometry"]["safety"]["epsilon"]),
    )
    suite.logger.info(
        "Benchmark figures exported: %s",
        {key: [str(p.resolve()) for p in value] for key, value in figure_paths.items()},
    )
    suite.logger.info("=== Campagna di Valutazione Scientifica Completata con Successo ===")


if __name__ == "__main__":
    main()
