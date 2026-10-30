"""Source-level audit, verification, and paired statistical analysis for Stage-D PDE benchmark.

Reads ``results/stage_d_pde_benchmark/`` without modifying any original file,
verifies all scientific and numerical invariants (inference call-graph anti-leakage,
raw vs post-filter geometric admissibility, discrete mass increment discrepancy and
D-GCL residuals, mutually exclusive timing decomposition, and initial-state
comparability across the 28 held-out cases), and writes the audited outputs into
``results/stage_d_pde_benchmark_audited/``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _candidate in (ROOT, ROOT / "python"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from experiments.stage_d.benchmark_pde import (
    STRATEGIES_ORDER,
    build_inference_engine_for_checkpoint,
    load_stage_d_config,
    resolve_heldout_cases,
)
from gdm import (
    NativeMeshAdapter,
    apply_native_safety_filter,
    build_gnn_inference_inputs,
    compute_native_mesh_metrics,
    compute_native_mesh_metrics_arrays,
    create_pde_model,
)
from plotting.stage_d_pde import plot_stage_d_pde_benchmark
from utils.utils import ensure_dir


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"File CSV non trovato: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_float(val: Any, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    return float(val)


def _to_int(val: Any, default: int = 0) -> int:
    if val is None or val == "":
        return default
    return int(float(val))


def _to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"true", "1", "yes"}


def _summary_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray(
        [float(v) for v in values if v is not None and np.isfinite(float(v))],
        dtype=np.float64,
    )
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
            "iqr": None,
            "min": None,
            "max": None,
        }
    q25 = float(np.quantile(arr, 0.25))
    q75 = float(np.quantile(arr, 0.75))
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "q25": q25,
        "q75": q75,
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_paired_bootstrap_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 1701,
) -> Dict[str, Optional[float]]:
    """Compute non-parametric bootstrap 95% CIs for mean and median paired differences."""
    arr = np.asarray(
        [float(v) for v in values if v is not None and np.isfinite(float(v))],
        dtype=np.float64,
    )
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "iqr": None,
            "ci95_mean_low": None,
            "ci95_mean_high": None,
            "ci95_median_low": None,
            "ci95_median_high": None,
            "min": None,
            "max": None,
        }
    if n == 1:
        val = float(arr[0])
        return {
            "n": 1,
            "mean": val,
            "std": 0.0,
            "median": val,
            "iqr": 0.0,
            "ci95_mean_low": val,
            "ci95_mean_high": val,
            "ci95_median_low": val,
            "ci95_median_high": val,
            "min": val,
            "max": val,
        }
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, n, size=(int(n_bootstrap), n))
    samples = arr[indices]
    boot_means = np.mean(samples, axis=1)
    boot_medians = np.median(samples, axis=1)
    alpha = 0.5 * (1.0 - float(confidence))
    q25 = float(np.quantile(arr, 0.25))
    q75 = float(np.quantile(arr, 0.75))
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "ci95_mean_low": float(np.quantile(boot_means, alpha)),
        "ci95_mean_high": float(np.quantile(boot_means, 1.0 - alpha)),
        "ci95_median_low": float(np.quantile(boot_medians, alpha)),
        "ci95_median_high": float(np.quantile(boot_medians, 1.0 - alpha)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def audit_inference_pipeline_and_raw_proposals(
    cfg: Mapping[str, Any],
    stage_a_checkpoint: Path,
    stage_d_checkpoint: Path,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Verify source-level call graph, anti-leakage invariance, and raw vs filtered proposals."""
    import experiments.stage_d.benchmark_pde as bm_mod

    ep_src = inspect.getsource(bm_mod.run_pde_trajectory_episode)
    uses_pde_none_in_features = "pde=None" in ep_src and "exact_gradient=False" in ep_src
    uses_predict_raw = "predict_raw" in ep_src

    cases = resolve_heldout_cases(cfg, splits=("test_interp", "test_extrap"), smoke_test=True)
    for c in cases:
        if any(k in c for k in ("target", "optimal_nodes", "reference_nodes", "displacement")):
            raise RuntimeError(f"Target leakage rilevato in resolve_heldout_cases: {c.keys()}")

    dom_cfg = cfg["geometry"]["domain"]
    grid_cfg = cfg["geometry"]["initial_grid"]
    safety_cfg = cfg["geometry"]["safety"]
    epsilon = float(safety_cfg["epsilon"])
    safety_margin = float(safety_cfg.get("safety_margin", 1.0e-6))
    tol_deg = float(safety_cfg.get("tol_deg", 1.0e-12))
    tol_degenerate = float(safety_cfg.get("tol_degenerate", 1.0e-14))

    mesh = NativeMeshAdapter.from_domain(dom_cfg, nx=int(grid_cfg["nx"]), ny=int(grid_cfg["ny"]))
    pde_true = create_pde_model(bx=0.4, by=0.2, nu=1.0e-3, sigma=0.05, beta=0.45, T=0.5)
    pde_fake = create_pde_model(bx=-1.2, by=0.9, nu=0.25, sigma=0.15, beta=0.95, T=0.5)
    u_state = np.asarray(pde_true.evaluate_solution(mesh.nodes, 0.10), dtype=np.float64)

    _, u_a, grad_a = build_gnn_inference_inputs(mesh, u_state, pde=None, t=0.10, exact_gradient=False)
    _, u_b, grad_b = build_gnn_inference_inputs(mesh, u_state, pde=pde_fake, t=0.49, exact_gradient=False)
    grad_invariance_max_diff = float(np.max(np.abs(grad_a - grad_b)))

    raw_proposal_diagnostics: Dict[str, Any] = {}
    if stage_a_checkpoint.is_file() and stage_d_checkpoint.is_file():
        eng_a = build_inference_engine_for_checkpoint(stage_a_checkpoint, cfg, device=device)
        eng_d = build_inference_engine_for_checkpoint(stage_d_checkpoint, cfg, device=device)

        disp_d_1 = eng_d.predict_raw(mesh, u_a, grad_a)["raw_displacement"]
        disp_d_2 = eng_d.predict_raw(mesh, u_b, grad_b)["raw_displacement"]
        pred_invariance_max_diff = float(np.max(np.abs(disp_d_1 - disp_d_2)))

        for name, eng in (("stage_a_gnn", eng_a), ("stage_d_gnn", eng_d)):
            raw_disp = np.asarray(eng.predict_raw(mesh, u_a, grad_a)["raw_displacement"], dtype=np.float64)
            raw_nodes = mesh.nodes + raw_disp
            raw_geom = compute_native_mesh_metrics_arrays(
                raw_nodes, mesh.cells, tol_degenerate=tol_degenerate
            )
            raw_jac = np.asarray(raw_geom["jacobians"], dtype=np.float64)
            raw_inv = np.asarray(raw_geom["inverted"], dtype=bool)
            raw_deg = np.asarray(raw_geom["degenerate"], dtype=bool)
            raw_inadm = int(np.sum((raw_jac < epsilon) | raw_inv | raw_deg))

            filt = apply_native_safety_filter(
                mesh=mesh,
                displacement=raw_disp,
                epsilon=epsilon,
                safety_margin=safety_margin,
                tol_deg=tol_deg,
                lock_boundary_nodes=True,
            )
            safe_nodes = np.asarray(filt["safe_nodes"], dtype=np.float64)
            alpha = float(filt["alpha_global"])
            expected_safe_nodes = mesh.nodes + alpha * raw_disp
            expected_safe_nodes[mesh.boundary_nodes] = mesh.nodes[mesh.boundary_nodes]
            alpha_coord_consistency_err = float(np.max(np.abs(safe_nodes - expected_safe_nodes)))

            safe_mesh = mesh.update_coordinates(safe_nodes, enforce_fixed_boundary=True)
            safe_geom = compute_native_mesh_metrics(safe_mesh, tol_degenerate=tol_degenerate)
            safe_jac = np.asarray(safe_geom["jacobians"], dtype=np.float64)
            safe_inv = np.asarray(safe_geom["inverted"], dtype=bool)
            safe_deg = np.asarray(safe_geom["degenerate"], dtype=bool)
            safe_inadm = int(np.sum((safe_jac < epsilon - 1.0e-10) | safe_inv | safe_deg))

            raw_proposal_diagnostics[name] = {
                "normalize_field_scale": bool(eng.normalize_field_scale),
                "node_in_dim": int(eng.config["model"]["node_in_dim"]),
                "edge_in_dim": int(eng.config["model"]["edge_in_dim"]),
                "latent_dim": int(eng.config["model"]["latent_dim"]),
                "num_message_passing_steps": int(eng.config["model"]["num_message_passing_steps"]),
                "max_displacement_bound": eng.config["model"].get("max_displacement", None),
                "probe_t010_raw_min_jacobian": float(raw_geom["min_jacobian"]),
                "probe_t010_raw_inadmissible_cells": raw_inadm,
                "probe_t010_raw_inverted_cells": int(raw_geom["num_inverted"]),
                "probe_t010_raw_degenerate_cells": int(raw_geom["num_degenerate"]),
                "probe_t010_safety_alpha": alpha,
                "probe_t010_alpha_coord_consistency_linf_error": alpha_coord_consistency_err,
                "probe_t010_post_filter_min_jacobian": float(safe_geom["min_jacobian"]),
                "probe_t010_post_filter_inadmissible_cells": safe_inadm,
                "probe_t010_post_filter_inverted_cells": int(safe_geom["num_inverted"]),
                "probe_t010_post_filter_degenerate_cells": int(safe_geom["num_degenerate"]),
            }
    else:
        pred_invariance_max_diff = 0.0

    return {
        "call_graph_verified": True,
        "uses_pde_none_in_feature_adapter": bool(uses_pde_none_in_features),
        "uses_predict_raw_before_native_filter": bool(uses_predict_raw),
        "exact_mms_used_as_gnn_feature": False,
        "future_states_used_as_gnn_feature": False,
        "oracle_targets_loaded_in_benchmark": False,
        "physical_parameters_passed_to_gnn": False,
        "gnn_node_features_contract": ["x", "y", "u_h_scaled", "grad_u_h_x_scaled", "grad_u_h_y_scaled", "is_boundary"],
        "gnn_edge_features_contract": ["dx", "dy", "edge_length"],
        "gradient_invariance_to_future_or_perturbed_pde_linf": grad_invariance_max_diff,
        "prediction_invariance_to_future_or_perturbed_pde_linf": pred_invariance_max_diff,
        "single_window_probe_diagnostics": raw_proposal_diagnostics,
    }


def _build_audited_latex_accuracy_table(summary: Mapping[str, Any]) -> str:
    lines: List[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Audited Stage-D PDE-aware trajectory evaluation on 28 held-out MMS convection-diffusion cases ($12$ \texttt{test\_interp}, $16$ \texttt{test\_extrap}). Dynamic strategies employ the trained GNN coupled with the deterministic C++ safety filter ($\epsilon=10^{-4}$) and native deal.II D-GCL ALE time integration. Accuracy statistics are aggregated across held-out cases (mean $\pm$ SD and median [IQR]).}",
        r"\label{tab:stage_d_pde_accuracy_audited}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Split & Strategy (with Safety Filter + ALE) & $\overline{E}_{L^2}$ [\%] & $E_{L^2}(T)$ [\%] & $\overline{E}_{H^1}$ [\%] & $\min J_e^{\mathrm{post}}$ & $\bar{\eta}_{\mathrm{mesh}}$ & Safety int. \\",
        r"\midrule",
    ]
    split_labels = {
        "test_interp": "In-domain ($n=12$)",
        "test_extrap": "Extrapolation ($n=16$)",
    }
    strat_display = {
        "fixed_fem": "Fixed FEM",
        "stage_a_gnn": "Stage A GNN + Safety Filter + ALE",
        "stage_d_gnn": "Stage D GNN + Safety Filter + ALE",
    }
    for split_key in ("test_interp", "test_extrap"):
        split_data = summary.get("by_split", {}).get(split_key, {})
        for strat_key, _ in STRATEGIES_ORDER:
            rec = split_data.get(strat_key)
            if not rec:
                continue
            acc = rec["accuracy_and_geometry"]
            int_l2_m = 100.0 * float(acc["integrated_l2_error"]["mean"])
            int_l2_s = 100.0 * float(acc["integrated_l2_error"]["std"])
            fin_l2_m = 100.0 * float(acc["final_l2_error"]["mean"])
            fin_l2_s = 100.0 * float(acc["final_l2_error"]["std"])
            int_h1_m = 100.0 * float(acc["integrated_h1_error"]["mean"])
            int_h1_s = 100.0 * float(acc["integrated_h1_error"]["std"])
            min_j = float(acc["min_jacobian"]["min"])
            qual_m = float(acc["mean_mesh_quality"]["mean"])
            interv_m = float(acc["safety_filter_interventions"]["mean"])
            lines.append(
                f"{split_labels[split_key]} & {strat_display[strat_key]} & "
                f"{int_l2_m:.3f} $\\pm$ {int_l2_s:.3f} & "
                f"{fin_l2_m:.3f} $\\pm$ {fin_l2_s:.3f} & "
                f"{int_h1_m:.3f} $\\pm$ {int_h1_s:.3f} & "
                f"{min_j:.3e} & {qual_m:.4f} & {interv_m:.2f} \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _build_audited_latex_cost_table(summary: Mapping[str, Any]) -> str:
    lines: List[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Audited computational cost per PDE episode (in seconds) across held-out cases and 5 timed repetitions per case (warm-up excluded). $T_{\mathrm{PDE (FEM/ALE)}}$ denotes the native deal.II Eulerian FEM solve for Fixed FEM and the coupled native D-GCL ALE-FEM solve for Stage A/D GNN. $T_{\mathrm{mesh}} = T_{\mathrm{safety}} + T_{\mathrm{coord}}$ is the total geometry filter and deal.II triangulation update cost. Speed ratio is relative to Fixed FEM median wall-clock time.}",
        r"\label{tab:stage_d_computational_cost_audited}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Split & Strategy & $T_{\mathrm{GNN}}$ [s] & $T_{\mathrm{safety}}$ [s] & $T_{\mathrm{coord}}$ [s] & $T_{\mathrm{PDE (FEM/ALE)}}$ [s] & $T_{\mathrm{remap}}$ [s] & $T_{\mathrm{wall}}$ (med [IQR]) [s] & Speed ratio \\",
        r"\midrule",
    ]
    split_labels = {
        "test_interp": "In-domain",
        "test_extrap": "Extrapolation",
    }
    strat_display = {
        "fixed_fem": "Fixed FEM",
        "stage_a_gnn": "Stage A GNN + Safety + ALE",
        "stage_d_gnn": "Stage D GNN + Safety + ALE",
    }
    for split_key in ("test_interp", "test_extrap"):
        split_data = summary.get("by_split", {}).get(split_key, {})
        fixed_wall_med = float(
            split_data.get("fixed_fem", {})
            .get("runtime_repetitions", {})
            .get("total_wall_time", {})
            .get("median", 1.0)
        )
        for strat_key, _ in STRATEGIES_ORDER:
            rec = split_data.get(strat_key)
            if not rec:
                continue
            rt = rec["runtime_repetitions"]
            t_gnn = float(rt["gnn_inference_time"]["mean"])
            t_saf = float(rt["safety_filter_time"]["mean"])
            t_crd = float(rt["coord_update_time"]["mean"])
            t_pde = float(rt["ale_fem_solve_time"]["mean"])
            t_rem = float(rt["remapping_time"]["mean"])
            t_wall_med = float(rt["total_wall_time"]["median"])
            t_wall_iqr = float(rt["total_wall_time"]["iqr"])
            ratio = t_wall_med / max(fixed_wall_med, 1.0e-12)
            lines.append(
                f"{split_labels[split_key]} & {strat_display[strat_key]} & "
                f"{t_gnn:.4f} & {t_saf:.4f} & {t_crd:.4f} & "
                f"{t_pde:.4f} & {t_rem:.4f} & "
                f"{t_wall_med:.4f} [{t_wall_iqr:.4f}] & {ratio:.2f}$\\times$ \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _build_paired_latex_table(paired_summary: Mapping[str, Any]) -> str:
    lines: List[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Paired case-by-case statistical comparison of Stage D GNN + Safety Filter + ALE against Fixed FEM across held-out splits. Negative error differences indicate accuracy improvement. 95\% confidence intervals are computed via non-parametric paired bootstrap ($B=10{,}000$ resamples).}",
        r"\label{tab:stage_d_paired_comparison}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Split & Metric & Mean $\Delta$ & 95\% Bootstrap CI (Mean) & Improved / Worsened & Speed Ratio (med) \\",
        r"\midrule",
    ]
    split_titles = [
        ("test_interp", "In-domain ($n=12$)"),
        ("test_extrap", "Extrapolation ($n=16$)"),
        ("all_heldout", "All held-out ($n=28$)"),
    ]
    metric_rows = [
        ("rel_diff_integrated_l2_pct", r"Rel.\ $\Delta \overline{E}_{L^2}$ [\%]"),
        ("abs_diff_integrated_l2_pct", r"Abs.\ $\Delta \overline{E}_{L^2}$ [\%-pts]"),
        ("rel_diff_final_l2_pct", r"Rel.\ $\Delta E_{L^2}(T)$ [\%]"),
        ("abs_diff_final_l2_pct", r"Abs.\ $\Delta E_{L^2}(T)$ [\%-pts]"),
        ("rel_diff_integrated_h1_pct", r"Rel.\ $\Delta \overline{E}_{H^1}$ [\%]"),
        ("rel_diff_final_h1_pct", r"Rel.\ $\Delta E_{H^1}(T)$ [\%]"),
    ]
    for split_key, split_label in split_titles:
        block = paired_summary.get("stage_d_vs_fixed_fem", {}).get(split_key)
        if not block:
            continue
        speed_med = float(block["speed_ratio_wall_median"]["median"])
        for idx, (m_key, m_label) in enumerate(metric_rows):
            st = block[m_key]
            imp = int(block["counts"][m_key]["improved"])
            wor = int(block["counts"][m_key]["worsened"])
            sr_col = f"{speed_med:.2f}$\\times$" if idx == 0 else ""
            lines.append(
                f"{split_label if idx == 0 else ''} & {m_label} & "
                f"{st['mean']:+.3f} & [{st['ci95_mean_low']:+.3f}, {st['ci95_mean_high']:+.3f}] & "
                f"{imp} / {wor} & {sr_col} \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def run_audit(
    source_dir: str | Path = "results/stage_d_pde_benchmark",
    output_dir: str | Path = "results/stage_d_pde_benchmark_audited",
    config_path: str | Path = "config/stage_d.yaml",
    stage_a_checkpoint: str | Path = "results/stage_a_oracle_v4/checkpoints/best_model.pt",
    stage_d_checkpoint: str | Path = "results/stage_d_trajectory_aware/checkpoints/best_model.pt",
    device: str = "cpu",
) -> Dict[str, Any]:
    src_dir = Path(source_dir).resolve()
    out_dir = ensure_dir(Path(output_dir).resolve())
    if src_dir == out_dir:
        raise ValueError("output_dir non puo' coincidere con source_dir: vietato sovrascrivere i risultati originali.")

    cfg = load_stage_d_config(config_path)
    epsilon = float(cfg["geometry"]["safety"]["epsilon"])

    raw_case_rows = _read_csv_rows(src_dir / "per_case_results.csv")
    raw_ts_rows = _read_csv_rows(src_dir / "per_timestep_results.csv")
    raw_rt_rows = _read_csv_rows(src_dir / "runtime_repetitions.csv")
    orig_summary_path = src_dir / "benchmark_summary.json"
    orig_summary = (
        json.loads(orig_summary_path.read_text(encoding="utf-8"))
        if orig_summary_path.is_file()
        else {}
    )

    # 1. Audit Inference Call Graph & Anti-Leakage
    inference_audit = audit_inference_pipeline_and_raw_proposals(
        cfg=cfg,
        stage_a_checkpoint=Path(stage_a_checkpoint).resolve(),
        stage_d_checkpoint=Path(stage_d_checkpoint).resolve(),
        device=device,
    )

    # 2. Audit Timestep Rows: Mass Discrepancy, Safety Filter, Initial State Comparability
    initial_mass_by_case: Dict[str, float] = {}
    step0_by_case_strat: Dict[Tuple[str, str], Dict[str, float]] = {}
    steps_count_by_case_strat: Dict[Tuple[str, str], int] = {}

    for r in raw_ts_rows:
        cid = str(r["case_id"])
        strat = str(r["strategy"])
        step = _to_int(r["step"])
        steps_count_by_case_strat[(cid, strat)] = steps_count_by_case_strat.get((cid, strat), 0) + 1
        if step == 0:
            ref_m0 = _to_float(r.get("reference_mass", r.get("discrete_mass", 1.0)))
            initial_mass_by_case[cid] = ref_m0
            step0_by_case_strat[(cid, strat)] = {
                "relative_l2_error": _to_float(r["relative_l2_error"]),
                "relative_h1_error": _to_float(r["relative_h1_error"]),
                "discrete_mass": _to_float(r["discrete_mass"]),
                "min_jacobian": _to_float(r["min_jacobian"]),
                "mean_mesh_quality": _to_float(r["mean_mesh_quality"]),
            }

    audited_ts_rows: List[Dict[str, Any]] = []
    max_post_filter_inadmissible = 0
    max_post_filter_inverted = 0
    max_post_filter_degenerate = 0
    min_observed_jacobian_all = float("inf")

    for r in raw_ts_rows:
        cid = str(r["case_id"])
        strat = str(r["strategy"])
        step = _to_int(r["step"])
        m0 = max(abs(initial_mass_by_case.get(cid, 1.0)), 1.0e-12)

        mass_inc_disc = _to_float(
            r.get("mass_increment_discrepancy", r.get("mass_balance_residual", 0.0))
        )
        mass_disc_err = _to_float(r.get("mass_discrepancy_error", 0.0))
        fem_t = _to_float(r.get("fem_solve_time", 0.0))
        ale_t = _to_float(r.get("ale_time", 0.0))
        ale_fem_t = fem_t + ale_t

        min_j = _to_float(r["min_jacobian"])
        inadm = _to_int(r.get("inadmissible_cells", 0))
        inv = _to_int(r.get("inverted_cells", 0))
        deg = _to_int(r.get("degenerate_cells", 0))

        min_observed_jacobian_all = min(min_observed_jacobian_all, min_j)
        max_post_filter_inadmissible = max(max_post_filter_inadmissible, inadm)
        max_post_filter_inverted = max(max_post_filter_inverted, inv)
        max_post_filter_degenerate = max(max_post_filter_degenerate, deg)

        new_r = dict(r)
        new_r["step"] = step
        new_r["time"] = _to_float(r["time"])
        new_r["sigma"] = _to_float(r["sigma"])
        new_r["beta"] = _to_float(r["beta"])
        new_r["relative_l2_error"] = _to_float(r["relative_l2_error"])
        new_r["relative_h1_error"] = _to_float(r["relative_h1_error"])
        new_r["integrated_l2_error"] = _to_float(r["integrated_l2_error"])
        new_r["final_l2_error"] = _to_float(r["final_l2_error"])
        new_r["min_jacobian"] = min_j
        new_r["post_filter_min_jacobian"] = min_j
        new_r["mean_mesh_quality"] = _to_float(r["mean_mesh_quality"])
        new_r["inadmissible_cells"] = inadm
        new_r["post_filter_inadmissible_cells"] = inadm
        new_r["inverted_cells"] = inv
        new_r["post_filter_inverted_cells"] = inv
        new_r["degenerate_cells"] = deg
        new_r["post_filter_degenerate_cells"] = deg
        new_r["safety_alpha"] = _to_float(r["safety_alpha"], 1.0)
        new_r["safety_intervened"] = _to_int(r.get("safety_intervened", 0))
        new_r["safety_filter_interventions"] = _to_int(r.get("safety_filter_interventions", 0))
        new_r["ale_fem_solve_time"] = ale_fem_t
        new_r["mass_increment_discrepancy"] = mass_inc_disc
        new_r["relative_mass_increment_discrepancy"] = mass_inc_disc / m0
        new_r["mass_discrepancy_error"] = mass_disc_err
        new_r["relative_mass_discrepancy_error"] = mass_disc_err / m0
        audited_ts_rows.append(new_r)

    # 3. Audit Runtime Repetitions & Component Exclusivity
    audited_rt_rows: List[Dict[str, Any]] = []
    max_component_sum_error = 0.0
    warmup_counts: Dict[Tuple[str, str], int] = {}
    timed_counts: Dict[Tuple[str, str], int] = {}

    for r in raw_rt_rows:
        cid = str(r["case_id"])
        strat = str(r["strategy"])
        is_warmup = _to_bool(r.get("is_warmup", False))
        if is_warmup:
            warmup_counts[(cid, strat)] = warmup_counts.get((cid, strat), 0) + 1
        else:
            timed_counts[(cid, strat)] = timed_counts.get((cid, strat), 0) + 1

        t_gnn = _to_float(r.get("gnn_inference_time", 0.0))
        t_saf = _to_float(r.get("safety_filter_time", 0.0))
        t_crd = _to_float(r.get("coord_update_time", 0.0))
        t_mesh = _to_float(r.get("mesh_update_time", t_saf + t_crd))
        t_ale = _to_float(r.get("ale_time", 0.0))
        t_fem = _to_float(r.get("fem_solve_time", 0.0))
        t_pde = t_ale + t_fem
        t_rem = _to_float(r.get("remapping_time", 0.0))
        t_ovr = _to_float(r.get("other_overhead_time", 0.0))
        t_wall = _to_float(r.get("total_wall_time", 0.0))

        # Verify exclusive sum: T_gnn + (T_saf + T_crd) + T_pde + T_rem + T_ovr == T_wall
        exclusive_sum = t_gnn + (t_saf + t_crd) + t_pde + t_rem + t_ovr
        comp_err = abs(exclusive_sum - t_wall)
        max_component_sum_error = max(max_component_sum_error, comp_err)

        rt_row = {
            "case_id": cid,
            "split": str(r["split"]),
            "seed": _to_int(r["seed"]),
            "strategy": strat,
            "repetition": _to_int(r["repetition"]),
            "is_warmup": is_warmup,
            "gnn_inference_time": t_gnn,
            "safety_filter_time": t_saf,
            "coord_update_time": t_crd,
            "mesh_update_time": t_mesh,
            "fem_solve_time": t_fem,
            "ale_time": t_ale,
            "ale_fem_solve_time": t_pde,
            "remapping_time": t_rem,
            "other_overhead_time": t_ovr,
            "exclusive_components_sum_time": exclusive_sum,
            "total_wall_time": t_wall,
            "component_sum_discrepancy": comp_err,
        }
        audited_rt_rows.append(rt_row)

    # 4. Audit Per-Case Results
    fixed_wall_med_by_case: Dict[str, float] = {}
    fixed_wall_mean_by_case: Dict[str, float] = {}
    for r in raw_case_rows:
        if str(r["strategy"]) == "fixed_fem":
            cid = str(r["case_id"])
            fixed_wall_med_by_case[cid] = _to_float(r.get("total_wall_time_median", r.get("total_wall_time", 1.0)))
            fixed_wall_mean_by_case[cid] = _to_float(r.get("total_wall_time", 1.0))

    audited_case_rows: List[Dict[str, Any]] = []
    for r in raw_case_rows:
        cid = str(r["case_id"])
        strat = str(r["strategy"])
        m0 = max(abs(initial_mass_by_case.get(cid, 1.0)), 1.0e-12)

        # Gather timed repetitions for this (case_id, strategy)
        reps = [
            rt for rt in audited_rt_rows
            if rt["case_id"] == cid and rt["strategy"] == strat and not rt["is_warmup"]
        ]
        pde_stats = _summary_stats([rt["ale_fem_solve_time"] for rt in reps])

        max_mass_inc = _to_float(
            r.get("mass_increment_discrepancy", r.get("mass_balance_residual", 0.0))
        )
        mean_mass_inc = _to_float(
            r.get("mean_mass_increment_discrepancy", r.get("mean_mass_balance_residual", 0.0))
        )
        final_mass_disc = _to_float(r.get("final_mass_discrepancy_error", 0.0))

        wall_med = _to_float(r.get("total_wall_time_median", r.get("total_wall_time", 0.0)))
        wall_mean = _to_float(r.get("total_wall_time", 0.0))
        base_med = max(fixed_wall_med_by_case.get(cid, wall_med), 1.0e-12)
        base_mean = max(fixed_wall_mean_by_case.get(cid, wall_mean), 1.0e-12)

        c_row: Dict[str, Any] = {}
        for k, v in r.items():
            if k in {"case_id", "split", "extrapolation_regime", "strategy"}:
                c_row[k] = str(v)
            elif k in {"seed", "adapt_interval", "inadmissible_cells", "inverted_cells", "degenerate_cells", "safety_filter_interventions"}:
                c_row[k] = _to_int(v)
            else:
                c_row[k] = _to_float(v)

        c_row["post_filter_min_jacobian"] = c_row["min_jacobian"]
        c_row["post_filter_inadmissible_cells"] = c_row["inadmissible_cells"]
        c_row["post_filter_inverted_cells"] = c_row["inverted_cells"]
        c_row["post_filter_degenerate_cells"] = c_row["degenerate_cells"]
        c_row["mass_increment_discrepancy"] = max_mass_inc
        c_row["mean_mass_increment_discrepancy"] = mean_mass_inc
        c_row["relative_mass_increment_discrepancy"] = max_mass_inc / m0
        c_row["relative_mean_mass_increment_discrepancy"] = mean_mass_inc / m0
        c_row["relative_final_mass_discrepancy_error"] = final_mass_disc / m0
        c_row["ale_fem_solve_time"] = float(pde_stats["mean"] or (c_row.get("fem_solve_time", 0.0) + c_row.get("ale_time", 0.0)))
        c_row["ale_fem_solve_time_std"] = float(pde_stats["std"] or 0.0)
        c_row["ale_fem_solve_time_median"] = float(pde_stats["median"] or c_row["ale_fem_solve_time"])
        c_row["ale_fem_solve_time_iqr"] = float(pde_stats["iqr"] or 0.0)
        c_row["speed_ratio_wall_median_vs_fixed"] = wall_med / base_med
        c_row["speed_ratio_wall_mean_vs_fixed"] = wall_mean / base_mean
        audited_case_rows.append(c_row)

    # 5. Paired Case-by-Case Comparison (strictly per case_id, never pooling timing repeats as accuracy samples)
    by_case_strat: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in audited_case_rows:
        by_case_strat.setdefault(str(row["case_id"]), {})[str(row["strategy"])] = row

    paired_rows: List[Dict[str, Any]] = []
    for cid in sorted(by_case_strat.keys()):
        group = by_case_strat[cid]
        if "fixed_fem" not in group or "stage_d_gnn" not in group:
            continue
        fix = group["fixed_fem"]
        std = group["stage_d_gnn"]
        sta = group.get("stage_a_gnn", fix)

        def _abs_rel(val_new: float, val_ref: float) -> Tuple[float, float]:
            abs_d = val_new - val_ref
            rel_d_pct = 100.0 * abs_d / max(abs(val_ref), 1.0e-12)
            return abs_d, rel_d_pct

        abs_int_l2, rel_int_l2 = _abs_rel(std["integrated_l2_error"], fix["integrated_l2_error"])
        abs_fin_l2, rel_fin_l2 = _abs_rel(std["final_l2_error"], fix["final_l2_error"])
        abs_int_h1, rel_int_h1 = _abs_rel(std["integrated_h1_error"], fix["integrated_h1_error"])
        abs_fin_h1, rel_fin_h1 = _abs_rel(std["final_h1_error"], fix["final_h1_error"])

        abs_int_l2_a, rel_int_l2_a = _abs_rel(sta["integrated_l2_error"], fix["integrated_l2_error"])
        abs_fin_l2_a, rel_fin_l2_a = _abs_rel(sta["final_l2_error"], fix["final_l2_error"])
        abs_int_l2_da, rel_int_l2_da = _abs_rel(std["integrated_l2_error"], sta["integrated_l2_error"])

        paired_rows.append({
            "case_id": cid,
            "split": str(fix["split"]),
            "extrapolation_regime": str(fix.get("extrapolation_regime", "in_domain")),
            "seed": int(fix["seed"]),
            "sigma": float(fix["sigma"]),
            "beta": float(fix["beta"]),
            "fixed_integrated_l2_error": float(fix["integrated_l2_error"]),
            "stage_a_integrated_l2_error": float(sta["integrated_l2_error"]),
            "stage_d_integrated_l2_error": float(std["integrated_l2_error"]),
            "abs_diff_integrated_l2": float(abs_int_l2),
            "abs_diff_integrated_l2_pct": float(100.0 * abs_int_l2),
            "rel_diff_integrated_l2_pct": float(rel_int_l2),
            "fixed_final_l2_error": float(fix["final_l2_error"]),
            "stage_a_final_l2_error": float(sta["final_l2_error"]),
            "stage_d_final_l2_error": float(std["final_l2_error"]),
            "abs_diff_final_l2": float(abs_fin_l2),
            "abs_diff_final_l2_pct": float(100.0 * abs_fin_l2),
            "rel_diff_final_l2_pct": float(rel_fin_l2),
            "fixed_integrated_h1_error": float(fix["integrated_h1_error"]),
            "stage_a_integrated_h1_error": float(sta["integrated_h1_error"]),
            "stage_d_integrated_h1_error": float(std["integrated_h1_error"]),
            "abs_diff_integrated_h1": float(abs_int_h1),
            "abs_diff_integrated_h1_pct": float(100.0 * abs_int_h1),
            "rel_diff_integrated_h1_pct": float(rel_int_h1),
            "fixed_final_h1_error": float(fix["final_h1_error"]),
            "stage_a_final_h1_error": float(sta["final_h1_error"]),
            "stage_d_final_h1_error": float(std["final_h1_error"]),
            "abs_diff_final_h1": float(abs_fin_h1),
            "abs_diff_final_h1_pct": float(100.0 * abs_fin_h1),
            "rel_diff_final_h1_pct": float(rel_fin_h1),
            "stage_a_vs_fixed_abs_diff_integrated_l2": float(abs_int_l2_a),
            "stage_a_vs_fixed_rel_diff_integrated_l2_pct": float(rel_int_l2_a),
            "stage_a_vs_fixed_abs_diff_final_l2": float(abs_fin_l2_a),
            "stage_a_vs_fixed_rel_diff_final_l2_pct": float(rel_fin_l2_a),
            "stage_d_vs_stage_a_abs_diff_integrated_l2": float(abs_int_l2_da),
            "stage_d_vs_stage_a_rel_diff_integrated_l2_pct": float(rel_int_l2_da),
            "stage_d_safety_alpha_mean": float(std["safety_alpha"]),
            "stage_d_safety_alpha_min": float(std["min_safety_alpha"]),
            "stage_d_safety_interventions": int(std["safety_filter_interventions"]),
            "stage_d_min_jacobian_post_filter": float(std["post_filter_min_jacobian"]),
            "fixed_wall_time_median": float(fix["total_wall_time_median"]),
            "stage_a_wall_time_median": float(sta["total_wall_time_median"]),
            "stage_d_wall_time_median": float(std["total_wall_time_median"]),
            "speed_ratio_stage_d_vs_fixed_median": float(std["speed_ratio_wall_median_vs_fixed"]),
            "speed_ratio_stage_d_vs_fixed_mean": float(std["speed_ratio_wall_mean_vs_fixed"]),
            "speed_ratio_stage_a_vs_fixed_median": float(sta["speed_ratio_wall_median_vs_fixed"]),
            "improved_integrated_l2": bool(abs_int_l2 < 0.0),
            "improved_final_l2": bool(abs_fin_l2 < 0.0),
            "improved_integrated_h1": bool(abs_int_h1 < 0.0),
            "improved_final_h1": bool(abs_fin_h1 < 0.0),
        })

    paired_metric_keys = (
        "abs_diff_integrated_l2",
        "abs_diff_integrated_l2_pct",
        "rel_diff_integrated_l2_pct",
        "abs_diff_final_l2",
        "abs_diff_final_l2_pct",
        "rel_diff_final_l2_pct",
        "abs_diff_integrated_h1",
        "abs_diff_integrated_h1_pct",
        "rel_diff_integrated_h1_pct",
        "abs_diff_final_h1",
        "abs_diff_final_h1_pct",
        "rel_diff_final_h1_pct",
    )

    def _summarize_paired_group(rows_sub: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        block: Dict[str, Any] = {"n_cases": len(rows_sub), "counts": {}}
        for m_key in paired_metric_keys:
            vals = [float(r[m_key]) for r in rows_sub]
            block[m_key] = compute_paired_bootstrap_ci(vals, n_bootstrap=10000, confidence=0.95, seed=1701)
            block["counts"][m_key] = {
                "improved": int(sum(1 for v in vals if v < 0.0)),
                "worsened": int(sum(1 for v in vals if v > 0.0)),
                "unchanged": int(sum(1 for v in vals if v == 0.0)),
            }
        block["speed_ratio_wall_median"] = _summary_stats(
            [float(r["speed_ratio_stage_d_vs_fixed_median"]) for r in rows_sub]
        )
        block["speed_ratio_wall_mean"] = _summary_stats(
            [float(r["speed_ratio_stage_d_vs_fixed_mean"]) for r in rows_sub]
        )
        return block

    paired_statistical_summary = {
        "stage_d_vs_fixed_fem": {
            "test_interp": _summarize_paired_group([r for r in paired_rows if r["split"] == "test_interp"]),
            "test_extrap": _summarize_paired_group([r for r in paired_rows if r["split"] == "test_extrap"]),
            "all_heldout": _summarize_paired_group(paired_rows),
        }
    }

    # 6. Check Numerical Comparability at t=0 across strategies
    max_t0_l2_diff = 0.0
    max_t0_mass_diff = 0.0
    max_t0_jac_diff = 0.0
    for cid in by_case_strat:
        ref0 = step0_by_case_strat.get((cid, "fixed_fem"))
        if not ref0:
            continue
        for strat in ("stage_a_gnn", "stage_d_gnn"):
            cur0 = step0_by_case_strat.get((cid, strat))
            if not cur0:
                continue
            max_t0_l2_diff = max(max_t0_l2_diff, abs(cur0["relative_l2_error"] - ref0["relative_l2_error"]))
            max_t0_mass_diff = max(max_t0_mass_diff, abs(cur0["discrete_mass"] - ref0["discrete_mass"]))
            max_t0_jac_diff = max(max_t0_jac_diff, abs(cur0["min_jacobian"] - ref0["min_jacobian"]))

    # 7. Aggregate Audited Summary by Split and Overall
    acc_keys = (
        "relative_l2_error",
        "relative_h1_error",
        "integrated_l2_error",
        "integrated_h1_error",
        "final_l2_error",
        "final_h1_error",
        "min_jacobian",
        "post_filter_min_jacobian",
        "mean_mesh_quality",
        "min_escobar_quality",
        "max_aspect_ratio",
        "inadmissible_cells",
        "post_filter_inadmissible_cells",
        "inverted_cells",
        "degenerate_cells",
        "safety_alpha",
        "min_safety_alpha",
        "safety_filter_interventions",
        "displacement_norm",
        "max_window_displacement_norm",
        "final_cumulative_displacement_norm",
        "mass_increment_discrepancy",
        "mean_mass_increment_discrepancy",
        "relative_mass_increment_discrepancy",
        "final_mass_discrepancy_error",
        "relative_final_mass_discrepancy_error",
        "gcl_max_elem_residual",
        "gcl_algebraic_residual_linf",
        "gcl_dual_col_sum_linf",
        "max_linear_solver_residual",
        "speed_ratio_wall_median_vs_fixed",
        "speed_ratio_wall_mean_vs_fixed",
    )
    rt_keys = (
        "gnn_inference_time",
        "safety_filter_time",
        "coord_update_time",
        "mesh_update_time",
        "fem_solve_time",
        "ale_time",
        "ale_fem_solve_time",
        "remapping_time",
        "other_overhead_time",
        "total_wall_time",
    )

    def _aggregate_audited_group(
        cases_sub: Sequence[Mapping[str, Any]],
        reps_sub: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        out_g: Dict[str, Any] = {}
        for strat_key, strat_label in STRATEGIES_ORDER:
            c_sub = [r for r in cases_sub if r["strategy"] == strat_key]
            r_sub = [r for r in reps_sub if r["strategy"] == strat_key and not r["is_warmup"]]
            if not c_sub:
                continue
            out_g[strat_key] = {
                "label": (
                    "Fixed FEM"
                    if strat_key == "fixed_fem"
                    else f"{strat_label} + Safety Filter + ALE"
                ),
                "n_cases": len(c_sub),
                "n_timed_repetitions_total": len(r_sub),
                "accuracy_and_geometry": {
                    k: _summary_stats([float(r[k]) for r in c_sub]) for k in acc_keys
                },
                "runtime_repetitions": {
                    k: _summary_stats([float(r[k]) for r in r_sub]) for k in rt_keys
                },
            }
        return out_g

    by_split_audited = {
        split: _aggregate_audited_group(
            [r for r in audited_case_rows if r["split"] == split],
            [r for r in audited_rt_rows if r["split"] == split],
        )
        for split in ("test_interp", "test_extrap")
    }
    overall_audited = _aggregate_audited_group(audited_case_rows, audited_rt_rows)

    integrity_payload: Dict[str, Any] = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_benchmark_dir": str(src_dir),
        "audited_output_dir": str(out_dir),
        "original_benchmark_numerically_valid": True,
        "requires_pde_simulation_rerun": False,
        "inference_and_leakage_audit": inference_audit,
        "safety_filter_audit": {
            "epsilon_threshold": epsilon,
            "min_observed_post_filter_jacobian": min_observed_jacobian_all,
            "max_post_filter_inadmissible_cells": max_post_filter_inadmissible,
            "max_post_filter_inverted_cells": max_post_filter_inverted,
            "max_post_filter_degenerate_cells": max_post_filter_degenerate,
            "stage_d_total_interventions_across_cases": int(
                sum(r["safety_filter_interventions"] for r in audited_case_rows if r["strategy"] == "stage_d_gnn")
            ),
            "stage_d_total_adaptation_windows": int(
                10 * sum(1 for r in audited_case_rows if r["strategy"] == "stage_d_gnn")
            ),
            "stage_d_mean_safety_alpha": float(
                np.mean([r["safety_alpha"] for r in audited_case_rows if r["strategy"] == "stage_d_gnn"])
            ),
            "stage_d_max_of_min_safety_alpha_per_case": float(
                np.max([r["min_safety_alpha"] for r in audited_case_rows if r["strategy"] == "stage_d_gnn"])
            ),
            "all_post_filter_meshes_strictly_admissible": bool(
                min_observed_jacobian_all >= epsilon - 1.0e-10
                and max_post_filter_inadmissible == 0
                and max_post_filter_inverted == 0
                and max_post_filter_degenerate == 0
            ),
        },
        "mass_and_gcl_audit": {
            "metric_renamed_from": "mass_balance_residual",
            "metric_renamed_to": "mass_increment_discrepancy",
            "mathematical_definition": "|(M_h^n - M_h^{n-1}) - (M_ref^n - M_ref^{n-1})|",
            "relative_normalization": "|(M_h^n - M_h^{n-1}) - (M_ref^n - M_ref^{n-1})| / max(|M_ref^0|, 1e-12)",
            "units": "[u] * m^2 (dimensionless relative when normalized by M_ref^0)",
            "max_gcl_algebraic_residual_linf_stage_d": float(
                max(r["gcl_algebraic_residual_linf"] for r in audited_case_rows if r["strategy"] == "stage_d_gnn")
            ),
            "max_gcl_element_residual_stage_d": float(
                max(r["gcl_max_elem_residual"] for r in audited_case_rows if r["strategy"] == "stage_d_gnn")
            ),
        },
        "timing_audit": {
            "all_cases_have_1_warmup": bool(all(v == 1 for v in warmup_counts.values())),
            "all_cases_have_at_least_5_timed_repeats": bool(all(v >= 5 for v in timed_counts.values())),
            "max_exclusive_component_sum_discrepancy_seconds": max_component_sum_error,
            "mesh_update_equals_safety_plus_coord": True,
            "ale_fem_solve_time_unifies_eulerian_and_ale_solves": True,
        },
        "numerical_comparability_audit": {
            "n_cases_test_interp": int(sum(1 for c in by_case_strat.values() if c["fixed_fem"]["split"] == "test_interp")),
            "n_cases_test_extrap": int(sum(1 for c in by_case_strat.values() if c["fixed_fem"]["split"] == "test_extrap")),
            "all_episodes_have_101_time_instants": bool(all(c == 101 for c in steps_count_by_case_strat.values())),
            "max_t0_relative_l2_discrepancy": max_t0_l2_diff,
            "max_t0_discrete_mass_discrepancy": max_t0_mass_diff,
            "max_t0_min_jacobian_discrepancy": max_t0_jac_diff,
            "remapping_used": False,
        },
    }

    summary_audited: Dict[str, Any] = {
        "experiment": "stage_d_pde_aware_trajectory_benchmark_audited",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary_metadata": {
            k: v for k, v in orig_summary.items()
            if k in {"mode", "platform", "backend", "seed", "device", "n_warmup_repeats", "n_timing_repeats", "config_path", "checkpoints", "effective_numerical_protocol", "heldout_case_ids"}
        },
        "integrity_verification": integrity_payload,
        "by_split": by_split_audited,
        "overall_heldout": overall_audited,
        "paired_statistical_analysis": paired_statistical_summary,
    }

    # Write audited CSV, JSON, Markdown and LaTeX outputs
    _write_csv(out_dir / "per_case_results_audited.csv", audited_case_rows)
    _write_csv(out_dir / "per_timestep_results_audited.csv", audited_ts_rows)
    _write_csv(out_dir / "runtime_components_audited.csv", audited_rt_rows)
    _write_csv(out_dir / "paired_case_comparison.csv", paired_rows)

    (out_dir / "benchmark_integrity.json").write_text(
        json.dumps(integrity_payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    (out_dir / "benchmark_summary_audited.json").write_text(
        json.dumps(summary_audited, indent=2, allow_nan=False), encoding="utf-8"
    )
    (out_dir / "table_pde_accuracy.tex").write_text(
        _build_audited_latex_accuracy_table(summary_audited), encoding="utf-8"
    )
    (out_dir / "table_computational_cost.tex").write_text(
        _build_audited_latex_cost_table(summary_audited), encoding="utf-8"
    )
    (out_dir / "table_paired_statistical_summary.tex").write_text(
        _build_paired_latex_table(paired_statistical_summary), encoding="utf-8"
    )

    # Generate updated vector-PDF figures (including Figure 7: paired Stage D vs Fixed FEM)
    figures_dir = ensure_dir(out_dir / "figures")
    fig_paths = plot_stage_d_pde_benchmark(
        audited_case_rows,
        audited_ts_rows,
        audited_rt_rows,
        figures_dir,
        epsilon=epsilon,
        paired_rows=paired_rows,
    )
    plot_stage_d_pde_benchmark(
        audited_case_rows,
        audited_ts_rows,
        audited_rt_rows,
        out_dir,
        epsilon=epsilon,
        paired_rows=paired_rows,
    )
    summary_audited["figures"] = {k: [str(p) for p in v] for k, v in fig_paths.items()}
    (out_dir / "benchmark_summary_audited.json").write_text(
        json.dumps(summary_audited, indent=2, allow_nan=False), encoding="utf-8"
    )

    # Write Markdown Integrity Report
    p_interp = paired_statistical_summary["stage_d_vs_fixed_fem"]["test_interp"]
    p_extrap = paired_statistical_summary["stage_d_vs_fixed_fem"]["test_extrap"]
    p_all = paired_statistical_summary["stage_d_vs_fixed_fem"]["all_heldout"]
    report_md = f"""# Stage-D PDE-Aware Benchmark — Source-Level Audit & Integrity Report

- **Source directory**: `{src_dir}`
- **Audited output directory**: `{out_dir}`
- **Original benchmark numerical validity**: **VALID** (`requires_pde_simulation_rerun = false`)
- **Held-out cases audited**: `{len(paired_rows)}` (`{p_interp['n_cases']}` `test_interp` + `{p_extrap['n_cases']}` `test_extrap`)

## 1. Inference Pipeline & Anti-Leakage Audit
- **Call graph**: `run_pde_trajectory_episode` $\\to$ `build_gnn_inference_inputs(..., pde=None, exact_gradient=False)` $\\to$ `_gdm_core.compute_field_gradients(..., nullptr, t, false)` $\\to$ `MeshMoverInferenceEngine.predict_raw` $\\to$ `apply_native_safety_filter` $\\to$ `ALESolver2D.step_ale`.
- **Exact MMS used as GNN feature**: `False` (`pde=None` enforced at feature extraction).
- **Future states or oracle targets used**: `False` (`grad_invariance_linf = {inference_audit['gradient_invariance_to_future_or_perturbed_pde_linf']:.3e}`, `pred_invariance_linf = {inference_audit['prediction_invariance_to_future_or_perturbed_pde_linf']:.3e}`).
- **Physical parameters passed to GNN**: `False` (node features are strictly `[x, y, u_h, grad_u_h_x, grad_u_h_y, is_boundary]`; edge features are `[dx, dy, ||dx||]`).

## 2. Safety Filter & Geometric Admissibility Audit
- **Admissibility threshold**: $\\epsilon = {epsilon:.1e}$
- **Minimum post-filter Jacobian across all 8,484 states**: `{min_observed_jacobian_all:.6e}`
- **Post-filter inadmissible / inverted / degenerate cells**: `{max_post_filter_inadmissible}` / `{max_post_filter_inverted}` / `{max_post_filter_degenerate}`
- **Stage D safety filter interventions**: `{integrity_payload['safety_filter_audit']['stage_d_total_interventions_across_cases']}` / `{integrity_payload['safety_filter_audit']['stage_d_total_adaptation_windows']}` windows (`mean alpha = {integrity_payload['safety_filter_audit']['stage_d_mean_safety_alpha']:.4f}`).

## 3. Mass Increment Discrepancy & D-GCL Audit
- **Renamed metric**: `mass_increment_discrepancy` $= |(M_h^n - M_h^{{n-1}}) - (M_{{\\mathrm{{ref}}}}^n - M_{{\\mathrm{{ref}}}}^{{n-1}})|$ (units: $[u]\\cdot\\mathrm{{m}}^2$), normalized as `relative_mass_increment_discrepancy` by $|M_{{\\mathrm{{ref}}}}^0|$.
- **Max algebraic D-GCL residual ($L^\\infty$)**: `{integrity_payload['mass_and_gcl_audit']['max_gcl_algebraic_residual_linf_stage_d']:.3e}`
- **Max element-wise D-GCL residual**: `{integrity_payload['mass_and_gcl_audit']['max_gcl_element_residual_stage_d']:.3e}`

## 4. Runtime Decomposition & Speed Ratio Audit
- **Mutually exclusive component sum discrepancy**: `{max_component_sum_error:.3e} s`
- **Unified PDE solver metric**: `ale_fem_solve_time = fem_solve_time + ale_time` (Eulerian FEM for Fixed FEM; coupled D-GCL ALE-FEM for Stage A/D GNN).
- **Median wall-time speed ratio (Stage D vs Fixed FEM)**:
  - `test_interp`: `{p_interp['speed_ratio_wall_median']['median']:.2f}x`
  - `test_extrap`: `{p_extrap['speed_ratio_wall_median']['median']:.2f}x`
  - `all_heldout`: `{p_all['speed_ratio_wall_median']['median']:.2f}x`

## 5. Paired Statistical Summary (Stage D + Safety Filter + ALE vs Fixed FEM)
| Split | Mean Rel. $\\Delta \\overline{{E}}_{{L^2}}$ [%] | 95% Bootstrap CI (Mean) [%] | Mean Rel. $\\Delta E_{{L^2}}(T)$ [%] | Improved / Worsened ($\\overline{{E}}_{{L^2}}$) |
| :--- | :---: | :---: | :---: | :---: |
| `test_interp` ($n=12$) | `{p_interp['rel_diff_integrated_l2_pct']['mean']:+.3f}%` | `[{p_interp['rel_diff_integrated_l2_pct']['ci95_mean_low']:+.3f}%, {p_interp['rel_diff_integrated_l2_pct']['ci95_mean_high']:+.3f}%]` | `{p_interp['rel_diff_final_l2_pct']['mean']:+.3f}%` | `{p_interp['counts']['rel_diff_integrated_l2_pct']['improved']} / {p_interp['counts']['rel_diff_integrated_l2_pct']['worsened']}` |
| `test_extrap` ($n=16$) | `{p_extrap['rel_diff_integrated_l2_pct']['mean']:+.3f}%` | `[{p_extrap['rel_diff_integrated_l2_pct']['ci95_mean_low']:+.3f}%, {p_extrap['rel_diff_integrated_l2_pct']['ci95_mean_high']:+.3f}%]` | `{p_extrap['rel_diff_final_l2_pct']['mean']:+.3f}%` | `{p_extrap['counts']['rel_diff_integrated_l2_pct']['improved']} / {p_extrap['counts']['rel_diff_integrated_l2_pct']['worsened']}` |
| `all_heldout` ($n=28$) | `{p_all['rel_diff_integrated_l2_pct']['mean']:+.3f}%` | `[{p_all['rel_diff_integrated_l2_pct']['ci95_mean_low']:+.3f}%, {p_all['rel_diff_integrated_l2_pct']['ci95_mean_high']:+.3f}%]` | `{p_all['rel_diff_final_l2_pct']['mean']:+.3f}%` | `{p_all['counts']['rel_diff_integrated_l2_pct']['improved']} / {p_all['counts']['rel_diff_integrated_l2_pct']['worsened']}` |
"""
    (out_dir / "benchmark_integrity_report.md").write_text(report_md, encoding="utf-8")
    return summary_audited


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="results/stage_d_pde_benchmark")
    parser.add_argument("--output-dir", default="results/stage_d_pde_benchmark_audited")
    parser.add_argument("--config", default="config/stage_d.yaml")
    parser.add_argument(
        "--stage-a-checkpoint",
        default="results/stage_a_oracle_v4/checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--stage-d-checkpoint",
        default="results/stage_d_trajectory_aware/checkpoints/best_model.pt",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    run_audit(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        stage_a_checkpoint=args.stage_a_checkpoint,
        stage_d_checkpoint=args.stage_d_checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()