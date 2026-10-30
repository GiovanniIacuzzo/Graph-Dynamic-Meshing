"""
generate_dataset.py
Script di generazione offline del dataset per Graph-Dynamic Meshing (Stage A).

Pipeline di esecuzione:
    1. Lettura configurazione globale config/stage_a.yaml
    2. Creazione della mesh iniziale triangolare non strutturata/uniforme
    3. Campionamento dei parametri fisici per le 4 partizioni:
       - train
       - validation
       - test_interp (interpolazione parametrica)
       - test_extrap (estrapolazione out-of-distribution)
    4. Valutazione del campo analitico e delle sue derivate
    5. Ottimizzazione variazionale a budget finito -> generazione di un riferimento X_ref e Delta X_ref
    6. Validazione geometrica di ammissibilità (min J_e >= epsilon)
    7. Serializzazione dei campioni in data/raw/<sample_id>.pkl
    8. Creazione del manifest formale data/manifest.json
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Inclusione root nel path di sistema per esecuzione diretta
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "python"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gdm.native_api import NativeMeshAdapter, compute_native_mesh_metrics_arrays
from oracle.field import GaussianBumpField
from oracle.optimizer import OfflineMeshOptimizer
from utils.utils import load_config, ensure_dir, save_pickle, set_seed, setup_logger, Timer
from plotting.dataset import audit_dataset


def create_initial_mesh_2d(
    domain: Dict[str, float],
    nx: int = 32,
    ny: int = 32
) -> NativeMeshAdapter:
    """
    Costruisce una triangolazione 2D conforme regolare sul dominio rettangolare.
    Suddivide ciascun rettangolo della griglia strutturata in due triangoli.
    """
    # La triangolazione è costruita dal MeshState nativo C++/deal.II.
    # Python resta responsabile del campionamento e della serializzazione del dataset.
    return NativeMeshAdapter.from_domain(domain, nx=nx, ny=ny)


def sample_parameters(config: Dict[str, Any], seed: int = 42) -> Dict[str, List[Dict[str, float]]]:
    """
    Genera i parametri fisici della Gaussian bump garantendo che i test set
    rispettino rigorosamente i requisiti di interpolazione ed estrapolazione.
    """
    rng = np.random.default_rng(seed)
    f_cfg = config["field"]["sampling_ranges"]
    splits_cfg = config["dataset"]["splits"]
    extrap_cfg = config["dataset"]["extrapolation_ranges"]

    sampled: Dict[str, List[Dict[str, float]]] = {
        "train": [],
        "validation": [],
        "test_interp": [],
        "test_extrap": []
    }

    # 1. Train, Validation e Test-Interp campionati all'interno del dominio standard
    for split_name in ["train", "validation", "test_interp"]:
        n_samples = splits_cfg[f"num_{split_name}"]
        for _ in range(n_samples):
            sampled[split_name].append({
                "mu_x": float(rng.uniform(f_cfg["mu_x_min"], f_cfg["mu_x_max"])),
                "mu_y": float(rng.uniform(f_cfg["mu_y_min"], f_cfg["mu_y_max"])),
                "sigma": float(rng.uniform(f_cfg["sigma_min"], f_cfg["sigma_max"])),
                "amplitude": float(rng.uniform(f_cfg["amplitude_min"], f_cfg["amplitude_max"]))
            })

    # 2. Test-Extrap: parametri campionati al di fuori del range (campi molto ripidi o molto dispersi)
    n_extrap = splits_cfg["num_test_extrap"]
    for i in range(n_extrap):
        if i % 2 == 0:
            # Regime a gradiente estremo (sigma molto piccolo)
            s_min, s_max = extrap_cfg["sigma_extrap_low"]
        else:
            # Regime diffuso (sigma molto grande)
            s_min, s_max = extrap_cfg["sigma_extrap_high"]

        sampled["test_extrap"].append({
            "mu_x": float(rng.uniform(f_cfg["mu_x_min"], f_cfg["mu_x_max"])),
            "mu_y": float(rng.uniform(f_cfg["mu_y_min"], f_cfg["mu_y_max"])),
            "sigma": float(rng.uniform(s_min, s_max)),
            "amplitude": float(rng.uniform(f_cfg["amplitude_min"], f_cfg["amplitude_max"]))
        })

    return sampled


def _mark_reference_target_acceptance(
    result: Dict[str, Any],
    *,
    initial_objective: float,
    initial_interpolation_error: float,
    minimum_objective_reduction_fraction: float,
    minimum_interpolation_error_reduction_fraction: float,
) -> bool:
    """Accept a finite-budget reference without claiming first-order convergence."""
    metrics = result.setdefault("final_metrics", {})
    final_objective = float(metrics.get("total_objective", float("nan")))
    final_interpolation_error = float(metrics.get("interpolation_error", float("nan")))
    objective_reduction = (
        (initial_objective - final_objective) / max(abs(initial_objective), 1.0e-15)
        if np.isfinite(initial_objective) and np.isfinite(final_objective) else float("nan")
    )
    interpolation_reduction = (
        (initial_interpolation_error - final_interpolation_error)
        / max(abs(initial_interpolation_error), 1.0e-15)
        if np.isfinite(initial_interpolation_error) and np.isfinite(final_interpolation_error)
        else float("nan")
    )
    candidate_valid = metrics.get("candidate_is_valid") is True
    safety_repaired = bool(metrics.get("safety_filter_active", False))
    displacement = np.asarray(result.get("target_displacement", np.array([np.nan])), dtype=np.float64)
    finite_diagnostics = bool(
        np.isfinite(final_objective) and np.isfinite(final_interpolation_error)
        and np.isfinite(objective_reduction) and np.isfinite(interpolation_reduction)
        and np.isfinite(displacement).all()
    )
    checks = {
        "finite_diagnostics": finite_diagnostics,
        "raw_candidate_valid": candidate_valid,
        "safety_filter_inactive": not safety_repaired,
        "objective_reduction_sufficient": bool(
            np.isfinite(objective_reduction)
            and objective_reduction >= minimum_objective_reduction_fraction
        ),
        "interpolation_reduction_sufficient": bool(
            np.isfinite(interpolation_reduction)
            and interpolation_reduction >= minimum_interpolation_error_reduction_fraction
        ),
    }
    accepted = all(checks.values())
    reason = "accepted_budgeted_reference" if accepted else ";".join(
        key for key, passed in checks.items() if not passed
    )
    result["target_accepted"] = bool(accepted)
    result["target_acceptance_checks"] = checks
    result["target_acceptance_reason"] = reason
    metrics["objective_reduction_fraction"] = float(objective_reduction)
    metrics["interpolation_error_reduction_fraction"] = float(interpolation_reduction)
    metrics["target_accepted"] = bool(accepted)
    metrics["target_acceptance_checks"] = checks
    metrics["target_acceptance_reason"] = reason
    metrics["first_order_stationary"] = bool(result.get("first_order_stationary", False))
    metrics["optimizer_success"] = bool(result.get("success", False))
    return bool(accepted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/stage_a.yaml", help="Experiment configuration YAML.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting samples at the configured output paths.")
    parser.add_argument("--pilot", action="store_true", help="Run an isolated 10-sample diagnostic pilot; it cannot be used for training.")
    parser.add_argument("--limit-per-split", type=int, default=None, help="Optional pilot-only cap for every split.")
    args = parser.parse_args()

    # 1. Caricamento configurazione e logger
    config_path = Path(args.config)
    assert config_path.exists(), f"File di configurazione mancante: {config_path.resolve()}"
    config = load_config(config_path)
    if args.limit_per_split is not None and not args.pilot:
        parser.error("--limit-per-split is only allowed with --pilot; production split sizes must come from config/stage_a.yaml.")
    if args.pilot:
        # Keep the four official splits represented, but isolate pilot files and
        # mark them as diagnostic-only so training cannot consume them.
        config["dataset"]["diagnostic_only"] = True
        config["dataset"]["raw_dir"] = "data/raw_oracle_v4_pilot"
        config["dataset"]["manifest_path"] = "data/manifest_oracle_v4_pilot.json"
        config["experiment"]["output_dir"] = "results/stage_a_oracle_v4_pilot"
        pilot_counts = {"num_train": 4, "num_validation": 2, "num_test_interp": 2, "num_test_extrap": 2}
        for key, default_count in pilot_counts.items():
            count = default_count if args.limit_per_split is None else min(default_count, max(1, int(args.limit_per_split)))
            config["dataset"]["splits"][key] = count

    set_seed(config["experiment"]["seed"])
    logger = setup_logger(
        name="DatasetGenerator",
        log_file=Path(config["experiment"]["output_dir"]) / "dataset_generation.log"
    )
    logger.info("=== Avvio Pipeline Offline di Generazione Dataset ===")

    raw_dir = Path(config["dataset"]["raw_dir"])
    manifest_path = Path(config["dataset"]["manifest_path"])
    has_existing_samples = raw_dir.exists() and any(raw_dir.glob("sample_*.pkl"))
    if (has_existing_samples or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite an existing dataset. Use a fresh config with new dataset paths, "
            "or pass --overwrite explicitly after backing up the old data. "
            f"raw_dir={raw_dir.resolve()}, manifest={manifest_path.resolve()}"
        )
    ensure_dir(raw_dir)
    ensure_dir(manifest_path.parent)

    # 2. Creazione della mesh iniziale
    dom = config["geometry"]["domain"]
    grid_cfg = config["geometry"]["initial_grid"]
    logger.info(f"Generazione mesh iniziale 2D ({grid_cfg['nx']}x{grid_cfg['ny']})...")
    base_mesh = create_initial_mesh_2d(dom, nx=grid_cfg["nx"], ny=grid_cfg["ny"])
    logger.info(f"Mesh inizializzata: {base_mesh.num_nodes} nodi, {base_mesh.num_cells} celle triangolari.")

    # 3. Campionamento parametri per tutti gli split
    parameters_by_split = sample_parameters(config, seed=config["experiment"]["seed"])
    manifest: Dict[str, List[str]] = {
        "train": [],
        "validation": [],
        "test_interp": [],
        "test_extrap": []
    }

    sample_counter = 0
    total_samples = sum(len(samples) for samples in parameters_by_split.values())
    logger.info(f"Pianificata la generazione di {total_samples} campioni totali.")

    # Oracle configuration. Fail closed if the config describes a rule that
    # is not the one implemented by this objective.
    oracle_cfg = config["oracle"]
    dataset_cfg = config["dataset"]
    minimum_objective_reduction = float(dataset_cfg.get("minimum_objective_reduction_fraction", 0.50))
    minimum_interpolation_reduction = float(dataset_cfg.get("minimum_interpolation_error_reduction_fraction", 0.50))
    geom_cfg = config["geometry"]["safety"]
    objective_cfg = oracle_cfg["objective"]
    if "quadrature_rule" not in objective_cfg or "quadrature_subdivision_level" not in objective_cfg:
        raise ValueError(
            "The oracle objective quadrature must be explicit in the config. "
            "The legacy Q3 objective is not used by this code; use "
            "config/stage_a.yaml and pass --pilot for a diagnostic-only run."
        )
    configured_rule = objective_cfg["quadrature_rule"]
    if configured_rule != "dunavant_7_composite":
        raise ValueError(
            f"Unsupported oracle quadrature_rule={configured_rule!r}; "
            "this build implements only 'dunavant_7_composite'."
        )
    raw_quadrature_level = objective_cfg["quadrature_subdivision_level"]
    if isinstance(raw_quadrature_level, bool) or int(raw_quadrature_level) != raw_quadrature_level:
        raise ValueError("oracle.objective.quadrature_subdivision_level must be an integer in [0, 4].")
    quadrature_level = int(raw_quadrature_level)
    if quadrature_level not in (0, 1, 2, 3, 4):
        raise ValueError("oracle.objective.quadrature_subdivision_level must be in [0, 4].")

    # 4. Loop di ottimizzazione offline e generazione dati
    with Timer() as total_timer:
        for split_name, params_list in parameters_by_split.items():
            logger.info(f"--- Inizio elaborazione split: '{split_name}' ({len(params_list)} campioni) ---")
            split_started = time.perf_counter()
            progress_block_started = split_started
            last_reported_index = 0

            for p_idx, params in enumerate(params_list):
                sample_started = time.perf_counter()
                sample_id = f"sample_{sample_counter:05d}_{split_name}"
                sample_counter += 1

                # Istanziazione campo continuo analitico
                field = GaussianBumpField(
                    mu_x=params["mu_x"],
                    mu_y=params["mu_y"],
                    sigma=params["sigma"],
                    amplitude=params["amplitude"]
                )

                # Valutazione campo e gradiente nodale iniziale
                u_init, grad_u_init, _, _ = field.evaluate(
                    base_mesh.nodes[:, 0],
                    base_mesh.nodes[:, 1]
                )

                # Ottimizzazione offline variazionale per il calcolo di X*
                optimizer = OfflineMeshOptimizer(
                    reference_field=field,
                    initial_nodes=base_mesh.nodes,
                    cells=base_mesh.cells,
                    boundary_nodes=base_mesh.boundary_nodes,
                    lambda_D=oracle_cfg["objective"]["lambda_D"],
                    lambda_Q=oracle_cfg["objective"]["lambda_Q"],
                    epsilon=geom_cfg["epsilon"],
                    max_iter=oracle_cfg["optimizer"]["max_iter"],
                    ftol=oracle_cfg["optimizer"]["ftol"],
                    gtol=oracle_cfg["optimizer"]["gtol"],
                    quadrature_subdivision_level=quadrature_level,
                )

                opt_result = optimizer.optimize()
                initial_iterations = int(opt_result["iterations"])
                iterations_total = initial_iterations
                nfev_total = int(opt_result.get("nfev", 0))
                njev_total = int(opt_result.get("njev", 0))
                # Fixed baselines from the undeformed reference mesh; continuation
                # passes are always judged against the same starting state.
                initial_objective = float(opt_result.get("starting_objective", np.nan))
                initial_interpolation_error = float(optimizer.objective.initial_interpolation_error)
                oracle_time_total = float(opt_result.get("final_metrics", {}).get("oracle_time_seconds", 0.0))
                retry_count = 0
                max_restarts = max(0, int(oracle_cfg["optimizer"].get("max_restarts", 1)))
                _mark_reference_target_acceptance(
                    opt_result, initial_objective=initial_objective,
                    initial_interpolation_error=initial_interpolation_error,
                    minimum_objective_reduction_fraction=minimum_objective_reduction,
                    minimum_interpolation_error_reduction_fraction=minimum_interpolation_reduction,
                )
                # The target-quality contract is objective descent + geometric
                # admissibility. First-order stationarity remains a separate metric.
                while not bool(opt_result.get("target_accepted", False)) and retry_count < max_restarts:
                    candidate_guess = np.ascontiguousarray(opt_result["candidate_nodes"], dtype=np.float64)
                    retry_result = optimizer.optimize(initial_guess_nodes=candidate_guess)
                    iterations_total += int(retry_result["iterations"])
                    nfev_total += int(retry_result.get("nfev", 0))
                    njev_total += int(retry_result.get("njev", 0))
                    oracle_time_total += float(retry_result.get("final_metrics", {}).get("oracle_time_seconds", 0.0))
                    retry_count += 1
                    opt_result = retry_result
                    _mark_reference_target_acceptance(
                        opt_result, initial_objective=initial_objective,
                        initial_interpolation_error=initial_interpolation_error,
                        minimum_objective_reduction_fraction=minimum_objective_reduction,
                        minimum_interpolation_error_reduction_fraction=minimum_interpolation_reduction,
                    )
                opt_result["iterations_last_pass"] = int(opt_result["iterations"])
                opt_result["iterations"] = int(iterations_total)
                opt_result["retry_count"] = int(retry_count)
                opt_result["final_metrics"]["iterations_total"] = int(iterations_total)
                opt_result["final_metrics"]["iterations_last_pass"] = int(opt_result["iterations_last_pass"])
                opt_result["final_metrics"]["nfev_total"] = int(nfev_total)
                opt_result["final_metrics"]["njev_total"] = int(njev_total)
                opt_result["final_metrics"]["retry_count"] = int(retry_count)
                opt_result["final_metrics"]["continuation_budget"] = int(max_restarts)
                # Record the total wall time across the initial pass and any
                # continuation passes; the last OptimizeResult alone understates
                # offline target-generation cost when retries are used.
                opt_result["final_metrics"]["oracle_time_seconds"] = float(oracle_time_total)
                opt_result["final_metrics"]["oracle_time_seconds_total"] = float(oracle_time_total)
                opt_result["final_metrics"]["initial_objective"] = initial_objective
                opt_result["final_metrics"]["initial_interpolation_error"] = initial_interpolation_error
                final_objective = float(opt_result["final_metrics"].get("total_objective", np.nan))
                final_interpolation_error = float(opt_result["final_metrics"].get("interpolation_error", np.nan))
                opt_result["final_metrics"]["objective_reduction_fraction"] = (
                    (initial_objective - final_objective) / max(abs(initial_objective), 1.0e-15)
                    if np.isfinite(initial_objective) and np.isfinite(final_objective) else float("nan")
                )
                opt_result["final_metrics"]["interpolation_error_reduction_fraction"] = (
                    (initial_interpolation_error - final_interpolation_error) / max(abs(initial_interpolation_error), 1.0e-15)
                    if np.isfinite(initial_interpolation_error) and np.isfinite(final_interpolation_error) else float("nan")
                )

                optimal_nodes = np.ascontiguousarray(opt_result["optimal_nodes"], dtype=np.float64)
                target_displacement = optimal_nodes - base_mesh.nodes

                # La validazione della proposta è eseguita da C++ su array, senza
                # costruire una triangolazione FEM prima di aver controllato l'orientamento.
                final_metrics_native = compute_native_mesh_metrics_arrays(
                    optimal_nodes, base_mesh.cells,
                    tol_degenerate=float(geom_cfg.get("tol_degenerate", 1.0e-14)),
                )
                final_metrics = dict(opt_result.get("final_metrics", {}))
                final_metrics.update({
                    "min_jacobian": float(final_metrics_native["min_jacobian"]),
                    "num_inverted_cells": int(final_metrics_native["num_inverted"]),
                    "num_degenerate_cells": int(final_metrics_native["num_degenerate"]),
                    "mean_escobar_quality": float(final_metrics_native["mean_escobar_quality"]),
                    "validation_backend": "C++/deal.II",
                })

                # Verifica formale di ammissibilità tramite il backend nativo.
                assert final_metrics["min_jacobian"] >= geom_cfg["epsilon"], (
                    f"Errore critico su {sample_id}: Jacobiano minimo "
                    f"{final_metrics['min_jacobian']:.3e} < epsilon"
                )
                assert final_metrics["num_inverted_cells"] == 0, (
                    f"Errore critico su {sample_id}: trovate celle invertite!"
                )
                assert final_metrics["num_degenerate_cells"] == 0, (
                    f"Errore critico su {sample_id}: trovate celle degeneri!"
                )

                opt_result["final_metrics"] = final_metrics
                _mark_reference_target_acceptance(
                    opt_result, initial_objective=initial_objective,
                    initial_interpolation_error=initial_interpolation_error,
                    minimum_objective_reduction_fraction=minimum_objective_reduction,
                    minimum_interpolation_error_reduction_fraction=minimum_interpolation_reduction,
                )
                final_metrics = opt_result["final_metrics"]

                # Composizione dizionario sample secondo il contratto formale
                sample_data = {
                    "mesh": {
                        "nodes": base_mesh.nodes,
                        "cells": base_mesh.cells
                    },
                    "state": {
                        "u": u_init,
                        "grad_u": grad_u_init
                    },
                    "target": {
                        # `reference_nodes` is the semantically correct name: the stored
                        # coordinates are quality-screened finite-budget references, not
                        # certified stationary/global optima. Keep `optimal_nodes` as a
                        # backwards-compatible alias for existing project consumers.
                        "reference_nodes": optimal_nodes,
                        "optimal_nodes": optimal_nodes,
                        "displacement": target_displacement
                    },
                    "metadata": {
                        "sample_id": sample_id,
                        "split": split_name,
                        "parameters": params,
                        "oracle_objective": {
                            "quadrature_rule": optimizer.objective.quadrature_rule,
                            "quadrature_subdivision_level": int(optimizer.objective.quadrature_subdivision_level),
                            "quadrature_points_per_cell": int(optimizer.objective.quad_bary.shape[0]),
                            "lambda_D": float(oracle_cfg["objective"]["lambda_D"]),
                            "lambda_Q": float(oracle_cfg["objective"]["lambda_Q"]),
                        },
                        # Strict stationarity and data-use acceptance are separate.
                        "converged": bool(opt_result["converged"]),
                        "target_accepted": bool(opt_result.get("target_accepted", False)),
                        "target_acceptance_reason": str(opt_result.get("target_acceptance_reason", "not_evaluated")),
                        "target_acceptance_checks": opt_result.get("target_acceptance_checks", {}),
                        "target_definition": str(dataset_cfg.get("target_definition", "finite_budget_reference_not_stationary_optimum")),
                        "iterations": opt_result["iterations"],
                        "oracle_metrics": final_metrics
                    }
                }

                # Salvataggio binario del campione
                sample_file = raw_dir / f"{sample_id}.pkl"
                save_pickle(sample_data, sample_file)
                manifest[split_name].append(sample_id)

                if (p_idx + 1) % 10 == 0 or (p_idx + 1) == len(params_list):
                    now = time.perf_counter()
                    block_samples = max(1, (p_idx + 1) - last_reported_index)
                    block_elapsed = now - progress_block_started
                    mean_sample_wall = block_elapsed / block_samples
                    remaining_samples = len(params_list) - (p_idx + 1)
                    eta_split_seconds = mean_sample_wall * remaining_samples
                    quadrature_points_per_eval = int(
                        base_mesh.num_cells * optimizer.objective.quad_bary.shape[0]
                    )
                    logger.info(
                        f"[{split_name}] Progresso: {p_idx + 1}/{len(params_list)} | "
                        f"Ultimo J_min: {final_metrics['min_jacobian']:.4e} | "
                        f"Iter: {opt_result['iterations']} (restarts={opt_result['retry_count']}, "
                        f"target_accepted={opt_result.get('target_accepted', False)}, "
                        f"stationary={opt_result['converged']}, nfev={nfev_total}) | "
                        f"Q-points/objective-eval={quadrature_points_per_eval:,} | "
                        f"oracle_time={oracle_time_total:.2f}s, "
                        f"wall_time_last_sample={now - sample_started:.2f}s, "
                        f"mean_wall_last_block={mean_sample_wall:.2f}s/sample, "
                        f"elapsed_split={now - split_started:.1f}s, "
                        f"ETA_split={eta_split_seconds/60.0:.1f}min"
                    )
                    progress_block_started = now
                    last_reported_index = p_idx + 1

    # 5. Salvataggio del manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest registrato con successo in: {manifest_path.resolve()}")
    provenance = {
        "mesh_backend": "C++/deal.II via gdm._gdm_core.MeshState",
        "geometry_validation_backend": "C++/deal.II via compute_mesh_metrics",
        "oracle_optimizer_backend": "Python/SciPy (not yet ported to C++)",
        "field_evaluation_backend": "Python analytic Gaussian field",
        "oracle_quadrature_rule": configured_rule,
        "oracle_quadrature_subdivision_level": quadrature_level,
        "oracle_quadrature_points_per_cell": int(7 * (4 ** quadrature_level)),
        "oracle_objective_version": "gaussian_interpolation_error_composite_q7_v1",
        "oracle_target_definition": str(config["dataset"].get("target_definition", "finite_budget_reference_not_stationary_optimum")),
        "target_acceptance_contract": {
            "minimum_objective_reduction_fraction": minimum_objective_reduction,
            "minimum_interpolation_error_reduction_fraction": minimum_interpolation_reduction,
            "minimum_target_acceptance_rate": float(config["dataset"].get("minimum_target_acceptance_rate", 0.95)),
            "first_order_stationarity_is_acceptance_gate": False,
        },
        "seed": int(config["experiment"]["seed"]),
        "sample_count": int(total_samples),
    }
    provenance_path = manifest_path.with_name(manifest_path.stem + "_provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    logger.info(f"Provenienza tecnica salvata in: {provenance_path.resolve()}")
    audit_dir = Path(config["experiment"]["output_dir"]) / "figures" / "dataset_audit"
    audit_summary = audit_dataset(
        manifest_path=manifest_path,
        raw_dir=raw_dir,
        output_dir=audit_dir,
        max_iter=int(oracle_cfg["optimizer"]["max_iter"]),
        minimum_target_acceptance_rate=float(config["dataset"].get("minimum_target_acceptance_rate", 0.95)),
        minimum_objective_reduction_fraction=minimum_objective_reduction,
        minimum_interpolation_error_reduction_fraction=minimum_interpolation_reduction,
        expected_quadrature_rule=configured_rule,
        expected_quadrature_subdivision_level=quadrature_level,
    )
    if audit_summary.get("audit_status") != "PASS":
        audit_path = audit_dir / "dataset_quality_audit.json"
        logger.warning(
            "Audit dataset con avvisi: controllare %s e sample_diagnostics.csv prima di utilizzare i target di riferimento nel training.",
            audit_path,
        )
        # Diagnostic pilots are allowed to complete so their full audit can be
        # inspected. Production datasets fail closed unless the configuration
        # explicitly opts into diagnostic-only use.
        diagnostic_only = bool(config["dataset"].get("diagnostic_only", False))
        allow_warnings = bool(config["dataset"].get("allow_dataset_audit_warnings", False))
        if not diagnostic_only and not allow_warnings:
            raise RuntimeError(
                f"Dataset quality audit failed (status={audit_summary.get('audit_status')}). "
                f"The manifest and diagnostic reports were written to {audit_path}, but this is a production configuration. "
                "Do not train on this dataset until the audit passes."
            )
    logger.info(
        f"=== Generazione Dataset completata in {total_timer.elapsed_seconds:.2f} s ==="
    )


if __name__ == "__main__":
    main()
