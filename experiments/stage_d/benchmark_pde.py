"""Stage-D PDE-aware trajectory evaluation campaign.

Compares three strategies under identical C++20/deal.II numerical conditions:
  A. Fixed FEM (`fixed_fem`): static mesh throughout [0, T].
  B. Stage A GNN (`stage_a_gnn`): static oracle-trained GNN coupled with native ALE.
  C. Stage D GNN (`stage_d_gnn`): trajectory-aware GNN coupled with native ALE.

All PDE operators, ALE time-stepping, continuous L2/H1 Dunavant degree-5 error
norms, discrete mass integrals, GCL diagnostics, mesh quality metrics, and
deterministic safety filtering run in ``gdm._gdm_core`` (C++20/deal.II).
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
for _candidate in (ROOT, ROOT / "python"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from gdm import (
    NativeMeshAdapter,
    apply_native_safety_filter,
    build_gnn_inference_inputs,
    compute_native_continuous_error_details,
    compute_native_discrete_mass,
    compute_native_mesh_metrics,
    create_ale_solver,
    create_fem_solver,
    create_pde_model,
)
from inference import MeshMoverInferenceEngine
from plotting.stage_d_pde import plot_stage_d_pde_benchmark
from utils.utils import ensure_dir, set_seed

STRATEGIES_ORDER: List[tuple[str, str]] = [
    ("fixed_fem", "Fixed FEM"),
    ("stage_a_gnn", "Stage A GNN"),
    ("stage_d_gnn", "Stage D GNN"),
]


def _trapezoid_integral(values: Sequence[float], times: Sequence[float]) -> float:
    integrator = getattr(np, "trapezoid", None)
    if integrator is None:
        integrator = np.trapz
    return float(integrator(np.asarray(values, dtype=np.float64), np.asarray(times, dtype=np.float64)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=np.float64)
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


def load_stage_d_config(config_path: str | Path = "config/stage_d.yaml") -> Dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configurazione Stage D non trovata: {path}")
    with path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"File YAML non valido: {path}")
    return cfg


def _infer_model_hyperparams_from_state_dict(
    state_dict: Mapping[str, Any], fallback_model_cfg: Mapping[str, Any]
) -> Dict[str, Any]:
    model_cfg = dict(fallback_model_cfg)
    enc_weight = state_dict.get("node_encoder.0.weight")
    if isinstance(enc_weight, torch.Tensor) and enc_weight.ndim == 2:
        model_cfg["latent_dim"] = int(enc_weight.shape[0])
        model_cfg["node_in_dim"] = int(enc_weight.shape[1])
    edge_weight = state_dict.get("edge_encoder.0.weight")
    if isinstance(edge_weight, torch.Tensor) and edge_weight.ndim == 2:
        model_cfg["edge_in_dim"] = int(edge_weight.shape[1])
    proc_indices = set()
    for key in state_dict.keys():
        if key.startswith("processor."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                proc_indices.add(int(parts[1]))
    if proc_indices:
        model_cfg["num_message_passing_steps"] = max(proc_indices) + 1
    return model_cfg


def build_inference_engine_for_checkpoint(
    checkpoint_path: str | Path,
    base_config: Mapping[str, Any],
    device: str = "cpu",
) -> MeshMoverInferenceEngine:
    """Load a GNN checkpoint preserving its model/feature contract and shared safety settings."""
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")

    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")

    effective_cfg = copy.deepcopy(dict(base_config))
    if isinstance(payload, dict):
        embedded_cfg = payload.get("config")
        if isinstance(embedded_cfg, dict):
            if isinstance(embedded_cfg.get("model"), dict):
                effective_cfg["model"] = copy.deepcopy(embedded_cfg["model"])
            if isinstance(embedded_cfg.get("features"), dict):
                effective_cfg["features"] = copy.deepcopy(embedded_cfg["features"])
        state_dict = payload.get("model_state_dict", payload.get("state_dict", payload))
        if isinstance(state_dict, dict):
            effective_cfg["model"] = _infer_model_hyperparams_from_state_dict(
                state_dict, effective_cfg.get("model", {})
            )
    # Enforce shared geometry and safety-filter configuration from config/stage_d.yaml
    effective_cfg["geometry"] = copy.deepcopy(dict(base_config["geometry"]))
    return MeshMoverInferenceEngine(
        checkpoint_path=ckpt_path,
        config=effective_cfg,
        device=device,
    )


def resolve_heldout_cases(
    config: Mapping[str, Any],
    splits: Sequence[str] = ("test_interp", "test_extrap"),
    smoke_test: bool = False,
    max_cases_per_split: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load held-out cases for test_interp and test_extrap without reading oracle targets.

    Only physical PDE parameters (sigma, beta) and case identifiers are extracted.
    Stored target displacements and optimal_nodes are never read or exposed.
    """
    allowed_splits = {"test_interp", "test_extrap"}
    for split in splits:
        if split not in allowed_splits:
            raise ValueError(
                f"Split '{split}' non consentito nel benchmark PDE: sono ammessi solo {sorted(allowed_splits)}."
            )

    stage_cfg = config["stage_d"]
    pde_cfg = stage_cfg["pde"]
    base_seed = int(config["experiment"]["seed"])
    raw_dir = (ROOT / str(config["dataset"]["raw_dir"])).resolve()
    manifest_path = (ROOT / str(config["dataset"]["manifest_path"])).resolve()

    cap = 1 if smoke_test and max_cases_per_split is None else max_cases_per_split
    cases_by_split: Dict[str, List[Dict[str, Any]]] = {s: [] for s in splits}

    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        for split in splits:
            sample_ids = list(manifest.get(split, []))
            if cap is not None:
                sample_ids = sample_ids[: max(0, int(cap))]
            for local_idx, sample_id in enumerate(sample_ids):
                pkl_path = raw_dir / f"{sample_id}.pkl"
                if not pkl_path.is_file():
                    raise FileNotFoundError(f"Campione held-out mancante: {pkl_path}")
                with pkl_path.open("rb") as stream:
                    raw_sample = pickle.load(stream)
                meta = raw_sample.get("metadata", {})
                recorded_split = str(meta.get("split", split))
                if recorded_split != split:
                    raise ValueError(
                        f"Inconsistenza di split per {sample_id}: atteso {split}, trovato {recorded_split}."
                    )
                params = meta.get("parameters", {})
                sigma = float(params["sigma"])
                beta = float(params["beta"])
                extrap_regime = "in_domain"
                if split == "test_extrap":
                    low_bounds = stage_cfg["splits"]["test_extrap"]["sigma_extrap_low"]
                    extrap_regime = "sigma_low" if sigma <= float(low_bounds[1]) + 1e-9 else "sigma_high"
                cases_by_split[split].append({
                    "case_id": str(sample_id),
                    "split": split,
                    "local_index": int(local_idx),
                    "case_seed": int(base_seed + 1000 * (1 if split == "test_interp" else 2) + local_idx),
                    "bx": float(pde_cfg["bx"]),
                    "by": float(pde_cfg["by"]),
                    "nu": float(pde_cfg["nu"]),
                    "sigma": sigma,
                    "beta": beta,
                    "T": float(pde_cfg["T"]),
                    "dt": float(stage_cfg["dt"]),
                    "adapt_interval": int(stage_cfg["adapt_interval"]),
                    "extrapolation_regime": extrap_regime,
                    "source": str(pkl_path),
                })
    else:
        # Deterministic fallback reproducing generate_trajectory_dataset.py sampling order
        dt = float(stage_cfg["dt"])
        adapt_interval = int(stage_cfg["adapt_interval"])
        horizon = dt * adapt_interval
        T = float(pde_cfg["T"])
        valid_start_times = np.arange(0.0, T - horizon + 0.5 * dt, horizon, dtype=np.float64)
        valid_start_times = valid_start_times[valid_start_times + horizon <= T + 1e-12]
        if valid_start_times.size == 0:
            valid_start_times = np.array([0.0], dtype=np.float64)

        rng = np.random.default_rng(base_seed)
        sample_index = 0
        for iter_split in ("train", "validation", "test_interp", "test_extrap"):
            spec = stage_cfg["splits"][iter_split]
            count = int(spec["count"])
            for local_idx in range(count):
                if iter_split == "test_extrap":
                    sigma_key = "sigma_extrap_low" if local_idx % 2 == 0 else "sigma_extrap_high"
                    bounds = spec[sigma_key]
                    sigma = float(rng.uniform(float(bounds[0]), float(bounds[1])))
                    extrap_regime = "sigma_low" if sigma_key == "sigma_extrap_low" else "sigma_high"
                else:
                    bounds = spec["sigma"]
                    sigma = float(rng.uniform(float(bounds[0]), float(bounds[1])))
                    extrap_regime = "in_domain"
                beta_bounds = spec["beta"]
                beta = float(rng.uniform(float(beta_bounds[0]), float(beta_bounds[1])))
                _ = float(rng.choice(valid_start_times))
                sample_id = f"stage_d_{sample_index:05d}_{iter_split}"
                sample_index += 1
                if iter_split in cases_by_split:
                    if cap is None or len(cases_by_split[iter_split]) < int(cap):
                        cases_by_split[iter_split].append({
                            "case_id": sample_id,
                            "split": iter_split,
                            "local_index": int(local_idx),
                            "case_seed": int(base_seed + 1000 * (1 if iter_split == "test_interp" else 2) + local_idx),
                            "bx": float(pde_cfg["bx"]),
                            "by": float(pde_cfg["by"]),
                            "nu": float(pde_cfg["nu"]),
                            "sigma": sigma,
                            "beta": beta,
                            "T": float(pde_cfg["T"]),
                            "dt": dt,
                            "adapt_interval": adapt_interval,
                            "extrapolation_regime": extrap_regime,
                            "source": "deterministic_seed_reconstruction",
                        })

    ordered_cases: List[Dict[str, Any]] = []
    for split in splits:
        if not cases_by_split[split]:
            raise RuntimeError(f"Nessun caso held-out trovato per lo split '{split}'.")
        ordered_cases.extend(cases_by_split[split])
    return ordered_cases


def run_pde_trajectory_episode(
    case_spec: Mapping[str, Any],
    strategy: str,
    config: Mapping[str, Any],
    inference_engine: Optional[MeshMoverInferenceEngine] = None,
    *,
    compute_diagnostics: bool = True,
) -> Dict[str, Any]:
    """Run a single transient PDE trajectory for a given strategy using native deal.II kernels.

    During inference, the GNN receives exclusively `(curr_mesh, u_curr, grad_u_curr)`
    at the current time step `t_curr`, with `pde=None` and `exact_gradient=False`
    so that no analytical or future information can ever be accessed.
    """
    if strategy not in {"fixed_fem", "stage_a_gnn", "stage_d_gnn"}:
        raise ValueError(f"Strategia non supportata: {strategy!r}")
    if strategy != "fixed_fem" and inference_engine is None:
        raise ValueError(f"La strategia {strategy!r} richiede un MeshMoverInferenceEngine valido.")

    domain_cfg = dict(config["geometry"]["domain"])
    grid_cfg = dict(config["geometry"]["initial_grid"])
    safety_cfg = dict(config["geometry"]["safety"])
    nx = int(grid_cfg["nx"])
    ny = int(grid_cfg["ny"])
    epsilon = float(safety_cfg["epsilon"])
    safety_margin = float(safety_cfg.get("safety_margin", 1.0e-6))
    tol_deg = float(safety_cfg.get("tol_deg", 1.0e-12))
    tol_degenerate = float(safety_cfg.get("tol_degenerate", 1.0e-14))

    bx = float(case_spec["bx"])
    by = float(case_spec["by"])
    nu = float(case_spec["nu"])
    sigma = float(case_spec["sigma"])
    beta = float(case_spec["beta"])
    T = float(case_spec["T"])
    dt = float(case_spec["dt"])
    adapt_interval = int(case_spec["adapt_interval"])
    n_steps = int(round(T / dt))
    if abs(n_steps * dt - T) > 1.0e-10 * T + 1.0e-12:
        raise ValueError(f"T={T} deve essere un multiplo intero di dt={dt}.")

    t_wall_start = time.perf_counter()

    # Identical initial state across all strategies
    init_mesh = NativeMeshAdapter.from_domain(domain_cfg, nx=nx, ny=ny)
    curr_mesh = init_mesh
    init_nodes = init_mesh.nodes.copy()
    pde = create_pde_model(bx=bx, by=by, nu=nu, sigma=sigma, beta=beta, T=T)
    u_curr = np.asarray(pde.evaluate_solution(curr_mesh.nodes, 0.0), dtype=np.float64)

    gnn_inference_time_total = 0.0
    mesh_update_time_total = 0.0
    safety_filter_time_total = 0.0
    coord_update_time_total = 0.0
    ale_time_total = 0.0
    fem_solve_time_total = 0.0
    remapping_time_total = 0.0

    if strategy == "fixed_fem":
        t0_setup = time.perf_counter()
        fem_solver = create_fem_solver(curr_mesh, pde)
        fem_solve_time_total += time.perf_counter() - t0_setup
        ale_solver = None
    else:
        fem_solver = None
        t0_setup = time.perf_counter()
        ale_solver = create_ale_solver(pde, nu=nu, bx=bx, by=by)
        ale_time_total += time.perf_counter() - t0_setup

    base_nodes = curr_mesh.nodes.copy()
    target_nodes = curr_mesh.nodes.copy()
    alpha_current_window = 1.0
    intervened_current_window = 0
    cumulative_safety_interventions = 0
    window_displacement_norm = 0.0

    timestep_rows: List[Dict[str, Any]] = []
    times_so_far: List[float] = []
    rel_l2_so_far: List[float] = []
    rel_h1_so_far: List[float] = []

    if compute_diagnostics:
        err0 = compute_native_continuous_error_details(curr_mesh, u_curr, pde, 0.0)
        geom0 = compute_native_mesh_metrics(curr_mesh, tol_degenerate=tol_degenerate)
        jacobians0 = np.asarray(geom0["jacobians"], dtype=np.float64)
        inverted0 = np.asarray(geom0["inverted"], dtype=bool)
        degenerate0 = np.asarray(geom0["degenerate"], dtype=bool)
        inadmissible0 = int(np.sum((jacobians0 < epsilon) | inverted0 | degenerate0))

        mass_h_prev = compute_native_discrete_mass(curr_mesh, u_curr)
        mass_ref_prev = mass_h_prev

        times_so_far.append(0.0)
        rel_l2_so_far.append(float(err0["rel_l2_error"]))
        rel_h1_so_far.append(float(err0["rel_h1_seminorm_error"]))

        timestep_rows.append({
            "case_id": str(case_spec["case_id"]),
            "split": str(case_spec["split"]),
            "seed": int(case_spec["case_seed"]),
            "strategy": strategy,
            "step": 0,
            "time": 0.0,
            "sigma": sigma,
            "beta": beta,
            "relative_l2_error": float(err0["rel_l2_error"]),
            "relative_h1_error": float(err0["rel_h1_seminorm_error"]),
            "abs_l2_error": float(err0["abs_l2_error"]),
            "abs_h1_error": float(err0["abs_h1_seminorm_error"]),
            "integrated_l2_error": float(err0["rel_l2_error"]),
            "final_l2_error": float(err0["rel_l2_error"]),
            "min_jacobian": float(geom0["min_jacobian"]),
            "mean_mesh_quality": float(geom0["mean_escobar_quality"]),
            "min_escobar_quality": float(geom0["min_escobar_quality"]),
            "max_aspect_ratio": float(geom0["max_aspect_ratio"]),
            "inadmissible_cells": inadmissible0,
            "inverted_cells": int(geom0["num_inverted"]),
            "degenerate_cells": int(geom0["num_degenerate"]),
            "safety_alpha": 1.0,
            "safety_intervened": 0,
            "safety_filter_interventions": 0,
            "displacement_norm": 0.0,
            "window_displacement_norm": 0.0,
            "cumulative_displacement_norm": 0.0,
            "gnn_inference_time": 0.0,
            "mesh_update_time": 0.0,
            "ale_time": 0.0,
            "fem_solve_time": 0.0,
            "remapping_time": 0.0,
            "total_wall_time": 0.0,
            "discrete_mass": float(mass_h_prev),
            "reference_mass": float(mass_ref_prev),
            "mass_balance_residual": 0.0,
            "mass_discrepancy_error": 0.0,
            "gcl_max_elem_residual": 0.0,
            "gcl_algebraic_residual_linf": 0.0,
            "gcl_dual_col_sum_linf": 0.0,
            "linear_solver_residual": 0.0,
        })
    else:
        mass_h_prev = 0.0
        mass_ref_prev = 0.0

    for step in range(1, n_steps + 1):
        t_step_wall_start = time.perf_counter()
        t_curr = (step - 1) * dt
        t_next = step * dt

        step_gnn_time = 0.0
        step_safety_time = 0.0
        step_coord_time = 0.0
        step_ale_time = 0.0
        step_fem_time = 0.0
        step_remap_time = 0.0
        intervened_this_step = 0

        # Window adaptation trigger at the beginning of each adaptation horizon
        if strategy != "fixed_fem" and (step - 1) % adapt_interval == 0:
            assert inference_engine is not None
            t0_gnn = time.perf_counter()
            # pde=None guarantees that C++ compute_field_gradients cannot access exact/future fields
            py_mesh, u_in, grad_in = build_gnn_inference_inputs(
                curr_mesh, u_curr, pde=None, t=float(t_curr), exact_gradient=False
            )
            raw_pred = inference_engine.predict_raw(py_mesh, u_in, grad_in)
            step_gnn_time = time.perf_counter() - t0_gnn
            gnn_inference_time_total += step_gnn_time

            raw_disp = np.asarray(raw_pred["raw_displacement"], dtype=np.float64)
            if raw_disp.shape != curr_mesh.nodes.shape or not np.all(np.isfinite(raw_disp)):
                raise RuntimeError(
                    f"Spostamento GNN non valido al passo {step} per la strategia {strategy}."
                )

            t0_safety = time.perf_counter()
            filt_res = apply_native_safety_filter(
                mesh=curr_mesh,
                displacement=raw_disp,
                epsilon=epsilon,
                safety_margin=safety_margin,
                tol_deg=tol_deg,
                lock_boundary_nodes=True,
            )
            step_safety_time = time.perf_counter() - t0_safety
            safety_filter_time_total += step_safety_time

            target_nodes = np.asarray(filt_res["safe_nodes"], dtype=np.float64)
            base_nodes = curr_mesh.nodes.copy()
            alpha_current_window = float(filt_res["alpha_global"])
            intervened_current_window = int(
                bool(alpha_current_window < 1.0 - 1.0e-12 or filt_res.get("filter_activated", False))
            )
            intervened_this_step = intervened_current_window
            cumulative_safety_interventions += intervened_current_window
            window_displacement_norm = float(np.linalg.norm(target_nodes - base_nodes))

        # Time integration step
        if strategy == "fixed_fem":
            assert fem_solver is not None
            t0_fem = time.perf_counter()
            u_next = np.asarray(
                fem_solver.step_backward_euler(u_curr, dt, t_next), dtype=np.float64
            )
            step_fem_time = time.perf_counter() - t0_fem
            fem_solve_time_total += step_fem_time
            next_mesh = curr_mesh
            step_disp_norm = 0.0
            gcl_max_elem = 0.0
            gcl_alg_linf = 0.0
            gcl_dual_col = 0.0
            lin_res = 0.0
        else:
            assert ale_solver is not None
            t0_coord = time.perf_counter()
            k_step = ((step - 1) % adapt_interval) + 1
            interp_nodes = base_nodes + (k_step / float(adapt_interval)) * (target_nodes - base_nodes)
            step_disp_norm = float(np.linalg.norm(interp_nodes - curr_mesh.nodes))
            next_mesh = init_mesh.update_coordinates(interp_nodes, enforce_fixed_boundary=True)
            step_coord_time = time.perf_counter() - t0_coord
            coord_update_time_total += step_coord_time

            t0_ale = time.perf_counter()
            u_next_raw, ale_diag = ale_solver.step_ale(
                curr_mesh.core, next_mesh.core, u_curr, dt, t_next, False, True
            )
            step_ale_time = time.perf_counter() - t0_ale
            ale_time_total += step_ale_time

            u_next = np.asarray(u_next_raw, dtype=np.float64)
            gcl_max_elem = float(ale_diag.max_elem_gcl_residual)
            gcl_alg_linf = float(ale_diag.algebraic_gcl_residual_linf)
            gcl_dual_col = float(ale_diag.dual_c_mesh_col_sum_linf)
            lin_res = float(ale_diag.linear_solver_residual)

        step_mesh_update_time = step_safety_time + step_coord_time
        mesh_update_time_total = safety_filter_time_total + coord_update_time_total
        u_curr = u_next
        curr_mesh = next_mesh
        step_wall_time = time.perf_counter() - t_step_wall_start

        if compute_diagnostics:
            err_step = compute_native_continuous_error_details(curr_mesh, u_curr, pde, t_next)
            geom_step = compute_native_mesh_metrics(curr_mesh, tol_degenerate=tol_degenerate)
            jacobians = np.asarray(geom_step["jacobians"], dtype=np.float64)
            inverted = np.asarray(geom_step["inverted"], dtype=bool)
            degenerate = np.asarray(geom_step["degenerate"], dtype=bool)
            inadmissible = int(np.sum((jacobians < epsilon) | inverted | degenerate))

            times_so_far.append(t_next)
            rel_l2_so_far.append(float(err_step["rel_l2_error"]))
            rel_h1_so_far.append(float(err_step["rel_h1_seminorm_error"]))
            running_int_l2 = _trapezoid_integral(rel_l2_so_far, times_so_far) / max(t_next, 1.0e-12)

            u_exact_next = np.asarray(
                pde.evaluate_solution(curr_mesh.nodes, t_next), dtype=np.float64
            )
            mass_h_next = compute_native_discrete_mass(curr_mesh, u_curr)
            mass_ref_next = compute_native_discrete_mass(curr_mesh, u_exact_next)
            delta_mass_h = mass_h_next - mass_h_prev
            delta_mass_ref = mass_ref_next - mass_ref_prev
            mass_balance_res = abs(delta_mass_h - delta_mass_ref)
            mass_discrepancy = abs(mass_h_next - mass_ref_next)
            mass_h_prev = mass_h_next
            mass_ref_prev = mass_ref_next

            cum_disp_norm = float(np.linalg.norm(curr_mesh.nodes - init_nodes))

            timestep_rows.append({
                "case_id": str(case_spec["case_id"]),
                "split": str(case_spec["split"]),
                "seed": int(case_spec["case_seed"]),
                "strategy": strategy,
                "step": int(step),
                "time": float(t_next),
                "sigma": sigma,
                "beta": beta,
                "relative_l2_error": float(err_step["rel_l2_error"]),
                "relative_h1_error": float(err_step["rel_h1_seminorm_error"]),
                "abs_l2_error": float(err_step["abs_l2_error"]),
                "abs_h1_error": float(err_step["abs_h1_seminorm_error"]),
                "integrated_l2_error": float(running_int_l2),
                "final_l2_error": float(err_step["rel_l2_error"]),
                "min_jacobian": float(geom_step["min_jacobian"]),
                "mean_mesh_quality": float(geom_step["mean_escobar_quality"]),
                "min_escobar_quality": float(geom_step["min_escobar_quality"]),
                "max_aspect_ratio": float(geom_step["max_aspect_ratio"]),
                "inadmissible_cells": inadmissible,
                "inverted_cells": int(geom_step["num_inverted"]),
                "degenerate_cells": int(geom_step["num_degenerate"]),
                "safety_alpha": float(alpha_current_window),
                "safety_intervened": int(intervened_this_step),
                "safety_filter_interventions": int(cumulative_safety_interventions),
                "displacement_norm": float(step_disp_norm),
                "window_displacement_norm": float(window_displacement_norm),
                "cumulative_displacement_norm": float(cum_disp_norm),
                "gnn_inference_time": float(step_gnn_time),
                "mesh_update_time": float(step_mesh_update_time),
                "ale_time": float(step_ale_time),
                "fem_solve_time": float(step_fem_time),
                "remapping_time": float(step_remap_time),
                "total_wall_time": float(step_wall_time),
                "discrete_mass": float(mass_h_next),
                "reference_mass": float(mass_ref_next),
                "mass_balance_residual": float(mass_balance_res),
                "mass_discrepancy_error": float(mass_discrepancy),
                "gcl_max_elem_residual": float(gcl_max_elem),
                "gcl_algebraic_residual_linf": float(gcl_alg_linf),
                "gcl_dual_col_sum_linf": float(gcl_dual_col),
                "linear_solver_residual": float(lin_res),
            })

    total_wall_time = time.perf_counter() - t_wall_start
    sum_components = (
        gnn_inference_time_total
        + mesh_update_time_total
        + ale_time_total
        + fem_solve_time_total
        + remapping_time_total
    )
    other_overhead = max(0.0, total_wall_time - sum_components)

    timings = {
        "gnn_inference_time": float(gnn_inference_time_total),
        "mesh_update_time": float(mesh_update_time_total),
        "safety_filter_time": float(safety_filter_time_total),
        "coord_update_time": float(coord_update_time_total),
        "ale_time": float(ale_time_total),
        "fem_solve_time": float(fem_solve_time_total),
        "remapping_time": float(remapping_time_total),
        "other_overhead_time": float(other_overhead),
        "total_wall_time": float(total_wall_time),
    }

    if not compute_diagnostics:
        return {"timings": timings}

    integrated_l2 = _trapezoid_integral(rel_l2_so_far, times_so_far) / T
    integrated_h1 = _trapezoid_integral(rel_h1_so_far, times_so_far) / T
    final_l2 = float(rel_l2_so_far[-1])
    final_h1 = float(rel_h1_so_far[-1])

    for row in timestep_rows:
        row["final_l2_error"] = final_l2

    case_summary = {
        "case_id": str(case_spec["case_id"]),
        "split": str(case_spec["split"]),
        "extrapolation_regime": str(case_spec.get("extrapolation_regime", "in_domain")),
        "seed": int(case_spec["case_seed"]),
        "strategy": strategy,
        "bx": bx,
        "by": by,
        "nu": nu,
        "sigma": sigma,
        "beta": beta,
        "T": T,
        "dt": dt,
        "adapt_interval": adapt_interval,
        "relative_l2_error": float(np.mean(rel_l2_so_far)),
        "relative_h1_error": float(np.mean(rel_h1_so_far)),
        "integrated_l2_error": float(integrated_l2),
        "integrated_h1_error": float(integrated_h1),
        "final_l2_error": final_l2,
        "final_h1_error": final_h1,
        "min_jacobian": float(min(r["min_jacobian"] for r in timestep_rows)),
        "mean_mesh_quality": float(np.mean([r["mean_mesh_quality"] for r in timestep_rows])),
        "min_escobar_quality": float(min(r["min_escobar_quality"] for r in timestep_rows)),
        "max_aspect_ratio": float(max(r["max_aspect_ratio"] for r in timestep_rows)),
        "inadmissible_cells": int(sum(r["inadmissible_cells"] for r in timestep_rows)),
        "inverted_cells": int(sum(r["inverted_cells"] for r in timestep_rows)),
        "degenerate_cells": int(sum(r["degenerate_cells"] for r in timestep_rows)),
        "safety_alpha": float(np.mean([r["safety_alpha"] for r in timestep_rows[1:]])),
        "min_safety_alpha": float(min(r["safety_alpha"] for r in timestep_rows[1:])),
        "safety_filter_interventions": int(cumulative_safety_interventions),
        "displacement_norm": float(np.mean([r["displacement_norm"] for r in timestep_rows[1:]])),
        "max_window_displacement_norm": float(max(r["window_displacement_norm"] for r in timestep_rows)),
        "final_cumulative_displacement_norm": float(timestep_rows[-1]["cumulative_displacement_norm"]),
        "mass_balance_residual": float(max(r["mass_balance_residual"] for r in timestep_rows)),
        "mean_mass_balance_residual": float(np.mean([r["mass_balance_residual"] for r in timestep_rows[1:]])),
        "final_mass_discrepancy_error": float(timestep_rows[-1]["mass_discrepancy_error"]),
        "gcl_max_elem_residual": float(max(r["gcl_max_elem_residual"] for r in timestep_rows)),
        "gcl_algebraic_residual_linf": float(max(r["gcl_algebraic_residual_linf"] for r in timestep_rows)),
        "gcl_dual_col_sum_linf": float(max(r["gcl_dual_col_sum_linf"] for r in timestep_rows)),
        "max_linear_solver_residual": float(max(r["linear_solver_residual"] for r in timestep_rows)),
    }
    case_summary.update(timings)
    return {
        "case_summary": case_summary,
        "timestep_rows": timestep_rows,
        "timings": timings,
        "final_nodes": curr_mesh.nodes.copy(),
        "final_u": u_curr.copy(),
    }


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


def _build_latex_accuracy_table(summary: Mapping[str, Any]) -> str:
    lines: List[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Stage-D PDE-aware trajectory evaluation on held-out MMS convection-diffusion cases. Accuracy and geometric metrics are aggregated across distinct held-out cases (mean $\pm$ SD and median [IQR]).}",
        r"\label{tab:stage_d_pde_accuracy}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Split & Strategy & $\overline{E}_{L^2}$ [\%] & $E_{L^2}(T)$ [\%] & $\overline{E}_{H^1}$ [\%] & $\min J_e$ & $\bar{\eta}_{\mathrm{mesh}}$ & Safety int. \\",
        r"\midrule",
    ]
    split_labels = {
        "test_interp": "In-domain (interp)",
        "test_extrap": "OOD (extrap)",
    }
    for split_key in ("test_interp", "test_extrap"):
        split_data = summary.get("by_split", {}).get(split_key, {})
        for strat_key, strat_label in STRATEGIES_ORDER:
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
                f"{split_labels[split_key]} & {strat_label} & "
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


def _build_latex_cost_table(summary: Mapping[str, Any]) -> str:
    lines: List[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Stage-D computational cost per PDE episode (in seconds) across held-out cases and timed repetitions (excluding warm-up). Fixed FEM uses the Eulerian deal.II solver; dynamic GNN strategies use the coupled native D-GCL ALE deal.II solver with zero remapping.}",
        r"\label{tab:stage_d_computational_cost}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Split & Strategy & $T_{\mathrm{GNN}}$ [s] & $T_{\mathrm{mesh}}$ [s] & $T_{\mathrm{ALE}}$ [s] & $T_{\mathrm{FEM}}$ [s] & $T_{\mathrm{remap}}$ [s] & $T_{\mathrm{wall}}$ (med [IQR]) [s] \\",
        r"\midrule",
    ]
    split_labels = {
        "test_interp": "In-domain (interp)",
        "test_extrap": "OOD (extrap)",
    }
    for split_key in ("test_interp", "test_extrap"):
        split_data = summary.get("by_split", {}).get(split_key, {})
        for strat_key, strat_label in STRATEGIES_ORDER:
            rec = split_data.get(strat_key)
            if not rec:
                continue
            rt = rec["runtime_repetitions"]
            t_gnn = float(rt["gnn_inference_time"]["mean"])
            t_mesh = float(rt["mesh_update_time"]["mean"])
            t_ale = float(rt["ale_time"]["mean"])
            t_fem = float(rt["fem_solve_time"]["mean"])
            t_rem = float(rt["remapping_time"]["mean"])
            t_wall_med = float(rt["total_wall_time"]["median"])
            t_wall_iqr = float(rt["total_wall_time"]["iqr"])
            lines.append(
                f"{split_labels[split_key]} & {strat_label} & "
                f"{t_gnn:.4f} & {t_mesh:.4f} & {t_ale:.4f} & "
                f"{t_fem:.4f} & {t_rem:.4f} & "
                f"{t_wall_med:.4f} [{t_wall_iqr:.4f}] \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def run_stage_d_pde_benchmark(
    config_path: str | Path = "config/stage_d.yaml",
    stage_a_checkpoint: str | Path = "results/stage_a_oracle_v4/checkpoints/best_model.pt",
    stage_d_checkpoint: str | Path = "results/stage_d_trajectory_aware/checkpoints/best_model.pt",
    output_dir: str | Path = "results/stage_d_pde_benchmark",
    *,
    smoke_test: bool = False,
    max_cases_per_split: Optional[int] = None,
    n_timing_repeats: int = 5,
    device: str = "cpu",
    splits: Sequence[str] = ("test_interp", "test_extrap"),
) -> Dict[str, Any]:
    if n_timing_repeats < 5:
        raise ValueError("Il protocollo richiede almeno 5 ripetizioni temporali (n_timing_repeats >= 5).")

    out_dir = ensure_dir(Path(output_dir).resolve())
    figures_dir = ensure_dir(out_dir / "figures")
    log_path = out_dir / "benchmark.log"

    logger = logging.getLogger("gdm.stage_d_pde_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    cfg = load_stage_d_config(config_path)
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed, deterministic_torch=True)

    stage_a_ckpt_path = Path(stage_a_checkpoint).resolve()
    stage_d_ckpt_path = Path(stage_d_checkpoint).resolve()
    logger.info("Caricamento checkpoint Stage A: %s", stage_a_ckpt_path)
    engine_stage_a = build_inference_engine_for_checkpoint(stage_a_ckpt_path, cfg, device=device)
    logger.info("Caricamento checkpoint Stage D: %s", stage_d_ckpt_path)
    engine_stage_d = build_inference_engine_for_checkpoint(stage_d_ckpt_path, cfg, device=device)

    engines: Dict[str, Optional[MeshMoverInferenceEngine]] = {
        "fixed_fem": None,
        "stage_a_gnn": engine_stage_a,
        "stage_d_gnn": engine_stage_d,
    }

    cases = resolve_heldout_cases(
        cfg,
        splits=splits,
        smoke_test=smoke_test,
        max_cases_per_split=max_cases_per_split,
    )
    logger.info(
        "Avvio Stage D PDE benchmark | modalita'=%s | casi held-out=%d | repeats=%d (+1 warm-up)",
        "SMOKE_TEST" if smoke_test else "FULL_HELDOUT",
        len(cases),
        n_timing_repeats,
    )

    per_case_rows: List[Dict[str, Any]] = []
    per_timestep_rows: List[Dict[str, Any]] = []
    runtime_repetition_rows: List[Dict[str, Any]] = []

    timing_keys = (
        "gnn_inference_time",
        "mesh_update_time",
        "safety_filter_time",
        "coord_update_time",
        "ale_time",
        "fem_solve_time",
        "remapping_time",
        "other_overhead_time",
        "total_wall_time",
    )

    for case_idx, case_spec in enumerate(cases, start=1):
        cid = str(case_spec["case_id"])
        split = str(case_spec["split"])
        logger.info(
            "[%02d/%02d] Caso %s (%s) | sigma=%.4f, beta=%.3f, seed=%d",
            case_idx,
            len(cases),
            cid,
            split,
            float(case_spec["sigma"]),
            float(case_spec["beta"]),
            int(case_spec["case_seed"]),
        )
        for strat_key, strat_label in STRATEGIES_ORDER:
            set_seed(int(case_spec["case_seed"]), deterministic_torch=True)
            # 1. Deterministic accuracy, geometry and physical consistency evaluation
            diag_out = run_pde_trajectory_episode(
                case_spec=case_spec,
                strategy=strat_key,
                config=cfg,
                inference_engine=engines[strat_key],
                compute_diagnostics=True,
            )
            case_row = dict(diag_out["case_summary"])
            per_timestep_rows.extend(diag_out["timestep_rows"])

            # 2. Warm-up repetition (excluded from timing statistics)
            warmup_out = run_pde_trajectory_episode(
                case_spec=case_spec,
                strategy=strat_key,
                config=cfg,
                inference_engine=engines[strat_key],
                compute_diagnostics=False,
            )
            warmup_row = {
                "case_id": cid,
                "split": split,
                "seed": int(case_spec["case_seed"]),
                "strategy": strat_key,
                "repetition": 0,
                "is_warmup": True,
            }
            warmup_row.update(warmup_out["timings"])
            runtime_repetition_rows.append(warmup_row)

            # 3. Timed repetitions (at least 5) without diagnostic integration overhead
            timed_samples: List[Dict[str, float]] = []
            for rep in range(1, n_timing_repeats + 1):
                rep_out = run_pde_trajectory_episode(
                    case_spec=case_spec,
                    strategy=strat_key,
                    config=cfg,
                    inference_engine=engines[strat_key],
                    compute_diagnostics=False,
                )
                timed_samples.append(rep_out["timings"])
                rep_row = {
                    "case_id": cid,
                    "split": split,
                    "seed": int(case_spec["case_seed"]),
                    "strategy": strat_key,
                    "repetition": int(rep),
                    "is_warmup": False,
                }
                rep_row.update(rep_out["timings"])
                runtime_repetition_rows.append(rep_row)

            for t_key in timing_keys:
                stats_t = _summary_stats([s[t_key] for s in timed_samples])
                case_row[t_key] = float(stats_t["mean"] or 0.0)
                case_row[f"{t_key}_std"] = float(stats_t["std"] or 0.0)
                case_row[f"{t_key}_median"] = float(stats_t["median"] or 0.0)
                case_row[f"{t_key}_iqr"] = float(stats_t["iqr"] or 0.0)
                case_row[f"{t_key}_q25"] = float(stats_t["q25"] or 0.0)
                case_row[f"{t_key}_q75"] = float(stats_t["q75"] or 0.0)

            per_case_rows.append(case_row)
            logger.info(
                "   -> %-12s | int_L2=%.4f%% | final_L2=%.4f%% | minJ=%.3e | alpha_mean=%.4f | wall_med=%.4fs (IQR=%.4fs)",
                strat_label,
                100.0 * case_row["integrated_l2_error"],
                100.0 * case_row["final_l2_error"],
                case_row["min_jacobian"],
                case_row["safety_alpha"],
                case_row["total_wall_time_median"],
                case_row["total_wall_time_iqr"],
            )

    accuracy_metric_keys = (
        "relative_l2_error",
        "relative_h1_error",
        "integrated_l2_error",
        "integrated_h1_error",
        "final_l2_error",
        "final_h1_error",
        "min_jacobian",
        "mean_mesh_quality",
        "min_escobar_quality",
        "max_aspect_ratio",
        "inadmissible_cells",
        "inverted_cells",
        "degenerate_cells",
        "safety_alpha",
        "min_safety_alpha",
        "safety_filter_interventions",
        "displacement_norm",
        "max_window_displacement_norm",
        "final_cumulative_displacement_norm",
        "mass_balance_residual",
        "mean_mass_balance_residual",
        "final_mass_discrepancy_error",
        "gcl_max_elem_residual",
        "gcl_algebraic_residual_linf",
        "gcl_dual_col_sum_linf",
        "max_linear_solver_residual",
    )

    def _aggregate_subset(
        subset_cases: Sequence[Mapping[str, Any]],
        subset_reps: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        out_group: Dict[str, Any] = {}
        for strat_key, strat_label in STRATEGIES_ORDER:
            c_rows = [r for r in subset_cases if r["strategy"] == strat_key]
            r_rows = [
                r for r in subset_reps
                if r["strategy"] == strat_key and not bool(r["is_warmup"])
            ]
            if not c_rows:
                continue
            acc_summary = {
                m_key: _summary_stats([float(r[m_key]) for r in c_rows])
                for m_key in accuracy_metric_keys
            }
            rt_summary = {
                t_key: _summary_stats([float(r[t_key]) for r in r_rows])
                for t_key in timing_keys
            }
            out_group[strat_key] = {
                "label": strat_label,
                "n_cases": len(c_rows),
                "n_timed_repetitions_total": len(r_rows),
                "accuracy_and_geometry": acc_summary,
                "runtime_repetitions": rt_summary,
            }
        return out_group

    by_split_summary: Dict[str, Any] = {}
    for split in splits:
        by_split_summary[split] = _aggregate_subset(
            [r for r in per_case_rows if r["split"] == split],
            [r for r in runtime_repetition_rows if r["split"] == split],
        )
    overall_summary = _aggregate_subset(per_case_rows, runtime_repetition_rows)

    # Flatten summary rows for benchmark_summary.csv
    summary_csv_rows: List[Dict[str, Any]] = []
    for scope_name, group_dict in [
        *[(s, by_split_summary[s]) for s in splits],
        ("all_heldout", overall_summary),
    ]:
        for strat_key, _ in STRATEGIES_ORDER:
            if strat_key not in group_dict:
                continue
            rec = group_dict[strat_key]
            acc = rec["accuracy_and_geometry"]
            rt = rec["runtime_repetitions"]
            flat: Dict[str, Any] = {
                "scope": scope_name,
                "strategy": strat_key,
                "n_cases": rec["n_cases"],
                "n_timed_repetitions_total": rec["n_timed_repetitions_total"],
            }
            for m_key in (
                "integrated_l2_error",
                "final_l2_error",
                "relative_l2_error",
                "relative_h1_error",
                "integrated_h1_error",
                "final_h1_error",
                "min_jacobian",
                "mean_mesh_quality",
                "inadmissible_cells",
                "safety_alpha",
                "safety_filter_interventions",
                "displacement_norm",
                "mass_balance_residual",
                "gcl_algebraic_residual_linf",
            ):
                st = acc[m_key]
                flat[f"{m_key}_mean"] = st["mean"]
                flat[f"{m_key}_std"] = st["std"]
                flat[f"{m_key}_median"] = st["median"]
                flat[f"{m_key}_iqr"] = st["iqr"]
            for t_key in (
                "gnn_inference_time",
                "mesh_update_time",
                "ale_time",
                "fem_solve_time",
                "remapping_time",
                "total_wall_time",
            ):
                st = rt[t_key]
                flat[f"{t_key}_mean"] = st["mean"]
                flat[f"{t_key}_std"] = st["std"]
                flat[f"{t_key}_median"] = st["median"]
                flat[f"{t_key}_iqr"] = st["iqr"]
            summary_csv_rows.append(flat)

    summary_payload: Dict[str, Any] = {
        "experiment": "stage_d_pde_aware_trajectory_benchmark",
        "mode": "smoke_test" if smoke_test else "full_heldout",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "backend": "C++20/deal.II (_gdm_core) for FEM, ALE, mesh, safety filter, and continuous error quadrature",
        "remapping_used": False,
        "physical_consistency_note": (
            "The MMS convection-diffusion problem has non-zero forcing and time-varying Dirichlet BCs, "
            "so total mass is not constant in time. We report the discrete step mass-balance residual "
            "|(M_h^{n} - M_h^{n-1}) - (M_ref^{n} - M_ref^{n-1})| coherent with forcing/BCs and the exact "
            "D-GCL residuals from deal.II ALESolver2D."
        ),
        "seed": seed,
        "device": device,
        "n_warmup_repeats": 1,
        "n_timing_repeats": n_timing_repeats,
        "config_path": str(Path(config_path).resolve()),
        "checkpoints": {
            "stage_a": {
                "path": str(stage_a_ckpt_path),
                "sha256": _sha256(stage_a_ckpt_path),
                "normalize_field_scale": bool(engine_stage_a.normalize_field_scale),
            },
            "stage_d": {
                "path": str(stage_d_ckpt_path),
                "sha256": _sha256(stage_d_ckpt_path),
                "normalize_field_scale": bool(engine_stage_d.normalize_field_scale),
            },
        },
        "effective_numerical_protocol": {
            "domain": dict(cfg["geometry"]["domain"]),
            "initial_grid": dict(cfg["geometry"]["initial_grid"]),
            "safety": dict(cfg["geometry"]["safety"]),
            "pde_fixed_coefficients": dict(cfg["stage_d"]["pde"]),
            "dt": float(cfg["stage_d"]["dt"]),
            "adapt_interval": int(cfg["stage_d"]["adapt_interval"]),
        },
        "heldout_case_ids": {
            split: [c["case_id"] for c in cases if c["split"] == split]
            for split in splits
        },
        "by_split": by_split_summary,
        "overall_heldout": overall_summary,
    }

    _write_csv(out_dir / "per_case_results.csv", per_case_rows)
    _write_csv(out_dir / "per_timestep_results.csv", per_timestep_rows)
    _write_csv(out_dir / "runtime_repetitions.csv", runtime_repetition_rows)
    _write_csv(out_dir / "benchmark_summary.csv", summary_csv_rows)
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(summary_payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    (out_dir / "table_pde_accuracy.tex").write_text(
        _build_latex_accuracy_table(summary_payload), encoding="utf-8"
    )
    (out_dir / "table_computational_cost.tex").write_text(
        _build_latex_cost_table(summary_payload), encoding="utf-8"
    )

    epsilon = float(cfg["geometry"]["safety"]["epsilon"])
    fig_paths_sub = plot_stage_d_pde_benchmark(
        per_case_rows, per_timestep_rows, runtime_repetition_rows, figures_dir, epsilon=epsilon
    )
    fig_paths_root = plot_stage_d_pde_benchmark(
        per_case_rows, per_timestep_rows, runtime_repetition_rows, out_dir, epsilon=epsilon
    )
    summary_payload["figures"] = {
        k: [str(p) for p in v] for k, v in fig_paths_sub.items()
    }
    summary_payload["figures_root"] = {
        k: [str(p) for p in v] for k, v in fig_paths_root.items()
    }
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(summary_payload, indent=2, allow_nan=False), encoding="utf-8"
    )

    logger.info("Benchmark completato. Risultati salvati in: %s", out_dir)
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/stage_d.yaml")
    parser.add_argument(
        "--stage-a-checkpoint",
        default="results/stage_a_oracle_v4/checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--stage-d-checkpoint",
        default="results/stage_d_trajectory_aware/checkpoints/best_model.pt",
    )
    parser.add_argument("--output-dir", default="results/stage_d_pde_benchmark")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Esegue 1 caso held-out per ciascuno split (test_interp e test_extrap).",
    )
    parser.add_argument(
        "--max-cases-per-split",
        type=int,
        default=None,
        help="Limite opzionale sul numero di casi per split (in ordine di manifest).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Numero di ripetizioni temporali (>= 5) oltre al warm-up iniziale.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test_interp", "test_extrap"],
        choices=["test_interp", "test_extrap"],
    )
    args = parser.parse_args()

    run_stage_d_pde_benchmark(
        config_path=args.config,
        stage_a_checkpoint=args.stage_a_checkpoint,
        stage_d_checkpoint=args.stage_d_checkpoint,
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
        max_cases_per_split=args.max_cases_per_split,
        n_timing_repeats=args.repeats,
        device=args.device,
        splits=args.splits,
    )


if __name__ == "__main__":
    main()