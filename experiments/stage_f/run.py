"""Stage-F1: fair comparison of classical variational r-adaptation and GDM.

The classical controllers use only the current numerical field and its native
recovered gradient. The Stage-D benchmark supplies the common PDE/ALE solver,
geometry checks and safety filter. Test-split labels/optimal-node targets are
never accessed by this module.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "python"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gdm.classical_r_adaptation import (  # noqa: E402
    MMPDEParameters,
    VariationalMMPDEController,
    structured_reference_nodes,
)

LOG = logging.getLogger("gdm.stage_f")


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configurazione YAML non valida: {path}")
    return value


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Tipo non serializzabile: {type(value)!r}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Keep a stable union of columns: optional diagnostic columns may be absent.
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key)); seen.add(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _parse_csv_value(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        number = float(value)
    except ValueError:
        return value
    if np.isfinite(number) and number.is_integer() and not any(ch in value.lower() for ch in (".", "e")):
        return int(number)
    return number


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [{key: _parse_csv_value(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_hashes() -> Dict[str, str]:
    # Hash the Python orchestration, native API/feature contract and source files
    # that define the FEM/ALE/safety-filter behaviour. This prevents a resumed
    # campaign from silently mixing results produced by different numerical cores.
    files = {
        "stage_f_runner": ROOT / "experiments/stage_f/run.py",
        "classical_r_adaptation": ROOT / "python/gdm/classical_r_adaptation.py",
        "stage_d_benchmark_backend": ROOT / "experiments/stage_d/benchmark_pde.py",
        "feature_adapter": ROOT / "python/gdm/feature_adapter.py",
        "native_api": ROOT / "python/gdm/native_api.py",
        "cmake_configuration": ROOT / "CMakeLists.txt",
        "native_mesh_state": ROOT / "cpp_core/src/mesh_state.cpp",
        "native_fem_solver": ROOT / "cpp_core/src/fem_solver.cpp",
        "native_ale_solver": ROOT / "cpp_core/src/ale_solver.cpp",
        "native_safety_filter": ROOT / "cpp_core/src/safety_filter.cpp",
        "native_bindings": ROOT / "cpp_core/bindings/pybind_module.cpp",
        "fem_solver_header": ROOT / "cpp_core/include/gdm/fem_solver.hpp",
        "ale_solver_header": ROOT / "cpp_core/include/gdm/ale_solver.hpp",
        "mesh_state_header": ROOT / "cpp_core/include/gdm/mesh_state.hpp",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Sorgenti necessari per la firma Stage F mancanti: {missing}")
    hashes = {name: _sha256(path) for name, path in files.items()}
    # Include the actual loaded extension binary too; source hashes alone do not
    # prove which compiled artifact was used for the campaign.
    try:
        import importlib
        native_module = importlib.import_module("gdm._gdm_core")
        native_path = Path(native_module.__file__).resolve()
        hashes["native_extension_binary_sha256"] = _sha256(native_path)
    except (ImportError, AttributeError, OSError):
        hashes["native_extension_binary_sha256"] = "unavailable"
    return hashes


def _tuning_signature(
    stage_f_cfg: Mapping[str, Any],
    stage_d_cfg: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    *,
    smoke: bool,
) -> Dict[str, Any]:
    return {
        "configuration_sha256": hashlib.sha256(json.dumps(
            {"stage_f": stage_f_cfg, "stage_d": stage_d_cfg}, sort_keys=True, default=str
        ).encode("utf-8")).hexdigest(),
        "implementation_sha256": _implementation_hashes(),
        "validation_case_specs_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "validation_case_ids": [case["case_id"] for case in cases],
        "smoke": bool(smoke),
    }


def _stage_d_config(stage_f: Mapping[str, Any]) -> Dict[str, Any]:
    source = (ROOT / str(stage_f["source"]["config"])).resolve()
    cfg = _read_yaml(source)
    protocol = stage_f["protocol"]
    cfg["geometry"]["initial_grid"] = {"nx": int(protocol["nx"]), "ny": int(protocol["ny"])}
    cfg["geometry"]["safety"]["epsilon"] = float(stage_f["safety"]["epsilon"])
    cfg["geometry"]["safety"]["safety_margin"] = float(stage_f["safety"].get("safety_margin", 1.0e-6))
    cfg["geometry"]["safety"]["tol_deg"] = float(stage_f["safety"].get("tol_deg", 1.0e-12))
    cfg["stage_d"]["dt"] = float(protocol["dt"])
    cfg["stage_d"]["adapt_interval"] = int(protocol["adapt_interval"])
    # Reuse the solver tolerances and Galerkin choice verified in Stage E1.
    numerics_path = (ROOT / str(stage_f["source"]["numerics_config"])).resolve()
    numerics_cfg = _read_yaml(numerics_path)
    cfg["stage_f_numerics"] = {
        "linear_solver": copy.deepcopy(numerics_cfg["linear_solver"]),
        "stabilization": copy.deepcopy(numerics_cfg["stabilization"]),
    }
    cfg["stage_f_numerics"]["stabilization"]["mode"] = "galerkin"
    return cfg


def _load_case_specs(stage_d_cfg: Mapping[str, Any], split: str, max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read only manifest IDs and physical metadata; no oracle target fields are consumed."""
    if split not in {"validation", "test_interp", "test_extrap"}:
        raise ValueError(f"Split non consentito: {split}")
    stage = stage_d_cfg["stage_d"]
    pde = stage["pde"]
    raw_dir = (ROOT / stage_d_cfg["dataset"]["raw_dir"]).resolve()
    manifest_path = (ROOT / stage_d_cfg["dataset"]["manifest_path"]).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest Stage D mancante: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_ids = list(manifest.get(split, []))
    if max_cases is not None:
        sample_ids = sample_ids[:max(0, int(max_cases))]
    if not sample_ids:
        raise RuntimeError(f"Nessun campione presente nello split {split!r}.")

    rows: List[Dict[str, Any]] = []
    base_seed = int(stage_d_cfg["experiment"]["seed"])
    split_offset = {"validation": 3000, "test_interp": 1000, "test_extrap": 2000}[split]
    for idx, sample_id in enumerate(sample_ids):
        path = raw_dir / f"{sample_id}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Campione {split} mancante: {path}")
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        metadata = payload.get("metadata", {})
        recorded_split = str(metadata.get("split", split))
        if recorded_split != split:
            raise ValueError(f"Split errato per {sample_id}: atteso {split}, trovato {recorded_split}.")
        parameters = metadata.get("parameters", {})
        sigma, beta = float(parameters["sigma"]), float(parameters["beta"])
        regime = "in_domain"
        if split == "test_extrap":
            low = stage_d_cfg["stage_d"]["splits"]["test_extrap"]["sigma_extrap_low"]
            regime = "sigma_low" if sigma <= float(low[1]) + 1e-9 else "sigma_high"
        rows.append({
            "case_id": str(sample_id), "split": split, "local_index": idx,
            "case_seed": base_seed + split_offset + idx,
            "bx": float(pde["bx"]), "by": float(pde["by"]), "nu": float(pde["nu"]),
            "sigma": sigma, "beta": beta, "T": float(pde["T"]),
            "dt": float(stage["dt"]), "adapt_interval": int(stage["adapt_interval"]),
            "extrapolation_regime": regime, "source": str(path),
            "case_file_sha256": _sha256(path),
        })
    return rows


def _structured_connectivity(nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    cells: List[Tuple[int, int, int]] = []
    boundary: List[int] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            if i == 0 or i == nx or j == 0 or j == ny:
                boundary.append(j * (nx + 1) + i)
    for j in range(ny):
        for i in range(nx):
            n00 = j * (nx + 1) + i
            n10 = n00 + 1
            n01 = (j + 1) * (nx + 1) + i
            n11 = n01 + 1
            cells.extend(((n00, n10, n11), (n00, n11, n01)))
    return np.asarray(cells, dtype=np.int64), np.asarray(boundary, dtype=np.int64)


def _make_controller(stage_f_cfg: Mapping[str, Any], domain: Mapping[str, float], method: str, strength: Optional[float] = None) -> VariationalMMPDEController:
    c = stage_f_cfg["classical_r"]
    nx, ny = int(stage_f_cfg["protocol"]["nx"]), int(stage_f_cfg["protocol"]["ny"])
    nodes = structured_reference_nodes(domain, nx, ny)
    cells, boundary = _structured_connectivity(nx, ny)
    monitor = method.removeprefix("mmpde_")
    params = MMPDEParameters(
        monitor_type=monitor,
        monitor_strength=float(strength if strength is not None else c["monitor_strength_candidates"][0]),
        monitor_smoothing_passes=int(c["monitor_smoothing_passes"]),
        smoothing_weight=float(c["smoothing_weight"]),
        metric_eigenvalue_min=float(c["metric_eigenvalue_min"]),
        metric_eigenvalue_max=float(c["metric_eigenvalue_max"]),
        q=int(c["q"]), theta=float(c["theta"]),
        pseudo_iterations=int(c["pseudo_iterations"]),
        max_move_fraction=float(c["max_move_fraction"]),
        armijo=float(c["armijo"]),
        max_backtracking=int(c["max_backtracking"]),
        relative_energy_tolerance=float(c["relative_energy_tolerance"]),
        safety_epsilon=float(stage_f_cfg["safety"]["epsilon"]),
    )
    return VariationalMMPDEController(nodes, cells, boundary, params)


def _canonical_method(method: str) -> str:
    return {"stage_d_gnn": "gdm", "fixed_fem": "fixed_fem"}.get(method, method)


def _episode(
    run_pde_trajectory_episode: Any,
    case: Mapping[str, Any],
    stage_d_cfg: Mapping[str, Any],
    method: str,
    domain: Mapping[str, float],
    stage_f_cfg: Mapping[str, Any],
    *,
    gdm_engine: Any = None,
    strength: Optional[float] = None,
    diagnostics: bool,
) -> Tuple[Dict[str, Any], Optional[VariationalMMPDEController]]:
    if method == "fixed_fem":
        result = run_pde_trajectory_episode(case, "fixed_fem", stage_d_cfg, None, compute_diagnostics=diagnostics)
        return result, None
    if method == "gdm":
        if gdm_engine is None:
            raise RuntimeError("Checkpoint GDM richiesto ma inference_engine assente.")
        result = run_pde_trajectory_episode(case, "stage_d_gnn", stage_d_cfg, gdm_engine, compute_diagnostics=diagnostics)
        return result, None
    controller = _make_controller(stage_f_cfg, domain, method, strength)
    result = run_pde_trajectory_episode(case, "stage_d_gnn", stage_d_cfg, controller, compute_diagnostics=diagnostics)
    return result, controller


def _load_runtime_modules():
    # Delayed import: permits syntax/unit tests without the compiled deal.II module.
    from experiments.stage_d.benchmark_pde import (
        ROOT as STAGE_D_ROOT,
        build_inference_engine_for_checkpoint,
        run_pde_trajectory_episode,
    )
    return STAGE_D_ROOT, build_inference_engine_for_checkpoint, run_pde_trajectory_episode


def _domain(stage_d_cfg: Mapping[str, Any]) -> Dict[str, float]:
    return {k: float(v) for k, v in stage_d_cfg["geometry"]["domain"].items()}


def _evaluate_validation(
    stage_f_cfg: Mapping[str, Any],
    stage_d_cfg: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    run_episode: Any,
    *,
    checkpoint_path: Optional[Path] = None,
    existing_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Tune only the predeclared monitor strength grid on validation cases.

    Per-case results are checkpointed after each completed candidate/case pair,
    allowing an interrupted tuning run to resume without recomputing finished
    trajectories. No held-out data enter this routine.
    """
    domain = _domain(stage_d_cfg)
    monitor_types = list(stage_f_cfg["classical_r"]["monitor_types"])
    per_case_rows: List[Dict[str, Any]] = [dict(row) for row in (existing_rows or [])]
    completed = {
        (str(row["case_id"]), str(row["split"]), str(row["method"]), round(float(row["monitor_strength"]), 12))
        for row in per_case_rows
    }

    for monitor in monitor_types:
        method = f"mmpde_{monitor}"
        candidate_key = "hessian_monitor_strength_candidates" if monitor == "hessian" else "monitor_strength_candidates"
        candidates = list(stage_f_cfg["classical_r"][candidate_key])
        for strength in candidates:
            for case_index, case in enumerate(cases, start=1):
                LOG.info("Validation tuning %s alpha=%g caso %d/%d (%s)",
                         method, strength, case_index, len(cases), case["case_id"])
                key = (str(case["case_id"]), str(case["split"]), method, round(float(strength), 12))
                if key in completed:
                    continue
                result, controller = _episode(
                    run_episode, case, stage_d_cfg, method, domain, stage_f_cfg,
                    strength=float(strength), diagnostics=True,
                )
                row = result["case_summary"]
                events = controller.events if controller is not None else []
                per_case_rows.append({
                    "case_id": str(case["case_id"]), "split": str(case["split"]),
                    "method": method, "monitor_strength": float(strength),
                    "integrated_l2_error": float(row["integrated_l2_error"]),
                    "final_l2_error": float(row["final_l2_error"]),
                    "integrated_h1_seminorm_error": float(row["integrated_h1_error"]),
                    "safety_filter_interventions": int(row["safety_filter_interventions"]),
                    "adaptation_event_count": int(len(events)),
                    "mean_accepted_pseudo_steps": float(np.mean([e["accepted_pseudo_steps"] for e in events])) if events else 0.0,
                    "mean_energy_decrease": float(np.mean([e["relative_energy_decrease"] for e in events])) if events else 0.0,
                    "line_search_failures": int(sum(e["line_search_failures"] for e in events)),
                    "pseudo_iteration_caps": int(sum(bool(e.get("pseudo_iterations_exhausted", False)) for e in events)),
                    "termination_relative_energy_tolerance": int(sum(e.get("termination_reason") == "relative_energy_tolerance" for e in events)),
                    "termination_gradient_tolerance": int(sum(e.get("termination_reason") == "gradient_tolerance" for e in events)),
                    "termination_max_iterations": int(sum(e.get("termination_reason") == "maximum_pseudo_iterations" for e in events)),
                    "termination_line_search_failure": int(sum(e.get("termination_reason") == "line_search_failure" for e in events)),
                    "energy_monotone_all_events": bool(all(bool(e["energy_monotone"]) for e in events)),
                })
                completed.add(key)
                if checkpoint_path is not None:
                    _write_csv(checkpoint_path, per_case_rows)

    expected_keys = {
        (str(case["case_id"]), str(case["split"]), f"mmpde_{monitor}", round(float(strength), 12))
        for monitor in monitor_types
        for strength in stage_f_cfg["classical_r"]["hessian_monitor_strength_candidates" if monitor == "hessian" else "monitor_strength_candidates"]
        for case in cases
    }
    observed_keys = {
        (str(row["case_id"]), str(row["split"]), str(row["method"]), round(float(row["monitor_strength"]), 12))
        for row in per_case_rows
    }
    if observed_keys != expected_keys or len(per_case_rows) != len(expected_keys):
        raise RuntimeError(
            "Tuning validation incompleto o con duplicati: "
            f"mancanti={len(expected_keys - observed_keys)}, extra={len(observed_keys - expected_keys)}."
        )

    trial_rows: List[Dict[str, Any]] = []
    for monitor in monitor_types:
        method = f"mmpde_{monitor}"
        candidate_key = "hessian_monitor_strength_candidates" if monitor == "hessian" else "monitor_strength_candidates"
        for strength in stage_f_cfg["classical_r"][candidate_key]:
            subset = [row for row in per_case_rows
                      if row["method"] == method and round(float(row["monitor_strength"]), 12) == round(float(strength), 12)]
            errors = np.asarray([float(row["integrated_l2_error"]) for row in subset], dtype=np.float64)
            pseudo = [float(row["mean_accepted_pseudo_steps"]) for row in subset]
            trial_rows.append({
                "method": method, "monitor_strength": float(strength), "n_validation_cases": len(subset),
                "mean_integrated_l2_error": float(np.mean(errors)),
                "median_integrated_l2_error": float(np.median(errors)),
                "std_integrated_l2_error": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
                "mean_safety_interventions": float(np.mean([r["safety_filter_interventions"] for r in subset])),
                "mean_accepted_pseudo_steps": float(np.mean(pseudo)) if pseudo else 0.0,
                "total_adaptation_events": int(sum(r["adaptation_event_count"] for r in subset)),
                "total_line_search_failures": int(sum(r["line_search_failures"] for r in subset)),
                "total_pseudo_iteration_caps": int(sum(r["pseudo_iteration_caps"] for r in subset)),
                "pseudo_iteration_cap_fraction": float(sum(r["pseudo_iteration_caps"] for r in subset) / max(1, sum(r["adaptation_event_count"] for r in subset))),
                "termination_relative_energy_tolerance": int(sum(r["termination_relative_energy_tolerance"] for r in subset)),
                "termination_gradient_tolerance": int(sum(r["termination_gradient_tolerance"] for r in subset)),
                "termination_max_iterations": int(sum(r["termination_max_iterations"] for r in subset)),
                "termination_line_search_failure": int(sum(r["termination_line_search_failure"] for r in subset)),
                "all_event_energies_monotone": bool(all(r["energy_monotone_all_events"] for r in subset)),
                "numerically_valid": bool(
                    all(r["energy_monotone_all_events"] for r in subset)
                    and sum(r["line_search_failures"] for r in subset) == 0
                ),
            })
            LOG.info("Validation %s strength=%g: mean integrated L2=%.6e", method, strength, np.mean(errors))

    selected: Dict[str, Any] = {}
    for method in (f"mmpde_{m}" for m in monitor_types):
        subset = [row for row in trial_rows if row["method"] == method]
        eligible = [row for row in subset if row["numerically_valid"]]
        if not eligible:
            raise RuntimeError(
                f"Nessun candidato {method} numericamente valido su validation: "
                "sono presenti line-search failure o energia non monotona."
            )
        best = min(eligible, key=lambda row: (row["mean_integrated_l2_error"], row["monitor_strength"]))
        selected[method] = {
            "monitor_strength": best["monitor_strength"],
            "validation_mean_integrated_l2_error": best["mean_integrated_l2_error"],
            "selection_metric": "mean_case_integrated_relative_L2_error",
            "selection_split": "validation",
        }
    return {"trials": trial_rows, "per_case": per_case_rows, "selected": selected}

def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    m = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.ones(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * float(p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def _paired_inference(rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    from scipy.stats import wilcoxon
    by_method_case = {
        (str(r["strategy"]), str(r["split"]), str(r["case_id"])): float(r["integrated_l2_error"])
        for r in rows
    }
    strategies = sorted({str(r["strategy"]) for r in rows})
    gdm_key = "gdm"
    if gdm_key not in strategies:
        return []
    groups = sorted({str(r["split"]) for r in rows})
    comparisons: List[Dict[str, Any]] = []
    split_case_keys: List[Tuple[str, List[Tuple[str, str]]]] = [
        (split, sorted({(split, str(r["case_id"])) for r in rows if str(r["split"]) == split}))
        for split in groups
    ]
    split_case_keys.append(("all", sorted({(str(r["split"]), str(r["case_id"])) for r in rows})))
    baseline_methods = [method for method in strategies if method != gdm_key]
    bootstrap_n = int(cfg["statistics"]["bootstrap_samples"])
    confidence = float(cfg["statistics"]["confidence_level"])
    rng = np.random.default_rng(int(cfg["experiment"]["seed"]))
    for split, case_keys in split_case_keys:
        for baseline in baseline_methods:
            keys = [key for key in case_keys
                    if (baseline, key[0], key[1]) in by_method_case and (gdm_key, key[0], key[1]) in by_method_case]
            if not keys:
                continue
            base = np.asarray([by_method_case[(baseline, key[0], key[1])] for key in keys], dtype=float)
            gdm = np.asarray([by_method_case[(gdm_key, key[0], key[1])] for key in keys], dtype=float)
            improvement = base - gdm  # positive => GDM lower error
            rel = 100.0 * improvement / np.maximum(base, np.finfo(float).tiny)
            idx = rng.integers(0, len(keys), size=(bootstrap_n, len(keys)))
            boot_abs = np.mean(improvement[idx], axis=1)
            boot_rel = np.mean(rel[idx], axis=1)
            lo_q, hi_q = (1.0 - confidence) / 2.0, 1.0 - (1.0 - confidence) / 2.0
            try:
                p = float(wilcoxon(improvement, zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                p = 1.0
            comparisons.append({
                "split": split, "baseline": baseline, "candidate": "gdm", "n_paired_cases": len(keys),
                "mean_baseline_error": float(np.mean(base)), "mean_gdm_error": float(np.mean(gdm)),
                "mean_absolute_error_reduction": float(np.mean(improvement)),
                "absolute_reduction_ci_low": float(np.quantile(boot_abs, lo_q)),
                "absolute_reduction_ci_high": float(np.quantile(boot_abs, hi_q)),
                "mean_relative_error_reduction_pct": float(np.mean(rel)),
                "relative_reduction_ci_low_pct": float(np.quantile(boot_rel, lo_q)),
                "relative_reduction_ci_high_pct": float(np.quantile(boot_rel, hi_q)),
                "gdm_win_fraction": float(np.mean(improvement > 0.0)),
                "wilcoxon_p_raw": p,
            })
    adjusted = _holm_adjust([r["wilcoxon_p_raw"] for r in comparisons]) if comparisons else []
    for row, p_adj in zip(comparisons, adjusted):
        row["wilcoxon_p_holm"] = float(p_adj)
    return comparisons


def _aggregate(rows: Sequence[Mapping[str, Any]], runtime_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    strategies = sorted({str(r["strategy"]) for r in rows})
    for method in strategies:
        subset = [r for r in rows if str(r["strategy"]) == method]
        run_subset = [r for r in runtime_rows if str(r["strategy"]) == method and not bool(r.get("warmup", False))]

        def case_runtime_medians(metric: str) -> np.ndarray:
            keys = sorted({(str(r["split"]), str(r["case_id"])) for r in run_subset})
            return np.asarray([
                np.median([float(r[metric]) for r in run_subset
                           if str(r["split"]) == split and str(r["case_id"]) == case_id])
                for split, case_id in keys
            ], dtype=np.float64)

        def runtime_summary(metric: str, stat: str) -> Optional[float]:
            values = case_runtime_medians(metric)
            if values.size == 0:
                return None
            if stat == "median":
                return float(np.median(values))
            if stat == "q25":
                return float(np.quantile(values, .25))
            if stat == "q75":
                return float(np.quantile(values, .75))
            raise ValueError(stat)

        out[method] = {
            "n_cases": len(subset),
            "integrated_l2_error_mean": float(np.mean([r["integrated_l2_error"] for r in subset])),
            "integrated_l2_error_median": float(np.median([r["integrated_l2_error"] for r in subset])),
            "integrated_l2_error_std": float(np.std([r["integrated_l2_error"] for r in subset], ddof=1)) if len(subset) > 1 else 0.0,
            "final_l2_error_mean": float(np.mean([r["final_l2_error"] for r in subset])),
            "integrated_h1_error_mean": float(np.mean([r["integrated_h1_error"] for r in subset])),
            "final_h1_error_mean": float(np.mean([r["final_h1_error"] for r in subset])),
            "min_jacobian_min": float(min(r["min_jacobian"] for r in subset)),
            "mean_mesh_quality": float(np.mean([r["mean_mesh_quality"] for r in subset])),
            "max_aspect_ratio_max": float(max(r["max_aspect_ratio"] for r in subset)),
            "inadmissible_cells_total": int(sum(r["inadmissible_cells"] for r in subset)),
            "safety_interventions_total": int(sum(r["safety_filter_interventions"] for r in subset)),
            "gcl_elem_residual_max": float(max(r["gcl_max_elem_residual"] for r in subset)),
            "gcl_algebraic_residual_max": float(max(r["gcl_algebraic_residual_linf"] for r in subset)),
            "linear_solver_residual_max": float(max(r["max_linear_solver_residual"] for r in subset)),
            "mass_balance_residual_max": float(max(r["mass_balance_residual"] for r in subset)),
            "timing_case_count": int(len(case_runtime_medians("total_wall_time"))),
            "total_wall_time_median_seconds": runtime_summary("total_wall_time", "median"),
            "total_wall_time_q25_seconds": runtime_summary("total_wall_time", "q25"),
            "total_wall_time_q75_seconds": runtime_summary("total_wall_time", "q75"),
            "strategy_compute_time_median_seconds": runtime_summary("strategy_compute_time", "median"),
            "ale_time_median_seconds": runtime_summary("ale_time", "median"),
            "fem_solve_time_median_seconds": runtime_summary("fem_solve_time", "median"),
        }
    return out

def _write_latex_tables(out_dir: Path, summary: Mapping[str, Any], comparisons: Sequence[Mapping[str, Any]]) -> None:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "fixed_fem": "Fixed FEM", "mmpde_winslow": "Variational r-adaptation (Winslow-type)",
        "mmpde_arclength": "Variational r-adaptation (arc-length)",
        "mmpde_hessian": "Variational r-adaptation (Hessian)", "gdm": "GDM (Stage D)",
    }
    row_end = r" \\"
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Stage-F1 accuracy and geometry on the held-out test cases.}",
             r"\label{tab:stage_f_accuracy}", r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Method & Cases & Mean integrated $L^2$ & Median integrated $L^2$ & Mean integrated $H^1$ seminorm & Mean final $L^2$ & Min. Jacobian " + r"\\",
             r"\hline"]
    for key in ("fixed_fem", "mmpde_winslow", "mmpde_arclength", "mmpde_hessian", "gdm"):
        if key not in summary:
            continue
        r = summary[key]
        lines.append(
            f"{labels[key]} & {r['n_cases']} & {r['integrated_l2_error_mean']:.4e} & "
            f"{r['integrated_l2_error_median']:.4e} & {r['integrated_h1_error_mean']:.4e} & "
            f"{r['final_l2_error_mean']:.4e} & {r['min_jacobian_min']:.4e}" + row_end
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (table_dir / "table_stage_f_accuracy.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Median online computational costs per trajectory; timing repetitions exclude warm-up.}",
             r"\label{tab:stage_f_cost}", r"\begin{tabular}{lrrrr}", r"\hline",
             r"Method & Median total [s] & Strategy compute [s] & ALE [s] & FEM [s] " + r"\\", r"\hline"]
    for key in ("fixed_fem", "mmpde_winslow", "mmpde_arclength", "mmpde_hessian", "gdm"):
        if key not in summary:
            continue
        r = summary[key]
        def fmt(v: Any) -> str:
            return "--" if v is None else f"{float(v):.4e}"
        lines.append(
            f"{labels[key]} & {fmt(r['total_wall_time_median_seconds'])} & "
            f"{fmt(r['strategy_compute_time_median_seconds'])} & {fmt(r['ale_time_median_seconds'])} & "
            f"{fmt(r['fem_solve_time_median_seconds'])}" + row_end
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (table_dir / "table_stage_f_cost.tex").write_text("\n".join(lines), encoding="utf-8")

    if comparisons:
        lines = [r"\begin{table}[t]", r"\centering\small",
                 r"\caption{Paired test-set improvement of GDM over each baseline. Positive values favour GDM.}",
                 r"\label{tab:stage_f_paired}", r"\begin{tabular}{llrrrr}", r"\hline",
                 r"Split & Baseline & $n$ & Mean reduction [95\% CI] & Win fraction & $p_{\mathrm{Holm}}$ " + r"\\", r"\hline"]
        for r in comparisons:
            if r["split"] == "all":
                lines.append(
                    f"All & {labels.get(r['baseline'], r['baseline'])} & {r['n_paired_cases']} & "
                    f"{r['mean_absolute_error_reduction']:.3e} "
                    f"[{r['absolute_reduction_ci_low']:.2e}, {r['absolute_reduction_ci_high']:.2e}] & "
                    f"{r['gdm_win_fraction']:.2f} & {r['wilcoxon_p_holm']:.3g}" + row_end
                )
        lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
        (table_dir / "table_stage_f_paired.tex").write_text("\n".join(lines), encoding="utf-8")

def _make_plots(out_dir: Path, rows: Sequence[Mapping[str, Any]], runtime_rows: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]]) -> List[str]:
    import matplotlib.pyplot as plt
    from plotting.style import save_figure, set_publication_style
    set_publication_style()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    order = ["fixed_fem", "mmpde_winslow", "mmpde_arclength", "mmpde_hessian", "gdm"]
    labels = ["Fixed", "Winslow-type", "Arc-length", "Hessian", "GDM"]
    # Accuracy grouped by held-out split (show distributions, not pseudo-replicates).
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    data, xpos, xticklabels = [], [], []
    position = 1
    for split in ("test_interp", "test_extrap"):
        for key, label in zip(order, labels):
            vals = [float(r["integrated_l2_error"]) for r in rows if r["split"] == split and r["strategy"] == key]
            if vals:
                data.append(vals); xpos.append(position); xticklabels.append(f"{label}\n{split.replace('test_', '')}")
            position += 1
        position += 0.6
    ax.boxplot(data, positions=xpos, widths=.65, showfliers=True)
    ax.set_xticks(xpos, xticklabels, rotation=20, ha="right")
    ax.set_ylabel(r"Time-integrated relative $L^2$ error")
    ax.set_title("Stage F1: held-out accuracy by strategy and split")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    p = save_figure(fig, fig_dir / "stage_f_accuracy_by_split")
    paths.append(str(p[0]))

    # Timing: aggregate repeated episode wall-clock measurements, excluding warm-ups.
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    medians, q1, q3 = [], [], []
    for key in order:
        subset = [r for r in runtime_rows if r["strategy"] == key and not r["warmup"]]
        case_keys = sorted({(str(r["split"]), str(r["case_id"])) for r in subset})
        case_medians = np.asarray([
            np.median([float(r["total_wall_time"]) for r in subset
                       if str(r["split"]) == split and str(r["case_id"]) == case_id])
            for split, case_id in case_keys
        ], dtype=float)
        medians.append(float(np.median(case_medians)) if case_medians.size else np.nan)
        q1.append(float(np.quantile(case_medians, .25)) if case_medians.size else np.nan)
        q3.append(float(np.quantile(case_medians, .75)) if case_medians.size else np.nan)
    xx = np.arange(len(order))
    ax.bar(xx, medians)
    ax.errorbar(xx, medians, yerr=[np.asarray(medians)-np.asarray(q1), np.asarray(q3)-np.asarray(medians)], fmt="none", capsize=4)
    ax.set_xticks(xx, labels, rotation=15, ha="right")
    ax.set_ylabel("Trajectory wall-clock time [s]")
    ax.set_title("Stage F1: per-case median computational cost (IQR across cases)")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    p = save_figure(fig, fig_dir / "stage_f_runtime")
    paths.append(str(p[0]))

    # Paired improvement, aggregate all test cases only.
    all_rows = [r for r in comparisons if r["split"] == "all"]
    if all_rows:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        names = [str(r["baseline"]).replace("mmpde_", "") for r in all_rows]
        vals = [float(r["mean_relative_error_reduction_pct"]) for r in all_rows]
        lo = [vals[i] - float(all_rows[i]["relative_reduction_ci_low_pct"]) for i in range(len(vals))]
        hi = [float(all_rows[i]["relative_reduction_ci_high_pct"]) - vals[i] for i in range(len(vals))]
        ax.bar(np.arange(len(vals)), vals)
        ax.errorbar(np.arange(len(vals)), vals, yerr=[lo, hi], fmt="none", capsize=4)
        ax.axhline(0.0, linestyle="--", linewidth=.9)
        ax.set_xticks(np.arange(len(vals)), names)
        ax.set_ylabel("Mean paired error reduction by GDM [%]")
        ax.set_title("Paired comparison (positive favours GDM)")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        p = save_figure(fig, fig_dir / "stage_f_paired_improvement")
        paths.append(str(p[0]))
    return paths


def run_tuning(stage_f_cfg: Mapping[str, Any], stage_d_cfg: Mapping[str, Any], *, smoke: bool = False) -> Path:
    _, _, run_episode = _load_runtime_modules()
    cases = _load_case_specs(stage_d_cfg, "validation", max_cases=1 if smoke else None)
    out_dir = (ROOT / str(stage_f_cfg["experiment"]["output_dir"]) / ("smoke_tuning" if smoke else "validation_tuning")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case_path = out_dir / "monitor_strength_validation_per_case.csv"
    trials_path = out_dir / "monitor_strength_validation_trials.csv"
    state_path = out_dir / "tuning_state.json"
    signature = _tuning_signature(stage_f_cfg, stage_d_cfg, cases, smoke=smoke)
    implementation_hashes = signature["implementation_sha256"]
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise RuntimeError("Tuning validation esistente con configurazione/codice/casi differenti. Usa un nuovo output_dir.")
    elif per_case_path.exists() or trials_path.exists():
        raise RuntimeError("Risultati tuning presenti senza tuning_state.json; archivia la directory o usa un nuovo output_dir.")
    else:
        _json_dump(state_path, {"status": "running", "signature": signature})

    snapshot_path = out_dir / "config_snapshot.json"
    snapshot = {"stage_f": dict(stage_f_cfg), "stage_d_effective": dict(stage_d_cfg)}
    if snapshot_path.is_file():
        previous_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if previous_snapshot != json.loads(json.dumps(snapshot, default=_json_default)):
            raise RuntimeError("config_snapshot.json di tuning non coincide con la configurazione corrente.")
    else:
        _json_dump(snapshot_path, snapshot)

    existing = _read_csv(per_case_path)
    _json_dump(state_path, {"status": "running", "signature": signature,
                            "completed_candidate_case_records": len(existing)})
    result = _evaluate_validation(
        stage_f_cfg, stage_d_cfg, cases, run_episode,
        checkpoint_path=per_case_path, existing_rows=existing,
    )
    _write_csv(trials_path, result["trials"])
    _write_csv(per_case_path, result["per_case"])
    _json_dump(out_dir / "selected_parameters.json", {
        "protocol": "selection on validation split only; no test labels accessed",
        "validation_case_ids": [case["case_id"] for case in cases],
        "selected": result["selected"],
        "all_trials": result["trials"],
        "implementation_sha256": implementation_hashes,
        "tuning_signature": signature,
    })
    _json_dump(state_path, {"status": "completed", "signature": signature,
                            "completed_candidate_case_records": len(result["per_case"]),
                            "completed_candidate_trials": len(result["trials"])})
    LOG.info("Parametri classici selezionati: %s", result["selected"])
    return out_dir / "selected_parameters.json"

def _load_selected(
    stage_f_cfg: Mapping[str, Any],
    stage_d_cfg: Mapping[str, Any],
    smoke: bool = False,
) -> Dict[str, Any]:
    suffix = "smoke_tuning" if smoke else "validation_tuning"
    directory = ROOT / str(stage_f_cfg["experiment"]["output_dir"]) / suffix
    path = directory / "selected_parameters.json"
    state_path = directory / "tuning_state.json"
    if not path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"Eseguire prima la selezione su validation: {path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status") != "completed":
        raise RuntimeError("La selezione dei parametri su validation non risulta completata.")
    cases = _load_case_specs(stage_d_cfg, "validation", max_cases=1 if smoke else None)
    expected_signature = _tuning_signature(stage_f_cfg, stage_d_cfg, cases, smoke=smoke)
    if state.get("signature") != expected_signature or saved.get("tuning_signature") != expected_signature:
        raise RuntimeError(
            "I parametri selezionati non corrispondono alla configurazione/codice/casi validation correnti. "
            "Rieseguire la fase tune prima di evaluate."
        )
    return saved["selected"]

def run_evaluation(stage_f_cfg: Mapping[str, Any], stage_d_cfg: Mapping[str, Any], *, smoke: bool = False, resume: bool = True) -> Path:
    _, build_engine, run_episode = _load_runtime_modules()
    from experiments.stage_d.benchmark_pde import _sha256 as stage_d_sha256
    protocol = stage_f_cfg["protocol"]
    checkpoint = (ROOT / stage_f_cfg["source"]["gdm_checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint GDM Stage D non trovato: {checkpoint}")
    engine = build_engine(checkpoint, stage_d_cfg, device="cpu")
    selected = _load_selected(stage_f_cfg, stage_d_cfg, smoke=smoke)
    domain = _domain(stage_d_cfg)
    splits = list(protocol["test_splits"])
    max_cases = 1 if smoke else None
    cases = [case for split in splits for case in _load_case_specs(stage_d_cfg, split, max_cases=max_cases)]
    if not smoke:
        expected_counts = {
            "test_interp": int(stage_d_cfg["dataset"]["splits"]["num_test_interp"]),
            "test_extrap": int(stage_d_cfg["dataset"]["splits"]["num_test_extrap"]),
        }
        observed_counts = {split: sum(c["split"] == split for c in cases) for split in splits}
        if observed_counts != expected_counts:
            raise RuntimeError(f"Numero casi held-out incoerente: osservato={observed_counts}, atteso={expected_counts}.")
    methods = ["fixed_fem", "mmpde_winslow", "mmpde_arclength", "mmpde_hessian", "gdm"]
    warmups = 1 if smoke else int(protocol["warmup_runs"])
    repetitions = 1 if smoke else int(protocol["repetitions"])
    if not smoke and repetitions < 5:
        raise ValueError("Il protocollo scientifico richiede almeno cinque ripetizioni temporali.")

    out_name = "smoke_evaluation" if smoke else "evaluation"
    out_dir = (ROOT / str(stage_f_cfg["experiment"]["output_dir"]) / out_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = _sha256(checkpoint)
    config_snapshot = {"stage_f": dict(stage_f_cfg), "stage_d_effective": dict(stage_d_cfg)}
    snapshot_path = out_dir / "config_snapshot.json"
    if snapshot_path.is_file():
        previous_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if previous_snapshot != json.loads(json.dumps(config_snapshot, default=_json_default)):
            raise RuntimeError("config_snapshot.json non coincide con la configurazione corrente.")
    else:
        _json_dump(snapshot_path, config_snapshot)
    config_fingerprint = hashlib.sha256(json.dumps(
        {"stage_f": stage_f_cfg, "stage_d": stage_d_cfg}, sort_keys=True, default=str
    ).encode("utf-8")).hexdigest()
    implementation_hashes = _implementation_hashes()
    run_signature = {
        "checkpoint_sha256": checkpoint_hash,
        "implementation_sha256": implementation_hashes,
        "configuration_sha256": config_fingerprint,
        "case_ids": [c["case_id"] for c in cases],
        "case_specs_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "methods": methods,
        "protocol": {"nx": int(protocol["nx"]), "ny": int(protocol["ny"]), "dt": float(protocol["dt"]),
                     "adapt_interval": int(protocol["adapt_interval"]), "warmups": warmups, "repetitions": repetitions},
        "selected_parameters": selected,
    }
    state_path = out_dir / "run_state.json"
    result_files = [out_dir / name for name in (
        "per_case_results.csv", "per_timestep_results.csv", "runtime_repetitions.csv", "adaptation_events.csv"
    )]
    if not state_path.is_file() and any(path.exists() for path in result_files):
        raise RuntimeError("Sono presenti risultati CSV senza run_state.json; non posso verificare la provenienza. "
                           "Archivia la directory oppure usa un nuovo output_dir.")
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("signature") != run_signature:
            raise RuntimeError("La directory Stage-F contiene un run_state con protocollo/checkpoint diversi. "
                               "Usa un nuovo output_dir oppure archivia i risultati parziali prima di cambiare configurazione.")
        if not resume:
            raise FileExistsError("Risultati Stage-F già presenti; la ripresa è abilitata per default. Scegli un nuovo output_dir per una nuova esecuzione.")
    else:
        _json_dump(state_path, {"status": "running", "signature": run_signature})

    case_rows = _read_csv(out_dir / "per_case_results.csv") if resume else []
    timestep_rows = _read_csv(out_dir / "per_timestep_results.csv") if resume else []
    event_rows = _read_csv(out_dir / "adaptation_events.csv") if resume else []
    runtime_rows = _read_csv(out_dir / "runtime_repetitions.csv") if resume else []
    completed_accuracy = {(str(r["case_id"]), str(r["split"]), str(r["strategy"])) for r in case_rows}
    completed_runtime = {
        (str(r["case_id"]), str(r["split"]), str(r["strategy"]), bool(r["warmup"]), int(r["repetition"]))
        for r in runtime_rows
    }

    for case_idx, case in enumerate(cases, start=1):
        LOG.info("Stage F: caso %d/%d (%s, %s)", case_idx, len(cases), case["split"], case["case_id"])
        for method in methods:
            strength = selected.get(method, {}).get("monitor_strength") if method.startswith("mmpde_") else None
            # One accuracy/diagnostic trajectory per case and method.
            if (str(case["case_id"]), str(case["split"]), method) not in completed_accuracy:
                result, controller = _episode(
                    run_episode, case, stage_d_cfg, method, domain, stage_f_cfg,
                    gdm_engine=engine, strength=strength, diagnostics=True,
                )
                summary = dict(result["case_summary"])
                # The common Stage-D backend calls this bucket gnn_inference_time;
                # for F1 it is the generic controller/strategy computation cost.
                summary["strategy_compute_time"] = float(summary.pop("gnn_inference_time", 0.0))
                summary["strategy"] = method
                summary["monitor_strength"] = float(strength) if strength is not None else None
                summary["evaluation_kind"] = "single_diagnostic_trajectory"
                case_rows.append(summary)
                for row in result["timestep_rows"]:
                    out = dict(row)
                    out["strategy_compute_time"] = float(out.pop("gnn_inference_time", 0.0))
                    out["strategy"] = method; out["monitor_strength"] = strength
                    timestep_rows.append(out)
                if controller is not None:
                    for event_idx, event in enumerate(controller.events):
                        event_rows.append({"case_id": case["case_id"], "split": case["split"], "strategy": method,
                                           "adaptation_event": event_idx, "monitor_strength": strength, **event})
                completed_accuracy.add((str(case["case_id"]), str(case["split"]), method))
                _write_csv(out_dir / "per_case_results.csv", case_rows)
                _write_csv(out_dir / "per_timestep_results.csv", timestep_rows)
                _write_csv(out_dir / "adaptation_events.csv", event_rows)

            # Dedicated wall-clock repetitions. Warm-up is stored but excluded from all summaries.
            for rep in range(warmups + repetitions):
                is_warmup = rep < warmups
                rep_id = rep - warmups if not is_warmup else rep
                runtime_key = (str(case["case_id"]), str(case["split"]), method, is_warmup, rep_id)
                if runtime_key in completed_runtime:
                    continue
                timed_result, _ = _episode(
                    run_episode, case, stage_d_cfg, method, domain, stage_f_cfg,
                    gdm_engine=engine, strength=strength, diagnostics=False,
                )
                timing = dict(timed_result["timings"])
                timing_row = {
                    "case_id": case["case_id"], "split": case["split"], "strategy": method,
                    "warmup": is_warmup, "repetition": rep_id,
                    "monitor_strength": strength,
                    "total_wall_time": float(timing["total_wall_time"]),
                    "strategy_compute_time": float(timing["gnn_inference_time"]),
                    "mesh_update_time": float(timing["mesh_update_time"]),
                    "safety_filter_time": float(timing["safety_filter_time"]),
                    "coord_update_time": float(timing["coord_update_time"]),
                    "ale_time": float(timing["ale_time"]),
                    "fem_solve_time": float(timing["fem_solve_time"]),
                    "other_overhead_time": float(timing["other_overhead_time"]),
                }
                runtime_rows.append(timing_row)
                completed_runtime.add(runtime_key)
                _write_csv(out_dir / "runtime_repetitions.csv", runtime_rows)

    _write_csv(out_dir / "per_case_results.csv", case_rows)
    _write_csv(out_dir / "per_timestep_results.csv", timestep_rows)
    _write_csv(out_dir / "runtime_repetitions.csv", runtime_rows)
    _write_csv(out_dir / "adaptation_events.csv", event_rows)

    expected_case_method = len(cases) * len(methods)
    expected_runtime = expected_case_method * (warmups + repetitions)
    expected_steps = int(round(float(cases[0]["T"]) / float(cases[0]["dt"]))) + 1
    expected_timestep_rows = expected_case_method * expected_steps
    expected_case_keys = {(str(c["case_id"]), str(c["split"]), method) for c in cases for method in methods}
    observed_case_keys = {(str(r["case_id"]), str(r["split"]), str(r["strategy"])) for r in case_rows}
    if len(case_rows) != expected_case_method or observed_case_keys != expected_case_keys:
        raise RuntimeError(
            f"Valutazioni accurate incomplete/duplicate: {len(case_rows)}/{expected_case_method}; "
            f"mancanti={len(expected_case_keys - observed_case_keys)}, extra={len(observed_case_keys - expected_case_keys)}."
        )
    expected_runtime_keys = {
        (str(c["case_id"]), str(c["split"]), method, rep < warmups, rep if rep < warmups else rep - warmups)
        for c in cases for method in methods for rep in range(warmups + repetitions)
    }
    if len(runtime_rows) != expected_runtime or completed_runtime != expected_runtime_keys:
        raise RuntimeError(
            f"Ripetizioni temporali incomplete/duplicate: {len(runtime_rows)}/{expected_runtime}; "
            f"mancanti={len(expected_runtime_keys - completed_runtime)}, extra={len(completed_runtime - expected_runtime_keys)}."
        )
    observed_timestep_keys = {
        (str(r["case_id"]), str(r["split"]), str(r["strategy"]), int(r["step"])) for r in timestep_rows
    }
    expected_timestep_keys = {
        (str(c["case_id"]), str(c["split"]), method, step)
        for c in cases for method in methods for step in range(expected_steps)
    }
    if len(timestep_rows) != expected_timestep_rows or observed_timestep_keys != expected_timestep_keys:
        raise RuntimeError(
            f"Diagnostiche temporali incomplete/duplicate: {len(timestep_rows)}/{expected_timestep_rows}; "
            f"mancanti={len(expected_timestep_keys - observed_timestep_keys)}, extra={len(observed_timestep_keys - expected_timestep_keys)}."
        )
    inadmissible = [
        (r["case_id"], r["strategy"], r["inadmissible_cells"], r["min_jacobian"])
        for r in case_rows if int(r["inadmissible_cells"]) > 0
    ]
    if inadmissible:
        raise RuntimeError(f"Rilevate celle non ammissibili nel benchmark; consultare per_case_results.csv: {inadmissible[:5]}")
    summary = _aggregate(case_rows, runtime_rows)
    comparisons = _paired_inference(case_rows, stage_f_cfg)
    _write_csv(out_dir / "paired_statistics.csv", comparisons)
    _write_latex_tables(out_dir, summary, comparisons)
    figures = _make_plots(out_dir, case_rows, runtime_rows, comparisons)

    checkpoint_hash = stage_d_sha256(checkpoint)
    metadata = {
        "stage": "F1", "protocol": "fixed-topology r-refinement; same nodes/DOFs/connectivity across strategies",
        "methods": methods, "case_ids": [c["case_id"] for c in cases],
        "split_counts": {s: sum(c["split"] == s for c in cases) for s in splits},
        "grid": {"nx": int(protocol["nx"]), "ny": int(protocol["ny"]),
                 "nodes": (int(protocol["nx"]) + 1) * (int(protocol["ny"]) + 1),
                 "triangles": 2 * int(protocol["nx"]) * int(protocol["ny"])},
        "dt": float(protocol["dt"]), "adapt_interval": int(protocol["adapt_interval"]),
        "timing": {"warmup_runs": warmups, "repetitions": repetitions, "cpu_threads": int(stage_f_cfg["experiment"]["cpu_threads"])},
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "implementation_sha256": implementation_hashes,
        "case_file_sha256": {f"{c['split']}:{c['case_id']}": c["case_file_sha256"] for c in cases},
        "platform": {"system": platform.platform(), "processor": platform.processor(), "python": sys.version,
                    "torch": torch.__version__, "numpy": np.__version__,
                    "torch_threads": torch.get_num_threads(),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                    "mkl_num_threads": os.environ.get("MKL_NUM_THREADS")},
        "selected_classical_parameters": selected,
        "aggregate": summary,
        "paired_statistics": comparisons,
        "figures_pdf": figures,
        "limitations": [
            "Classical methods use lagged elementwise monitors reconstructed from the current numerical P1 field.",
            "The variational flow is a discrete mass-lumped gradient flow of the Huang q=2 functional, not a claim of identity with every MMPDE discretization.",
            "The same deal.II safety filter and ALE solver are applied to every moving-mesh strategy.",
            "Oracle mesh targets are not accessed during validation tuning or test evaluation.",
            "Each classical mesh-flow event logs its termination reason, pseudo-iteration cap and line-search failures.",
        ],
    }
    _json_dump(out_dir / "benchmark_summary.json", metadata)
    _json_dump(state_path, {"status": "completed", "signature": run_signature,
                            "completed_accuracy_records": len(case_rows),
                            "completed_timing_records": len(runtime_rows)})
    (out_dir / "experiment.log").write_text("Stage F1 evaluation completed. See benchmark_summary.json and CSV outputs.\n", encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage F1 classical r-refinement vs GDM")
    parser.add_argument("mode", choices=("tune", "evaluate", "all"))
    parser.add_argument("--config", default="config/stage_f.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="one validation/test case per split and one timing repeat")
    parser.add_argument("--no-resume", action="store_true", help="fail when an evaluation output already exists")
    args = parser.parse_args()
    cfg_path = (ROOT / args.config).resolve()
    stage_f_cfg = _read_yaml(cfg_path)
    stage_d_cfg = _stage_d_config(stage_f_cfg)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cpu_threads = int(stage_f_cfg["experiment"]["cpu_threads"])
    torch.set_num_threads(cpu_threads)
    try:
        from threadpoolctl import threadpool_limits
        _threadpool_limiter = threadpool_limits(limits=cpu_threads)
    except ImportError:
        LOG.warning("threadpoolctl non installato: le librerie BLAS potrebbero non rispettare cpu_threads.")
    np.random.seed(int(stage_f_cfg["experiment"]["seed"]))
    torch.manual_seed(int(stage_f_cfg["experiment"]["seed"]))
    if args.mode in {"tune", "all"}:
        run_tuning(stage_f_cfg, stage_d_cfg, smoke=args.smoke_test)
    if args.mode in {"evaluate", "all"}:
        run_evaluation(stage_f_cfg, stage_d_cfg, smoke=args.smoke_test, resume=not args.no_resume)


if __name__ == "__main__":
    main()
