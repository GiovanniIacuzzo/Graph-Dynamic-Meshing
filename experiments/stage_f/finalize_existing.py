"""Post-hoc finalization of a completed Stage-F1 run interrupted by a
strict-vs-tolerant Jacobian-threshold check.

This script performs no FEM/ALE solve and never alters the raw CSV files.
It validates their completeness, reclassifies only roundoff-level threshold
flags using the same tolerance as the native safety filter, and produces the
normal Stage-F1 summaries/statistics/tables/figures.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import json
import math
import os
import platform
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

# Import Stage-F pure-Python post-processing helpers without executing
# python/gdm/__init__.py (the latter imports the compiled native extension).
ROOT = Path(__file__).resolve().parents[2]
if "gdm" not in sys.modules:
    _gdm_pkg = types.ModuleType("gdm")
    _gdm_pkg.__path__ = [str(ROOT / "python" / "gdm")]
    sys.modules["gdm"] = _gdm_pkg

from experiments.stage_f.run import (
    _aggregate,
    _holm_adjust,
    _json_dump,
    _make_plots,
    _paired_inference,
    _read_csv,
    _read_yaml,
    _sha256,
    _stage_d_config,
    _write_csv,
    _write_latex_tables,
)

METHODS = ["fixed_fem", "mmpde_winslow", "mmpde_arclength", "mmpde_hessian", "gdm"]
TIMED_METHODS = METHODS


def _assert_unique(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], label: str) -> None:
    seen = set()
    duplicates = []
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise RuntimeError(f"{label}: chiavi duplicate, esempi={duplicates[:5]}")


def _roundoff_classification(
    timestep_rows: Sequence[Mapping[str, Any]],
    *,
    epsilon: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Preserve strict flags, but separate native-tolerated roundoff flags."""
    roundoff_tol = max(1.0e-12, 1.0e-9 * abs(float(epsilon)))
    corrected_rows: List[Dict[str, Any]] = []
    per_case: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    audit_rows: List[Dict[str, Any]] = []

    strict_flags_total = 0
    roundoff_flags_total = 0
    material_flags_total = 0
    inverted_total = 0
    degenerate_total = 0
    affected_time_rows = 0
    materially_bad_rows = 0
    min_seen = math.inf

    for raw in timestep_rows:
        row = dict(raw)
        case_id, split, strategy = str(row["case_id"]), str(row["split"]), str(row["strategy"])
        key = (case_id, split, strategy)
        strict_count = int(row.get("inadmissible_cells", 0) or 0)
        inverted = int(row.get("inverted_cells", 0) or 0)
        degenerate = int(row.get("degenerate_cells", 0) or 0)
        min_j = float(row["min_jacobian"])
        min_seen = min(min_seen, min_j)
        strict_flags_total += strict_count
        inverted_total += inverted
        degenerate_total += degenerate

        roundoff_only = (
            strict_count > 0
            and min_j >= float(epsilon) - roundoff_tol
            and inverted == 0
            and degenerate == 0
        )
        tolerated_count = strict_count if roundoff_only else 0
        material_count = 0 if roundoff_only else strict_count
        if roundoff_only:
            roundoff_flags_total += tolerated_count
            affected_time_rows += 1
        elif strict_count > 0 or inverted > 0 or degenerate > 0:
            materially_bad_rows += 1
            material_flags_total += material_count

        # Also catch a material threshold violation if the raw count field is
        # inconsistent with min(J), which would indicate a separate diagnostic bug.
        if min_j < float(epsilon) - roundoff_tol and strict_count == 0:
            materially_bad_rows += 1
            material_flags_total += 1
            material_count = 1

        corrected = dict(row)
        corrected["inadmissible_cells"] = material_count
        corrected_rows.append(corrected)

        acc = per_case.setdefault(key, {
            "case_id": case_id, "split": split, "strategy": strategy,
            "strict_below_epsilon_cell_time_flags": 0,
            "roundoff_tolerated_cell_time_flags": 0,
            "materially_inadmissible_cell_time_flags": 0,
            "inverted_cell_time_flags": 0,
            "degenerate_cell_time_flags": 0,
            "min_jacobian": math.inf,
            "roundoff_tolerated_time_rows": 0,
            "materially_inadmissible_time_rows": 0,
        })
        acc["strict_below_epsilon_cell_time_flags"] += strict_count
        acc["roundoff_tolerated_cell_time_flags"] += tolerated_count
        acc["materially_inadmissible_cell_time_flags"] += material_count
        acc["inverted_cell_time_flags"] += inverted
        acc["degenerate_cell_time_flags"] += degenerate
        acc["min_jacobian"] = min(float(acc["min_jacobian"]), min_j)
        acc["roundoff_tolerated_time_rows"] += int(roundoff_only)
        acc["materially_inadmissible_time_rows"] += int(
            strict_count > 0 and not roundoff_only or inverted > 0 or degenerate > 0
        )

    for record in per_case.values():
        record["min_jacobian"] = float(record["min_jacobian"])
        audit_rows.append(record)

    summary = {
        "epsilon": float(epsilon),
        "native_roundoff_tolerance": float(roundoff_tol),
        "strict_below_epsilon_cell_time_flags": int(strict_flags_total),
        "roundoff_tolerated_cell_time_flags": int(roundoff_flags_total),
        "materially_inadmissible_cell_time_flags": int(material_flags_total),
        "inverted_cell_time_flags": int(inverted_total),
        "degenerate_cell_time_flags": int(degenerate_total),
        "roundoff_tolerated_time_rows": int(affected_time_rows),
        "materially_inadmissible_time_rows": int(materially_bad_rows),
        "global_min_jacobian": float(min_seen),
        "classification": (
            "A strict J<epsilon flag is reclassified as roundoff-tolerated only when "
            "J_min >= epsilon-roundoff_tolerance and there are no inverted or degenerate cells. "
            "Raw CSV files remain unchanged."
        ),
    }
    return corrected_rows, summary, sorted(audit_rows, key=lambda r: (r["split"], r["case_id"], r["strategy"]))


def _safety_alpha_audit(
    timestep_rows: Sequence[Mapping[str, Any]], *, adapt_interval: int,
) -> Dict[str, Any]:
    # The Stage-D backend applies a new proposal at steps 1, 1+interval, ... .
    events = [
        r for r in timestep_rows
        if str(r["strategy"]) == "gdm" and (int(r["step"]) - 1) % int(adapt_interval) == 0
    ]
    result: Dict[str, Any] = {"n_events": len(events), "by_split": {}}
    for split in sorted({str(r["split"]) for r in events}):
        subset = [r for r in events if str(r["split"]) == split]
        alpha = np.asarray([float(r["safety_alpha"]) for r in subset], dtype=float)
        per_case: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in subset:
            per_case[str(r["case_id"])].append({
                "step": int(r["step"]), "alpha": float(r["safety_alpha"]),
                "min_jacobian": float(r["min_jacobian"]),
            })
        result["by_split"][split] = {
            "n_events": int(len(subset)),
            "alpha_min": float(np.min(alpha)),
            "alpha_median": float(np.median(alpha)),
            "alpha_mean": float(np.mean(alpha)),
            "alpha_q10": float(np.quantile(alpha, 0.10)),
            "alpha_q25": float(np.quantile(alpha, 0.25)),
            "fraction_intervened": float(np.mean(alpha < 1.0 - 1.0e-12)),
            "n_alpha_le_1e-6": int(np.sum(alpha <= 1.0e-6)),
            "n_alpha_le_1e-10": int(np.sum(alpha <= 1.0e-10)),
            "n_alpha_eq_zero": int(np.sum(alpha == 0.0)),
            "cases_with_alpha_le_1e-8": {
                case_id: values for case_id, values in per_case.items()
                if any(v["alpha"] <= 1.0e-8 for v in values)
            },
        }
    return result


def _mmpde_event_audit(event_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for method in sorted({str(r["strategy"]) for r in event_rows}):
        subset = [r for r in event_rows if str(r["strategy"]) == method]
        term = collections.Counter(str(r.get("termination_reason", "unknown")) for r in subset)
        result[method] = {
            "n_events": len(subset),
            "termination_reason_counts": dict(term),
            "pseudo_iteration_caps": int(sum(str(r.get("pseudo_iterations_exhausted", "False")).lower() == "true" for r in subset)),
            "line_search_failures": int(sum(int(r.get("line_search_failures", 0) or 0) for r in subset)),
            "all_event_energies_monotone": bool(all(str(r.get("energy_monotone", "False")).lower() == "true" for r in subset)),
            "mean_accepted_pseudo_steps": float(np.mean([float(r["accepted_pseudo_steps"]) for r in subset])) if subset else 0.0,
            "max_accepted_pseudo_steps": int(max(float(r["accepted_pseudo_steps"]) for r in subset)) if subset else 0,
        }
    return result


def _verify_completeness(
    case_rows: Sequence[Mapping[str, Any]], timestep_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]], event_rows: Sequence[Mapping[str, Any]],
    signature: Mapping[str, Any],
) -> Dict[str, Any]:
    case_keys = [(str(r["case_id"]), str(r["split"])) for r in case_rows]
    case_pairs = sorted(set(case_keys))
    if len(case_pairs) != len(signature.get("case_ids", [])):
        raise RuntimeError("Numero di casi per-case non coerente con run_state.signature.")
    _assert_unique(case_rows, ("case_id", "split", "strategy"), "per_case_results")
    _assert_unique(timestep_rows, ("case_id", "split", "strategy", "step"), "per_timestep_results")
    _assert_unique(runtime_rows, ("case_id", "split", "strategy", "warmup", "repetition"), "runtime_repetitions")

    methods = list(signature.get("methods", []))
    if methods != METHODS:
        raise RuntimeError(f"Metodi nel run_state non attesi: {methods}")
    protocol = signature.get("protocol", {})
    nx, ny = int(protocol["nx"]), int(protocol["ny"])
    dt = float(protocol["dt"])
    # T is recorded in the per-case table and should be identical across cases.
    T_values = {float(r["T"]) for r in case_rows}
    if len(T_values) != 1:
        raise RuntimeError(f"Intervalli temporali non uniformi nel benchmark: {T_values}")
    T = next(iter(T_values))
    n_steps = int(round(T / dt))
    n_time_rows_per_trajectory = n_steps + 1
    warmups, repetitions = int(protocol["warmups"]), int(protocol["repetitions"])
    expected_case = len(case_pairs) * len(methods)
    expected_timestep = expected_case * n_time_rows_per_trajectory
    expected_runtime = expected_case * (warmups + repetitions)
    expected_event_count = len(case_pairs) * 3 * math.ceil(n_steps / int(protocol["adapt_interval"]))

    counts = {
        "n_cases": len(case_pairs), "methods": methods,
        "per_case_rows": len(case_rows), "expected_per_case_rows": expected_case,
        "per_timestep_rows": len(timestep_rows), "expected_per_timestep_rows": expected_timestep,
        "runtime_rows": len(runtime_rows), "expected_runtime_rows": expected_runtime,
        "adaptation_event_rows": len(event_rows), "expected_classical_event_rows": expected_event_count,
        "n_steps": n_steps, "time_rows_per_trajectory": n_time_rows_per_trajectory,
        "warmups": warmups, "timed_repetitions": repetitions,
        "grid": {"nx": nx, "ny": ny, "nodes": (nx + 1) * (ny + 1), "triangles": 2 * nx * ny},
    }
    if len(case_rows) != expected_case or len(timestep_rows) != expected_timestep or len(runtime_rows) != expected_runtime:
        raise RuntimeError(f"Output incompleti: {counts}")
    observed_case_keys = {(str(r["case_id"]), str(r["split"]), str(r["strategy"])) for r in case_rows}
    expected_case_keys = {(c, s, m) for c, s in case_pairs for m in methods}
    if observed_case_keys != expected_case_keys:
        raise RuntimeError("Chiavi per-case mancanti o extra.")
    expected_timestep_keys = {
        (c, s, m, step) for c, s in case_pairs for m in methods for step in range(n_time_rows_per_trajectory)
    }
    observed_timestep_keys = {
        (str(r["case_id"]), str(r["split"]), str(r["strategy"]), int(r["step"])) for r in timestep_rows
    }
    if observed_timestep_keys != expected_timestep_keys:
        raise RuntimeError("Chiavi per-timestep mancanti o extra.")
    expected_runtime_keys = {
        (c, s, m, rep < warmups, rep if rep < warmups else rep - warmups)
        for c, s in case_pairs for m in methods for rep in range(warmups + repetitions)
    }
    observed_runtime_keys = {
        (str(r["case_id"]), str(r["split"]), str(r["strategy"]),
         str(r["warmup"]).lower() == "true", int(r["repetition"])) for r in runtime_rows
    }
    if observed_runtime_keys != expected_runtime_keys:
        raise RuntimeError("Ripetizioni temporali mancanti o extra.")
    return counts


def finalize(config_path: str = "config/stage_f.yaml", output_dir: str | None = None) -> Path:
    cfg_path = (ROOT / config_path).resolve()
    stage_f_cfg = _read_yaml(cfg_path)
    output = Path(output_dir) if output_dir else Path(stage_f_cfg["experiment"]["output_dir"]) / "evaluation"
    out_dir = (ROOT / output).resolve()
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Directory evaluation non trovata: {out_dir}")

    state_path = out_dir / "run_state.json"
    if not state_path.is_file():
        raise FileNotFoundError("run_state.json mancante: finalizzazione non sicura.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    signature = state.get("signature", {})
    if state.get("status") not in {"running", "completed", "completed_posthoc"}:
        raise RuntimeError(f"Stato non finalizzabile: {state.get('status')}")

    # Check that the F1 runner and Stage-D backend used for post-processing are
    # exactly those recorded by the interrupted campaign.
    expected_hashes = signature.get("implementation_sha256", {})
    for key, rel in {
        "stage_f_runner": "experiments/stage_f/run.py",
        "stage_d_benchmark_backend": "experiments/stage_d/benchmark_pde.py",
    }.items():
        current = _sha256(ROOT / rel)
        if expected_hashes.get(key) != current:
            raise RuntimeError(f"Codice {key} diverso da quello firmato dal run interrotto.")

    case_rows = _read_csv(out_dir / "per_case_results.csv")
    timestep_rows = _read_csv(out_dir / "per_timestep_results.csv")
    runtime_rows = _read_csv(out_dir / "runtime_repetitions.csv")
    event_rows = _read_csv(out_dir / "adaptation_events.csv")
    completeness = _verify_completeness(case_rows, timestep_rows, runtime_rows, event_rows, signature)

    epsilon = float(stage_f_cfg["safety"]["epsilon"])
    corrected_timestep, geometry_audit, geometry_by_case = _roundoff_classification(timestep_rows, epsilon=epsilon)
    if geometry_audit["materially_inadmissible_cell_time_flags"] > 0:
        raise RuntimeError(
            "Sono presenti violazioni geometriche oltre la tolleranza nativa: "
            f"{geometry_audit['materially_inadmissible_cell_time_flags']} flags. "
            "Nessun riepilogo finale è stato prodotto."
        )

    # Reconstruct tolerance-aware case summaries in memory. Original CSVs remain
    # untouched and retain the strict J<epsilon diagnostic count.
    corrected_count_by_key: Dict[Tuple[str, str, str], int] = collections.Counter()
    for row in corrected_timestep:
        key = (str(row["case_id"]), str(row["split"]), str(row["strategy"]))
        corrected_count_by_key[key] += int(row["inadmissible_cells"])
    analysis_case_rows = []
    for raw in case_rows:
        row = dict(raw)
        key = (str(row["case_id"]), str(row["split"]), str(row["strategy"]))
        row["inadmissible_cells"] = int(corrected_count_by_key[key])
        analysis_case_rows.append(row)

    # Standard Stage-F1 outputs, generated from the completed raw trajectories.
    summary = _aggregate(analysis_case_rows, runtime_rows)
    comparisons = _paired_inference(analysis_case_rows, stage_f_cfg)
    _write_csv(out_dir / "paired_statistics.csv", comparisons)
    _write_latex_tables(out_dir, summary, comparisons)
    figures = _make_plots(out_dir, analysis_case_rows, runtime_rows, comparisons)
    _write_csv(out_dir / "geometry_tolerance_audit.csv", geometry_by_case)

    adapt_interval = int(signature["protocol"]["adapt_interval"])
    alpha_audit = _safety_alpha_audit(timestep_rows, adapt_interval=adapt_interval)
    mmpde_audit = _mmpde_event_audit(event_rows)
    # Threadpoolctl was explicitly missing in the launch log supplied by the user.
    timing_caveat = (
        "The launch log reports threadpoolctl was not installed, so cpu_threads=1 "
        "was not guaranteed for BLAS. Runtime numbers are implementation- and "
        "threading-environment-specific and should be treated as provisional."
    )

    case_ids = [str(x) for x in signature.get("case_ids", [])]
    split_counts = collections.Counter(str(r["split"]) for r in case_rows)
    metadata = {
        "stage": "F1",
        "status": "completed_posthoc_roundoff_consistent",
        "protocol": "fixed-topology r-refinement; same nodes/DOFs/connectivity across strategies",
        "methods": METHODS,
        "case_ids": case_ids,
        "split_counts": dict(split_counts),
        "grid": completeness["grid"],
        "dt": float(signature["protocol"]["dt"]),
        "adapt_interval": adapt_interval,
        "timing": {
            "warmup_runs": int(signature["protocol"]["warmups"]),
            "repetitions": int(signature["protocol"]["repetitions"]),
            "cpu_threads_requested": int(stage_f_cfg["experiment"]["cpu_threads"]),
            "threadpoolctl_warning_observed": True,
            "caveat": timing_caveat,
        },
        "checkpoint_sha256": signature.get("checkpoint_sha256"),
        "implementation_sha256": expected_hashes,
        "configuration_sha256": signature.get("configuration_sha256"),
        "selected_classical_parameters": signature.get("selected_parameters", {}),
        "completeness": completeness,
        "aggregate": summary,
        "paired_statistics": comparisons,
        "geometry_tolerance_audit": geometry_audit,
        "gdm_safety_filter_audit": alpha_audit,
        "classical_mmpde_event_audit": mmpde_audit,
        "figures_pdf": figures,
        "limitations": [
            "The raw per_timestep_results.csv and per_case_results.csv are not modified.",
            "The strict raw count J<epsilon is separated from roundoff-level values within the native tolerance.",
            "A nonzero but tiny safety alpha indicates global displacement suppression and is reported, not reclassified as a geometry failure.",
            "MMPDE relative_energy_tolerance is a one-step relative energy-drop stopping rule, not proof of a stationary point.",
            "The logged final_gradient_l2 in the MMPDE event implementation is evaluated before the final accepted pseudo-step; it must not be interpreted as the gradient norm at the returned mesh.",
            timing_caveat,
        ],
    }
    _json_dump(out_dir / "benchmark_summary.json", metadata)
    _json_dump(out_dir / "posthoc_finalization_audit.json", {
        "status": metadata["status"],
        "run_state_before": state.get("status"),
        "raw_csv_sha256": {
            name: _sha256(out_dir / name) for name in (
                "per_case_results.csv", "per_timestep_results.csv",
                "runtime_repetitions.csv", "adaptation_events.csv",
            )
        },
        "completeness": completeness,
        "geometry_tolerance_audit": geometry_audit,
        "gdm_safety_filter_audit": alpha_audit,
        "classical_mmpde_event_audit": mmpde_audit,
        "paired_comparison_count": len(comparisons),
        "paired_comparisons": comparisons,
        "note": "Post-processing only: no trajectory, mesh, PDE, or runtime experiment was rerun.",
    })

    state["status"] = "completed_posthoc"
    state["posthoc_finalization"] = {
        "status": "passed_roundoff_consistent_geometry_gate",
        "materially_inadmissible_cell_time_flags": geometry_audit["materially_inadmissible_cell_time_flags"],
        "roundoff_tolerated_cell_time_flags": geometry_audit["roundoff_tolerated_cell_time_flags"],
        "raw_csvs_preserved": True,
    }
    _json_dump(state_path, state)
    (out_dir / "experiment.log").write_text(
        "Stage F1 outputs post-hoc finalized. No PDE/ALE trajectories were rerun. "
        "See posthoc_finalization_audit.json for the tolerance-aware geometry audit.\n",
        encoding="utf-8",
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc validation/finalization of an interrupted Stage-F1 output")
    parser.add_argument("--config", default="config/stage_f.yaml")
    parser.add_argument("--output-dir", default=None, help="relative to repository root; default results/stage_f1/evaluation")
    args = parser.parse_args()
    out = finalize(args.config, args.output_dir)
    print(f"Stage F1 finalizzata senza rieseguire gli esperimenti: {out}")
    print(f"Audit: {out / 'posthoc_finalization_audit.json'}")


if __name__ == "__main__":
    main()
