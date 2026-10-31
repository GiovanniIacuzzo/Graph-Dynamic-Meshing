"""Audit SUPG tau balance and parameter sensitivity on the fixed 32x32 case."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math

import numpy as np

from .closeout_common import (
    ensure_closeout_tree, native_fixed_mesh_metrics,
    tau_decomposition, append_or_replace_csv_row,
)
from .common import configure_cpu_threads, load_config, run_fixed_case


def _stats(values: np.ndarray) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"min": float(np.min(a)), "median": float(np.median(a)), "mean": float(np.mean(a)),
            "p95": float(np.quantile(a, .95)), "max": float(np.max(a))}


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    configure_cpu_threads(int(cfg["experiment"].get("cpu_threads", 1)))
    root = ensure_closeout_tree(cfg) / "supg"
    root.mkdir(parents=True, exist_ok=True)
    spec = cfg["closeout"].get("supg_tau_audit", {})
    nx = int(spec.get("resolution", cfg["supg_comparison"]["resolution"]))
    dt = float(spec.get("dt", cfg["supg_comparison"]["dt"]))
    scales = [float(x) for x in spec.get("tau_scales", [0.5, 1.0, 2.0, 4.0])]
    if any(not math.isfinite(s) or s <= 0.0 for s in scales):
        raise ValueError("All tau scales must be finite and positive.")

    mesh, _, beta_cells, h_beta, pe = native_fixed_mesh_metrics(cfg, nx)
    p = cfg["pde"]
    speed = np.linalg.norm(beta_cells, axis=1)
    base_stab = cfg["stabilization"]
    base = tau_decomposition(
        speed, h_beta, dt, float(p["nu"]), float(base_stab["c_time"]),
        float(base_stab["c_advection"]), float(base_stab["c_diffusion"]),
    )
    rows: list[dict] = []
    galerkin = run_fixed_case(cfg, nx, dt, "galerkin", retain_snapshots=True)
    baseline = {
        "mode": "galerkin", "tau_scale": 0.0,
        "time_average_absolute_l2_error": galerkin["time_average_absolute_l2_error"],
        "time_average_absolute_h1_seminorm_error": galerkin["time_average_absolute_h1_seminorm_error"],
        "overshoot": galerkin["overshoot"], "undershoot": galerkin["undershoot"],
        "wall_seconds": galerkin["total_wall_seconds"],
    }
    rows.append(baseline)

    for scale in scales:
        trial = copy.deepcopy(cfg)
        # tau is homogeneous of degree -1 in (c_time,c_advection,c_diffusion).
        # Dividing all three coefficients by scale multiplies tau by scale.
        trial["stabilization"]["mode"] = "supg"
        for key in ("c_time", "c_advection", "c_diffusion"):
            trial["stabilization"][key] = float(base_stab[key]) / scale
        run_result = run_fixed_case(trial, nx, dt, "supg", retain_snapshots=True)
        tau = base["tau"] * scale
        row = {
            "mode": "supg", "tau_scale": scale,
            "c_time_effective": trial["stabilization"]["c_time"],
            "c_advection_effective": trial["stabilization"]["c_advection"],
            "c_diffusion_effective": trial["stabilization"]["c_diffusion"],
            "tau_min": float(np.min(tau)), "tau_median": float(np.median(tau)),
            "tau_p95": float(np.quantile(tau, .95)), "tau_max": float(np.max(tau)),
            "Pe_median": float(np.median(pe)), "Pe_p95": float(np.quantile(pe, .95)),
            "temporal_frequency_fraction_median": float(np.median(base["temporal_fraction"])),
            "advective_frequency_fraction_median": float(np.median(base["advective_fraction"])),
            "diffusive_frequency_fraction_median": float(np.median(base["diffusive_fraction"])),
            "time_average_absolute_l2_error": run_result["time_average_absolute_l2_error"],
            "time_average_absolute_h1_seminorm_error": run_result["time_average_absolute_h1_seminorm_error"],
            "overshoot": run_result["overshoot"], "undershoot": run_result["undershoot"],
            "wall_seconds": run_result["total_wall_seconds"],
            "relative_l2_change_vs_galerkin": (
                run_result["time_average_absolute_l2_error"] / max(galerkin["time_average_absolute_l2_error"], 1e-300) - 1.0
            ),
        }
        rows.append(row)

    csv_path = root / "supg_tau_parameter_audit.csv"
    for row in rows:
        append_or_replace_csv_row(csv_path, {"nx": nx, "dt": dt, **row}, ("nx", "dt", "mode", "tau_scale"))
    summary = {
        "resolution": nx, "dt": dt,
        "tau_definition": "[ (c_time/dt)^2 + (c_advection*|b|/h_beta)^2 + (c_diffusion*nu/h_beta^2)^2 ]^(-1/2)",
        "tau_scale_method": "divide all three c coefficients by s, which multiplies tau by s without changing their relative component balance",
        "base_tau_statistics": _stats(base["tau"]),
        "peclet_statistics": _stats(pe),
        "base_tau_frequency_fraction_median": {
            "temporal": float(np.median(base["temporal_fraction"])),
            "advection": float(np.median(base["advective_fraction"])),
            "diffusion": float(np.median(base["diffusive_fraction"])),
        },
        "results": rows,
        "interpretation": "Sensitivity audit only. It does not select an optimal tau or justify tuning on the held-out benchmark.",
    }
    (root / "supg_tau_parameter_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    run(parser.parse_args().config)
