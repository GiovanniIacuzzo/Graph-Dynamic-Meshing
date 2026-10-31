"""E1-A temporal convergence of Backward Euler on a fixed P1 mesh."""
from __future__ import annotations

import argparse
import gc

import numpy as np

from .common import (
    configure_cpu_threads, configure_logging, format_float, load_config, l2_p1_difference,
    merge_csv, number_of_steps, observed_order, prepare_output_tree, run_fixed_case, csv_contains_keys,
    write_latex_table,
)


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    configure_cpu_threads(int(cfg["experiment"].get("cpu_threads", 1)))
    logger = configure_logging(cfg)
    out = prepare_output_tree(cfg)
    spec = cfg["temporal_convergence"]
    nx = int(spec["resolution"])
    base_dts = [float(v) for v in spec["dts"]]
    reference_dt = float(spec["reference_dt"])
    temporal_csv = out / "solver_verification" / "temporal_convergence.csv"
    if csv_contains_keys(temporal_csv, ["nx", "dt"], [[nx, dt] for dt in base_dts]):
        logger.info("E1 temporal: risultati già completi; riuso dei CSV (resume).")
        import json
        saved = out / "solver_verification" / "temporal_observed_orders.json"
        if saved.exists():
            return json.loads(saved.read_text(encoding="utf-8"))
    all_dts = sorted(set(base_dts + [v / 2.0 for v in base_dts] + [reference_dt]), reverse=True)
    results: dict[float, dict] = {}
    rows_by_dt: dict[float, dict] = {}
    for dt in all_dts:
        number_of_steps(float(cfg["pde"]["T"]), dt)
        logger.info("E1 temporal: nx=%d dt=%g", nx, dt)
        result = run_fixed_case(cfg, nx, dt, "galerkin", retain_snapshots=True)
        results[dt] = result
        rows_by_dt[dt] = {
            "nx": nx, "dt": dt, "n_steps": number_of_steps(float(cfg["pde"]["T"]), dt),
            "n_nodes": result["n_nodes"], "n_elements": result["n_elements"], "n_dofs": result["n_dofs"],
            "relative_l2_error_final": result["relative_l2_error_final"],
            "relative_h1_seminorm_error_final": result["relative_h1_seminorm_error_final"],
            "absolute_l2_error_final": result["absolute_l2_error_final"],
            "absolute_h1_seminorm_error_final": result["absolute_h1_seminorm_error_final"],
            "time_average_relative_l2_error": result["time_average_relative_l2_error"],
            "time_average_relative_h1_seminorm_error": result["time_average_relative_h1_seminorm_error"],
            "time_average_absolute_l2_error": result["time_average_absolute_l2_error"],
            "time_average_absolute_h1_seminorm_error": result["time_average_absolute_h1_seminorm_error"],
            "assembly_seconds": result["assembly_seconds"], "solve_seconds": result["solve_seconds"],
            "total_wall_seconds": result["total_wall_seconds"],
        }

    rows = []
    for dt in base_dts:
        coarse, fine = results[dt], results[dt / 2.0]
        cauchy = l2_p1_difference(fine["mesh"], coarse["final_solution"], fine["final_solution"])
        row = dict(rows_by_dt[dt])
        row["cauchy_l2_difference_dt_dt_half"] = cauchy
        row["comparison_dt"] = dt / 2.0
        row["reference_l2_difference"] = l2_p1_difference(
            fine["mesh"], coarse["final_solution"], results[reference_dt]["final_solution"])
        # BE's formal order is first order; this is inferred from consecutive Cauchy differences.
        rows.append(row)
    rows.sort(key=lambda r: r["dt"], reverse=True)
    for i, row in enumerate(rows):
        if i == 0:
            row["observed_order_cauchy"] = None
        else:
            prev = rows[i-1]
            row["observed_order_cauchy"] = observed_order(
                prev["cauchy_l2_difference_dt_dt_half"], row["cauchy_l2_difference_dt_dt_half"],
                prev["dt"], row["dt"])
    target = out / "solver_verification"
    merge_csv(target / "temporal_convergence.csv", rows, ["nx", "dt"])
    summary = {
        "time_integrator_identified_from_source": "Backward Euler (first-order implicit Euler)",
        "expected_asymptotic_order": 1,
        "cauchy_metric": "L2 norm on the same fixed P1 mesh of u_h(dt)-u_h(dt/2) at final time",
        "reference_dt": reference_dt,
        "reference_solution_error": rows_by_dt[reference_dt]["absolute_l2_error_final"] if reference_dt in rows_by_dt else None,
        "reference_difference_definition": "L2 norm at T between each coarse-dt solution and the configured finer reference solution",
        "observed_orders": [{"dt": r["dt"], "cauchy_l2": r["cauchy_l2_difference_dt_dt_half"], "q_obs": r["observed_order_cauchy"]} for r in rows],
    }
    from .common import atomic_write_json
    atomic_write_json(target / "temporal_observed_orders.json", summary, merge=True)
    write_latex_table(
        out / "tables" / "table_temporal_convergence.tex",
        "Temporal convergence on a fixed mesh. Cauchy differences compare consecutive Backward Euler time steps.",
        "tab:gdm_stage_e_temporal",
        [r"$\Delta t$", "Steps", r"$E_{L^2}(T)$", r"$E_{H^1}(T)$", r"$\|u_{\Delta t}-u_{\Delta t/2}\|_{L^2}$", r"$q_{\rm obs}$"],
        [[format_float(r["dt"]), r["n_steps"], format_float(r["absolute_l2_error_final"]),
          format_float(r["absolute_h1_seminorm_error_final"]), format_float(r["cauchy_l2_difference_dt_dt_half"]),
          format_float(r["observed_order_cauchy"])] for r in rows],
    )
    from python.plotting.stage_e import plot_temporal_convergence
    plot_temporal_convergence(rows, out / "figures" / "temporal_convergence.pdf")
    # Release several large time histories after all Cauchy differences are computed.
    results.clear()
    gc.collect()
    logger.info("E1 temporal completata; confrontare q_obs con l'ordine 1 atteso da Backward Euler.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(args.config)
