"""E1-A spatial convergence with an explicit temporal-contamination audit."""
from __future__ import annotations

import argparse
import gc
import json

import numpy as np

from .common import (
    atomic_write_json, configure_cpu_threads, configure_logging, format_float,
    load_config, l2_p1_difference, maximum_element_diameter, merge_csv, observed_order, prepare_output_tree,
    run_fixed_case, write_latex_table, csv_contains_keys, trapezoid_average,
)


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    configure_cpu_threads(int(cfg["experiment"].get("cpu_threads", 1)))
    logger = configure_logging(cfg)
    out = prepare_output_tree(cfg)
    spec = cfg["spatial_convergence"]
    dt = float(spec["dt"])
    threshold = float(spec["temporal_contamination_limit"])
    spatial_csv = out / "solver_verification" / "spatial_convergence.csv"
    if csv_contains_keys(spatial_csv, ["nx", "dt_used"], [[int(n), dt / 4.0] for n in spec["resolutions"]]):
        logger.info("E1 spatial: risultati già completi; riuso dei CSV (resume).")
        import json
        saved = out / "solver_verification" / "observed_orders.json"
        if saved.exists():
            return json.loads(saved.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for nx in map(int, spec["resolutions"]):
        logger.info("E1-A spatial: nx=%d, dt/2=%g e dt/4=%g", nx, dt / 2.0, dt / 4.0)
        medium = run_fixed_case(cfg, nx, dt / 2.0, "galerkin", retain_snapshots=True)
        fine = run_fixed_case(cfg, nx, dt / 4.0, "galerkin", retain_snapshots=True)
        times_fine = np.asarray(fine["history"]["time"], dtype=float)
        medium_hist = medium["solution_history"]
        fine_hist = fine["solution_history"][::2]
        times = times_fine[::2]
        if len(medium_hist) != len(fine_hist) or len(times) != len(fine_hist):
            raise RuntimeError("Le storie dt/2 e dt/4 non sono allineate: impossibile stimare la contaminazione temporale.")
        diff_l2 = [l2_p1_difference(fine["mesh"], a, b) for a, b in zip(medium_hist, fine_hist)]
        integrated_diff = trapezoid_average(times, diff_l2, float(cfg["pde"]["T"]))
        denominator = float(fine["time_average_absolute_l2_error"])
        ratio = integrated_diff / max(denominator, np.finfo(float).tiny)
        row = {
            "nx": nx, "ny": nx, "h": maximum_element_diameter(fine["mesh"]),
            "n_nodes": fine["n_nodes"], "n_elements": fine["n_elements"], "n_dofs": fine["n_dofs"],
            "dt_temporal_comparison_coarse": dt / 2.0, "dt_used": dt / 4.0,
            "time_average_relative_l2_error": fine["time_average_relative_l2_error"],
            "time_average_relative_h1_seminorm_error": fine["time_average_relative_h1_seminorm_error"],
            "time_average_absolute_l2_error": fine["time_average_absolute_l2_error"],
            "time_average_absolute_h1_seminorm_error": fine["time_average_absolute_h1_seminorm_error"],
            "relative_l2_error_final": fine["relative_l2_error_final"],
            "relative_h1_seminorm_error_final": fine["relative_h1_seminorm_error_final"],
            "absolute_l2_error_final": fine["absolute_l2_error_final"],
            "absolute_h1_seminorm_error_final": fine["absolute_h1_seminorm_error_final"],
            "temporal_difference_time_average_l2": integrated_diff,
            "temporal_contamination_ratio": ratio,
            "temporal_contamination_limit": threshold,
            "temporal_separation_pass": bool(ratio <= threshold),
            "setup_seconds": fine["setup_seconds"],
            "assembly_seconds": fine["assembly_seconds"],
            "solve_seconds": fine["solve_seconds"],
            "total_wall_seconds": fine["total_wall_seconds"],
            "solution_min": fine["solution_min"], "solution_max": fine["solution_max"],
            "exact_sampled_min": fine["exact_sampled_min"], "exact_sampled_max": fine["exact_sampled_max"],
            "overshoot": fine["overshoot"], "undershoot": fine["undershoot"],
            "error_residual_local_extrema_count": fine["error_residual_local_extrema_count"],
            "cells_with_local_error_extrema": fine["cells_with_local_error_extrema"],
            "fraction_cells_with_local_error_extrema": fine["fraction_cells_with_local_error_extrema"],
            "high_gradient_node_fraction": fine["high_gradient_node_fraction"],
            "high_gradient_mean_absolute_nodal_error": fine["high_gradient_mean_absolute_nodal_error"],
            "high_gradient_maximum_absolute_nodal_error": fine["high_gradient_maximum_absolute_nodal_error"],
            "max_linear_solver_residual": fine["max_linear_solver_residual"],
        }
        rows.append(row)
        logger.info("nx=%d: avg L2=%g, avg H1-semi=%g, temporal contamination=%.3g (%s)",
                    nx, row["time_average_absolute_l2_error"], row["time_average_absolute_h1_seminorm_error"],
                    ratio, "PASS" if row["temporal_separation_pass"] else "FAIL")
        del medium, fine, medium_hist, fine_hist
        gc.collect()

    rows.sort(key=lambda r: r["nx"])
    for i, row in enumerate(rows):
        row["observed_order_l2"] = None if i == 0 else observed_order(
            rows[i-1]["time_average_absolute_l2_error"], row["time_average_absolute_l2_error"],
            rows[i-1]["h"], row["h"])
        row["observed_order_h1_seminorm"] = None if i == 0 else observed_order(
            rows[i-1]["time_average_absolute_h1_seminorm_error"], row["time_average_absolute_h1_seminorm_error"],
            rows[i-1]["h"], row["h"])
    target = out / "solver_verification"
    merge_csv(target / "spatial_convergence.csv", rows, ["nx", "dt_used"])
    orders = {
        "metric_definitions": {
            "h": "maximum physical triangle diameter h=max_K diam(K), used for observed spatial orders",
            "L2": "time average of continuous absolute L2 error, using native deal.II quadrature",
            "H1": "time average of continuous H1 seminorm error (not the full H1 norm)",
            "temporal_contamination_ratio": "time-average L2 norm of u(dt/2)-u(dt/4), divided by time-average absolute L2 error of the dt/4 solution; first-order Cauchy proxy, not a rigorous upper bound",
        },
        "temporal_contamination_limit": threshold,
        "temporal_separation_method": "same fixed mesh; align u(dt/2) with every second state of u(dt/4); use the dt/4 trajectory for spatial errors; the Cauchy difference estimates remaining first-order temporal error and is diagnostic rather than a rigorous bound",
        "all_spatial_levels_temporally_separated": all(r["temporal_separation_pass"] for r in rows),
        "rows": [{k: r.get(k) for k in ("nx", "observed_order_l2", "observed_order_h1_seminorm", "temporal_contamination_ratio", "temporal_separation_pass")} for r in rows],
    }
    atomic_write_json(target / "observed_orders.json", orders, merge=True)
    write_latex_table(
        out / "tables" / "table_spatial_convergence.tex",
        "Fixed-grid spatial convergence for the Galerkin P1 method. The H1 column denotes the H1 seminorm.",
        "tab:gdm_stage_e_spatial",
        [r"$N_x$", "DOFs", r"$E_{L^2}$", r"$p_{L^2}$", r"$|e|_{H^1}$", r"$p_{H^1}$", "Temporal ratio"],
        [[r["nx"], r["n_dofs"], format_float(r["time_average_absolute_l2_error"]), format_float(r["observed_order_l2"]),
          format_float(r["time_average_absolute_h1_seminorm_error"]), format_float(r["observed_order_h1_seminorm"]),
          format_float(r["temporal_contamination_ratio"])] for r in rows],
    )
    from python.plotting.stage_e import plot_spatial_convergence
    plot_spatial_convergence(rows, out / "figures" / "spatial_convergence.pdf")
    return orders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(args.config)
