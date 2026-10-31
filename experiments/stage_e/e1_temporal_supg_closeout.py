"""Temporal convergence audit of the optional fixed-grid SUPG path."""
from __future__ import annotations

import argparse
import json

from .closeout_common import ensure_closeout_tree, append_or_replace_csv_row
from .common import configure_cpu_threads, load_config, l2_p1_difference, number_of_steps, run_fixed_case, observed_order, format_float, write_latex_table


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    configure_cpu_threads(int(cfg["experiment"].get("cpu_threads", 1)))
    root = ensure_closeout_tree(cfg) / "supg"
    root.mkdir(parents=True, exist_ok=True)
    nx = int(cfg["temporal_convergence"]["resolution"])
    base_dts = [float(x) for x in cfg["temporal_convergence"]["dts"]]
    reference_dt = float(cfg["temporal_convergence"]["reference_dt"])
    all_dts = sorted(set(base_dts + [x / 2.0 for x in base_dts] + [reference_dt]), reverse=True)
    results = {}
    for dt in all_dts:
        number_of_steps(float(cfg["pde"]["T"]), dt)
        print(f"[SUPG temporal] nx={nx}, dt={dt:g}")
        results[dt] = run_fixed_case(cfg, nx, dt, "supg", retain_snapshots=False)

    rows = []
    for dt in base_dts:
        coarse, fine = results[dt], results[dt / 2.0]
        cauchy = l2_p1_difference(fine["mesh"], coarse["final_solution"], fine["final_solution"])
        rows.append({
            "mode": "supg", "nx": nx, "dt": dt, "dt_half": dt / 2.0,
            "n_steps": number_of_steps(float(cfg["pde"]["T"]), dt),
            "absolute_l2_error_final": coarse["absolute_l2_error_final"],
            "absolute_h1_seminorm_error_final": coarse["absolute_h1_seminorm_error_final"],
            "cauchy_l2_difference_dt_dt_half": cauchy,
        })
    rows.sort(key=lambda x: x["dt"], reverse=True)
    for i, row in enumerate(rows):
        row["observed_order_cauchy"] = None if i == 0 else observed_order(
            rows[i-1]["cauchy_l2_difference_dt_dt_half"], row["cauchy_l2_difference_dt_dt_half"],
            rows[i-1]["dt"], row["dt"],
        )
    csv_path = root / "supg_temporal_convergence.csv"
    for row in rows:
        append_or_replace_csv_row(csv_path, row, ("mode", "nx", "dt"))
    summary = {
        "mode": "supg", "nx": nx,
        "integrator": "Backward Euler, formally first order",
        "metric": "final-time exact P1 L2 norm of u_h(dt)-u_h(dt/2) on the same fixed mesh",
        "expected_order": 1,
        "rows": rows,
    }
    (root / "supg_temporal_convergence.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_latex_table(
        root / "table_supg_temporal_convergence.tex",
        "Temporal convergence of the fixed-grid SUPG variant; Cauchy differences compare consecutive Backward Euler steps.",
        "tab:gdm_stage_e_supg_temporal",
        [r"$\Delta t$", "Steps", r"$E_{L^2}(T)$", r"$E_{H^1}(T)$", r"$\|u_{\Delta t}-u_{\Delta t/2}\|_{L^2}$", r"$q_{obs}$"],
        [[format_float(r["dt"]), r["n_steps"], format_float(r["absolute_l2_error_final"]),
          format_float(r["absolute_h1_seminorm_error_final"]), format_float(r["cauchy_l2_difference_dt_dt_half"]),
          format_float(r["observed_order_cauchy"])] for r in rows],
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    run(parser.parse_args().config)
