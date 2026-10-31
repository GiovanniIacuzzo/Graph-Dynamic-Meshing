"""Temporally resolved Galerkin work-precision extension for the E1 close-out."""
from __future__ import annotations

import argparse
import json
import statistics

from .closeout_common import (
    closeout_metadata, ensure_closeout_tree, read_rows, append_or_replace_csv_row,
)
from .common import configure_cpu_threads, load_config, run_fixed_case, format_float, write_latex_table


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    configure_cpu_threads(int(cfg["experiment"].get("cpu_threads", 1)))
    root = ensure_closeout_tree(cfg)
    source = root / "solver_verification" / "spatial_convergence_refined.csv"
    source_rows = read_rows(source)
    close_cfg = cfg.get("closeout", {})
    fine_dt = float(close_cfg.get("spatial_fine_dt", 0.0000390625))
    resolutions = [int(x) for x in cfg["work_precision"]["resolutions"]]
    if not source_rows:
        raise RuntimeError("Manca spatial_convergence_refined.csv: eseguire prima e1_spatial_closeout.")

    spatial = {}
    for row in source_rows:
        if row.get("mode") == "galerkin" and abs(float(row.get("dt_used", "nan")) - fine_dt) <= 1e-15:
            spatial[int(row["nx"])] = row
    missing = sorted(set(resolutions) - set(spatial))
    if missing:
        raise RuntimeError(f"Mancano righe spaziali Galerkin temporally refined per nx={missing}.")

    target = root / "work_precision"
    target.mkdir(parents=True, exist_ok=True)
    raw_path = target / "runtime_repetitions_refined.csv"
    summary_path = target / "work_precision_galerkin_refined.json"
    repetitions = int(close_cfg.get("work_precision_repetitions", 5))
    warmups = int(close_cfg.get("work_precision_warmups", 1))
    if repetitions < 5 or warmups < 1:
        raise ValueError("Closeout work-precision requires at least 5 repetitions and 1 warm-up.")

    previous = read_rows(raw_path)
    done = {(int(float(r["nx"])), int(float(r["repetition"]))) for r in previous if r.get("mode") == "galerkin"}
    summary_rows = []
    for nx in resolutions:
        accuracy = spatial[nx]
        for _ in range(warmups):
            # Warm-ups are intentionally not entered in the measured repetition CSV.
            # They are repeated on restart; only measured repetitions are resumable.
            run_fixed_case(cfg, nx, fine_dt, "galerkin", retain_snapshots=False)
        for repetition in range(1, repetitions + 1):
            if (nx, repetition) in done:
                continue
            result = run_fixed_case(cfg, nx, fine_dt, "galerkin", retain_snapshots=False)
            row = {
                "mode": "galerkin", "nx": nx, "dt": fine_dt, "repetition": repetition,
                "setup_seconds": result["setup_seconds"],
                "assembly_seconds": result["assembly_seconds"],
                "solve_seconds": result["solve_seconds"],
                "total_wall_seconds": result["total_wall_seconds"],
                "n_dofs": result["n_dofs"], "n_elements": result["n_elements"],
                "max_linear_solver_residual": result["max_linear_solver_residual"],
                "max_linear_solver_tolerance": result["max_linear_solver_tolerance"],
            }
            append_or_replace_csv_row(raw_path, row, ("mode", "nx", "dt", "repetition"))
            done.add((nx, repetition))

        timed = [r for r in read_rows(raw_path) if r.get("mode") == "galerkin" and int(float(r["nx"])) == nx]
        timed = [r for r in timed if abs(float(r["dt"]) - fine_dt) <= 1e-15]
        timed.sort(key=lambda r: int(float(r["repetition"])))
        if len(timed) < repetitions:
            raise RuntimeError(f"Timing repetitions incomplete for nx={nx}: {len(timed)}/{repetitions}")
        med = lambda key: statistics.median(float(r[key]) for r in timed)
        summary_rows.append({
            "mode": "galerkin", "nx": nx, "dt": fine_dt,
            "n_dofs": int(float(accuracy["n_dofs"])), "n_elements": int(float(accuracy["n_elements"])),
            "time_average_absolute_l2_error": float(accuracy["time_average_absolute_l2_error"]),
            "time_average_absolute_h1_seminorm_error": float(accuracy["time_average_absolute_h1_seminorm_error"]),
            "temporal_contamination_ratio": float(accuracy["temporal_contamination_ratio"]),
            "setup_seconds_median": med("setup_seconds"),
            "assembly_seconds_median": med("assembly_seconds"),
            "solve_seconds_median": med("solve_seconds"),
            "total_wall_seconds_median": med("total_wall_seconds"),
            "total_wall_seconds_min": min(float(r["total_wall_seconds"]) for r in timed),
            "total_wall_seconds_max": max(float(r["total_wall_seconds"]) for r in timed),
            "timing_repetitions": repetitions, "warmup_runs": warmups,
            "accuracy_source": "temporally refined spatial closeout trajectory; same dt and same mesh",
        })

    from python.plotting.stage_e import plot_work_precision
    plot_work_precision(summary_rows, root / "figures" / "work_precision_galerkin_temporally_resolved.pdf")
    write_latex_table(
        target / "table_work_precision_galerkin_temporally_resolved.tex",
        "Galerkin work--precision at a temporally resolved time step; accuracy and runtime use the same mesh and time step.",
        "tab:gdm_stage_e_work_precision_refined",
        [r"$N_x$", "DOFs", r"$E_{L^2}$", r"$|e|_{H^1}$", "Temporal ratio", "CPU time [s]"],
        [[r["nx"], r["n_dofs"], format_float(r["time_average_absolute_l2_error"]),
          format_float(r["time_average_absolute_h1_seminorm_error"]),
          format_float(r["temporal_contamination_ratio"]), format_float(r["total_wall_seconds_median"])] for r in summary_rows],
    )
    report = {
        "mode": "galerkin", "dt": fine_dt, "rows": summary_rows,
        "warmup_runs": warmups, "timing_repetitions": repetitions,
        "accuracy_and_runtime_same_discretization": True,
        **closeout_metadata(),
        "interpretation": "This is the temporally resolved Galerkin work-precision curve. The previously reported curve at dt=0.0025 remains an operational fixed-step benchmark and is not replaced.",
    }
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    run(parser.parse_args().config)
