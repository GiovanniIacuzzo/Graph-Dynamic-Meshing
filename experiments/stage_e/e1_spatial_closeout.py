"""Close the Stage-E1 spatial study with a streaming, temporally refined pair."""
from __future__ import annotations

import argparse
import json
import logging

from .closeout_common import (
    closeout_root, ensure_closeout_tree, read_rows, run_paired_fixed_streaming,
    append_or_replace_csv_row,
)
from .common import configure_cpu_threads, load_config, observed_order, format_float, write_latex_table


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    configure_cpu_threads(int(cfg["experiment"].get("cpu_threads", 1)))
    root = ensure_closeout_tree(cfg)
    out = root / "solver_verification"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "spatial_convergence_refined.csv"
    close_cfg = cfg.get("closeout", {})
    fine_dt = float(close_cfg.get("spatial_fine_dt", 0.0000390625))
    modes = [str(x) for x in close_cfg.get("spatial_modes", ["galerkin", "supg"])]
    resolutions = [int(x) for x in cfg["spatial_convergence"]["resolutions"]]
    logger = logging.getLogger("gdm.stage_e.closeout")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    for mode in modes:
        for nx in resolutions:
            previous = read_rows(csv_path)
            key_present = any(
                row.get("mode") == mode and row.get("nx") == str(nx)
                and abs(float(row.get("dt_used", "nan")) - fine_dt) <= 1e-15
                for row in previous
            )
            if key_present:
                logger.info("Reuse spatial closeout: mode=%s nx=%d dt=%g", mode, nx, fine_dt)
                continue
            logger.info("Spatial closeout: mode=%s nx=%d, pair dt=%g/%g", mode, nx, 2*fine_dt, fine_dt)
            row = run_paired_fixed_streaming(cfg, nx, fine_dt, mode)
            append_or_replace_csv_row(csv_path, row, ("mode", "nx", "dt_used"))

    all_rows = read_rows(csv_path)
    output_summaries = {}
    for mode in modes:
        rows = [r for r in all_rows if r.get("mode") == mode and abs(float(r.get("dt_used", "nan")) - fine_dt) <= 1e-15]
        rows.sort(key=lambda r: int(r["nx"]))
        if len(rows) != len(resolutions):
            raise RuntimeError(f"Incomplete closeout rows for {mode}: {len(rows)}/{len(resolutions)}")
        for i, row in enumerate(rows):
            row["observed_order_l2"] = "" if i == 0 else observed_order(
                float(rows[i-1]["time_average_absolute_l2_error"]), float(row["time_average_absolute_l2_error"]),
                float(rows[i-1]["h"]), float(row["h"]),
            )
            row["observed_order_h1_seminorm"] = "" if i == 0 else observed_order(
                float(rows[i-1]["time_average_absolute_h1_seminorm_error"]), float(row["time_average_absolute_h1_seminorm_error"]),
                float(rows[i-1]["h"]), float(row["h"]),
            )
            row["temporal_separation_pass"] = str(
                float(row["temporal_contamination_ratio"]) <= float(close_cfg.get("temporal_contamination_limit", 0.10))
            )
            # Persist the derived orders and acceptance flag in the CSV as well
            # as in the LaTeX/JSON summaries, so the raw tabular output is self-contained.
            append_or_replace_csv_row(csv_path, row, ("mode", "nx", "dt_used"))
        output_summaries[mode] = {
            "fine_dt": fine_dt,
            "coarse_comparison_dt": 2.0 * fine_dt,
            "temporal_contamination_limit": float(close_cfg.get("temporal_contamination_limit", 0.10)),
            "all_levels_temporally_separated": all(str(r["temporal_separation_pass"]).lower() == "true" for r in rows),
            "rows": [{k: (float(r[k]) if r[k] not in ("", None) else None) for k in (
                "nx", "time_average_absolute_l2_error", "time_average_absolute_h1_seminorm_error",
                "observed_order_l2", "observed_order_h1_seminorm", "temporal_contamination_ratio",
            )} for r in rows],
        }
        write_latex_table(
            root / "tables" / f"table_spatial_convergence_{mode}_refined.tex",
            f"Temporally refined fixed-grid spatial convergence ({mode}); H1 denotes the H1 seminorm.",
            f"tab:gdm_stage_e_spatial_{mode}_refined",
            [r"$N_x$", "DOFs", r"$E_{L^2}$", r"$p_{L^2}$", r"$|e|_{H^1}$", r"$p_{H^1}$", "Temporal ratio"],
            [[r["nx"], r["n_dofs"], format_float(float(r["time_average_absolute_l2_error"])),
              format_float(float(r["observed_order_l2"])) if r["observed_order_l2"] != "" else "--",
              format_float(float(r["time_average_absolute_h1_seminorm_error"])),
              format_float(float(r["observed_order_h1_seminorm"])) if r["observed_order_h1_seminorm"] != "" else "--",
              format_float(float(r["temporal_contamination_ratio"]))] for r in rows],
        )
        from python.plotting.stage_e import plot_spatial_convergence
        plot_spatial_convergence(rows, root / "figures" / f"spatial_convergence_{mode}_refined.pdf")

    report = {
        "experiment": "stage_e1_spatial_closeout_streaming",
        "fine_dt": fine_dt,
        "coarse_comparison_dt": 2.0 * fine_dt,
        "method": "lockstep BE solves; online time integration of exact errors and aligned coarse/fine L2 differences; no full trajectory storage",
        "temporal_difference_interpretation": "Cauchy proxy for the first-order temporal error; not a rigorous upper bound",
        "modes": output_summaries,
        "pass": all(v["all_levels_temporally_separated"] for v in output_summaries.values()),
    }
    report_path = out / "spatial_closeout_summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    logger.info("Spatial closeout: %s", "PASS" if report["pass"] else "FAIL")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(args.config)
