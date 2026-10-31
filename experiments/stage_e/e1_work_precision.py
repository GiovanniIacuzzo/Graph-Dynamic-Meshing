"""E1-E fixed-grid CPU work-precision with warm-ups and resumable timings."""
from __future__ import annotations

import argparse
import csv
import gc
import statistics

from .common import (
    configure_cpu_threads, configure_logging, format_float, load_config, merge_csv,
    prepare_output_tree, run_fixed_case, write_latex_table, read_csv_rows,
)


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    configure_cpu_threads(int(cfg["experiment"].get("cpu_threads", 1)))
    logger = configure_logging(cfg)
    out = prepare_output_tree(cfg)
    spec = cfg["work_precision"]
    warmups, repetitions = int(spec["warmup_runs"]), int(spec["repetitions"])
    if repetitions < 5:
        raise ValueError("La campagna E1-E richiede almeno cinque ripetizioni temporali CPU.")
    if warmups < 1:
        raise ValueError("La campagna E1-E richiede almeno un warm-up per configurazione.")

    target = out / "work_precision"
    accuracy_path = target / "accuracy_samples.csv"
    runtime_path = target / "runtime_repetitions.csv"
    previous_accuracy = read_csv_rows(accuracy_path)
    previous_runtime = read_csv_rows(runtime_path)

    expected = {
        (str(mode), str(int(nx)), str(float(spec["dt"])))
        for mode in spec["modes"] for nx in spec["resolutions"]
    }
    have_accuracy = {
        (r.get("mode", ""), r.get("nx", ""), r.get("dt", ""))
        for r in previous_accuracy
    }
    repetition_ids: dict[tuple[str, str, str], set[int]] = {}
    for r in previous_runtime:
        key = (r.get("mode", ""), r.get("nx", ""), r.get("dt", ""))
        repetition_ids.setdefault(key, set()).add(int(float(r.get("repetition", "0"))))

    summary_path = target / "work_precision_summary.json"
    fixed_results_path = target / "fixed_grid_results.csv"
    if (expected.issubset(have_accuracy) and
        all(set(range(1, repetitions + 1)).issubset(repetition_ids.get(key, set())) for key in expected) and
        summary_path.exists() and fixed_results_path.exists()):
        logger.info("Work-precision già completo; riuso CSV e summary (resume).")
        import json
        return json.loads(summary_path.read_text(encoding="utf-8"))

    for mode in map(str, spec["modes"]):
        if mode not in {"galerkin", "supg"}:
            raise ValueError(f"Modo non supportato: {mode}")
        for nx in map(int, spec["resolutions"]):
            dt = float(spec["dt"])
            key = (mode, str(nx), str(float(dt)))
            ids = repetition_ids.setdefault(key, set())
            logger.info(
                "Work-precision mode=%s nx=%d: accuracy=%s, timing %d/%d già presenti",
                mode, nx, "presente" if key in have_accuracy else "da calcolare", len(ids), repetitions,
            )

            # Accuracy is one independent numerical solve, never pooled across repeats.
            if key not in have_accuracy:
                accuracy = run_fixed_case(cfg, nx, dt, mode, retain_snapshots=True)
                accuracy_row = {
                    "mode": mode, "nx": nx, "dt": dt,
                    "n_nodes": accuracy["n_nodes"], "n_elements": accuracy["n_elements"],
                    "n_dofs": accuracy["n_dofs"],
                    "time_average_absolute_l2_error": accuracy["time_average_absolute_l2_error"],
                    "relative_l2_error_final": accuracy["relative_l2_error_final"],
                    "time_average_absolute_h1_seminorm_error": accuracy["time_average_absolute_h1_seminorm_error"],
                    "relative_h1_seminorm_error_final": accuracy["relative_h1_seminorm_error_final"],
                    "solution_min": accuracy["solution_min"], "solution_max": accuracy["solution_max"],
                    "overshoot": accuracy["overshoot"], "undershoot": accuracy["undershoot"],
                    "accuracy_sample_policy": "one dedicated full-trajectory solve; timing repetitions are not accuracy samples",
                }
                del accuracy
                merge_csv(accuracy_path, [accuracy_row], ["mode", "nx", "dt"])
                have_accuracy.add(key)
                gc.collect()

            # The warm-up is never recorded as a measured timing repetition.
            if len(ids) < repetitions:
                for w in range(warmups):
                    _ = run_fixed_case(cfg, nx, dt, mode, retain_snapshots=False)
                    logger.info("warm-up %d/%d completato", w + 1, warmups)
                    gc.collect()

            for rep_no in range(1, repetitions + 1):
                if rep_no in ids:
                    continue
                measured = run_fixed_case(cfg, nx, dt, mode, retain_snapshots=False)
                runtime_row = {
                    "mode": mode, "nx": nx, "dt": dt, "repetition": rep_no,
                    "cpu_threads": int(cfg["experiment"].get("cpu_threads", 1)),
                    "setup_seconds": measured["setup_seconds"],
                    "assembly_seconds": measured["assembly_seconds"],
                    "solve_seconds": measured["solve_seconds"],
                    "other_seconds": measured["other_seconds"],
                    "total_wall_seconds": measured["total_wall_seconds"],
                    "max_linear_solver_residual": measured["max_linear_solver_residual"],
                }
                # Commit each completed repetition immediately. A later interruption
                # cannot erase it, and resume fills the missing repetition indices.
                merge_csv(runtime_path, [runtime_row], ["mode", "nx", "dt", "repetition"])
                ids.add(rep_no)
                logger.info(
                    "rep %d/%d total %.3fs (assembly %.3fs; solve %.3fs)",
                    rep_no, repetitions, measured["total_wall_seconds"],
                    measured["assembly_seconds"], measured["solve_seconds"],
                )
                del measured
                gc.collect()

    accuracy_rows = read_csv_rows(accuracy_path)
    runtime_rows = read_csv_rows(runtime_path)
    summary_rows = []
    for accuracy in accuracy_rows:
        mode, nx, dt_s = accuracy["mode"], accuracy["nx"], accuracy["dt"]
        reps = [r for r in runtime_rows if
                r["mode"] == mode and r["nx"] == nx and r["dt"] == dt_s]
        reps.sort(key=lambda r: int(float(r["repetition"])))
        if len(reps) < repetitions:
            raise RuntimeError(f"Timing incompleti per {mode}, nx={nx}: {len(reps)}/{repetitions}.")
        reps = reps[:repetitions]
        setup = [float(r["setup_seconds"]) for r in reps]
        assembly = [float(r["assembly_seconds"]) for r in reps]
        solve = [float(r["solve_seconds"]) for r in reps]
        total = [float(r["total_wall_seconds"]) for r in reps]
        row = dict(accuracy)
        row.update({
            "setup_seconds_median": statistics.median(setup),
            "assembly_seconds_median": statistics.median(assembly),
            "solve_seconds_median": statistics.median(solve),
            "total_wall_seconds_median": statistics.median(total),
            "total_wall_seconds_min": min(total), "total_wall_seconds_max": max(total),
            "timing_repetitions": repetitions, "warmup_runs": warmups,
        })
        summary_rows.append(row)

    # Materialize the combined manuscript-ready CSV only after every configured
    # timing repetition is available. Accuracy and raw timings remain immutable
    # source files and permit an interrupted campaign to resume safely.
    if summary_rows:
        fields = list(dict.fromkeys(k for row in summary_rows for k in row.keys()))
        temporary = fixed_results_path.with_suffix(fixed_results_path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        temporary.replace(fixed_results_path)

    summary = {
        "hardware_policy": "CPU only; configured CPU thread limit; same process/environment per campaign invocation",
        "warmup_runs": warmups, "timing_repetitions": repetitions,
        "accuracy_not_pooled_from_repetitions": True,
        "accuracy_and_timing_separated": True,
        "rows": summary_rows,
    }
    from .common import atomic_write_json
    atomic_write_json(target / "work_precision_summary.json", summary, merge=False)
    write_latex_table(
        out / "tables" / "table_work_precision.tex",
        "Fixed-grid work--precision results. Runtime entries are medians over independent CPU repetitions; accuracy comes from one dedicated full-trajectory solve per configuration.",
        "tab:gdm_stage_e_work_precision",
        ["Method", r"$N_x$", "DOFs", r"$E_{L^2}$", "Setup (s)", "Assembly (s)", "Solve (s)", "Total (s)"],
        [[r["mode"], r["nx"], r["n_dofs"], format_float(r["time_average_absolute_l2_error"]),
          format_float(r["setup_seconds_median"], 4), format_float(r["assembly_seconds_median"], 4),
          format_float(r["solve_seconds_median"], 4), format_float(r["total_wall_seconds_median"], 4)]
         for r in summary_rows],
    )
    from python.plotting.stage_e import plot_work_precision
    plot_work_precision(summary_rows, out / "figures" / "fixed_grid_work_precision.pdf")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    run(parser.parse_args().config)
