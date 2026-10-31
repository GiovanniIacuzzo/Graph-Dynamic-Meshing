"""E1-B elementwise directional Péclet statistics on fixed and ALE meshes."""
from __future__ import annotations

import argparse

import numpy as np

from .common import (
    configure_cpu_threads, configure_logging, directional_element_metrics,
    exact_mesh_motion, load_config, merge_csv, prepare_output_tree, csv_contains_keys,
)


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    configure_cpu_threads(int(cfg["experiment"].get("cpu_threads", 1)))
    logger = configure_logging(cfg)
    out = prepare_output_tree(cfg)
    from .common import native_modules, make_domain
    NativeMeshAdapter, *_ = native_modules()
    nx = int(cfg["temporal_convergence"]["resolution"])
    base_mesh = NativeMeshAdapter.from_domain(make_domain(cfg, nx), nx=nx, ny=nx)
    nodes0 = np.asarray(base_mesh.nodes, dtype=float)
    cells = np.asarray(base_mesh.cells, dtype=np.int64)
    pde = cfg["pde"]
    b = np.array([float(pde["bx"]), float(pde["by"])], dtype=float)
    nu = float(pde["nu"])
    spec = cfg["peclet"]
    expected = [[kind, float(t)] for kind in ("fixed", "ale") for t in spec["sample_times"]]
    peclet_csv = out / "solver_verification" / "peclet_distribution.csv"
    if csv_contains_keys(peclet_csv, ["mesh_type", "time"], expected):
        logger.info("E1 Péclet: distribuzioni già complete; riuso dei CSV (resume).")
        return {"status": "reused_existing_results", "rows": len(expected)}
    amp = float(spec["ale_motion_amplitude"])
    period = float(spec["motion_period"])
    # Use exactly the secant velocity over an ALE solver step from the
    # SUPG/ALE comparison, rather than a derivative approximation with an
    # unrelated auxiliary time increment.
    dv = float(cfg["supg_comparison"]["dt"])
    final_time = float(cfg["pde"]["T"])
    if abs(round(final_time / dv) * dv - final_time) > 1.0e-10 * max(1.0, final_time):
        raise ValueError("Il passo ALE configurato non divide T.")
    cell_rows: list[dict] = []
    summary_rows: list[dict] = []

    for t in map(float, spec["sample_times"]):
        # Fixed mesh: w=0 exactly.
        beta_fixed = np.repeat(b[None, :], cells.shape[0], axis=0)
        h_fixed, pe_fixed = directional_element_metrics(nodes0, cells, beta_fixed, nu)
        summary_rows.append({"mesh_type": "fixed", "time": t, **_stats(h_fixed, pe_fixed)})
        for e in range(cells.shape[0]):
            cell_rows.append({"mesh_type": "fixed", "time": t, "cell_id": e,
                              "beta_x": b[0], "beta_y": b[1], "h_beta": h_fixed[e], "Pe_K": pe_fixed[e]})

        # ALE mesh velocity is the actual nodal secant used by the ALE solver.
        # Sample times are aligned to its time grid. At T, use the final interval
        # [T-dt,T] rather than extrapolating the trajectory beyond the simulation.
        if abs(t - final_time) <= 1.0e-12 * max(1.0, final_time):
            t0, t1 = final_time - dv, final_time
        else:
            if abs(round(t / dv) * dv - t) > 1.0e-10 * max(1.0, final_time):
                raise ValueError(f"Il tempo Pe t={t} non è allineato al passo ALE dt={dv}.")
            t0, t1 = t, t + dv
        nodes_t = exact_mesh_motion(nodes0, t0, period, amp)
        nodes_tp = exact_mesh_motion(nodes0, t1, period, amp)
        nodes_mid = 0.5 * (nodes_t + nodes_tp)
        w_nodes = (nodes_tp - nodes_t) / (t1 - t0)
        beta_cells = np.empty((cells.shape[0], 2), dtype=float)
        for e, tri in enumerate(cells):
            beta_cells[e] = b - np.mean(w_nodes[tri], axis=0)
        h_ale, pe_ale = directional_element_metrics(nodes_mid, cells, beta_cells, nu)
        summary_rows.append({"mesh_type": "ale", "time": t, **_stats(h_ale, pe_ale)})
        for e in range(cells.shape[0]):
            cell_rows.append({"mesh_type": "ale", "time": t, "cell_id": e,
                              "beta_x": beta_cells[e, 0], "beta_y": beta_cells[e, 1],
                              "h_beta": h_ale[e], "Pe_K": pe_ale[e]})
        logger.info("Pe t=%g: fixed median/p95/max %.3g/%.3g/%.3g; ALE %.3g/%.3g/%.3g",
                    t, np.median(pe_fixed), np.quantile(pe_fixed, .95), np.max(pe_fixed),
                    np.median(pe_ale), np.quantile(pe_ale, .95), np.max(pe_ale))

    target = out / "solver_verification"
    merge_csv(target / "peclet_cell_statistics.csv", cell_rows, ["mesh_type", "time", "cell_id"])
    merge_csv(target / "peclet_distribution.csv", summary_rows, ["mesh_type", "time"])
    from python.plotting.stage_e import plot_peclet_distribution
    plot_peclet_distribution(summary_rows, out / "figures" / "peclet_distribution.pdf")
    return {"sample_times": list(map(float, spec["sample_times"])), "n_cells": int(cells.shape[0]),
            "ale_velocity_time_step": dv,
            "definition": "Pe_K=||b-w|| h_K^(b-w)/(2 nu); h_K=2||b-w||/sum_i |(b-w).grad(phi_i)|; w is the actual nodal secant velocity over the ALE solver step (at T, the final backward interval); zero speed -> Pe=0 and h_K=element diameter"}


def _stats(h: np.ndarray, pe: np.ndarray) -> dict:
    return {
        "n_elements": int(pe.size),
        "h_beta_min": float(np.min(h)), "h_beta_median": float(np.median(h)),
        "h_beta_mean": float(np.mean(h)), "h_beta_p95": float(np.quantile(h, .95)), "h_beta_max": float(np.max(h)),
        "Pe_min": float(np.min(pe)), "Pe_median": float(np.median(pe)),
        "Pe_mean": float(np.mean(pe)), "Pe_p95": float(np.quantile(pe, .95)), "Pe_max": float(np.max(pe)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(run(args.config))
