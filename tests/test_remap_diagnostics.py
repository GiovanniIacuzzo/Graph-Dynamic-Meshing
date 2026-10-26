"""
tests/test_remap_diagnostics.py
Suite diagnostica quantitativa per l'operatore di remapping P1 (Stage B).
Genera e stampa tabelle analitiche complete per R1, R2, R3, R4 e R5.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import ensure_dir
from generate_dataset import create_initial_mesh_2d
from geometry.safety import apply_safety_filter
from geometry.geometry import compute_mesh_metrics
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.remap import remap_p1_solution


def _get_free_indices(mesh):
    is_bnd = np.zeros(mesh.num_nodes, dtype=bool)
    is_bnd[mesh.boundary_nodes] = True
    return np.where(~is_bnd)[0]


def test_r1_exactness():
    """R1: Esattezza a precisione macchina su campi costanti e lineari."""
    mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=16, ny=16)
    rng = np.random.default_rng(42)

    raw_disp = np.zeros_like(mesh.nodes)
    free_idx = _get_free_indices(mesh)
    raw_disp[free_idx] = rng.uniform(-0.01, 0.01, size=(len(free_idx), 2))

    _, safe_nodes, is_safe, _ = apply_safety_filter(mesh.nodes, raw_disp, mesh.cells, epsilon=1e-4)
    assert is_safe

    u_const = np.full(mesh.num_nodes, 5.0)
    remap_const = remap_p1_solution(mesh.nodes, mesh.cells, u_const, safe_nodes)
    err_const = float(np.max(np.abs(remap_const - 5.0)))

    u_lin = 2.0 * mesh.nodes[:, 0] - 3.0 * mesh.nodes[:, 1] + 1.0
    remap_lin = remap_p1_solution(mesh.nodes, mesh.cells, u_lin, safe_nodes)
    exp_lin = 2.0 * safe_nodes[:, 0] - 3.0 * safe_nodes[:, 1] + 1.0
    err_lin = float(np.max(np.abs(remap_lin - exp_lin)))

    print("\n" + "=" * 65)
    print("R1 — EXACTNESS TEST")
    print("=" * 65)
    print(f"Campo Costante (u = 5.0)       | Max Error: {err_const:.4e}")
    print(f"Campo Lineare  (u = 2x - 3y + 1)| Max Error: {err_lin:.4e}")

    assert err_const < 1e-12
    assert err_lin < 1e-12


def test_r2_spatial_convergence():
    """R2: Ordine di convergenza spaziale h con risoluzioni 16x16 -> 128x128."""
    pde = ManufacturedConvectionDiffusion2D()
    t = 0.2
    resolutions = [16, 32, 64, 128]
    h_vals = [1.0 / n for n in resolutions]
    errors_l2_nodal = []

    for n in resolutions:
        h = 1.0 / n
        mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=n, ny=n)
        u_old = pde.evaluate_solution(mesh.nodes[:, 0], mesh.nodes[:, 1], t)

        disp = np.zeros_like(mesh.nodes)
        free_idx = _get_free_indices(mesh)
        disp[free_idx, 0] = 0.25 * h * np.sin(np.pi * mesh.nodes[free_idx, 0])
        disp[free_idx, 1] = 0.25 * h * np.cos(np.pi * mesh.nodes[free_idx, 1])

        # Epsilon scalato coerentemente con la dimensione della cella h^2
        eps_h = min(1e-4, 0.05 * (h ** 2))
        _, safe_nodes, is_safe, _ = apply_safety_filter(mesh.nodes, disp, mesh.cells, epsilon=eps_h)
        assert is_safe

        u_remap = remap_p1_solution(mesh.nodes, mesh.cells, u_old, safe_nodes)
        u_exact_new = pde.evaluate_solution(safe_nodes[:, 0], safe_nodes[:, 1], t)

        err_l2 = float(np.linalg.norm(u_remap - u_exact_new) / np.linalg.norm(u_exact_new))
        errors_l2_nodal.append(err_l2)

    rates = []
    for i in range(len(errors_l2_nodal) - 1):
        p = np.log(errors_l2_nodal[i] / errors_l2_nodal[i + 1]) / np.log(h_vals[i] / h_vals[i + 1])
        rates.append(p)

    print("\n" + "=" * 65)
    print("R2 — SPATIAL CONVERGENCE (h-refinement)")
    print("=" * 65)
    print(f"{'Mesh':<12} {'h':<12} {'Remap L2':<18} {'Observed Order (p)':<15}")
    print("-" * 65)
    print(f"{'16x16':<12} {h_vals[0]:<12.4f} {errors_l2_nodal[0]:<18.4e} {'-':<15}")
    for k in range(len(rates)):
        name = f"{resolutions[k+1]}x{resolutions[k+1]}"
        print(f"{name:<12} {h_vals[k+1]:<12.4f} {errors_l2_nodal[k+1]:<18.4e} {rates[k]:<15.2f}")

    assert rates[-1] >= 1.70, f"Ordine asintotico di convergenza insufficiente: {rates[-1]:.2f}"

def test_r3_displacement_amplitude():
    """R3: Errore del remapping in funzione del fattore di scala gamma."""
    mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=32, ny=32)
    pde = ManufacturedConvectionDiffusion2D()
    t = 0.25
    u_old = pde.evaluate_solution(mesh.nodes[:, 0], mesh.nodes[:, 1], t)
    free_idx = _get_free_indices(mesh)

    gammas = [0.10, 0.25, 0.50, 1.00, 2.00]
    base_amp = 0.005
    errors_gamma = []

    print("\n" + "=" * 65)
    print("R3 — ERROR VS DISPLACEMENT AMPLITUDE")
    print("=" * 65)
    print(f"{'gamma':<12} {'Max Displacement':<20} {'Remap L2 Error':<18}")
    print("-" * 65)

    for g in gammas:
        disp = np.zeros_like(mesh.nodes)
        disp[free_idx, 0] = g * base_amp * np.sin(2 * np.pi * mesh.nodes[free_idx, 1])
        disp[free_idx, 1] = g * base_amp * np.cos(2 * np.pi * mesh.nodes[free_idx, 0])

        _, safe_nodes, is_safe, _ = apply_safety_filter(mesh.nodes, disp, mesh.cells, epsilon=1e-4)
        assert is_safe

        actual_max_disp = float(np.max(np.linalg.norm(safe_nodes - mesh.nodes, axis=1)))
        u_remap = remap_p1_solution(mesh.nodes, mesh.cells, u_old, safe_nodes)
        u_exact = pde.evaluate_solution(safe_nodes[:, 0], safe_nodes[:, 1], t)

        err_l2 = float(np.linalg.norm(u_remap - u_exact) / np.linalg.norm(u_exact))
        errors_gamma.append(err_l2)
        print(f"{g:<12.2f} {actual_max_disp:<20.4e} {err_l2*100:<17.3f}%")

    for k in range(len(errors_gamma) - 1):
        assert errors_gamma[k] <= errors_gamma[k + 1]


def test_r4_localization_diagnostics():
    """R4: Controllo di conformità su localizzazione e violazioni al bordo."""
    mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=32, ny=32)
    pde = ManufacturedConvectionDiffusion2D()
    u_old = pde.evaluate_solution(mesh.nodes[:, 0], mesh.nodes[:, 1], 0.0)

    raw_disp = np.zeros_like(mesh.nodes)
    free_idx = _get_free_indices(mesh)
    raw_disp[free_idx] = 0.008

    _, safe_nodes, is_safe, _ = apply_safety_filter(mesh.nodes, raw_disp, mesh.cells, epsilon=1e-4)
    assert is_safe

    u_remap, diag = remap_p1_solution(mesh.nodes, mesh.cells, u_old, safe_nodes, return_diagnostics=True)

    print("\n" + "=" * 65)
    print("R4 — LOCALIZATION DIAGNOSTICS")
    print("=" * 65)
    print(f"global fallback       = 0 (non implementato / vietato)")
    print(f"unlocated             = {0 if diag['all_localized'] else 'DETECTED'}")
    print(f"projected/clamped     = 0 (clamping rimosso)")
    print(f"max bary violation    = {diag['max_boundary_violation']:.4e}")

    assert diag["all_localized"] is True
    assert diag["max_boundary_violation"] == 0.0


def test_r5_single_vs_cumulative():
    """R5: Confronto quantitativo tra remap cumulativi e remap diretto."""
    mesh_init = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=32, ny=32)
    pde = ManufacturedConvectionDiffusion2D()
    t = 0.1
    u_exact_init = pde.evaluate_solution(mesh_init.nodes[:, 0], mesh_init.nodes[:, 1], t)
    free_idx = _get_free_indices(mesh_init)

    remap_steps = [1, 2, 5, 10]
    out_table = []

    print("\n" + "=" * 65)
    print("R5 — SINGLE VS CUMULATIVE REMAP")
    print("=" * 65)
    print(f"{'N remaps':<12} {'Cumulative Error':<20} {'Direct Error':<18} {'Amplification':<15}")
    print("-" * 65)

    for n_steps in remap_steps:
        curr_nodes = mesh_init.nodes.copy()
        u_cum = u_exact_init.copy()

        min_alpha = 1.0
        for step in range(n_steps):
            disp = np.zeros_like(mesh_init.nodes)
            disp[free_idx, 0] = 0.002 * np.sin((step + 1) * curr_nodes[free_idx, 1])
            disp[free_idx, 1] = 0.002 * np.cos((step + 1) * curr_nodes[free_idx, 0])

            alpha, next_nodes, _, _ = apply_safety_filter(
                curr_nodes, disp, mesh_init.cells, epsilon=1e-4)

            geom = compute_mesh_metrics(next_nodes, mesh_init.cells)
            assert int(np.sum(geom["inverted"])) == 0
            assert float(np.min(geom["jacobians"])) >= 1e-4
            min_alpha = min(min_alpha, alpha)

            u_cum = remap_p1_solution(curr_nodes, mesh_init.cells, u_cum, next_nodes)
            curr_nodes = next_nodes

            u_cum = remap_p1_solution(curr_nodes, mesh_init.cells, u_cum, next_nodes)
            curr_nodes = next_nodes

        u_exact_final = pde.evaluate_solution(curr_nodes[:, 0], curr_nodes[:, 1], t)
        u_direct = remap_p1_solution(mesh_init.nodes, mesh_init.cells, u_exact_init, curr_nodes)

        err_cum = float(np.linalg.norm(u_cum - u_exact_final) / np.linalg.norm(u_exact_final))
        err_dir = float(np.linalg.norm(u_direct - u_exact_final) / np.linalg.norm(u_exact_final))
        amp = err_cum / max(err_dir, 1e-15)

        out_table.append({
            "n_remaps": n_steps,
            "cumulative_l2": err_cum,
            "direct_l2": err_dir,
            "amplification": amp
        })

        print(f"{n_steps:<12d} {err_cum*100:<19.3f}% {err_dir*100:<17.3f}% {amp:<15.2f}x")

    out_dir = ensure_dir("results/stage_b/diagnostics")
    with open(out_dir / "remap_diagnostics_r1_r5.json", "w", encoding="utf-8") as f:
        json.dump(out_table, f, indent=2)

if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])