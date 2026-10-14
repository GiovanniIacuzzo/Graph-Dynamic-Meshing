"""
tests/test_remap_l2.py
Suite diagnostica comparativa formale: P1 Pointwise vs L2-Projection Remap.
Implementa i test L1, L2, L3, L4 (homotopy a target fisso), L5 (costo computazionale).
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from generate_dataset import create_initial_mesh_2d
from geometry.safety import apply_safety_filter
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.remap import remap_p1_solution
from numerics.remap_l2 import remap_l2_projection


def _get_free_indices(mesh):
    is_bnd = np.zeros(mesh.num_nodes, dtype=bool)
    is_bnd[mesh.boundary_nodes] = True
    return np.where(~is_bnd)[0]


def test_l1_exactness():
    """L1: Esattezza su costante e lineare."""
    mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=16, ny=16)
    rng = np.random.default_rng(42)

    raw_disp = np.zeros_like(mesh.nodes)
    free_idx = _get_free_indices(mesh)
    raw_disp[free_idx] = rng.uniform(-0.01, 0.01, size=(len(free_idx), 2))

    _, safe_nodes, is_safe, _ = apply_safety_filter(mesh.nodes, raw_disp, mesh.cells, epsilon=1e-4)
    assert is_safe

    # 1. Costante u = 1.0
    u_const = np.ones(mesh.num_nodes)
    u_l2_const = remap_l2_projection(mesh.nodes, mesh.cells, u_const, safe_nodes, mesh.cells)
    err_const = float(np.max(np.abs(u_l2_const - 1.0)))

    # 2. Lineare u = 2x - 3y + 1
    u_lin = 2.0 * mesh.nodes[:, 0] - 3.0 * mesh.nodes[:, 1] + 1.0
    u_l2_lin = remap_l2_projection(mesh.nodes, mesh.cells, u_lin, safe_nodes, mesh.cells)
    exp_lin = 2.0 * safe_nodes[:, 0] - 3.0 * safe_nodes[:, 1] + 1.0
    err_lin = float(np.max(np.abs(u_l2_lin - exp_lin)))

    print("\n" + "=" * 65)
    print("L1 — L2 PROJECTION: EXACTNESS TEST")
    print("=" * 65)
    print(f"Costante (u = 1.0)       | Max Error: {err_const:.4e}")
    print(f"Lineare  (u = 2x - 3y + 1)| Max Error: {err_lin:.4e}")

    assert err_const < 1e-10
    assert err_lin < 1e-10


def test_l2_spatial_convergence():
    """
    L2: Confronto della convergenza spaziale tra P1 pointwise
    e proiezione L2.

    La metrica è la norma L2 relativa continua sul dominio,
    integrata sulla nuova mesh mediante quadratura di grado 5.

    Non viene utilizzata la norma euclidea dei soli valori nodali.
    """

    from numerics.fem import _get_triangle_quadrature_degree_5

    pde = ManufacturedConvectionDiffusion2D()
    t = 0.2

    resolutions = [16, 32, 64, 128]
    h_vals = [1.0 / n for n in resolutions]

    err_p1_list = []
    err_l2_list = []

    # -------------------------------------------------------------
    # Errore L2 relativo continuo
    # -------------------------------------------------------------
    def relative_continuous_l2_error(mesh, u_approx):
        nodes = mesh.nodes
        cells = mesh.cells

        tri_pts = nodes[cells]

        x1 = tri_pts[:, 0, 0]
        y1 = tri_pts[:, 0, 1]

        x2 = tri_pts[:, 1, 0]
        y2 = tri_pts[:, 1, 1]

        x3 = tri_pts[:, 2, 0]
        y3 = tri_pts[:, 2, 1]

        det_J = (
            (x2 - x1) * (y3 - y1)
            - (x3 - x1) * (y2 - y1)
        )

        if np.any(det_J <= 1e-14):
            raise ValueError(
                "Mesh non valida nel test di convergenza L2: "
                "sono presenti elementi invertiti o degeneri."
            )

        areas = 0.5 * det_J
        u_elem = u_approx[cells]

        qp_ref, qw_ref = _get_triangle_quadrature_degree_5()

        err_sq = 0.0
        ref_sq = 0.0

        for (xi, eta), w in zip(qp_ref, qw_ref):
            phi1 = 1.0 - xi - eta
            phi2 = xi
            phi3 = eta

            # Coordinate fisiche dei punti di quadratura.
            xq = phi1 * x1 + phi2 * x2 + phi3 * x3
            yq = phi1 * y1 + phi2 * y2 + phi3 * y3

            # Soluzione analitica continua.
            u_exact = pde.evaluate_solution(xq, yq, t)

            # Interpolazione P1 della soluzione rimappata.
            u_h = (
                phi1 * u_elem[:, 0]
                + phi2 * u_elem[:, 1]
                + phi3 * u_elem[:, 2]
            )

            factor = 2.0 * w * areas

            err_sq += np.sum(
                factor * (u_h - u_exact) ** 2
            )

            ref_sq += np.sum(
                factor * u_exact ** 2
            )

        if ref_sq <= 1e-14:
            raise ValueError(
                "Norma L2 della soluzione di riferimento "
                "troppo piccola per calcolare l'errore relativo."
            )

        return float(
            np.sqrt(max(err_sq, 0.0) / ref_sq)
        )

    # -------------------------------------------------------------
    # Esperimenti su mesh progressivamente più fini
    # -------------------------------------------------------------
    for n in resolutions:
        h = 1.0 / n

        mesh = create_initial_mesh_2d(
            {
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
            },
            nx=n,
            ny=n,
        )

        u_old = pde.evaluate_solution(
            mesh.nodes[:, 0],
            mesh.nodes[:, 1],
            t,
        )

        disp = np.zeros_like(mesh.nodes)
        free_idx = _get_free_indices(mesh)

        disp[free_idx, 0] = (
            0.25 * h
            * np.sin(np.pi * mesh.nodes[free_idx, 0])
        )

        disp[free_idx, 1] = (
            0.25 * h
            * np.cos(np.pi * mesh.nodes[free_idx, 1])
        )

        eps_h = min(1e-4, 0.05 * h**2)

        _, safe_nodes, is_safe, _ = apply_safety_filter(
            mesh.nodes,
            disp,
            mesh.cells,
            epsilon=eps_h,
        )

        assert is_safe, (
            f"Safety filter fallito per la mesh {n}x{n}."
        )

        # Mesh geometrica di destinazione.
        mesh_new = mesh.update_coordinates(safe_nodes)

        # Remapping pointwise P1.
        u_p1 = remap_p1_solution(
            old_nodes=mesh.nodes,
            old_cells=mesh.cells,
            old_u=u_old,
            new_nodes=safe_nodes,
        )

        # Proiezione L2 con integrazione sulle intersezioni.
        u_l2 = remap_l2_projection(
            old_nodes=mesh.nodes,
            old_cells=mesh.cells,
            old_u=u_old,
            new_nodes=safe_nodes,
            new_cells=mesh.cells,
        )

        # Errori L2 relativi continui, non errori nodali.
        err_p1 = relative_continuous_l2_error(
            mesh_new, u_p1
        )

        err_l2 = relative_continuous_l2_error(
            mesh_new, u_l2
        )

        err_p1_list.append(err_p1)
        err_l2_list.append(err_l2)

    # -------------------------------------------------------------
    # Calcolo dei ratei sperimentali
    # -------------------------------------------------------------
    rates_p1 = [
        np.log(err_p1_list[i] / err_p1_list[i + 1])
        / np.log(h_vals[i] / h_vals[i + 1])
        for i in range(len(h_vals) - 1)
    ]

    rates_l2 = [
        np.log(err_l2_list[i] / err_l2_list[i + 1])
        / np.log(h_vals[i] / h_vals[i + 1])
        for i in range(len(h_vals) - 1)
    ]

    print("\n" + "=" * 82)
    print("L2 — CONTINUOUS SPATIAL CONVERGENCE: P1 vs L2 PROJECTION")
    print("=" * 82)

    print(
        f"{'Mesh':<10} {'h':<10} "
        f"{'P1 rel. L2':<15} {'p(P1)':<10} "
        f"{'Proj. rel. L2':<17} {'p(Proj.)':<10}"
    )

    print("-" * 82)

    for i, n in enumerate(resolutions):
        rate_p1 = "-" if i == 0 else f"{rates_p1[i - 1]:.2f}"
        rate_l2 = "-" if i == 0 else f"{rates_l2[i - 1]:.2f}"

        print(
            f"{n}x{n:<6} "
            f"{h_vals[i]:<10.5f} "
            f"{err_p1_list[i]:<15.6e} "
            f"{rate_p1:<10} "
            f"{err_l2_list[i]:<17.6e} "
            f"{rate_l2:<10}"
        )

    # -------------------------------------------------------------
    # Controlli minimi automatici
    # -------------------------------------------------------------
    for name, errors in [
        ("P1 pointwise", err_p1_list),
        ("L2 projection", err_l2_list),
    ]:
        assert np.all(np.isfinite(errors)), (
            f"{name}: sono stati rilevati errori non finiti."
        )

        assert np.all(np.asarray(errors) > 0.0), (
            f"{name}: errori nulli o non validi."
        )

        assert errors[-1] < errors[0], (
            f"{name}: la mesh più fine non riduce "
            "l'errore L2 rispetto alla mesh più grossolana."
        )

def test_l3_displacement_amplitude():
    """L3: Errore vs Displacement Amplitude: P1 vs L2 Projection."""
    mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=32, ny=32)
    pde = ManufacturedConvectionDiffusion2D()
    t = 0.25
    u_old = pde.evaluate_solution(mesh.nodes[:, 0], mesh.nodes[:, 1], t)
    free_idx = _get_free_indices(mesh)

    gammas = [0.10, 0.25, 0.50, 1.00, 2.00]
    base_amp = 0.005

    print("\n" + "=" * 65)
    print("L3 — DISPLACEMENT AMPLITUDE: P1 vs L2 PROJECTION")
    print("=" * 65)
    print(f"{'gamma':<8} {'Disp Max':<14} {'P1 Error [%]':<16} {'L2-Proj Error [%]':<16}")
    print("-" * 65)

    for g in gammas:
        disp = np.zeros_like(mesh.nodes)
        disp[free_idx, 0] = g * base_amp * np.sin(2 * np.pi * mesh.nodes[free_idx, 1])
        disp[free_idx, 1] = g * base_amp * np.cos(2 * np.pi * mesh.nodes[free_idx, 0])

        _, safe_nodes, is_safe, _ = apply_safety_filter(mesh.nodes, disp, mesh.cells, epsilon=1e-4)
        assert is_safe

        actual_max_disp = float(np.max(np.linalg.norm(safe_nodes - mesh.nodes, axis=1)))
        u_p1 = remap_p1_solution(mesh.nodes, mesh.cells, u_old, safe_nodes)
        u_l2 = remap_l2_projection(mesh.nodes, mesh.cells, u_old, safe_nodes, mesh.cells)
        u_exact = pde.evaluate_solution(safe_nodes[:, 0], safe_nodes[:, 1], t)

        norm_ex = np.linalg.norm(u_exact)
        err_p1 = float(np.linalg.norm(u_p1 - u_exact) / norm_ex)
        err_l2 = float(np.linalg.norm(u_l2 - u_exact) / norm_ex)

        print(f"{g:<8.2f} {actual_max_disp:<14.4e} {err_p1*100:<16.3f} {err_l2*100:<16.3f}")


def test_l4_rigorous_cumulative_remapping_fixed_target():
    """
    L4: Test rigoroso di accumulo:
    Fissiamo la mesh target finale X_target e dividiamo lo spostamento in K step intermedi.
    Confrontiamo: P1 cumulativo vs L2-Projection cumulativo vs Remap Diretto.
    """
    mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=32, ny=32)
    pde = ManufacturedConvectionDiffusion2D()
    t = 0.1
    u_exact_init = pde.evaluate_solution(mesh.nodes[:, 0], mesh.nodes[:, 1], t)
    free_idx = _get_free_indices(mesh)

    # 1. Definizione dello spostamento totale target
    total_disp = np.zeros_like(mesh.nodes)
    total_disp[free_idx, 0] = 0.015 * np.sin(np.pi * mesh.nodes[free_idx, 0]) * np.sin(np.pi * mesh.nodes[free_idx, 1])
    total_disp[free_idx, 1] = 0.015 * np.cos(np.pi * mesh.nodes[free_idx, 0]) * np.sin(np.pi * mesh.nodes[free_idx, 1])

    _, final_target_nodes, is_safe, _ = apply_safety_filter(mesh.nodes, total_disp, mesh.cells, epsilon=1e-4)
    assert is_safe

    u_exact_final = pde.evaluate_solution(final_target_nodes[:, 0], final_target_nodes[:, 1], t)
    norm_final = np.linalg.norm(u_exact_final)

    # Remap diretti iniziali (1 step)
    u_dir_p1 = remap_p1_solution(mesh.nodes, mesh.cells, u_exact_init, final_target_nodes)
    u_dir_l2 = remap_l2_projection(mesh.nodes, mesh.cells, u_exact_init, final_target_nodes, mesh.cells)
    err_dir_p1 = float(np.linalg.norm(u_dir_p1 - u_exact_final) / norm_final)
    err_dir_l2 = float(np.linalg.norm(u_dir_l2 - u_exact_final) / norm_final)

    k_steps = [1, 2, 5, 10]

    print("\n" + "=" * 75)
    print("L4 — RIGOROUS CUMULATIVE REMAP (FIXED FINAL TARGET X_target)")
    print("=" * 75)
    print(f"{'K Steps':<10} {'P1 Cumul [%]':<15} {'P1 Amp':<10} {'L2 Cumul [%]':<15} {'L2 Amp':<10}")
    print("-" * 75)

    for K in k_steps:
        # Omotopia lineare: nodi intermedi X_k = X_0 + (k/K) * (X_target - X_0)
        curr_nodes_p1 = mesh.nodes.copy()
        curr_nodes_l2 = mesh.nodes.copy()
        u_k_p1 = u_exact_init.copy()
        u_k_l2 = u_exact_init.copy()

        for step in range(1, K + 1):
            next_nodes = mesh.nodes + (step / K) * (final_target_nodes - mesh.nodes)

            u_k_p1 = remap_p1_solution(curr_nodes_p1, mesh.cells, u_k_p1, next_nodes)
            u_k_l2 = remap_l2_projection(curr_nodes_l2, mesh.cells, u_k_l2, next_nodes, mesh.cells)

            curr_nodes_p1 = next_nodes
            curr_nodes_l2 = next_nodes

        err_cum_p1 = float(np.linalg.norm(u_k_p1 - u_exact_final) / norm_final)
        err_cum_l2 = float(np.linalg.norm(u_k_l2 - u_exact_final) / norm_final)
        amp_p1 = err_cum_p1 / err_dir_p1
        amp_l2 = err_cum_l2 / err_dir_l2

        print(f"{K:<10d} {err_cum_p1*100:<15.3f} {amp_p1:<10.2f}x {err_cum_l2*100:<15.3f} {amp_l2:<10.2f}x")


def test_l5_computational_cost():
    """L5: Confronto cronometrico per singolo step di remapping su 32x32."""
    mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=32, ny=32)
    pde = ManufacturedConvectionDiffusion2D()
    u_old = pde.evaluate_solution(mesh.nodes[:, 0], mesh.nodes[:, 1], 0.0)

    disp = np.zeros_like(mesh.nodes)
    free_idx = _get_free_indices(mesh)
    disp[free_idx] = 0.005
    _, safe_nodes, _, _ = apply_safety_filter(mesh.nodes, disp, mesh.cells, epsilon=1e-4)

    # Warmup
    _ = remap_p1_solution(mesh.nodes, mesh.cells, u_old, safe_nodes)
    _ = remap_l2_projection(mesh.nodes, mesh.cells, u_old, safe_nodes, mesh.cells)

    # Benchmark P1
    t0 = time.perf_counter()
    n_runs = 5
    for _ in range(n_runs):
        _ = remap_p1_solution(mesh.nodes, mesh.cells, u_old, safe_nodes)
    t_p1 = (time.perf_counter() - t0) / n_runs * 1000.0

    # Benchmark L2
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = remap_l2_projection(mesh.nodes, mesh.cells, u_old, safe_nodes, mesh.cells)
    t_l2 = (time.perf_counter() - t0) / n_runs * 1000.0

    print("\n" + "=" * 65)
    print("L5 — COMPUTATIONAL RUNTIME (32x32 MESH)")
    print("=" * 65)
    print(f"P1 Pointwise Remap Time : {t_p1:.2f} ms")
    print(f"L2-Projection Remap Time: {t_l2:.2f} ms")
    print(f"Rapporto di costo T_L2 / T_P1: {t_l2 / t_p1:.2f}x")

def test_l6_integral_mass_conservation():
    """
    L6: Verifica conservazione della massa integrale per la L2-projection.
    Verifica sia l'identità variazionale a precisione macchina:
        1^T M^{new} u_{new} == 1^T b = int_Omega u_{old} dOmega
    sia la conservazione fisica rispetto alla massa discreta iniziale (< 0.05%).
    """
    from numerics.remap import compute_mesh_mass
    from numerics.remap_l2 import assemble_l2_projection_system

    mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=32, ny=32)
    pde = ManufacturedConvectionDiffusion2D()
    u_old = pde.evaluate_solution(mesh.nodes[:, 0], mesh.nodes[:, 1], 0.1)

    disp = np.zeros_like(mesh.nodes)
    free_idx = _get_free_indices(mesh)
    disp[free_idx, 0] = 0.008 * np.sin(2 * np.pi * mesh.nodes[free_idx, 1])
    disp[free_idx, 1] = 0.008 * np.cos(2 * np.pi * mesh.nodes[free_idx, 0])

    _, safe_nodes, is_safe, _ = apply_safety_filter(mesh.nodes, disp, mesh.cells, epsilon=1e-4)
    assert is_safe

    # 1. Risoluzione L2-projection e conservazione variazionale interna
    M_new, b = assemble_l2_projection_system(mesh.nodes, mesh.cells, u_old, safe_nodes, mesh.cells)
    import scipy.sparse.linalg as spla
    u_new_l2 = spla.spsolve(M_new, b)

    mass_projected_exact = float(np.sum(b))
    mass_numerical_fem = float(np.sum(M_new @ u_new_l2))

    # Identità variazionale esatta (risoluzione del sistema M * u = b)
    var_rel_err = abs(mass_numerical_fem - mass_projected_exact) / abs(mass_projected_exact)
    np.testing.assert_allclose(mass_numerical_fem, mass_projected_exact, rtol=1e-12)

    # 2. Conservazione fisica rispetto alla massa iniziale calcolata su mesh_old
    mass_old = compute_mesh_mass(mesh, u_old)
    mesh_new = mesh.update_coordinates(safe_nodes)
    mass_new = compute_mesh_mass(mesh_new, u_new_l2)

    rel_mass_err = abs(mass_new - mass_old) / abs(mass_old)

    print("\n" + "=" * 65)
    print("L6 — MASS CONSERVATION TEST (L2-PROJECTION)")
    print("=" * 65)
    print(f"Massa Discreta Iniziale (1-point)  : {mass_old:.8e}")
    print(f"Massa Proiettata Esatta (Dunavant): {mass_projected_exact:.8e}")
    print(f"Massa Finale Calcolata (FEM)       : {mass_numerical_fem:.8e}")
    print(f"Errore Variazionale Interno        : {var_rel_err:.4e}")
    print(f"Scarto Relativo di Discretizzazione: {rel_mass_err:.4e} ({rel_mass_err*100:.3f}%)")

    # Il sistema garantisce identità variazionale e scarto fisico al di sotto dello 0.05%
    assert rel_mass_err < 5e-4, f"Conservazione di massa fisica oltre la tolleranza: {rel_mass_err:.3e}"