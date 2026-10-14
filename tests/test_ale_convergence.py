"""
tests/test_ale_convergence.py
Test C3 (Manufactured Verification), C4 (Spatial Convergence su Mesh Deformata), C5 (Temporal Convergence).
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from generate_dataset import create_initial_mesh_2d
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.fem import ConvectionDiffusionFEM2D
from numerics.dynamic_solver import compute_continuous_errors
from numerics.ale import ALESolver2D
from tests.test_ale_gcl import analytical_mesh_motion


def test_c3_ale_one_step_manufactured():
    """Test C3: Verifica a singolo time-step su Manufactured Solution."""
    pde = ManufacturedConvectionDiffusion2D(bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=0.5)
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    mesh_0 = create_initial_mesh_2d(domain, nx=32, ny=32)

    dt = 0.005
    t_n = 0.10
    t_next = t_n + dt

    fem_fixed = ConvectionDiffusionFEM2D(mesh_0, pde)
    u_n_fix = pde.evaluate_solution(mesh_0.nodes[:, 0], mesh_0.nodes[:, 1], t_n)
    u_next_fix = fem_fixed.step_backward_euler(u_n_fix, dt=dt, t_next=t_next)
    err_fix, _ = compute_continuous_errors(mesh_0, u_next_fix, pde, t_next)

    nodes_n = analytical_mesh_motion(mesh_0.nodes, t_n, T=0.5, A=0.015)
    nodes_next = analytical_mesh_motion(mesh_0.nodes, t_next, T=0.5, A=0.015)
    m_n = mesh_0.update_coordinates(nodes_n)
    m_next = mesh_0.update_coordinates(nodes_next)

    ale_solver = ALESolver2D(pde, nu=1e-3, bx=0.4, by=0.2)
    u_n_ale = pde.evaluate_solution(m_n.nodes[:, 0], m_n.nodes[:, 1], t_n)
    u_next_ale, _ = ale_solver.step_ale(m_n, m_next, u_n_ale, dt=dt, t_next=t_next, pure_gcl=False)
    err_ale, _ = compute_continuous_errors(m_next, u_next_ale, pde, t_next)

    print("\n" + "=" * 65)
    print("TEST C3 — ALE ONE-STEP MANUFACTURED SOLUTION VERIFICATION")
    print("=" * 65)
    print(f"Errore L2 Passo Fixed Mesh : {err_fix * 100:.4f}%")
    print(f"Errore L2 Passo ALE Mesh   : {err_ale * 100:.4f}%")

    # Verifica di consistenza rispetto alla mesh fissa
    assert abs(err_ale - err_fix) < 0.02


def test_c4_ale_spatial_convergence():
    """
    Test C4: Convergenza spaziale h partendo da geometria deformata a t_n = 0.10.
    Per evitare che l'errore temporale O(dt) contamini l'ordine spaziale O(h^2),
    il passo dt viene scalato proporzionalmente a h^2.
    """
    pde = ManufacturedConvectionDiffusion2D(bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=0.5)
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    resolutions = [8, 16, 32, 64]
    h_vals = [1.0 / n for n in resolutions]

    t_n = 0.10
    dt_base = 0.0005  # su griglia 8x8, scalato con (h / h_0)^2

    ale_solver = ALESolver2D(pde, nu=1e-3, bx=0.4, by=0.2)
    errors_l2 = []
    errors_h1 = []

    for idx, n in enumerate(resolutions):
        dt = dt_base * ((h_vals[idx] / h_vals[0]) ** 2)
        t_next = t_n + dt

        m0 = create_initial_mesh_2d(domain, nx=n, ny=n)
        nodes_n = analytical_mesh_motion(m0.nodes, t_n, T=0.5, A=0.015)
        nodes_next = analytical_mesh_motion(m0.nodes, t_next, T=0.5, A=0.015)
        m_n = m0.update_coordinates(nodes_n)
        m_next = m0.update_coordinates(nodes_next)

        u_n = pde.evaluate_solution(m_n.nodes[:, 0], m_n.nodes[:, 1], t_n)
        u_1, _ = ale_solver.step_ale(m_n, m_next, u_n, dt=dt, t_next=t_next, pure_gcl=False)

        l2_err, h1_err = compute_continuous_errors(m_next, u_1, pde, t_next)
        errors_l2.append(l2_err)
        errors_h1.append(h1_err)

    rates_l2 = [
        np.log(errors_l2[i] / errors_l2[i + 1]) / np.log(h_vals[i] / h_vals[i + 1])
        for i in range(len(h_vals) - 1)
    ]
    rates_h1 = [
        np.log(errors_h1[i] / errors_h1[i + 1]) / np.log(h_vals[i] / h_vals[i + 1])
        for i in range(len(h_vals) - 1)
    ]

    print("\n" + "=" * 70)
    print("TEST C4 — ALE SPATIAL CONVERGENCE (h-refinement on deformed state)")
    print("=" * 70)
    print(f"{'Mesh':<10} {'h':<10} {'L2 Error':<14} {'p(L2)':<8} {'H1 Error':<14} {'p(H1)':<8}")
    print("-" * 70)
    print(f"{'8x8':<10} {h_vals[0]:<10.4f} {errors_l2[0]:<14.4e} {'-':<8} {errors_h1[0]:<14.4e} {'-':<8}")
    for k in range(len(rates_l2)):
        name = f"{resolutions[k + 1]}x{resolutions[k + 1]}"
        print(
            f"{name:<10} {h_vals[k + 1]:<10.4f} {errors_l2[k + 1]:<14.4e} "
            f"{rates_l2[k]:<8.2f} {errors_h1[k + 1]:<14.4e} {rates_h1[k]:<8.2f}"
        )

    assert rates_l2[-1] >= 1.85, f"Convergenza L2 insufficiente: {rates_l2[-1]:.2f}"
    assert rates_h1[-1] >= 0.90, f"Convergenza H1 insufficiente: {rates_h1[-1]:.2f}"


def test_c5_ale_temporal_convergence():
    """
    Test C5: Convergenza temporale dt via Cauchy Self-Convergence.
    Misura l'errore di auto-convergenza nella norma discreta L2(Omega) indotta
    dalla matrice di massa M(T_eval) sulla configurazione finale della mesh.
    """
    pde = ManufacturedConvectionDiffusion2D(bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=0.5)
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    mesh_0 = create_initial_mesh_2d(domain, nx=32, ny=32)
    nodes_0 = mesh_0.nodes.copy()

    ale_solver = ALESolver2D(pde, nu=1e-3, bx=0.4, by=0.2)
    dt_list = [0.010, 0.005, 0.0025, 0.00125]
    T_eval = 0.02
    solutions_at_T = []

    for dt in dt_list:
        n_steps = int(round(T_eval / dt))
        # Parte esattamente dalla mesh indeformata a t = 0
        curr_mesh = mesh_0.update_coordinates(analytical_mesh_motion(nodes_0, 0.0, T=0.5, A=0.01))
        u_curr = pde.evaluate_solution(curr_mesh.nodes[:, 0], curr_mesh.nodes[:, 1], 0.0)

        for s in range(1, n_steps + 1):
            t_next = s * dt
            nodes_next = analytical_mesh_motion(nodes_0, t_next, T=0.5, A=0.01)
            next_mesh = mesh_0.update_coordinates(nodes_next)
            u_curr, _ = ale_solver.step_ale(curr_mesh, next_mesh, u_curr, dt=dt, t_next=t_next, pure_gcl=False)
            curr_mesh = next_mesh

        solutions_at_T.append(u_curr)

    # Matrice di massa sulla configurazione comune a t = T_eval per la norma L2(Omega)
    final_mesh = mesh_0.update_coordinates(analytical_mesh_motion(nodes_0, T_eval, T=0.5, A=0.01))
    M_final = ale_solver.assemble_mass_matrix(final_mesh)

    def l2_mass_norm(vec: np.ndarray) -> float:
        return float(np.sqrt(max(0.0, float(vec @ (M_final @ vec)))))

    cauchy_errors = []
    for k in range(len(dt_list) - 1):
        diff = solutions_at_T[k] - solutions_at_T[k + 1]
        err = l2_mass_norm(diff) / l2_mass_norm(solutions_at_T[k + 1])
        cauchy_errors.append(err)

    rates_dt = []
    for k in range(len(cauchy_errors) - 1):
        rate = np.log(cauchy_errors[k] / cauchy_errors[k + 1]) / np.log(dt_list[k] / dt_list[k + 1])
        rates_dt.append(rate)

    print("\n" + "=" * 65)
    print("TEST C5 — ALE TEMPORAL CONVERGENCE (Cauchy L2 Mass-Matrix Norm)")
    print("=" * 65)
    print(f"{'dt':<12} {'Cauchy L2(Omega) Error':<24} {'Observed Order':<15}")
    print("-" * 65)
    print(f"{dt_list[0]:<12.5f} {cauchy_errors[0]:<24.4e} {'-':<15}")
    for k in range(len(rates_dt)):
        print(f"{dt_list[k + 1]:<12.5f} {cauchy_errors[k + 1]:<24.4e} {rates_dt[k]:<15.2f}")

    assert rates_dt[-1] >= 0.90, f"Ordine temporale insufficiente: {rates_dt[-1]:.2f}"