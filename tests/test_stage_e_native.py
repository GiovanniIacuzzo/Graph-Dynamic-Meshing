"""Native Stage-E regression tests; skipped unless deal.II extension is built."""
from __future__ import annotations

import numpy as np
import pytest

from tests.native_test_support import (
    make_mesh, create_pde_model, create_fem_solver, create_ale_solver,
    compute_native_continuous_errors,
)
from gdm import create_stabilization_parameters


def test_stage_e_supg_fixed_and_static_ale_are_consistent():
    mesh = make_mesh(10)
    pde = create_pde_model(sigma=0.12)
    u0 = np.asarray(pde.evaluate_solution(mesh.nodes, 0.0), dtype=float)
    dt = 0.0025
    solver_cfg = {"right_preconditioning": True, "ale_rhs_norm_floor": 0.0}
    g = create_fem_solver(mesh, pde, stabilization_mode="galerkin", linear_solver_config=solver_cfg)
    s = create_fem_solver(mesh, pde, stabilization_mode="supg", linear_solver_config=solver_cfg)
    result_g = g.step_backward_euler_with_diagnostics(u0, dt, dt)
    result_s = s.step_backward_euler_with_diagnostics(u0, dt, dt)
    ug = np.asarray(result_g["u_next"], dtype=float)
    us = np.asarray(result_s["u_next"], dtype=float)
    assert result_g["linear_solver_residual"] <= 10.0 * result_g["linear_solver_tolerance"]
    assert result_s["linear_solver_residual"] <= 10.0 * result_s["linear_solver_tolerance"]
    assert np.all(np.isfinite(ug)) and np.all(np.isfinite(us))
    assert compute_native_continuous_errors(mesh, us, pde, dt)[0] > 0.0

    ag = create_ale_solver(pde=pde, stabilization_mode="galerkin", linear_solver_config=solver_cfg)
    ass = create_ale_solver(pde=pde, stabilization_mode="supg", linear_solver_config=solver_cfg)
    uag, diag_g = ag.step_ale(mesh.core, mesh.core, u0, dt, dt, False, True)
    uas, diag_s = ass.step_ale(mesh.core, mesh.core, u0, dt, dt, False, True)
    assert diag_g.linear_solver_residual <= 10.0 * diag_g.linear_solver_tolerance
    assert diag_s.linear_solver_residual <= 10.0 * diag_s.linear_solver_tolerance
    assert np.linalg.norm(ug - np.asarray(uag)) < 1.0e-8
    assert np.linalg.norm(us - np.asarray(uas)) < 1.0e-8


def test_stage_e_supg_zero_relative_velocity_is_inactive():
    mesh = make_mesh(6)
    dt = 0.005
    b = np.array([0.4, 0.2])
    moved = mesh.nodes.copy() + dt * b[None, :]
    shifted = mesh.update_coordinates(moved, enforce_fixed_boundary=False)
    x, y = mesh.nodes[:, 0] - .5, mesh.nodes[:, 1] - .5
    u0 = np.exp(-30.0 * (x*x + y*y))
    ag = create_ale_solver(pde=None, nu=1e-3, bx=.4, by=.2, stabilization_mode="galerkin")
    ass = create_ale_solver(pde=None, nu=1e-3, bx=.4, by=.2, stabilization_mode="supg")
    ug, _ = ag.step_ale(mesh.core, shifted.core, u0, dt, dt, False, False)
    us, _ = ass.step_ale(mesh.core, shifted.core, u0, dt, dt, False, False)
    np.testing.assert_allclose(ug, us, rtol=1e-10, atol=1e-11)


def _solve_to_final(mesh, pde, dt: float, mode: str, final_time: float = 0.05) -> np.ndarray:
    steps = int(round(final_time / dt))
    assert abs(steps * dt - final_time) < 1e-12
    solver = create_fem_solver(mesh, pde, stabilization_mode=mode)
    u = np.asarray(pde.evaluate_solution(mesh.nodes, 0.0), dtype=float)
    for k in range(1, steps + 1):
        u = np.asarray(solver.step_backward_euler(u, dt, k * dt), dtype=float)
    return u


def test_stage_e_supg_spatial_refinement_decreases_error():
    from tests.native_test_support import compute_native_continuous_errors

    pde = create_pde_model(sigma=0.12, beta=0.5, T=0.05)
    errors = []
    for n, dt in ((8, 0.00625), (16, 0.0015625)):
        mesh = make_mesh(n)
        u = _solve_to_final(mesh, pde, dt, "supg")
        errors.append(float(compute_native_continuous_errors(mesh, u, pde, 0.05)[0]))
    assert np.isfinite(errors).all()
    assert errors[1] < errors[0]


def test_stage_e_supg_temporal_cauchy_difference_decreases():
    from experiments.stage_e.common import l2_p1_difference

    mesh = make_mesh(8)
    pde = create_pde_model(sigma=0.12, beta=0.5, T=0.05)
    coarse = _solve_to_final(mesh, pde, 0.01, "supg")
    medium = _solve_to_final(mesh, pde, 0.005, "supg")
    fine = _solve_to_final(mesh, pde, 0.0025, "supg")
    coarse_difference = l2_p1_difference(mesh, coarse, medium)
    fine_difference = l2_p1_difference(mesh, medium, fine)
    assert np.isfinite([coarse_difference, fine_difference]).all()
    assert fine_difference < coarse_difference
