"""Spatial refinement check for the native FEM solver.

The single-step time increment scales as h^2, so temporal truncation error does
not dominate the expected P1 spatial convergence trend.
"""
import numpy as np

from tests.native_test_support import make_mesh, create_pde_model, create_fem_solver, compute_native_continuous_errors


def test_native_fem_error_decreases_under_mesh_refinement():
    errors = []
    pde = create_pde_model(bx=0.4, by=0.2, nu=1e-3, sigma=0.12, beta=0.5, T=0.5)
    for n in (8, 16, 32):
        mesh = make_mesh(n)
        u0 = np.asarray(pde.evaluate_solution(mesh.nodes, 0.0), dtype=np.float64)
        dt = 0.01 * (8.0 / n) ** 2
        solver = create_fem_solver(mesh, pde)
        u1 = np.asarray(solver.step_backward_euler(u0, dt, dt), dtype=np.float64)
        err_l2, _ = compute_native_continuous_errors(mesh, u1, pde, dt)
        assert np.isfinite(err_l2)
        errors.append(err_l2)
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
