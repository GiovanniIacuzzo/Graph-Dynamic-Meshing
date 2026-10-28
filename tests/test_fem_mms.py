"""One-step P1 backward-Euler FEM validation through C++/deal.II."""
import numpy as np

from tests.native_test_support import (
    make_mesh, create_pde_model, create_fem_solver,
    compute_native_continuous_errors, compute_native_field_gradients,
)


def test_native_fem_backward_euler_and_gradient_are_finite():
    mesh = make_mesh(24)
    pde = create_pde_model(bx=0.4, by=0.2, nu=1e-3, sigma=0.08, beta=0.5, T=0.5)
    u0 = np.asarray(pde.evaluate_solution(mesh.nodes, 0.0), dtype=np.float64)
    grad_fem = compute_native_field_gradients(mesh, u0, pde, 0.0, exact_gradient=False)
    grad_exact = compute_native_field_gradients(mesh, u0, pde, 0.0, exact_gradient=True)
    assert grad_fem.shape == grad_exact.shape == (mesh.num_nodes, 2)
    assert np.all(np.isfinite(grad_fem)) and np.all(np.isfinite(grad_exact))

    solver = create_fem_solver(mesh, pde)
    dt = 0.0025
    u1 = np.asarray(solver.step_backward_euler(u0, dt, dt), dtype=np.float64)
    assert u1.shape == (mesh.num_nodes,)
    assert np.all(np.isfinite(u1))
    err_l2, err_h1 = compute_native_continuous_errors(mesh, u1, pde, dt)
    assert np.isfinite(err_l2) and np.isfinite(err_h1)
    assert err_l2 < 0.15
    assert err_h1 < 1.5


def test_native_fem_rebuild_after_mesh_motion():
    mesh = make_mesh(12)
    pde = create_pde_model(sigma=0.1, T=0.5)
    u0 = np.asarray(pde.evaluate_solution(mesh.nodes, 0.0), dtype=np.float64)
    solver = create_fem_solver(mesh, pde)
    moved_nodes = mesh.nodes.copy()
    phi = np.sin(np.pi*moved_nodes[:, 0]) * np.sin(np.pi*moved_nodes[:, 1])
    moved_nodes[:, 0] += 0.004 * phi
    moved_nodes[mesh.boundary_nodes] = mesh.nodes[mesh.boundary_nodes]
    moved = mesh.update_coordinates(moved_nodes, enforce_fixed_boundary=True)
    solver.rebuild_for_mesh(moved.core)
    u1 = solver.step_backward_euler(u0, 0.002, 0.002)
    assert np.all(np.isfinite(np.asarray(u1)))
