from __future__ import annotations
import numpy as np
import pytest

pytest.importorskip("gdm._gdm_core", reason="Test d'integrazione nativa: compilare prima il core deal.II.", exc_type=ImportError)
from gdm import NativeMeshAdapter, create_pde_model
from oracle.mms_field import MMSFieldAtTime
from oracle.trajectory_objective import TrajectoryOracleObjective
from oracle.trajectory_optimizer import TrajectoryMeshOptimizer


def _small_mesh(n: int = 4) -> NativeMeshAdapter:
    x = np.linspace(0.0, 1.0, n + 1)
    y = np.linspace(0.0, 1.0, n + 1)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    cells = []
    for j in range(n):
        for i in range(n):
            n0 = j * (n + 1) + i
            n1 = n0 + 1
            n2 = n0 + (n + 1)
            n3 = n2 + 1
            cells.extend(([n0, n1, n3], [n0, n3, n2]))
    return NativeMeshAdapter.from_arrays(nodes, np.asarray(cells, dtype=np.int64))


def test_mms_adapter_matches_exact_solution_and_gradient():
    pde = create_pde_model(sigma=0.12, beta=0.4, T=0.5)
    x = np.array([0.2, 0.45, 0.7], dtype=np.float64)
    y = np.array([0.3, 0.55, 0.4], dtype=np.float64)
    adapter = MMSFieldAtTime(pde, 0.15)
    u, grad, lap, _ = adapter.evaluate(x, y)
    ue, ux, uy, uxx, uyy, _, _ = pde.evaluate_exact(x, y, 0.15)
    np.testing.assert_allclose(u, ue, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(grad[:, 0], ux, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(grad[:, 1], uy, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(lap, uxx + uyy, rtol=0.0, atol=1e-13)


def test_trajectory_objective_directional_gradient_and_boundary_lock():
    mesh = _small_mesh(4)
    pde = create_pde_model(sigma=0.12, beta=0.4, T=0.5)
    objective = TrajectoryOracleObjective(
        pde=pde,
        initial_nodes=mesh.nodes,
        cells=mesh.cells,
        boundary_nodes=mesh.boundary_nodes,
        t_start=0.10,
        horizon=0.05,
        lambda_D=0.01,
        lambda_Q=0.001,
        epsilon=1e-4,
        time_quadrature_order=3,
    )
    rng = np.random.default_rng(123)
    candidate = mesh.nodes.copy()
    is_boundary = mesh.boundary_mask
    candidate[~is_boundary] += 5e-4 * rng.normal(size=candidate[~is_boundary].shape)
    candidate[mesh.boundary_nodes] = mesh.nodes[mesh.boundary_nodes]

    value, gradient = objective.value_and_gradient(candidate)
    assert np.isfinite(value)
    assert gradient.shape == candidate.shape
    assert np.all(np.isfinite(gradient))

    direction = np.zeros_like(candidate)
    direction[~is_boundary] = rng.normal(size=direction[~is_boundary].shape)
    direction /= max(np.linalg.norm(direction), 1e-15)
    h = 2e-6
    plus = candidate + h * direction
    minus = candidate - h * direction
    plus[mesh.boundary_nodes] = mesh.nodes[mesh.boundary_nodes]
    minus[mesh.boundary_nodes] = mesh.nodes[mesh.boundary_nodes]
    fp = objective.value_and_gradient(plus)[0]
    fm = objective.value_and_gradient(minus)[0]
    fd = (fp - fm) / (2.0 * h)
    analytic = float(np.sum(gradient * direction))
    scale = max(1.0, abs(fd), abs(analytic))
    assert abs(fd - analytic) / scale < 5e-3, (fd, analytic)


def test_trajectory_optimizer_returns_valid_mesh_and_keeps_boundary_fixed():
    mesh = _small_mesh(4)
    pde = create_pde_model(sigma=0.12, beta=0.4, T=0.5)
    optimizer = TrajectoryMeshOptimizer(
        pde=pde,
        initial_nodes=mesh.nodes,
        cells=mesh.cells,
        boundary_nodes=mesh.boundary_nodes,
        t_start=0.05,
        horizon=0.05,
        lambda_D=0.02,
        lambda_Q=0.001,
        epsilon=1e-4,
        max_iter=3,
        time_quadrature_order=2,
    )
    result = optimizer.optimize()
    assert result["optimal_nodes"].shape == mesh.nodes.shape
    assert np.all(np.isfinite(result["optimal_nodes"]))
    np.testing.assert_allclose(
        result["optimal_nodes"][mesh.boundary_nodes],
        mesh.nodes[mesh.boundary_nodes],
        rtol=0.0,
        atol=1e-12,
    )
    assert result["final_metrics"]["min_jacobian"] >= 1e-4 - 1e-10
