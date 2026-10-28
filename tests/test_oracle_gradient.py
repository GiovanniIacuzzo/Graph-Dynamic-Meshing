
import numpy as np
import pytest

from oracle.field import GaussianBumpField
from oracle.optimizer import OfflineMeshOptimizer


def test_oracle_analytic_gradient_matches_finite_differences():
    pytest.importorskip("gdm._gdm_core", reason="Oracle geometry is delegated to the native core; compile deal.II first.", exc_type=ImportError)
    nx = 4
    ny = 4

    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = (j + 1) * (nx + 1) + i
            n3 = n2 + 1

            cells.append([n0, n1, n3])
            cells.append([n0, n3, n2])

    cells = np.asarray(cells, dtype=np.int64)

    boundary_mask = (
        np.isclose(nodes[:, 0], 0.0)
        | np.isclose(nodes[:, 0], 1.0)
        | np.isclose(nodes[:, 1], 0.0)
        | np.isclose(nodes[:, 1], 1.0)
    )
    boundary_nodes = np.flatnonzero(boundary_mask)
    free_nodes = np.flatnonzero(~boundary_mask)

    field = GaussianBumpField(
        mu_x=0.43,
        mu_y=0.57,
        sigma=0.17,
        amplitude=1.1
    )

    optimizer = OfflineMeshOptimizer(
        reference_field=field,
        initial_nodes=nodes,
        cells=cells,
        boundary_nodes=boundary_nodes,
        lambda_D=0.01,
        lambda_Q=0.001,
        epsilon=1e-4,
        max_iter=5
    )
    objective = optimizer.objective

    # Piccola perturbazione interna per evitare una configurazione
    # perfettamente simmetrica durante il confronto numerico.
    candidate = nodes.copy()
    rng = np.random.default_rng(123)
    candidate[free_nodes] += rng.normal(
        loc=0.0,
        scale=0.003,
        size=(len(free_nodes), 2)
    )

    value, grad_full = objective.value_and_gradient(candidate)

    assert np.isfinite(value)
    assert objective.evaluate(candidate)["is_valid"]

    analytic = grad_full[free_nodes].reshape(-1)

    # Differenze finite centrali sulle sole coordinate libere.
    h = 1e-6
    finite_difference = []

    for node_idx in free_nodes:
        for dim in range(2):
            plus = candidate.copy()
            minus = candidate.copy()

            plus[node_idx, dim] += h
            minus[node_idx, dim] -= h

            f_plus = objective.evaluate(plus)["total_objective"]
            f_minus = objective.evaluate(minus)["total_objective"]

            finite_difference.append(
                (f_plus - f_minus) / (2.0 * h)
            )

    finite_difference = np.asarray(finite_difference)

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=5e-4,
        atol=1e-7
    )
