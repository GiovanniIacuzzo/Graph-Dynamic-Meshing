from math import factorial

import numpy as np
import pytest

from oracle.quadrature import composite_triangle_rule, dunavant_7_rule


def _exact_monomial_integral(p: int, q: int) -> float:
    # Integral over the reference triangle x>=0, y>=0, x+y<=1.
    return factorial(p) * factorial(q) / factorial(p + q + 2)


@pytest.mark.parametrize("level, expected_points", [(0, 7), (1, 28), (2, 112), (3, 448)])
def test_composite_rule_weights_and_barycentric_coordinates(level: int, expected_points: int) -> None:
    bary, weights = composite_triangle_rule(level)
    assert bary.shape == (expected_points, 3)
    assert weights.shape == (expected_points,)
    assert np.all(np.isfinite(bary)) and np.all(np.isfinite(weights))
    assert np.all(bary >= -1e-14)
    np.testing.assert_allclose(bary.sum(axis=1), 1.0, atol=1e-13)
    assert np.all(weights > 0.0)
    assert np.sum(weights) == pytest.approx(0.5, abs=1e-12)


def test_dunavant7_is_exact_for_polynomials_through_degree_five() -> None:
    bary, weights = dunavant_7_rule()
    x, y = bary[:, 1], bary[:, 2]
    for p in range(6):
        for q in range(6 - p):
            numerical = np.sum(weights * x**p * y**q)
            exact = _exact_monomial_integral(p, q)
            assert numerical == pytest.approx(exact, rel=2e-11, abs=2e-13), (p, q, numerical, exact)


def test_level_above_supported_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        composite_triangle_rule(5)


def test_composite_q7_objective_gradient_matches_finite_differences(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the analytic integration gradient without needing the compiled extension.

    The geometry function is stubbed here only to isolate the Python objective
    calculus. Separate integration tests continue to exercise the native API.
    """
    from oracle import objective as objective_module
    from oracle.field import GaussianBumpField
    from oracle.objective import OracleObjective

    def geometry_metrics(nodes: np.ndarray, cells: np.ndarray) -> dict[str, np.ndarray]:
        tri = np.asarray(nodes, dtype=np.float64)[np.asarray(cells, dtype=np.int64)]
        p1, p2, p3 = tri[:, 0], tri[:, 1], tri[:, 2]
        jac = (
            (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1])
            - (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
        )
        edge_sum_sq = (
            np.sum((p2 - p1) ** 2, axis=1)
            + np.sum((p3 - p2) ** 2, axis=1)
            + np.sum((p1 - p3) ** 2, axis=1)
        )
        quality = np.clip(2.0 * np.sqrt(3.0) * jac / np.maximum(edge_sum_sq, 1e-14), 0.0, 1.0)
        return {
            "jacobians": jac,
            "escobar_quality": quality,
            "aspect_ratio": np.ones_like(jac),
            "inverted": jac <= 0.0,
        }

    monkeypatch.setattr(objective_module, "compute_mesh_metrics", geometry_metrics)
    n = 4
    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, 1.0, n + 1), indexing="xy")
    nodes = np.column_stack((xx.ravel(), yy.ravel()))
    cells = []
    for j in range(n):
        for i in range(n):
            n0 = j * (n + 1) + i
            n1 = n0 + 1
            n2 = (j + 1) * (n + 1) + i
            n3 = n2 + 1
            cells.extend(((n0, n1, n3), (n0, n3, n2)))
    cells_arr = np.asarray(cells, dtype=np.int64)
    boundary = np.flatnonzero(
        np.isclose(nodes[:, 0], 0.0) | np.isclose(nodes[:, 0], 1.0)
        | np.isclose(nodes[:, 1], 0.0) | np.isclose(nodes[:, 1], 1.0)
    )
    free = np.setdiff1d(np.arange(len(nodes)), boundary)
    objective = OracleObjective(
        reference_field=GaussianBumpField(0.43, 0.57, 0.17, 1.1),
        initial_nodes=nodes,
        cells=cells_arr,
        boundary_nodes=boundary,
        quadrature_subdivision_level=2,
    )
    candidate = nodes.copy()
    candidate[free] += np.random.default_rng(123).normal(0.0, 0.003, size=(len(free), 2))
    value, grad = objective.value_and_gradient(candidate)
    assert np.isfinite(value)

    analytic = grad[free].reshape(-1)
    h = 1e-6
    finite_difference = []
    for node_idx in free:
        for dim in range(2):
            plus = candidate.copy()
            minus = candidate.copy()
            plus[node_idx, dim] += h
            minus[node_idx, dim] -= h
            f_plus = objective.evaluate(plus)["total_objective"]
            f_minus = objective.evaluate(minus)["total_objective"]
            finite_difference.append((f_plus - f_minus) / (2.0 * h))
    np.testing.assert_allclose(analytic, np.asarray(finite_difference), rtol=5e-4, atol=1e-7)
