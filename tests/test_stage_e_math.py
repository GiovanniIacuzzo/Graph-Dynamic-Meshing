"""Pure-Python unit tests for Stage-E diagnostic identities (no native extension)."""
from __future__ import annotations

import math

import numpy as np

from experiments.stage_e.common import (
    directional_element_metrics,
    l2_p1_difference,
    maximum_element_diameter,
    number_of_steps,
    trapezoid_average,
)


class _Mesh:
    def __init__(self, nodes, cells):
        self.nodes = np.asarray(nodes, dtype=float)
        self.cells = np.asarray(cells, dtype=np.int64)


def test_directional_length_and_element_peclet_on_reference_triangle():
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]])
    cells = np.array([[0, 1, 2]], dtype=np.int64)
    beta = np.array([[1., 0.]])
    h, pe = directional_element_metrics(nodes, cells, beta, nu=0.5)
    # For beta=(1,0), sum_i |beta.grad(phi_i)|=2 -> h_beta=1.
    assert math.isclose(h[0], 1.0, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(pe[0], 1.0, rel_tol=0.0, abs_tol=1e-14)


def test_zero_relative_velocity_returns_zero_peclet_and_finite_length():
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]])
    cells = np.array([[0, 1, 2]], dtype=np.int64)
    h, pe = directional_element_metrics(nodes, cells, np.zeros((1, 2)), nu=1e-3)
    assert math.isclose(h[0], math.sqrt(2.0), rel_tol=0.0, abs_tol=1e-14)
    assert pe[0] == 0.0


def test_exact_p1_l2_difference_on_single_triangle():
    mesh = _Mesh([[0., 0.], [1., 0.], [0., 1.]], [[0, 1, 2]])
    value = l2_p1_difference(mesh, np.array([1., 0., 0.]), np.zeros(3))
    # Integral of phi_0^2 over area 1/2 is area/6=1/12.
    assert math.isclose(value, math.sqrt(1.0/12.0), rel_tol=0.0, abs_tol=1e-14)


def test_time_grid_and_trapezoidal_average():
    assert number_of_steps(0.5, 0.01) == 50
    assert math.isclose(trapezoid_average([0., .5, 1.], [0., 1., 2.], 1.0), 1.0)


def test_maximum_element_diameter_uses_physical_triangle_diameter():
    mesh = _Mesh([[0., 0.], [1., 0.], [0., 1.]], [[0, 1, 2]])
    assert math.isclose(maximum_element_diameter(mesh), math.sqrt(2.0), rel_tol=0.0, abs_tol=1e-14)
