"""Manufactured-solution API checks executed against the C++/deal.II implementation."""
from __future__ import annotations

import numpy as np

from tests.native_test_support import create_pde_model


def test_native_mms_solution_and_derivatives_are_consistent():
    pde = create_pde_model(bx=0.4, by=0.2, nu=1.0e-3, sigma=0.12, beta=0.5, T=0.5)
    x = np.array([0.31, 0.44, 0.63], dtype=np.float64)
    y = np.array([0.27, 0.52, 0.61], dtype=np.float64)
    t = 0.12
    u, ux, uy, uxx, uyy, ut, forcing = pde.evaluate_exact(x, y, t)
    h = 1.0e-5
    u_xp = pde.evaluate_solution(x + h, y, t)
    u_xm = pde.evaluate_solution(x - h, y, t)
    u_yp = pde.evaluate_solution(x, y + h, t)
    u_ym = pde.evaluate_solution(x, y - h, t)
    u_tp = pde.evaluate_solution(x, y, t + h)
    u_tm = pde.evaluate_solution(x, y, t - h)
    np.testing.assert_allclose(ux, (u_xp - u_xm) / (2*h), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(uy, (u_yp - u_ym) / (2*h), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(ut, (u_tp - u_tm) / (2*h), rtol=2e-5, atol=2e-6)
    assert np.all(np.isfinite(np.concatenate([u, ux, uy, uxx, uyy, ut, forcing])))
    assert pde.nu == 1.0e-3 and pde.T == 0.5


def test_native_mms_vector_and_xy_apis_agree():
    pde = create_pde_model(sigma=0.1, T=0.5)
    nodes = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]], dtype=np.float64)
    np.testing.assert_allclose(pde.evaluate_solution(nodes, 0.1), pde.evaluate_solution(nodes[:, 0], nodes[:, 1], 0.1))
