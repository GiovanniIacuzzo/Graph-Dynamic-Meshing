from __future__ import annotations

import numpy as np
import pytest

from oracle.mms_field import MMSFieldAtTime


class _AnalyticPDE:
    bx = 0.4
    by = 0.2
    nu = 1.0e-3
    sigma = 0.06
    beta = 0.4
    T = 0.5

    def evaluate_exact(self, x, y, t):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        decay = np.exp(-self.beta * t)
        u = decay * (np.sin(np.pi * x) * np.sin(np.pi * y))
        ux = decay * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
        uy = decay * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
        uxx = -np.pi**2 * u
        uyy = -np.pi**2 * u
        ut = -self.beta * u
        forcing = np.zeros_like(u)
        return u, ux, uy, uxx, uyy, ut, forcing


def test_mms_value_gradient_fast_path_matches_full_evaluation():
    pde = _AnalyticPDE()
    field = MMSFieldAtTime(pde, 0.17)
    x = np.array([0.13, 0.42, 0.79], dtype=np.float64)
    y = np.array([0.21, 0.57, 0.83], dtype=np.float64)

    u_full, grad_full, _lap, _metadata = field.evaluate(x, y)
    u_fast, grad_fast = field.evaluate_value_gradient(x, y)

    np.testing.assert_allclose(u_fast, u_full, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(grad_fast, grad_full, rtol=0.0, atol=1e-14)
    assert u_fast.dtype == np.float64
    assert grad_fast.shape == (x.size, 2)
    assert grad_fast.flags.c_contiguous


def test_mms_value_gradient_rejects_invalid_coordinate_shapes():
    field = MMSFieldAtTime(_AnalyticPDE(), 0.1)
    with pytest.raises(ValueError, match="vettori 1D"):
        field.evaluate_value_gradient(np.zeros((2, 1)), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="vettori 1D"):
        field.evaluate_value_gradient(np.zeros(2), np.zeros(3))
