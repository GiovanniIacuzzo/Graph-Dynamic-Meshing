from __future__ import annotations

import numpy as np

from oracle.field import GaussianBumpField


def test_value_gradient_hot_path_matches_full_evaluation() -> None:
    field = GaussianBumpField(mu_x=0.31, mu_y=0.67, sigma=0.021, amplitude=1.17)
    x = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    y = np.linspace(1.0, 0.0, 101, dtype=np.float64)
    u_ref, grad_ref, _, _ = field.evaluate(x, y)
    u_fast, grad_fast = field.evaluate_value_gradient(x, y)
    np.testing.assert_allclose(u_fast, u_ref, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(grad_fast, grad_ref, rtol=2e-15, atol=0.0)
