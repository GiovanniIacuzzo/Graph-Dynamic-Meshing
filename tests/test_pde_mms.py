"""
tests/test_pde_mms.py
Test di consistenza algebrica ed esattezza differenziale per la Manufactured Solution (MMS).
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from numerics.pde import ManufacturedConvectionDiffusion2D


def test_mms_exact_residual_vanishes():
    """
    Verifica che su punti casuali interni a [0, 1]^2 e tempi arbitrari
    il residuo analitico (ut + b · grad(u) - nu * Laplacian(u) - f) sia nullo
    entro la precisione di macchina (1e-14).
    """
    pde = ManufacturedConvectionDiffusion2D(
        bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=0.5
    )

    rng = np.random.default_rng(seed=42)
    N_points = 500
    x = rng.uniform(0.01, 0.99, size=N_points)
    y = rng.uniform(0.01, 0.99, size=N_points)

    for t in [0.0, 0.125, 0.25, 0.375, 0.5]:
        u, ux, uy, uxx, uyy, ut, f = pde.evaluate_exact(x, y, t)

        convection = pde.bx * ux + pde.by * uy
        diffusion = pde.nu * (uxx + uyy)
        residual = ut + convection - diffusion - f

        max_residual = np.max(np.abs(residual))
        assert max_residual < 1e-14, f"Residuo analitico non nullo al tempo t={t}: {max_residual:.3e}"


def test_mms_boundary_conditions_zero_dirichlet():
    """
    Verifica che per costruzione la soluzione esatta soddisfi u(x, y, t) = 0
    lungo tutti e 4 i bordi di Omega = [0, 1]^2 grazie al termine sin(pi*x)*sin(pi*y).
    """
    pde = ManufacturedConvectionDiffusion2D()
    s = np.linspace(0.0, 1.0, 100)

    for t in [0.0, 0.25, 0.5]:
        # Bordo inferiore (y = 0) e superiore (y = 1)
        u_bottom = pde.evaluate_solution(s, np.zeros_like(s), t)
        u_top = pde.evaluate_solution(s, np.ones_like(s), t)
        # Bordo sinistro (x = 0) e destro (x = 1)
        u_left = pde.evaluate_solution(np.zeros_like(s), s, t)
        u_right = pde.evaluate_solution(np.ones_like(s), s, t)

        assert np.allclose(u_bottom, 0.0, atol=1e-15)
        assert np.allclose(u_top, 0.0, atol=1e-15)
        assert np.allclose(u_left, 0.0, atol=1e-15)
        assert np.allclose(u_right, 0.0, atol=1e-15)


def test_mms_finite_difference_cross_check():
    """
    Cross-check indipendente: verifica le derivate analitiche ut, ux, uy
    tramite differenze finite centrate.
    """
    pde = ManufacturedConvectionDiffusion2D()
    x = np.array([0.4, 0.6])
    y = np.array([0.45, 0.55])
    t = 0.2
    h = 1e-6

    u, ux_an, uy_an, _, _, ut_an, _ = pde.evaluate_exact(x, y, t)

    # Differenze finite su x
    u_x_plus = pde.evaluate_solution(x + h, y, t)
    u_x_minus = pde.evaluate_solution(x - h, y, t)
    ux_num = (u_x_plus - u_x_minus) / (2.0 * h)

    # Differenze finite su y
    u_y_plus = pde.evaluate_solution(x, y + h, t)
    u_y_minus = pde.evaluate_solution(x, y - h, t)
    uy_num = (u_y_plus - u_y_minus) / (2.0 * h)

    # Differenze finite su t
    u_t_plus = pde.evaluate_solution(x, y, t + h)
    u_t_minus = pde.evaluate_solution(x, y, t - h)
    ut_num = (u_t_plus - u_t_minus) / (2.0 * h)

    np.testing.assert_allclose(ux_an, ux_num, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(uy_an, uy_num, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(ut_an, ut_num, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])