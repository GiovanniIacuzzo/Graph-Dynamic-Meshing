"""
tests/test_fem_mms.py
Unit test per la correttezza algebrica del solutore FEM P1 e del gradiente nodale.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from generate_dataset import create_initial_mesh_2d
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.fem import ConvectionDiffusionFEM2D, compute_nodal_gradient, compute_field_gradients


def test_fem_matrix_properties_and_mass_conservation():
    """
    Verifica che su mesh unitaria conforme:
    - La somma di tutti gli elementi di M sia pari all'area del dominio (1.0)
    - M e K siano matrici simmetriche
    """
    mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=8, ny=8)
    pde = ManufacturedConvectionDiffusion2D()
    fem = ConvectionDiffusionFEM2D(mesh, pde)

    # 1. Conservazione della massa: integrale(1) = 1.0
    ones = np.ones(mesh.num_nodes)
    total_area = ones @ fem.M @ ones
    assert np.isclose(total_area, 1.0, atol=1e-12)

    # 2. Simmetria di M e K
    diff_M = fem.M - fem.M.T
    diff_K = fem.K - fem.K.T
    assert diff_M.nnz == 0 or np.max(np.abs(diff_M.data)) < 1e-14
    assert diff_K.nnz == 0 or np.max(np.abs(diff_K.data)) < 1e-14


def test_nodal_gradient_linear_exactness():
    """
    Verifica che il gradiente P1 mediato per area riproduca ESATTAMENTE
    il gradiente di un campo lineare: u = 3x - 2y -> grad_u = [3, -2].
    """
    mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=16, ny=16)
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]

    u_linear = 3.0 * x - 2.0 * y
    grad_u = compute_nodal_gradient(mesh.nodes, mesh.cells, u_linear)

    expected_gx = np.full(mesh.num_nodes, 3.0)
    expected_gy = np.full(mesh.num_nodes, -2.0)

    np.testing.assert_allclose(grad_u[:, 0], expected_gx, atol=1e-12)
    np.testing.assert_allclose(grad_u[:, 1], expected_gy, atol=1e-12)


def test_fem_backward_euler_one_step_consistency():
    """
    Esegue un singolo passo Backward Euler con dt=1e-4 a partire dalla soluzione esatta u(t=0)
    e verifica che la soluzione numerica rimanga prossima alla soluzione esatta u(dt).
    """
    mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=32, ny=32)
    pde = ManufacturedConvectionDiffusion2D()
    fem = ConvectionDiffusionFEM2D(mesh, pde)

    u0 = pde.evaluate_solution(mesh.nodes[:, 0], mesh.nodes[:, 1], t=0.0)
    dt = 1e-4
    u1_num = fem.step_backward_euler(u_n=u0, dt=dt, t_next=dt)

    u1_exact = pde.evaluate_solution(mesh.nodes[:, 0], mesh.nodes[:, 1], t=dt)

    # Errore relativo L2 su singolo step piccolo deve essere nell'ordine di O(h^2 + dt)
    l2_err = np.sqrt(np.mean((u1_num - u1_exact) ** 2))
    assert l2_err < 5e-4, f"Errore Backward Euler su singolo passo troppo elevato: {l2_err:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])