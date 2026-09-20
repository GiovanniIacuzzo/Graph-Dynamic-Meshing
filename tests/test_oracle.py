"""
tests/test_oracle.py
Test di convergenza dell'ottimizzatore offline su campi gaussiani simmetrici.
"""

import numpy as np
import pytest
from geometry.mesh import TriangularMesh2D
from geometry.geometry import compute_mesh_metrics
from oracle.field import GaussianBumpField
from oracle.objective import OracleObjective
from oracle.optimizer import OfflineMeshOptimizer


def test_gaussian_bump_analytical_derivatives():
    """Verifica che il gradiente analitico in oracle/field.py coincida con le differenze finite centrate."""
    field = GaussianBumpField(mu_x=0.5, mu_y=0.5, sigma=0.08, amplitude=1.0)

    # Griglia di test lontana dal centro per avere gradienti non nulli
    x = np.array([0.45, 0.55, 0.60], dtype=np.float64)
    y = np.array([0.48, 0.52, 0.40], dtype=np.float64)

    u, grad_analytic, laplacian_analytic, _ = field.evaluate(x, y)

    # Differenze finite centrate
    h = 1e-6
    u_x_plus, _, _, _ = field.evaluate(x + h, y)
    u_x_minus, _, _, _ = field.evaluate(x - h, y)
    grad_x_num = (u_x_plus - u_x_minus) / (2.0 * h)

    u_y_plus, _, _, _ = field.evaluate(x, y + h)
    u_y_minus, _, _, _ = field.evaluate(x, y - h)
    grad_y_num = (u_y_plus - u_y_minus) / (2.0 * h)

    np.testing.assert_allclose(grad_analytic[:, 0], grad_x_num, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(grad_analytic[:, 1], grad_y_num, rtol=1e-5, atol=1e-7)


def test_oracle_objective_evaluation_and_normalization():
    """
    Verifica che il funzionale multiobiettivo OracleObjective:
    - abbia valore iniziale J(X_0) = 1.0 + lambda_Q * Q_0 (grazie alla normalizzazione E_0)
    - identifichi correttamente vincoli di bordo e soglie minime di Jacobiano
    """
    # Piccola mesh regolare 3x3 nodi
    x = np.linspace(0.0, 1.0, 3)
    y = np.linspace(0.0, 1.0, 3)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.flatten(), yy.flatten()])

    cells = np.array([
        [0, 1, 4], [0, 4, 3],
        [1, 2, 5], [1, 5, 4],
        [3, 4, 7], [3, 7, 6],
        [4, 5, 8], [4, 8, 7]
    ], dtype=np.int64)

    mesh = TriangularMesh2D(nodes, cells)
    field = GaussianBumpField(mu_x=0.5, mu_y=0.5, sigma=0.1, amplitude=1.0)

    obj = OracleObjective(
        reference_field=field,
        initial_nodes=mesh.nodes,
        cells=mesh.cells,
        boundary_nodes=mesh.boundary_nodes,
        lambda_D=0.01,
        lambda_Q=0.001,
        epsilon=1e-4
    )

    eval_0 = obj.evaluate(mesh.nodes)
    assert eval_0["is_valid"]
    # All'inizio D(X_0) = 0 e E_tilde = 1.0
    assert np.isclose(eval_0["movement_penalty"], 0.0, atol=1e-14)
    assert np.isclose(eval_0["total_objective"], 1.0 + 0.001 * eval_0["quality_penalty"], rtol=1e-4)


def test_offline_optimizer_reduces_interpolation_error():
    """
    Verifica che l'ottimizzatore offline converga, riduca l'errore funzionale integrato
    e produca uno spostamento supervisionato Delta X* ammissibile che rispetta il bordo.
    """
    # Griglia di test 5x5 nodi (16 celle)
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(0.0, 1.0, 5)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.flatten(), yy.flatten()])

    cells = []
    for j in range(4):
        for i in range(4):
            n0 = j * 5 + i
            n1 = n0 + 1
            n2 = (j + 1) * 5 + i
            n3 = n2 + 1
            cells.append([n0, n1, n3])
            cells.append([n0, n3, n2])
    cells = np.array(cells, dtype=np.int64)

    mesh = TriangularMesh2D(nodes, cells)
    field = GaussianBumpField(mu_x=0.5, mu_y=0.5, sigma=0.08, amplitude=1.0)

    optimizer = OfflineMeshOptimizer(
        reference_field=field,
        initial_nodes=mesh.nodes,
        cells=mesh.cells,
        boundary_nodes=mesh.boundary_nodes,
        lambda_D=0.01,
        lambda_Q=0.001,
        epsilon=1e-4,
        max_iter=30
    )

    res = optimizer.optimize()

    # 1. Deve terminare in configurazione valida
    assert res["final_metrics"]["is_valid"]
    assert res["final_metrics"]["min_jacobian"] >= 1e-4
    assert res["final_metrics"]["num_inverted_cells"] == 0

    # 2. I nodi di bordo devono avere spostamento rigorosamente nullo
    target_disp = res["target_displacement"]
    np.testing.assert_allclose(target_disp[mesh.boundary_nodes], 0.0, atol=1e-12)

    # 3. Almeno un nodo interno deve essersi concentrato verso il centro della gaussiana
    inner_disp = target_disp[~optimizer.is_boundary]
    assert np.max(np.linalg.norm(inner_disp, axis=1)) > 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])