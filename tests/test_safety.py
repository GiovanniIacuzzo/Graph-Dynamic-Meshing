"""
tests/test_safety.py
Test deterministici di non-inversione del safety filter sotto deformazioni critiche.
"""

import numpy as np
import pytest
from geometry.safety import apply_safety_filter
from geometry.geometry import compute_mesh_metrics


def test_safety_filter_zero_displacement():
    """Con spostamento nullo il filtro deve restituire alpha_global = 1.0 e is_safe = True."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)
    disp = np.zeros_like(nodes)

    alpha, nodes_safe, is_safe, min_j = apply_safety_filter(nodes, disp, cells, epsilon=1e-4)

    assert alpha == 1.0
    assert is_safe
    assert np.isclose(min_j, 1.0, atol=1e-12)
    np.testing.assert_allclose(nodes_safe, nodes)


def test_safety_filter_admissible_displacement():
    """Spostamenti moderati che mantengono J_e > epsilon non devono essere limitati."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)
    # Traslazione rigida uniforme: il determinante resta inalterato
    disp = np.ones_like(nodes) * 0.25

    alpha, nodes_safe, is_safe, min_j = apply_safety_filter(nodes, disp, cells, epsilon=1e-4)

    assert alpha == 1.0
    assert is_safe
    assert np.isclose(min_j, 1.0, atol=1e-12)


def test_safety_filter_prevents_critical_inversion():
    """
    Imponendo uno spostamento distruttivo che rovescerebbe l'orientamento,
    il filtro deve calcolare alpha_global in (0, 1) preservando min_j >= epsilon.
    """
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)

    # Muove il nodo 0 oltre l'ipotenusa causando J_e < 0
    disp = np.zeros_like(nodes)
    disp[0] = [2.0, 2.0]

    epsilon = 1e-3
    alpha, nodes_safe, is_safe, min_j = apply_safety_filter(
        nodes, disp, cells, epsilon=epsilon, safety_margin=1e-6
    )

    assert not is_safe
    assert 0.0 < alpha < 1.0
    # Verifica a posteriori formale
    metrics = compute_mesh_metrics(nodes_safe, cells)
    assert metrics["jacobians"][0] >= epsilon * (1.0 - 1e-5)
    assert min_j >= epsilon * (1.0 - 1e-5)
    assert not metrics["inverted"][0]


def test_safety_filter_linear_regime_and_quadratic_regime():
    """
    Verifica che il filtro gestisca separatamente elementi con dinamica lineare (|A_e| ~ 0)
    ed elementi con dinamica quadratica genuina (|A_e| > 0).
    """
    # Due celle adiacenti
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float64)
    cells = np.array([
        [0, 1, 2],  # Triangolo 1
        [0, 2, 3]   # Triangolo 2
    ], dtype=np.int64)

    # Spostamento differenziale eterogeneo
    disp = np.array([
        [0.8, 0.8],
        [0.0, 0.0],
        [-0.5, 0.0],
        [0.0, -0.7]
    ], dtype=np.float64)

    epsilon = 1e-4
    alpha, nodes_safe, is_safe, min_j = apply_safety_filter(
        nodes, disp, cells, epsilon=epsilon
    )

    assert not is_safe
    assert 0.0 < alpha < 1.0
    metrics = compute_mesh_metrics(nodes_safe, cells)
    assert np.all(metrics["jacobians"] >= epsilon * (1.0 - 1e-5))
    assert not np.any(metrics["inverted"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])