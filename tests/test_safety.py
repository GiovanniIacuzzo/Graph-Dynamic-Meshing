"""
tests/test_safety.py
Test deterministici di non-inversione e gestione dei casi limite del safety filter.
"""

import numpy as np
import pytest
from geometry.safety import apply_safety_filter
from geometry.geometry import compute_mesh_metrics


def test_safety_filter_zero_displacement():
    """Con spostamento nullo e mesh valida il filtro deve restituire alpha_global = 1.0 e is_safe = True."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)
    disp = np.zeros_like(nodes)

    alpha, nodes_safe, is_safe, min_j = apply_safety_filter(nodes, disp, cells, epsilon=1e-4)

    assert alpha == 1.0
    assert is_safe
    assert np.isclose(min_j, 1.0, atol=1e-12)
    np.testing.assert_allclose(nodes_safe, nodes)


def test_safety_filter_rejects_invalid_initial_mesh_even_with_zero_displacement():
    """
    Se la mesh iniziale viola già J_e(0) >= epsilon, il filtro deve sollevare ValueError
    anche quando lo spostamento proposto è identicamente nullo.
    """
    # Triangolo invertito (J_0 = -1.0)
    nodes_inv = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)
    disp_zero = np.zeros_like(nodes_inv)

    with pytest.raises(ValueError, match="Mesh iniziale non ammissibile"):
        apply_safety_filter(nodes_inv, disp_zero, cells, epsilon=1e-4)

    # Triangolo positivo ma sotto soglia (J_0 = 0.5 * 1e-4 < epsilon = 1e-4)
    nodes_sub = np.array([[0.0, 0.0], [1e-2, 0.0], [0.0, 0.5e-2]], dtype=np.float64)
    with pytest.raises(ValueError, match="Mesh iniziale non ammissibile"):
        apply_safety_filter(nodes_sub, disp_zero, cells, epsilon=1e-4)


def test_safety_filter_admissible_displacement():
    """Spostamenti moderati che mantengono J_e > epsilon non devono essere limitati."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)
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

    disp = np.zeros_like(nodes)
    disp[0] = [2.0, 2.0]

    epsilon = 1e-3
    alpha, nodes_safe, is_safe, min_j = apply_safety_filter(
        nodes, disp, cells, epsilon=epsilon, safety_margin=1e-6
    )

    assert not is_safe
    assert 0.0 < alpha < 1.0
    metrics = compute_mesh_metrics(nodes_safe, cells)
    assert metrics["jacobians"][0] >= epsilon
    assert min_j >= epsilon
    assert not metrics["inverted"][0]


def test_safety_filter_linear_regime_and_quadratic_regime():
    """
    Verifica che il filtro gestisca separatamente elementi con dinamica lineare (|A_e| ~ 0)
    ed elementi con dinamica quadratica genuina (|A_e| > 0).
    """
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float64)
    cells = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=np.int64)

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
    assert np.all(metrics["jacobians"] >= epsilon)
    assert not np.any(metrics["inverted"])


def test_safety_filter_tiny_positive_root():
    """
    Verifica che una radice positiva estremamente piccola (r ~ 1e-16) non venga
    scartata lasciando passare alpha = 1.0 con conseguente inversione dell'elemento.
    """
    epsilon = 1e-4
    # Costruiamo un triangolo con J_0 = epsilon + 1e-16
    h_y = epsilon + 1e-16
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, h_y]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)

    # Spostamento del terzo vertice verso y = -1.0 -> B = -1.0, A = 0, radice r = 1e-16
    disp = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, -1.0]], dtype=np.float64)

    alpha, nodes_safe, is_safe, min_j = apply_safety_filter(
        nodes, disp, cells, epsilon=epsilon
    )

    assert not is_safe
    assert 0.0 <= alpha < 1e-14
    assert min_j >= epsilon - 1e-12


def test_safety_filter_near_zero_discriminant_tangent():
    """
    Verifica il caso quadratico con discriminante prossimo a zero (parabola con A > 0
    che tocca la soglia epsilon all'interno dell'intervallo (0, 1)).
    """
    epsilon = 0.2
    # Triangolo iniziale con J_0 = 1.0 -> C_tilde = 1.0 - 0.2 = 0.8
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)

    # Scegliamo du1 = -s, dv2 = -s su P2 e P3 -> J(alpha) = (1 - s*alpha)^2
    # Vogliamo che il minimo tocchi epsilon = 0.2 leggermente prima o alla tangenza:
    # Con du1 = -1.8, dv2 = -1.8 -> J(alpha) = (1 - 1.8*alpha)^2, che si annulla a alpha = 1/1.8 ~ 0.555
    disp = np.array([[0.0, 0.0], [-1.8, 0.0], [0.0, -1.8]], dtype=np.float64)

    alpha, nodes_safe, is_safe, min_j = apply_safety_filter(
        nodes, disp, cells, epsilon=epsilon, safety_margin=1e-6
    )

    expected_root = (1.0 - np.sqrt(epsilon)) / 1.8
    assert not is_safe
    assert np.isclose(alpha, expected_root * (1.0 - 1e-6), rtol=1e-5)
    assert min_j >= epsilon


if __name__ == "__main__":
    pytest.main([__file__, "-v"])