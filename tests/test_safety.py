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
    nodes_inv = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)
    disp_zero = np.zeros_like(nodes_inv)

    with pytest.raises(ValueError, match="Mesh iniziale non ammissibile"):
        apply_safety_filter(nodes_inv, disp_zero, cells, epsilon=1e-4)

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
    h_y = epsilon + 1e-16
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, h_y]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)

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
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)

    disp = np.array([[0.0, 0.0], [-1.8, 0.0], [0.0, -1.8]], dtype=np.float64)

    alpha, nodes_safe, is_safe, min_j = apply_safety_filter(
        nodes, disp, cells, epsilon=epsilon, safety_margin=1e-6
    )

    expected_root = (1.0 - np.sqrt(epsilon)) / 1.8
    assert not is_safe
    assert np.isclose(alpha, expected_root * (1.0 - 1e-6), rtol=1e-5)
    assert min_j >= epsilon


def test_safety_filter_boundary_micro_negative_derivative_trajectory():
    """
    Verifica il caso limite in cui un elemento parte esattamente sulla soglia C_e = epsilon,
    con derivata iniziale negativa piccola (-tol_deg < B_e < 0) e curvatura positiva (A_e > 0).
    Anche se a fine passo (alpha = 1) il Jacobiano tornerebbe > epsilon, lungo la traiettoria
    per alpha in (0, -B_e / A_e) si avrebbe J_e(alpha) < epsilon.
    Il filtro deve bloccare lo spostamento restituendo alpha_global = 0.0.
    """
    epsilon = 1e-4
    tol_deg = 1e-12

    # Configurazione iniziale con C_e = u1 * v2 = 1.0 * epsilon = epsilon
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, epsilon]], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)

    # Scegliamo du1 = 1.0, dv2 = 1.0e-4 - 0.5e-13 ->
    # B_e = u1 * dv2 + du1 * v2 = dv2 + epsilon * du1
    # Per avere B_e = -1e-13 (quindi -tol_deg < B_e < 0) e A_e = du1 * dv2 > 0:
    du1 = 1.0
    dv2 = -epsilon * du1 - 1e-13  # B_e = -1e-13 < 0, ma |B_e| < tol_deg = 1e-12
    # Rendiamo A_e = du1 * dv2 - du2 * dv1 > 0 usando du2 = -1.0, dv1 = 1.0 -> A_e = dv2 + 1.0 > 0
    disp = np.array([
        [0.0, 0.0],
        [du1, -1.0],
        [1.0, dv2]
    ], dtype=np.float64)

    alpha, nodes_safe, is_safe, min_j = apply_safety_filter(
        nodes, disp, cells, epsilon=epsilon, tol_deg=tol_deg
    )

    assert not is_safe
    assert alpha == 0.0
    assert min_j >= epsilon - 1e-12

    # Verifica lungo l'intera traiettoria ammessa [0, alpha]
    for s in np.linspace(0.0, alpha, 25):
        n_s = nodes + s * disp
        j_s = compute_mesh_metrics(n_s, cells)["jacobians"][0]
        assert j_s >= epsilon - 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])