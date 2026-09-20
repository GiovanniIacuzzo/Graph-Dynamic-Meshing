"""
tests/test_geometry.py
Test per metriche di qualità, aree e determinanti Jacobiani.
"""

import numpy as np
import pytest
from geometry.mesh import TriangularMesh2D
from geometry.geometry import compute_mesh_metrics, compute_mesh_quality_summary


def test_equilateral_triangle_metrics():
    """
    Verifica che un triangolo equilatero perfetto abbia qualità Escobar-Knupp pari a 1.0,
    aspect ratio unitario e angoli interni esattamente pari a pi/3 radianti (60 gradi).
    """
    # Coordinate di un triangolo equilatero con lato l = 1.0
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3.0) / 2.0]
    ], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)

    metrics = compute_mesh_metrics(nodes, cells)

    expected_jacobian = np.sqrt(3.0) / 2.0
    expected_area = np.sqrt(3.0) / 4.0

    assert np.isclose(metrics["jacobians"][0], expected_jacobian, atol=1e-12)
    assert np.isclose(metrics["areas"][0], expected_area, atol=1e-12)
    assert np.isclose(metrics["escobar_quality"][0], 1.0, atol=1e-12)
    assert np.isclose(metrics["aspect_ratio"][0], 1.0, atol=1e-12)
    assert np.isclose(metrics["min_angle"][0], np.pi / 3.0, atol=1e-12)
    assert np.isclose(metrics["max_angle"][0], np.pi / 3.0, atol=1e-12)
    assert not metrics["inverted"][0]
    assert not metrics["degenerate"][0]


def test_right_isosceles_triangle_orientation_and_area():
    """
    Verifica orientamento, segno del Jacobiano e inversione:
    - vertici in senso antiorario -> J > 0
    - vertici invertiti (senso orario) -> J < 0 (inverted=True)
    """
    # Triangolo rettangolo isoscele unitario antiorario
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=np.float64)
    cells_ccw = np.array([[0, 1, 2]], dtype=np.int64)
    cells_cw = np.array([[0, 2, 1]], dtype=np.int64)

    metrics_ccw = compute_mesh_metrics(nodes, cells_ccw)
    assert np.isclose(metrics_ccw["jacobians"][0], 1.0, atol=1e-14)
    assert np.isclose(metrics_ccw["areas"][0], 0.5, atol=1e-14)
    assert not metrics_ccw["inverted"][0]

    metrics_cw = compute_mesh_metrics(nodes, cells_cw)
    assert np.isclose(metrics_cw["jacobians"][0], -1.0, atol=1e-14)
    assert metrics_cw["inverted"][0]
    assert metrics_cw["escobar_quality"][0] == 0.0


def test_degenerate_collinear_triangle():
    """Verifica che tre punti allineati producano area nulla e flag degenerate=True."""
    nodes = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0]
    ], dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)

    metrics = compute_mesh_metrics(nodes, cells)
    assert np.isclose(metrics["jacobians"][0], 0.0, atol=1e-14)
    assert metrics["degenerate"][0]
    assert metrics["inverted"][0]


def test_mesh_topology_and_boundary_extraction():
    """
    Verifica la coerenza topologica in TriangularMesh2D:
    due triangoli che formano un quadrato unitario con 4 nodi, 2 celle e 4 nodi di bordo.
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

    mesh = TriangularMesh2D(nodes, cells)
    assert mesh.num_nodes == 4
    assert mesh.num_cells == 2

    # Tutti i 4 vertici giacciono sul contorno esterno
    bnd_nodes = mesh.boundary_nodes
    assert len(bnd_nodes) == 4
    np.testing.assert_array_equal(bnd_nodes, np.array([0, 1, 2, 3]))

    # Il lato [0, 2] è condiviso: 5 segmenti totali unici -> 10 archi bidirezionali
    edges = mesh.edges
    assert edges.shape == (10, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])