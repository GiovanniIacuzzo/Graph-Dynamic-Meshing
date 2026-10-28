"""Mesh topology and geometry metrics are exercised through the native backend."""
import numpy as np

from tests.native_test_support import make_mesh, deform_mesh, compute_native_mesh_metrics, compute_native_mesh_metrics_arrays


def test_native_structured_mesh_topology_and_metrics():
    mesh = make_mesh(12, 8)
    assert mesh.num_nodes == (12 + 1) * (8 + 1)
    assert mesh.num_cells == 2 * 12 * 8
    assert mesh.edges.ndim == 2 and mesh.edges.shape[1] == 2
    assert mesh.boundary_nodes.size == 2 * 12 + 2 * 8
    m = compute_native_mesh_metrics(mesh)
    assert m["num_inverted"] == 0
    assert m["num_degenerate"] == 0
    assert m["min_jacobian"] > 0.0
    assert 0.0 < m["mean_escobar_quality"] <= 1.0


def test_native_metric_array_api_can_diagnose_invalid_proposal_without_building_mesh():
    mesh = make_mesh(5)
    proposed = mesh.nodes.copy()
    interior = np.flatnonzero(mesh.interior_mask)
    proposed[interior[0]] += np.array([0.25, -0.1])
    m = compute_native_mesh_metrics_arrays(proposed, mesh.cells)
    assert len(m["jacobians"]) == mesh.num_cells
    assert np.all(np.isfinite(m["jacobians"]))
    assert int(m["num_inverted"]) >= 0


def test_native_mesh_update_preserves_topology_and_validity():
    mesh = make_mesh(16)
    moved = deform_mesh(mesh, amplitude=0.01)
    assert np.array_equal(moved.cells, mesh.cells)
    assert np.array_equal(moved.boundary_nodes, mesh.boundary_nodes)
    assert compute_native_mesh_metrics(moved)["num_inverted"] == 0
    assert moved.core.geometry_revision >= mesh.core.geometry_revision
