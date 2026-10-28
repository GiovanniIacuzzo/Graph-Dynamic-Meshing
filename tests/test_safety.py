"""Safety filter behavior is tested through the compiled C++ implementation."""
import numpy as np

from tests.native_test_support import make_mesh, apply_native_safety_filter, compute_native_mesh_metrics


def test_native_safety_filter_accepts_small_valid_motion():
    mesh = make_mesh(8)
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    disp = np.column_stack((0.002 * np.sin(np.pi*x) * np.sin(np.pi*y), -0.001 * np.sin(np.pi*x) * np.sin(np.pi*y)))
    result = apply_native_safety_filter(mesh, disp, epsilon=1e-6, lock_boundary_nodes=True)
    assert result["is_safe"]
    assert result["alpha_global"] == 1.0
    np.testing.assert_allclose(result["safe_nodes"][mesh.boundary_nodes], mesh.nodes[mesh.boundary_nodes], atol=1e-14)
    moved = mesh.update_coordinates(result["safe_nodes"], enforce_fixed_boundary=True)
    assert compute_native_mesh_metrics(moved)["num_inverted"] == 0


def test_native_safety_filter_reduces_invalidating_displacement():
    mesh = make_mesh(6)
    disp = np.zeros_like(mesh.nodes)
    center = int(np.argmin(np.sum((mesh.nodes - np.array([0.5, 0.5]))**2, axis=1)))
    disp[center] = np.array([0.7, 0.0])
    result = apply_native_safety_filter(mesh, disp, epsilon=1e-5, safety_margin=1e-5, lock_boundary_nodes=True)
    assert 0.0 <= result["alpha_global"] <= 1.0
    assert np.isfinite(result["min_jacobian_after"])
    assert result["min_jacobian_after"] >= 1e-5 - 1e-9
    assert np.all(np.isfinite(result["safe_nodes"]))
