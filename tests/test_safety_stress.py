"""Deterministic randomized stress test for the native safety filter."""
import numpy as np

from tests.native_test_support import make_mesh, apply_native_safety_filter, compute_native_mesh_metrics


def test_native_safety_filter_randomized_displacements_remain_admissible():
    mesh = make_mesh(8)
    rng = np.random.default_rng(20261028)
    for scale in (0.001, 0.01, 0.05, 0.15):
        disp = rng.normal(size=mesh.nodes.shape) * scale
        disp[mesh.boundary_nodes] = 0.0
        safe = apply_native_safety_filter(mesh, disp, epsilon=1e-6, safety_margin=1e-5, lock_boundary_nodes=True)
        assert safe["min_jacobian_after"] >= 1e-6 - 1e-9
        moved = mesh.update_coordinates(safe["safe_nodes"], enforce_fixed_boundary=True)
        metrics = compute_native_mesh_metrics(moved)
        assert metrics["num_inverted"] == 0
        assert metrics["num_degenerate"] == 0
