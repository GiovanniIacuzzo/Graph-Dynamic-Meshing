"""L2 projection tests against the native deal.II remapping implementation."""
import numpy as np

from tests.native_test_support import make_mesh, deform_mesh, remap_l2_native, remap_p1_native


def test_native_l2_projection_preserves_constant_field():
    source = make_mesh(8)
    target = deform_mesh(source, amplitude=0.003)
    u = np.full(source.num_nodes, 2.5, dtype=np.float64)
    result = remap_l2_native(source, u, target)
    transferred = np.asarray(result["remapped_field"])
    assert np.all(np.isfinite(transferred))
    np.testing.assert_allclose(transferred, 2.5, rtol=1e-7, atol=1e-7)
    assert abs(float(result["relative_mass_drift"])) < 1e-6


def test_native_l2_projection_identity_and_comparison_with_p1():
    mesh = make_mesh(8)
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    u = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / (2 * 0.15**2))
    l2 = remap_l2_native(mesh, u, mesh)
    p1 = remap_p1_native(mesh, u, mesh)
    np.testing.assert_allclose(l2["remapped_field"], u, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(p1["remapped_field"], u, rtol=1e-10, atol=1e-10)
