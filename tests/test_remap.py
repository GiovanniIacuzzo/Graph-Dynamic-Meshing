"""P1 interpolation remapping is performed and diagnosed by the C++ core."""
import numpy as np

from tests.native_test_support import make_mesh, deform_mesh, create_pde_model, remap_p1_native, compute_native_continuous_errors, compute_native_mesh_metrics


def test_native_p1_remap_returns_finite_target_field():
    source = make_mesh(12)
    target = deform_mesh(source, amplitude=0.006)
    pde = create_pde_model(sigma=0.1, T=0.5)
    u = np.asarray(pde.evaluate_solution(source.nodes, 0.1), dtype=np.float64)
    out = remap_p1_native(source, u, target)
    transferred = np.asarray(out["remapped_field"], dtype=np.float64)
    assert transferred.shape == (target.num_nodes,)
    assert np.all(np.isfinite(transferred))
    assert np.isfinite(out["relative_mass_drift"])
    err_l2, err_h1 = compute_native_continuous_errors(target, transferred, pde, 0.1)
    assert np.isfinite(err_l2) and np.isfinite(err_h1)
    assert compute_native_mesh_metrics(target)["num_inverted"] == 0


def test_native_p1_remap_identity_on_same_mesh():
    mesh = make_mesh(8)
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    u = np.sin(np.pi*x) * np.sin(np.pi*y)
    result = remap_p1_native(mesh, u, mesh)
    np.testing.assert_allclose(result["remapped_field"], u, rtol=1e-10, atol=1e-10)
    assert abs(float(result["relative_mass_drift"])) < 1e-10
