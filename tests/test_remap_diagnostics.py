"""Native remapping diagnostics and invalid-input guard checks."""
import numpy as np
import pytest

from tests.native_test_support import make_mesh, deform_mesh, create_pde_model, remap_p1_native, remap_l2_native


@pytest.mark.parametrize("remapper", [remap_p1_native, remap_l2_native])
def test_native_remapper_exposes_mass_and_timing_diagnostics(remapper):
    source = make_mesh(8)
    target = deform_mesh(source, amplitude=0.003)
    pde = create_pde_model(sigma=0.12, T=0.5)
    u = np.asarray(pde.evaluate_solution(source.nodes, 0.0), dtype=np.float64)
    result = remapper(source, u, target)
    assert set(("remapped_field", "mass_before", "mass_after", "relative_mass_drift", "elapsed_ms")).issubset(result)
    assert result["remapped_field"].shape == (target.num_nodes,)
    assert np.all(np.isfinite(result["remapped_field"]))
    assert np.isfinite(result["mass_before"]) and np.isfinite(result["mass_after"])
    assert np.isfinite(result["relative_mass_drift"]) and result["elapsed_ms"] >= 0.0


def test_native_remapper_rejects_wrong_field_shape():
    source = make_mesh(6)
    target = deform_mesh(source, amplitude=0.002)
    with pytest.raises(ValueError):
        remap_p1_native(source, np.zeros(source.num_nodes - 1), target)
    with pytest.raises(ValueError):
        remap_l2_native(source, np.zeros(source.num_nodes - 1), target)
