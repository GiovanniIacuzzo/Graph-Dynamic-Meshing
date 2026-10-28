"""Controlled-target replay test using the native C++ numerical runtime."""
import numpy as np
import pytest

from tests.native_test_support import make_mesh, make_replay_targets, NativeDynamicRuntime


def test_native_replay_uses_identical_target_sequence_for_all_transfer_methods():
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    mesh = make_mesh(8, domain=domain)
    targets = make_replay_targets(mesh, n_targets=3, amplitude=0.002)
    runtime = NativeDynamicRuntime(
        domain_cfg=domain, nx=8, ny=8, T=0.015, dt=0.005, adapt_interval=1,
        sigma=0.12,
    )
    rows = {}
    for method in ("gdm_p1", "gdm_l2", "gdm_ale"):
        rows[method] = runtime.run_episode(method, replay_targets=targets, record_targets=True)
        assert len(rows[method]["adaptation_steps"]) == 3
        assert rows[method]["final_l2"] >= 0.0
        assert rows[method]["min_jacobian"] > 0.0
    for method in rows:
        assert np.all(np.isfinite(rows[method]["final_u"]))


def test_native_replay_rejects_wrong_number_of_targets():
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    runtime = NativeDynamicRuntime(domain_cfg=domain, nx=4, ny=4, T=0.01, dt=0.005, adapt_interval=1)
    with pytest.raises(ValueError, match="esattamente"):
        runtime.run_episode("gdm_ale", replay_targets=[np.zeros((25, 2))])
