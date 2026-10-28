"""Integration checks for native runtime coupling without requiring a checkpoint.

Saved target coordinates are replayed to isolate native remapping/ALE/FEM coupling
from the neural inference path. A separate runtime test exercises GNN inference.
"""
import numpy as np

from tests.native_test_support import make_mesh, make_replay_targets, NativeDynamicRuntime


def test_native_runtime_one_step_coupling_for_p1_l2_and_ale():
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    base = make_mesh(8, domain=domain)
    targets = make_replay_targets(base, n_targets=2, amplitude=0.002)
    runtime = NativeDynamicRuntime(
        domain_cfg=domain, nx=8, ny=8, T=0.01, dt=0.005, adapt_interval=1,
        bx=0.4, by=0.2, nu=1e-3, sigma=0.12, beta=0.5,
    )
    fixed = runtime.run_episode("fixed", compute_diagnostics=True)
    assert np.isfinite(fixed["final_l2"])
    for method in ("gdm_p1", "gdm_l2", "gdm_ale"):
        result = runtime.run_episode(method, replay_targets=targets, compute_diagnostics=True)
        assert np.isfinite(result["final_l2"])
        assert np.isfinite(result["final_h1"])
        assert result["min_jacobian"] > 0.0
        assert np.all(np.isfinite(result["final_u"]))
        assert result["total_wall_time_s"] > 0.0
