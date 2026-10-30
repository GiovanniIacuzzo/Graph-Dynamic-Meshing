"""Regression test for restarting the offline optimizer from a prior candidate."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "gdm._gdm_core",
    reason="Build the C++/deal.II extension to run native oracle integration tests",
    exc_type=ImportError,
)

from gdm import NativeMeshAdapter  # noqa: E402
from oracle.field import GaussianBumpField  # noqa: E402
from oracle.optimizer import OfflineMeshOptimizer  # noqa: E402


def test_oracle_continuation_preserves_input_and_fixed_boundary() -> None:
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    mesh = NativeMeshAdapter.from_domain(domain, nx=3, ny=3)
    nodes = np.asarray(mesh.nodes, dtype=np.float64).copy()
    cells = np.asarray(mesh.cells, dtype=np.int64)
    boundary_nodes = np.asarray(mesh.boundary_nodes, dtype=np.int64)

    field = GaussianBumpField(mu_x=0.43, mu_y=0.57, sigma=0.18, amplitude=1.0)
    optimizer = OfflineMeshOptimizer(
        reference_field=field,
        initial_nodes=nodes,
        cells=cells,
        boundary_nodes=boundary_nodes,
        max_iter=2,
    )

    # Nontrivial but small internal perturbation, leaving the boundary untouched.
    candidate = nodes.copy()
    interior = np.setdiff1d(np.arange(nodes.shape[0]), boundary_nodes)
    candidate[interior, 0] += 1.0e-3
    candidate[boundary_nodes] = nodes[boundary_nodes]
    candidate_before = candidate.copy()

    result = optimizer.optimize(initial_guess_nodes=candidate, max_iter_override=1)

    # The optimizer must not mutate the caller's continuation array.
    np.testing.assert_array_equal(candidate, candidate_before)
    # Its returned target must preserve the original Dirichlet coordinates exactly.
    np.testing.assert_allclose(result["optimal_nodes"][boundary_nodes], nodes[boundary_nodes], rtol=0.0, atol=1.0e-14)
    assert np.isfinite(result["optimal_nodes"]).all()
    assert np.isfinite(result["target_displacement"]).all()
