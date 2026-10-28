"""Shared helpers for integration tests of the compiled deal.II backend."""
from __future__ import annotations

import numpy as np
import pytest

# Import the actual extension before importing gdm.__init__, so a machine without
# a compiled deal.II build skips only native integration tests, not the full suite.
pytest.importorskip("gdm._gdm_core", reason="Build the C++/deal.II extension to run native integration tests", exc_type=ImportError)

from gdm import (  # noqa: E402
    NativeMeshAdapter,
    create_pde_model,
    create_fem_solver,
    create_ale_solver,
    compute_native_mesh_metrics,
    compute_native_mesh_metrics_arrays,
    compute_native_continuous_errors,
    compute_native_field_gradients,
    apply_native_safety_filter,
    remap_p1_native,
    remap_l2_native,
    NativeDynamicRuntime,
)


def make_mesh(nx: int = 16, ny: int | None = None, domain: dict | None = None) -> NativeMeshAdapter:
    ny = nx if ny is None else ny
    domain = domain or {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    return NativeMeshAdapter.from_domain(domain, nx=nx, ny=ny)


def deform_mesh(mesh: NativeMeshAdapter, amplitude: float = 0.01) -> NativeMeshAdapter:
    nodes = mesh.nodes.copy()
    x, y = nodes[:, 0], nodes[:, 1]
    phi = np.sin(np.pi * x) * np.sin(np.pi * y)
    nodes[:, 0] += float(amplitude) * phi
    nodes[:, 1] -= 0.5 * float(amplitude) * phi
    nodes[mesh.boundary_nodes] = mesh.nodes[mesh.boundary_nodes]
    return mesh.update_coordinates(nodes, enforce_fixed_boundary=True)


def analytical_mesh_motion(nodes_0: np.ndarray, t: float, T: float = 0.5, amplitude: float = 0.02) -> np.ndarray:
    x, y = nodes_0[:, 0], nodes_0[:, 1]
    spatial = np.sin(np.pi * x) * np.sin(np.pi * y)
    omega = 2.0 * np.pi / T
    moved = nodes_0.copy()
    moved[:, 0] += amplitude * spatial * np.sin(omega * t)
    moved[:, 1] += 0.5 * amplitude * spatial * (np.cos(omega * t) - 1.0)
    return moved


def make_replay_targets(mesh: NativeMeshAdapter, n_targets: int, amplitude: float = 0.005) -> list[np.ndarray]:
    base = mesh.nodes
    targets: list[np.ndarray] = []
    x, y = base[:, 0], base[:, 1]
    phi = np.sin(np.pi * x) * np.sin(np.pi * y)
    for i in range(n_targets):
        nodes = base.copy()
        a = amplitude * (i + 1) / max(n_targets, 1)
        nodes[:, 0] += a * phi
        nodes[:, 1] -= 0.3 * a * phi
        nodes[mesh.boundary_nodes] = base[mesh.boundary_nodes]
        # The fixture is small by construction; if a platform-specific mesh guard rejects it,
        # the exception should make the test fail rather than hide invalid fixture geometry.
        moved = mesh.update_coordinates(nodes, enforce_fixed_boundary=True)
        targets.append(moved.nodes.copy())
    return targets
