"""D-GCL, free-stream preservation and mesh-motion checks through native deal.II."""
import numpy as np

from tests.native_test_support import (
    make_mesh, analytical_mesh_motion, create_ale_solver,
    compute_native_mesh_metrics,
)


def test_native_mesh_motion_is_identity_at_initial_time_and_boundary_fixed():
    mesh = make_mesh(16)
    nodes0 = mesh.nodes.copy()
    np.testing.assert_allclose(analytical_mesh_motion(nodes0, 0.0), nodes0, atol=1e-15)
    nodes1 = analytical_mesh_motion(nodes0, 0.1)
    np.testing.assert_allclose(nodes1[mesh.boundary_nodes], nodes0[mesh.boundary_nodes], atol=1e-15)
    assert compute_native_mesh_metrics(mesh.update_coordinates(nodes1, enforce_fixed_boundary=True))["num_inverted"] == 0


def test_native_ale_discrete_gcl_and_free_stream_preservation():
    mesh0 = make_mesh(12)
    ale = create_ale_solver(None, nu=0.0, bx=0.0, by=0.0)
    curr = mesh0
    u = np.ones(mesh0.num_nodes, dtype=np.float64)
    dt, T = 0.005, 0.05
    max_elem, max_alg, max_fs = 0.0, 0.0, 0.0
    for k in range(1, int(round(T/dt)) + 1):
        t_next = k * dt
        target_nodes = analytical_mesh_motion(mesh0.nodes, t_next)
        nxt = mesh0.update_coordinates(target_nodes, enforce_fixed_boundary=True)
        u, diag = ale.step_ale(curr.core, nxt.core, u, dt, t_next, True, False)
        u = np.asarray(u, dtype=np.float64)
        max_elem = max(max_elem, float(diag.max_elem_gcl_residual))
        max_alg = max(max_alg, float(diag.algebraic_gcl_residual_linf))
        max_fs = max(max_fs, float(np.max(np.abs(u - 1.0))))
        curr = nxt
    assert max_elem < 1e-12
    assert max_alg < 1e-12
    assert max_fs < 1e-9


def test_native_ale_mass_diagnostic_for_nonuniform_field():
    mesh0 = make_mesh(10)
    ale = create_ale_solver(None, nu=0.0, bx=0.0, by=0.0)
    x, y = mesh0.nodes[:, 0], mesh0.nodes[:, 1]
    u = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / (2 * 0.08**2))
    curr = mesh0
    m0 = None
    max_drift = 0.0
    for k in range(1, 6):
        t_next = k * 0.005
        nxt = mesh0.update_coordinates(analytical_mesh_motion(mesh0.nodes, t_next), enforce_fixed_boundary=True)
        u_next, diag = ale.step_ale(curr.core, nxt.core, u, 0.005, t_next, True, False)
        if m0 is None:
            m0 = float(diag.discrete_mass_n)
        mass_next = float(diag.discrete_mass_next)
        max_drift = max(max_drift, abs(mass_next - m0) / max(abs(m0), 1e-14))
        u = np.asarray(u_next, dtype=np.float64)
        curr = nxt
    assert max_drift < 1e-8
