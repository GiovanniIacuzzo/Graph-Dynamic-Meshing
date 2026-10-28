"""Native ALE solver consistency and mesh-refinement smoke test."""
import numpy as np

from tests.native_test_support import make_mesh, create_pde_model, create_ale_solver, compute_native_continuous_errors


def _one_step_error(n: int) -> float:
    mesh = make_mesh(n)
    pde = create_pde_model(bx=0.4, by=0.2, nu=1e-3, sigma=0.12, beta=0.5, T=0.5)
    u0 = np.asarray(pde.evaluate_solution(mesh.nodes, 0.0), dtype=np.float64)
    dt = 0.01 * (8.0 / n) ** 2
    # Stationary-mesh ALE still exercises the native ALE assembly/solve path.
    ale = create_ale_solver(pde)
    u1, _diag = ale.step_ale(mesh.core, mesh.core, u0, dt, dt, False, True)
    err, _ = compute_native_continuous_errors(mesh, np.asarray(u1), pde, dt)
    assert np.isfinite(err)
    return err


def test_native_ale_stationary_mesh_refinement_reduces_error():
    errors = [_one_step_error(n) for n in (8, 16, 32)]
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
