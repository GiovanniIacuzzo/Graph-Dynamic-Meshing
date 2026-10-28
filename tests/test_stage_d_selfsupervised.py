"""Unit tests for the differentiable Stage-D trajectory objective."""
import sys
from pathlib import Path
import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "python"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

pytest.importorskip("gdm._gdm_core", reason="Test d'integrazione nativa: compilare prima il core deal.II.", exc_type=ImportError)
from gdm import create_pde_model
from oracle.trajectory_objective import TrajectoryOracleObjective
from oracle.torch_trajectory_objective import (
    trajectory_objective_torch,
    compute_baseline_interpolation_errors,
    fixed_mesh_objective,
    pathwise_minimum_jacobians,
    pathwise_jacobian_barrier,
)


def _mesh(n=5):
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    nodes = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
    cells = []
    for j in range(n - 1):
        for i in range(n - 1):
            v00 = j * n + i
            v10 = v00 + 1
            v01 = (j + 1) * n + i
            v11 = v01 + 1
            cells.append((v00, v10, v11))
            cells.append((v00, v11, v01))
    boundary = np.array([j*n+i for j in range(n) for i in range(n) if i == 0 or i == n-1 or j == 0 or j == n-1], dtype=np.int64)
    return nodes, np.asarray(cells, dtype=np.int64), boundary


def _params():
    return {"bx": 0.4, "by": 0.2, "nu": 1e-3, "sigma": 0.12, "beta": 0.5, "T": 0.5}


def test_torch_objective_matches_numpy_at_fixed_mesh():
    nodes, cells, boundary = _mesh(7)
    p = _params()
    pde = create_pde_model(**p)
    np_obj = TrajectoryOracleObjective(pde, nodes, cells, boundary, 0.0, 0.05, lambda_D=0.01, lambda_Q=0.001, epsilon=1e-4, time_quadrature_order=3)
    tn = torch.tensor(nodes, dtype=torch.float64)
    tc = torch.tensor(cells, dtype=torch.long)
    disp = torch.zeros_like(tn, requires_grad=True)
    result = trajectory_objective_torch(tn, disp, tc, p, 0.0, 0.05, lambda_D=0.01, lambda_Q=0.001, epsilon=1e-4, time_quadrature_order=3)
    expected = np_obj.evaluate(nodes)["total_objective"]
    assert abs(float(result["objective"].detach()) - expected) < 1e-9
    assert abs(float(result["normalized_interpolation_error"].detach()) - 1.0) < 1e-10


def test_torch_objective_has_finite_gradients():
    nodes, cells, boundary = _mesh(7)
    p = _params()
    tn = torch.tensor(nodes, dtype=torch.float64)
    tc = torch.tensor(cells, dtype=torch.long)
    disp = torch.zeros_like(tn)
    free = np.ones(len(nodes), dtype=bool)
    free[boundary] = False
    rng = np.random.default_rng(42)
    perturb = np.zeros_like(nodes)
    perturb[free] = rng.normal(0.0, 0.001, size=(int(free.sum()), 2))
    disp = torch.tensor(perturb, dtype=torch.float64, requires_grad=True)
    result = trajectory_objective_torch(tn, disp, tc, p, 0.0, 0.05, epsilon=1e-4)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert disp.grad is not None and torch.isfinite(disp.grad).all()
    assert float(disp.grad.norm()) > 0.0


def test_small_interior_displacement_has_finite_objective():
    nodes, cells, boundary = _mesh(7)
    p = _params()
    tn = torch.tensor(nodes, dtype=torch.float64)
    tc = torch.tensor(cells, dtype=torch.long)
    disp_np = np.zeros_like(nodes)
    free = np.ones(len(nodes), dtype=bool)
    free[boundary] = False
    disp_np[free, 0] = 0.001
    disp = torch.tensor(disp_np, dtype=torch.float64, requires_grad=True)
    result = trajectory_objective_torch(tn, disp, tc, p, 0.0, 0.05, epsilon=1e-4)
    assert torch.isfinite(result["loss"])
    assert float(result["jacobian_barrier"].detach()) >= 0.0


def test_cached_baseline_matches_uncached_objective_and_fixed_objective():
    nodes, cells, boundary = _mesh(7)
    p = _params()
    tn = torch.tensor(nodes, dtype=torch.float64)
    tc = torch.tensor(cells, dtype=torch.long)
    zero = torch.zeros_like(tn)
    baseline = compute_baseline_interpolation_errors(tn, tc, p, 0.0, 0.05, 3)
    uncached = trajectory_objective_torch(
        tn, zero, tc, p, 0.0, 0.05, lambda_D=0.01, lambda_Q=0.001,
        epsilon=1e-4, time_quadrature_order=3,
    )
    cached = trajectory_objective_torch(
        tn, zero, tc, p, 0.0, 0.05, lambda_D=0.01, lambda_Q=0.001,
        epsilon=1e-4, time_quadrature_order=3, baseline_errors=baseline,
    )
    fixed = fixed_mesh_objective(tn, tc, lambda_Q=0.001)
    assert torch.allclose(uncached["objective"], cached["objective"], atol=1e-10, rtol=1e-10)
    assert torch.allclose(cached["objective"], fixed["objective"], atol=1e-8, rtol=1e-8)
    assert torch.allclose(cached["normalized_interpolation_error"], torch.ones((), dtype=tn.dtype), atol=1e-10)


def test_pathwise_minimum_catches_bad_endpoint_missed_by_interior_quadrature():
    # For X(s)=X0+s*dX, this triangle has J(s)=1-(s/0.95)^2.
    # The three Gauss-Legendre nodes used by order-3 time quadrature are all
    # inside the positive region, while the endpoint s=1 is inverted. A
    # quadrature-only barrier therefore misses precisely this failure.
    a = 0.95
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    cells = torch.tensor([[0, 1, 2]], dtype=torch.long)
    displacement = torch.tensor([[0.0, 0.0], [0.0, 1.0 / a], [1.0 / a, 0.0]], dtype=torch.float64)
    quadrature_s = [0.1127016653792583, 0.5, 0.8872983346207417]
    sampled = [1.0 - (s / a) ** 2 for s in quadrature_s]
    assert min(sampled) > 0.0
    assert 1.0 - (1.0 / a) ** 2 < 0.0

    path_min = pathwise_minimum_jacobians(nodes, displacement, cells)
    assert path_min.shape == (1,)
    assert abs(float(path_min[0]) - (1.0 - (1.0 / a) ** 2)) < 1e-12
    barrier, _, floor = pathwise_jacobian_barrier(nodes, displacement, cells, epsilon=1e-4)
    assert float(barrier) > 0.0
    assert float(path_min[0]) < float(floor[0])


def test_pathwise_barrier_has_finite_nonzero_gradient_for_inverted_endpoint():
    a = 0.95
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    cells = torch.tensor([[0, 1, 2]], dtype=torch.long)
    displacement = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0 / a], [1.0 / a, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    barrier, _, _ = pathwise_jacobian_barrier(nodes, displacement, cells, epsilon=1e-4)
    barrier.backward()
    assert torch.isfinite(barrier)
    assert displacement.grad is not None
    assert torch.isfinite(displacement.grad).all()
    assert float(displacement.grad.norm()) > 0.0


def test_pathwise_barrier_is_zero_for_zero_motion_on_valid_mesh():
    nodes, cells, _ = _mesh(5)
    nodes_t = torch.tensor(nodes, dtype=torch.float64)
    cells_t = torch.tensor(cells, dtype=torch.long)
    zero = torch.zeros_like(nodes_t)
    barrier, path_min, floor = pathwise_jacobian_barrier(nodes_t, zero, cells_t, epsilon=1e-4)
    assert torch.all(path_min > floor)
    assert abs(float(barrier)) < 1e-14
