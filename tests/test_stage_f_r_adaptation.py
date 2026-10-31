"""Independent numerical checks for the classical r-adaptation controller.

The module is loaded from its file path to keep these algebraic tests independent
of the deal.II extension eagerly imported by the top-level `gdm` package.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "python" / "gdm" / "classical_r_adaptation.py"
_spec = importlib.util.spec_from_file_location("gdm_stage_f_classical_r", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

MMPDEParameters = _module.MMPDEParameters
VariationalMMPDEController = _module.VariationalMMPDEController
build_monitor_tensors = _module.build_monitor_tensors
recover_nodal_hessian = _module.recover_nodal_hessian
structured_reference_nodes = _module.structured_reference_nodes
_triangle_double_areas = _module._triangle_double_areas


def _mesh(nx: int = 8, ny: int = 8):
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    nodes = structured_reference_nodes(domain, nx, ny)
    cells, boundary = [], []
    for j in range(ny + 1):
        for i in range(nx + 1):
            if i == 0 or i == nx or j == 0 or j == ny:
                boundary.append(j * (nx + 1) + i)
    for j in range(ny):
        for i in range(nx):
            n00 = j * (nx + 1) + i
            n10 = n00 + 1
            n01 = (j + 1) * (nx + 1) + i
            n11 = n01 + 1
            cells.extend(((n00, n10, n11), (n00, n11, n01)))
    return nodes, np.asarray(cells, dtype=np.int64), np.asarray(boundary, dtype=np.int64)


def _smooth_gradient(nodes):
    x, y = nodes[:, 0], nodes[:, 1]
    value = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.04)
    return np.column_stack((-(x - 0.5) * value / 0.02, -(y - 0.5) * value / 0.02))


def test_monitor_metrics_are_spd_and_respect_eigenvalue_bounds():
    nodes, cells, _ = _mesh()
    grad = _smooth_gradient(nodes)
    for kind in ("winslow", "arclength", "hessian"):
        params = MMPDEParameters(monitor_type=kind, monitor_strength=0.005 if kind == "hessian" else 0.05,
                                 metric_eigenvalue_min=1.0, metric_eigenvalue_max=20.0)
        metric = build_monitor_tensors(grad, cells, params, nodes=nodes)
        eigs = np.linalg.eigvalsh(metric)
        assert metric.shape == (len(cells), 2, 2)
        assert np.all(np.isfinite(metric))
        assert np.all(eigs >= 1.0 - 1.0e-12)
        assert np.all(eigs <= 20.0 + 1.0e-12)


def test_winslow_monitor_is_isotropic():
    nodes, cells, _ = _mesh()
    metric = build_monitor_tensors(_smooth_gradient(nodes), cells,
                                   MMPDEParameters(monitor_type="winslow", monitor_strength=0.05), nodes=nodes)
    assert np.max(np.abs(metric[:, 0, 1])) < 1.0e-12
    assert np.max(np.abs(metric[:, 0, 0] - metric[:, 1, 1])) < 1.0e-12



def test_recovered_hessian_is_exact_for_a_quadratic_polynomial():
    nodes, cells, _ = _mesh(8, 8)
    x, y = nodes[:, 0], nodes[:, 1]
    # u = x^2 + 0.5 y^2 + 0.4 xy; exact Hessian [[2, 0.4], [0.4, 1]].
    grad = np.column_stack((2.0 * x + 0.4 * y, y + 0.4 * x))
    recovered = recover_nodal_hessian(grad, nodes, cells, ring_depth=2)
    exact = np.asarray([[2.0, 0.4], [0.4, 1.0]])
    assert np.max(np.abs(recovered - exact[None, :, :])) < 1.0e-10


def test_analytic_variational_energy_gradient_matches_autodiff_and_finite_difference():
    nodes, cells, boundary = _mesh(5, 5)
    params = MMPDEParameters(monitor_type="arclength", monitor_strength=0.03,
                             metric_eigenvalue_max=20.0, pseudo_iterations=2)
    controller = VariationalMMPDEController(nodes, cells, boundary, params)
    metric_np = build_monitor_tensors(_smooth_gradient(nodes), cells, params, nodes=nodes)

    # Independent Torch-autodiff reference, implemented from the mathematical functional.
    x = torch.as_tensor(nodes.copy(), dtype=torch.float64).clone().requires_grad_(True)
    cells_t = torch.as_tensor(cells, dtype=torch.long)
    inv_dref_t = torch.as_tensor(controller.inv_dref, dtype=torch.float64)
    area_t = torch.as_tensor(controller.reference_areas, dtype=torch.float64)
    metric_t = torch.as_tensor(metric_np, dtype=torch.float64)
    dx = torch.stack((x[cells_t[:, 1]] - x[cells_t[:, 0]],
                      x[cells_t[:, 2]] - x[cells_t[:, 0]]), dim=2)
    jac = torch.matmul(dx, inv_dref_t)
    det_j = torch.linalg.det(jac)
    det_g = torch.linalg.det(metric_t)
    sqrt_g = torch.sqrt(det_g)
    inv_j = torch.linalg.inv(jac)
    gbar_inverse = torch.linalg.inv(metric_t) * det_g[:, None, None].pow(0.25)
    a = torch.matmul(torch.matmul(inv_j, gbar_inverse), inv_j.transpose(1, 2))
    beta = torch.diagonal(a, dim1=1, dim2=2).sum(dim=1)
    theta = params.theta
    energy = torch.sum(area_t * (theta * det_j * sqrt_g * beta.pow(2)
                                  + (1.0 - 2.0 * theta) * 4.0 / (det_j * sqrt_g)))
    (grad_t,) = torch.autograd.grad(energy, x)

    grad_analytic = controller._energy_gradient(nodes, metric_np)
    assert np.allclose(grad_analytic, grad_t.detach().numpy(), rtol=2.0e-10, atol=2.0e-10)

    node_idx = (5 // 2) * (5 + 1) + (5 // 2)
    eps = 1.0e-6
    xp = nodes.copy(); xm = nodes.copy()
    xp[node_idx, 0] += eps; xm[node_idx, 0] -= eps
    fd = (controller.evaluate_energy(xp, metric_np) - controller.evaluate_energy(xm, metric_np)) / (2.0 * eps)
    assert np.isclose(grad_analytic[node_idx, 0], fd, rtol=2.0e-5, atol=2.0e-7)


def test_gradient_flow_decreases_energy_and_preserves_boundary_and_orientation():
    nodes, cells, boundary = _mesh()
    params = MMPDEParameters(monitor_type="arclength", monitor_strength=0.05,
                             pseudo_iterations=20, max_move_fraction=0.1, safety_epsilon=1.0e-4)
    controller = VariationalMMPDEController(nodes, cells, boundary, params)
    displacement, event = controller.propose(nodes, _smooth_gradient(nodes))
    moved = nodes + displacement
    assert event["energy_after"] <= event["energy_before"] + 1.0e-12
    assert event["energy_monotone"]
    assert event["accepted_pseudo_steps"] >= 0
    assert np.max(np.abs(displacement[boundary])) < 1.0e-13
    assert np.min(_triangle_double_areas(moved, cells)) > params.safety_epsilon
    assert np.all(np.isfinite(displacement))


def test_zero_gradient_keeps_uniform_mesh_stationary():
    nodes, cells, boundary = _mesh()
    params = MMPDEParameters(monitor_type="winslow", monitor_strength=0.05,
                             pseudo_iterations=5, safety_epsilon=1.0e-4)
    controller = VariationalMMPDEController(nodes, cells, boundary, params)
    displacement, event = controller.propose(nodes, np.zeros_like(nodes))
    assert np.linalg.norm(displacement) < 1.0e-10
    assert np.isclose(event["energy_before"], event["energy_after"], rtol=0.0, atol=1.0e-12)



def test_hessian_controller_generates_admissible_descent_step():
    nodes, cells, boundary = _mesh()
    params = MMPDEParameters(monitor_type="hessian", monitor_strength=0.002,
                             metric_eigenvalue_max=50.0, pseudo_iterations=10, safety_epsilon=1.0e-4)
    controller = VariationalMMPDEController(nodes, cells, boundary, params)
    displacement, event = controller.propose(nodes, _smooth_gradient(nodes))
    moved = nodes + displacement
    assert event["energy_after"] <= event["energy_before"] + 1.0e-12
    assert np.max(np.abs(displacement[boundary])) < 1.0e-13
    assert np.min(_triangle_double_areas(moved, cells)) > params.safety_epsilon


def test_invalid_monitor_configuration_is_rejected():
    try:
        MMPDEParameters(monitor_type="unknown").validate()
    except ValueError:
        pass
    else:
        raise AssertionError("Monitor non valido non rifiutato.")


def test_huang_q2_energy_at_identity_map_has_expected_normalization():
    nodes, cells, boundary = _mesh(6, 6)
    params = MMPDEParameters(monitor_type="winslow", monitor_strength=0.05)
    controller = VariationalMMPDEController(nodes, cells, boundary, params)
    identity_metric = np.broadcast_to(np.eye(2), (len(cells), 2, 2)).copy()
    # For n=q=2, theta=1/4, J=I and G=I, the element density is 3.
    assert np.isclose(controller.evaluate_energy(nodes, identity_metric), 3.0, rtol=0.0, atol=1.0e-12)


def test_arc_length_and_winslow_monitors_match_declared_cell_formulas():
    nodes, cells, _ = _mesh(6, 6)
    grad = _smooth_gradient(nodes)
    alpha = 0.02
    grad_cell = grad[cells].mean(axis=1)
    expected_arc = np.eye(2)[None, :, :] + alpha * grad_cell[:, :, None] * grad_cell[:, None, :]
    expected_winslow = np.sqrt(np.linalg.det(expected_arc))[:, None, None] * np.eye(2)[None, :, :]

    arc = build_monitor_tensors(
        grad, cells,
        MMPDEParameters(monitor_type="arclength", monitor_strength=alpha,
                        monitor_smoothing_passes=0, metric_eigenvalue_min=1.0e-12,
                        metric_eigenvalue_max=1.0e6),
        nodes=nodes,
    )
    winslow = build_monitor_tensors(
        grad, cells,
        MMPDEParameters(monitor_type="winslow", monitor_strength=alpha,
                        monitor_smoothing_passes=0, metric_eigenvalue_min=1.0e-12,
                        metric_eigenvalue_max=1.0e6),
        nodes=nodes,
    )
    assert np.allclose(arc, expected_arc, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(winslow, expected_winslow, rtol=1.0e-12, atol=1.0e-12)


def test_mesh_flow_logs_termination_reason_and_pseudo_iteration_budget():
    nodes, cells, boundary = _mesh(8, 8)
    params = MMPDEParameters(monitor_type="arclength", monitor_strength=0.05,
                             pseudo_iterations=3, max_move_fraction=0.1)
    controller = VariationalMMPDEController(nodes, cells, boundary, params)
    _, event = controller.propose(nodes, _smooth_gradient(nodes))
    assert event["termination_reason"] in {
        "relative_energy_tolerance", "gradient_tolerance", "maximum_pseudo_iterations",
        "line_search_failure", "zero_direction", "non_descent_direction",
    }
    assert event["pseudo_iterations_requested"] == 3
    assert event["pseudo_iterations_exhausted"] == (event["termination_reason"] == "maximum_pseudo_iterations")
