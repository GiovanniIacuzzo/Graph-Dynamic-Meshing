"""Differentiable trajectory-aware mesh objective for self-supervised GDM training.

This objective matches the normalized space-time P1 interpolation-error proxy
used by the Stage-D oracle, plus mesh-quality and displacement terms. A smooth
pathwise Jacobian barrier is added only to the training loss. The deterministic
safety filter remains mandatory for deployment/evaluation.
"""
from __future__ import annotations
from typing import Dict, Any
import math
import torch


def _spatial_quadrature(device: torch.device, dtype: torch.dtype):
    a = 0.059715871789770
    b = 0.470142064105115
    c = 0.797426985353087
    d = 0.101286507323456
    bary = torch.tensor([
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [a, b, b], [b, a, b], [b, b, a],
        [c, d, d], [d, c, d], [d, d, c],
    ], device=device, dtype=dtype)
    weights = torch.tensor(
        [0.1125] + [0.066197076394253] * 3 + [0.062969590272414] * 3,
        device=device, dtype=dtype,
    )
    return bary, weights


def _exact_mms_u(x: torch.Tensor, y: torch.Tensor, t: float, params: Dict[str, Any]) -> torch.Tensor:
    sigma = float(params["sigma"])
    beta = float(params["beta"])
    T = float(params["T"])
    if sigma <= 0.0 or T <= 0.0:
        raise ValueError("sigma e T devono essere positivi.")
    cx = 0.3 + 0.4 * (float(t) / T)
    cy = 0.35 + 0.3 * (float(t) / T)
    r2 = (x - cx).square() + (y - cy).square()
    gaussian = torch.exp(-0.5 * r2 / (sigma * sigma))
    sine = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    return gaussian * sine * math.exp(-beta * float(t))


def _signed_jacobians(nodes: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    tri = nodes[cells]
    e1 = tri[:, 1, :] - tri[:, 0, :]
    e2 = tri[:, 2, :] - tri[:, 0, :]
    return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]


def pathwise_minimum_jacobians(
    initial_nodes: torch.Tensor,
    displacement: torch.Tensor,
    cells: torch.Tensor,
) -> torch.Tensor:
    """Exact per-cell minimum signed Jacobian on X(s)=X0+s*dX, s in [0,1].

    For linear nodal motion, each triangle determinant is quadratic in s:
    J_e(s) = c0 + c1*s + c2*s^2. Its minimum on the closed interval occurs
    at an endpoint, or at the stationary point -c1/(2*c2) when c2>0 and
    that point lies strictly inside (0,1). The returned tensor has one value
    per cell and is differentiable almost everywhere with respect to dX.
    """
    if initial_nodes.ndim != 2 or initial_nodes.shape[-1] != 2:
        raise ValueError("initial_nodes deve avere shape [N,2].")
    if displacement.shape != initial_nodes.shape:
        raise ValueError("displacement deve avere la stessa shape di initial_nodes.")
    if cells.ndim != 2 or cells.shape[-1] != 3:
        raise ValueError("cells deve avere shape [M,3].")
    tri0 = initial_nodes[cells]
    trid = displacement[cells]
    e10 = tri0[:, 1, :] - tri0[:, 0, :]
    e20 = tri0[:, 2, :] - tri0[:, 0, :]
    de1 = trid[:, 1, :] - trid[:, 0, :]
    de2 = trid[:, 2, :] - trid[:, 0, :]

    def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    c0 = cross(e10, e20)
    c1 = cross(de1, e20) + cross(e10, de2)
    c2 = cross(de1, de2)
    j_end = c0 + c1 + c2
    endpoint_min = torch.minimum(c0, j_end)

    # Avoid division by zero on linear/constant determinants. The condition
    # below also prevents the masked branch from being interpreted as a real
    # stationary point for near-zero quadratic coefficients.
    tiny = 1.0e-14 if initial_nodes.dtype in (torch.float32, torch.float64) else 1.0e-8
    valid_quadratic = c2.abs() > tiny
    safe_c2 = torch.where(valid_quadratic, c2, torch.ones_like(c2))
    s_star = -c1 / (2.0 * safe_c2)
    interior_minimum = (c2 > tiny) & (s_star > 0.0) & (s_star < 1.0)
    j_star = c0 + c1 * s_star + c2 * s_star.square()
    return torch.where(interior_minimum, torch.minimum(endpoint_min, j_star), endpoint_min)


def pathwise_jacobian_barrier(
    initial_nodes: torch.Tensor,
    displacement: torch.Tensor,
    cells: torch.Tensor,
    epsilon: float = 1.0e-4,
    relative_jacobian_floor: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (mean squared floor violation, per-cell path min, per-cell floor)."""
    j0 = _signed_jacobians(initial_nodes, cells).detach()
    if torch.any(j0 <= 0.0):
        raise ValueError("La mesh iniziale deve avere Jacobiani positivi.")
    j_path_min = pathwise_minimum_jacobians(initial_nodes, displacement, cells)
    target_floor = torch.maximum(
        torch.full_like(j0, 3.0 * float(epsilon)),
        float(relative_jacobian_floor) * j0.abs(),
    )
    violation = torch.relu(target_floor - j_path_min) / target_floor.clamp_min(1e-14)
    return torch.mean(violation.square()), j_path_min, target_floor


def _interpolation_error(
    nodes: torch.Tensor,
    cells: torch.Tensor,
    t: float,
    params: Dict[str, Any],
    bary: torch.Tensor,
    qweights: torch.Tensor,
) -> torch.Tensor:
    tri = nodes[cells]  # [M,3,2]
    xq = torch.einsum("qv,mvd->mqd", bary, tri)
    u_nodes = _exact_mms_u(nodes[:, 0], nodes[:, 1], t, params)
    u_tri = u_nodes[cells]
    u_interp = torch.einsum("qv,mv->mq", bary, u_tri)
    u_exact_q = _exact_mms_u(xq[..., 0], xq[..., 1], t, params)
    det = _signed_jacobians(nodes, cells).abs()
    return torch.sum((u_exact_q - u_interp).square() * det[:, None] * qweights[None, :])


def _time_quadrature(horizon: float, order: int, device: torch.device, dtype: torch.dtype):
    """Return mapped Gauss-Legendre nodes/weights on the normalized interval [0,1]."""
    import numpy as np
    z_np, w_np = np.polynomial.legendre.leggauss(int(order))
    s_values = 0.5 * (z_np + 1.0)
    s_nodes = torch.as_tensor(s_values, device=device, dtype=dtype)
    s_weights = torch.as_tensor(0.5 * w_np, device=device, dtype=dtype)
    # Keep host-side quadrature coordinates to avoid synchronizing MPS/CUDA just
    # to recover a Python float for the analytic time-dependent MMS.
    return s_nodes, s_weights, s_values.tolist()


def compute_baseline_interpolation_errors(
    initial_nodes: torch.Tensor,
    cells: torch.Tensor,
    params: Dict[str, Any],
    t_start: float,
    horizon: float,
    time_quadrature_order: int = 3,
) -> torch.Tensor:
    """Precompute the fixed-mesh interpolation-error denominators for one sample.

    These denominators depend only on the sample's initial mesh, PDE parameters,
    and time window; caching them avoids recomputing the same exact-field
    integration at every optimizer epoch.
    """
    if initial_nodes.ndim != 2 or initial_nodes.shape[-1] != 2:
        raise ValueError("initial_nodes deve avere shape [N,2].")
    if cells.ndim != 2 or cells.shape[-1] != 3:
        raise ValueError("cells deve avere shape [M,3].")
    if horizon <= 0.0:
        raise ValueError("horizon deve essere strettamente positivo.")
    order = int(time_quadrature_order)
    if order not in (2, 3, 4, 5):
        raise ValueError("time_quadrature_order deve essere 2, 3, 4 o 5.")
    bary, qweights = _spatial_quadrature(initial_nodes.device, initial_nodes.dtype)
    s_nodes, _, s_values = _time_quadrature(horizon, order, initial_nodes.device, initial_nodes.dtype)
    with torch.no_grad():
        values = [
            _interpolation_error(
                initial_nodes, cells,
                float(t_start) + float(horizon) * float(s.detach().cpu()),
                params, bary, qweights,
            )
            for s in s_nodes
        ]
    result = torch.stack(values).detach()
    if not torch.isfinite(result).all() or torch.any(result <= 0.0):
        raise ValueError("Gli errori di interpolazione baseline devono essere finiti e positivi.")
    return result


def fixed_mesh_objective(
    initial_nodes: torch.Tensor,
    cells: torch.Tensor,
    lambda_Q: float = 0.001,
) -> Dict[str, torch.Tensor]:
    """Return the objective for zero displacement without repeating MMS quadrature.

    The trajectory-normalized interpolation term is exactly one for a fixed mesh
    (up to roundoff, provided the denominator is nonzero); the displacement cost
    is zero. Only the geometry-quality term needs evaluation.
    """
    tri = initial_nodes[cells]
    edge01 = tri[:, 1] - tri[:, 0]
    edge12 = tri[:, 2] - tri[:, 1]
    edge20 = tri[:, 0] - tri[:, 2]
    sum_l2 = (
        edge01.square().sum(dim=1)
        + edge12.square().sum(dim=1)
        + edge20.square().sum(dim=1)
    ).clamp_min(1e-14)
    det = _signed_jacobians(initial_nodes, cells)
    eta = ((2.0 * math.sqrt(3.0)) * det / sum_l2).clamp(0.0, 1.0)
    quality = torch.mean(1.0 - eta)
    interpolation = torch.ones((), device=initial_nodes.device, dtype=initial_nodes.dtype)
    displacement_cost = torch.zeros((), device=initial_nodes.device, dtype=initial_nodes.dtype)
    objective = interpolation + float(lambda_Q) * quality
    return {
        "objective": objective,
        "normalized_interpolation_error": interpolation,
        "mesh_quality_penalty": quality,
        "displacement_cost": displacement_cost,
    }

def _mesh_quality(nodes: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Mean element-shape penalty at one point on the trajectory."""
    tri = nodes[cells]
    det = _signed_jacobians(nodes, cells)
    edge01 = tri[:, 1] - tri[:, 0]
    edge12 = tri[:, 2] - tri[:, 1]
    edge20 = tri[:, 0] - tri[:, 2]
    sum_l2 = (
        edge01.square().sum(dim=1)
        + edge12.square().sum(dim=1)
        + edge20.square().sum(dim=1)
    ).clamp_min(1e-14)
    eta = (2.0 * math.sqrt(3.0)) * det / sum_l2
    eta = eta.clamp(0.0, 1.0)
    return torch.mean(1.0 - eta)


def trajectory_objective_torch(
    initial_nodes: torch.Tensor,
    displacement: torch.Tensor,
    cells: torch.Tensor,
    params: Dict[str, Any],
    t_start: float,
    horizon: float,
    lambda_D: float = 0.01,
    lambda_Q: float = 0.001,
    epsilon: float = 1e-4,
    time_quadrature_order: int = 3,
    barrier_weight: float = 1.0,
    relative_jacobian_floor: float = 0.15,
    baseline_errors: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Evaluate a differentiable trajectory-aware objective for one graph.

    The trajectory is X(s)=X0+s*dX and t(s)=t0+s*H. The interpolation error
    at each temporal quadrature point is normalized by the same error on X0,
    matching the scale-free Stage-D oracle objective. `objective` excludes the
    extra training-only barrier; `loss` includes it.
    """
    if initial_nodes.ndim != 2 or initial_nodes.shape[-1] != 2:
        raise ValueError("initial_nodes deve avere shape [N,2].")
    if displacement.shape != initial_nodes.shape:
        raise ValueError("displacement deve avere la stessa shape di initial_nodes.")
    if cells.ndim != 2 or cells.shape[-1] != 3:
        raise ValueError("cells deve avere shape [M,3].")
    if not torch.isfinite(initial_nodes).all() or not torch.isfinite(displacement).all():
        raise ValueError("Coordinate o spostamenti non finiti.")
    T = float(params["T"])
    if horizon <= 0.0 or t_start < 0.0 or t_start + horizon > T + 1e-10:
        raise ValueError("La finestra [t_start,t_start+horizon] deve essere contenuta in [0,T].")
    if time_quadrature_order not in (2, 3, 4, 5):
        raise ValueError("time_quadrature_order deve essere 2, 3, 4 o 5.")

    dtype, device = initial_nodes.dtype, initial_nodes.device
    bary, qweights = _spatial_quadrature(device, dtype)
    s_nodes, s_weights, s_values = _time_quadrature(horizon, int(time_quadrature_order), device, dtype)
    if baseline_errors is not None:
        baseline_errors = torch.as_tensor(baseline_errors, device=device, dtype=dtype).reshape(-1)
        if baseline_errors.numel() != s_nodes.numel():
            raise ValueError("baseline_errors deve contenere un valore per ogni nodo di quadratura temporale.")
        if not torch.isfinite(baseline_errors).all() or torch.any(baseline_errors <= 0.0):
            raise ValueError("baseline_errors deve contenere valori finiti e positivi.")

    j0 = _signed_jacobians(initial_nodes, cells).detach()
    if torch.any(j0 <= 0.0):
        raise ValueError("La mesh iniziale deve avere Jacobiani positivi.")

    interp_terms = []
    quality_terms = []
    for q_index, (s, w) in enumerate(zip(s_nodes, s_weights)):
        t = float(t_start) + float(horizon) * float(s_values[q_index])
        path_nodes = initial_nodes + s * displacement
        err = _interpolation_error(path_nodes, cells, t, params, bary, qweights)
        if baseline_errors is None:
            with torch.no_grad():
                err0 = _interpolation_error(initial_nodes, cells, t, params, bary, qweights)
        else:
            err0 = baseline_errors[q_index]
        interp_terms.append(w * err / err0.clamp_min(1e-12))
        quality = _mesh_quality(path_nodes, cells)
        quality_terms.append(w * quality)

    interpolation = torch.stack(interp_terms).sum()
    quality = torch.stack(quality_terms).sum()
    # Crucial: geometry admissibility is a path constraint, not a time-quadrature
    # constraint. Check the exact quadratic minimum for every element over the
    # entire closed segment, including s=0 and s=1.
    barrier, path_min_jacobians, path_target_floor = pathwise_jacobian_barrier(
        initial_nodes, displacement, cells, epsilon, relative_jacobian_floor
    )
    min_path_jacobian = torch.min(path_min_jacobians)
    min_path_normalized_jacobian = torch.min(
        path_min_jacobians / _signed_jacobians(initial_nodes, cells).detach().abs().clamp_min(1e-14)
    )
    displacement_cost = displacement.square().sum() / max(float(initial_nodes.shape[0]), 1.0)
    objective = interpolation + float(lambda_Q) * quality + float(lambda_D) * displacement_cost
    loss = objective + float(barrier_weight) * barrier
    return {
        "loss": loss,
        "objective": objective,
        "normalized_interpolation_error": interpolation,
        "mesh_quality_penalty": quality,
        "displacement_cost": displacement_cost,
        "jacobian_barrier": barrier,
        "path_jacobian_barrier": barrier,
        "path_min_jacobian": min_path_jacobian,
        "path_min_normalized_jacobian": min_path_normalized_jacobian,
        "path_floor_violation_fraction": torch.mean(
            (path_min_jacobians < path_target_floor).to(dtype=dtype)
        ),
    }
