"""Classical variational r-adaptation controllers for GDM.

The physical PDE is never solved here.  This module constructs elementwise
monitor tensors from the *current numerical P1 field* and integrates a
finite-dimensional gradient flow for a variational mesh functional. The
nodal functional gradient is assembled analytically in NumPy (and independently
checked against finite differences and an independent autodiff reference in the tests). The resulting nodal
displacement is passed to the existing native safety filter
and deal.II ALE solver by the Stage-F benchmark.

Functional: Huang (2001), q=2, n=2, theta in (0, 1/2].  For each reference
triangle Khat and its affine map x = x(xi),

  I_K = |Khat| [ theta J sqrt(g) beta^2
                 + (1-2 theta) 4 / (J sqrt(g)) ],

  J = det(dx/dxi) > 0, g = det(G),
  Gbar = G / g^(1/4),
  beta = tr( J^(-1) Gbar^(-1) J^(-T) ).

The pseudo-time flow is dX/ds = -M_lump^(-1) grad_X I_h, with fixed
boundary nodes.  A monotone backtracking line search and a determinant guard
are used for the explicit pseudo-time integration.  This is a discrete,
lagged-monitor gradient-flow realization; it is not claimed to be the exact
fully coupled MMPDE discretization in Huang & Russell (1999).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MMPDEParameters:
    monitor_type: str = "winslow"  # winslow | arclength | hessian
    monitor_strength: float = 0.05
    monitor_smoothing_passes: int = 1
    smoothing_weight: float = 0.5
    metric_eigenvalue_min: float = 1.0
    metric_eigenvalue_max: float = 50.0
    q: int = 2
    theta: float = 0.25
    pseudo_iterations: int = 200
    max_move_fraction: float = 0.10
    armijo: float = 1.0e-4
    max_backtracking: int = 16
    relative_energy_tolerance: float = 1.0e-8
    safety_epsilon: float = 1.0e-4

    def validate(self) -> None:
        if self.monitor_type not in {"winslow", "arclength", "hessian"}:
            raise ValueError("monitor_type deve essere 'winslow', 'arclength' oppure 'hessian'.")
        positive = {
            "monitor_strength": self.monitor_strength,
            "metric_eigenvalue_min": self.metric_eigenvalue_min,
            "metric_eigenvalue_max": self.metric_eigenvalue_max,
            "max_move_fraction": self.max_move_fraction,
            "armijo": self.armijo,
            "safety_epsilon": self.safety_epsilon,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} deve essere finito e > 0.")
        if self.metric_eigenvalue_max < self.metric_eigenvalue_min:
            raise ValueError("metric_eigenvalue_max deve essere >= metric_eigenvalue_min.")
        if self.monitor_smoothing_passes < 0 or self.pseudo_iterations < 1:
            raise ValueError("Numero di smoothing pass non valido o pseudo_iterations < 1.")
        if not (0.0 <= self.smoothing_weight <= 1.0):
            raise ValueError("smoothing_weight deve appartenere a [0,1].")
        if self.q != 2:
            raise ValueError("Questa implementazione Stage-F verifica q=2; altri q non sono supportati.")
        if not (0.0 < self.theta <= 0.5):
            raise ValueError("theta deve appartenere a (0, 0.5].")
        if self.max_backtracking < 1:
            raise ValueError("max_backtracking deve essere >= 1.")
        if not np.isfinite(self.relative_energy_tolerance) or self.relative_energy_tolerance < 0.0:
            raise ValueError("relative_energy_tolerance deve essere finito e >= 0.")


def structured_reference_nodes(domain: Mapping[str, float], nx: int, ny: int) -> np.ndarray:
    """Reference coordinates with the exact node ordering of MeshState.create_structured_mesh."""
    if nx < 2 or ny < 2:
        raise ValueError("nx e ny devono essere >= 2.")
    x = np.linspace(float(domain["x_min"]), float(domain["x_max"]), nx + 1)
    y = np.linspace(float(domain["y_min"]), float(domain["y_max"]), ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.ascontiguousarray(np.column_stack((xx.ravel(), yy.ravel())), dtype=np.float64)


def _triangle_double_areas(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    p0 = nodes[cells[:, 0]]
    p1 = nodes[cells[:, 1]]
    p2 = nodes[cells[:, 2]]
    return ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))


def _cell_neighbours(cells: np.ndarray) -> List[np.ndarray]:
    edge_to_cells: Dict[Tuple[int, int], List[int]] = {}
    for k, cell in enumerate(cells):
        for a, b in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edge_to_cells.setdefault(key, []).append(k)
    neighbours: List[set[int]] = [set([i]) for i in range(len(cells))]
    for incident in edge_to_cells.values():
        if len(incident) == 2:
            a, b = incident
            neighbours[a].add(b)
            neighbours[b].add(a)
    return [np.asarray(sorted(v), dtype=np.int64) for v in neighbours]


def recover_nodal_hessian(
    grad_u_nodal: np.ndarray,
    nodes: np.ndarray,
    cells: np.ndarray,
    *,
    ring_depth: int = 2,
) -> np.ndarray:
    """Recover nodal Hessians by weighted local least-squares fitting of gradients.

    For each node i, fit grad(u)(x_j)-grad(u)(x_i) ~= H_i (x_j-x_i)
    over a graph patch.  The recovered matrix is symmetrized.  This is a
    recovery operator, not the elementwise Hessian of a P1 function (which
    vanishes inside each affine triangle).
    """
    grad = np.ascontiguousarray(grad_u_nodal, dtype=np.float64)
    xy = np.ascontiguousarray(nodes, dtype=np.float64)
    tri = np.ascontiguousarray(cells, dtype=np.int64)
    if grad.ndim != 2 or grad.shape[1] != 2 or xy.shape != grad.shape:
        raise ValueError("grad_u_nodal e nodes devono avere shape (N,2).")
    if not np.all(np.isfinite(grad)) or not np.all(np.isfinite(xy)):
        raise ValueError("Gradiente e coordinate devono essere finiti.")
    if tri.ndim != 2 or tri.shape[1] != 3 or np.any(tri < 0) or np.any(tri >= len(grad)):
        raise ValueError("cells deve avere shape (M,3) e indici validi.")
    if ring_depth < 1:
        raise ValueError("ring_depth deve essere >= 1.")

    adjacency: List[set[int]] = [set() for _ in range(len(xy))]
    for a, b, c in tri:
        adjacency[int(a)].update((int(b), int(c)))
        adjacency[int(b)].update((int(a), int(c)))
        adjacency[int(c)].update((int(a), int(b)))

    hessian = np.zeros((len(xy), 2, 2), dtype=np.float64)
    for i in range(len(xy)):
        frontier = {i}
        patch: set[int] = set()
        for _ in range(ring_depth):
            next_frontier: set[int] = set()
            for node in frontier:
                next_frontier.update(adjacency[node])
            next_frontier.difference_update(patch)
            next_frontier.discard(i)
            patch.update(next_frontier)
            frontier = next_frontier
        ids = np.asarray(sorted(patch), dtype=np.int64)
        if ids.size < 2:
            raise RuntimeError(f"Patch insufficiente per il recupero Hessiano al nodo {i}.")
        offsets = xy[ids] - xy[i]
        radii = np.linalg.norm(offsets, axis=1)
        valid = radii > 100.0 * np.finfo(np.float64).eps
        offsets, radii, ids = offsets[valid], radii[valid], ids[valid]
        if ids.size < 2:
            raise RuntimeError(f"Patch degenere per il recupero Hessiano al nodo {i}.")
        # Weighted LS with weight 1/r^2; rows are normalized by r.
        design = offsets / radii[:, None]
        rhs = (grad[ids] - grad[i]) / radii[:, None]
        if np.linalg.matrix_rank(design) < 2:
            raise RuntimeError(f"Patch geometricamente degenere per il recupero Hessiano al nodo {i}.")
        coeff, *_ = np.linalg.lstsq(design, rhs, rcond=None)
        # coeff is (coordinate direction, gradient component); transpose to H.
        h_i = coeff.T
        hessian[i] = 0.5 * (h_i + h_i.T)
    if not np.all(np.isfinite(hessian)):
        raise FloatingPointError("Il recupero Hessiano ha prodotto valori non finiti.")
    return hessian


def build_monitor_tensors(
    grad_u_nodal: np.ndarray,
    cells: np.ndarray,
    params: MMPDEParameters,
    nodes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build classical first-derivative and Hessian-based SPD monitors.

    * arclength: G3 = I + alpha grad(u) grad(u)^T;
    * winslow:   G4 = sqrt(det(G3)) I;
    * hessian:   G1 = I + alpha |H(u)|, with |H| defined spectrally.

    Cell gradients/Hessians are reconstructed from the current numerical P1
    state; no exact solution or exact derivatives are accessed.
    """
    params.validate()
    grad = np.ascontiguousarray(grad_u_nodal, dtype=np.float64)
    tri = np.ascontiguousarray(cells, dtype=np.int64)
    if grad.ndim != 2 or grad.shape[1] != 2 or not np.all(np.isfinite(grad)):
        raise ValueError("grad_u_nodal deve essere finito e avere shape (N,2).")
    if tri.ndim != 2 or tri.shape[1] != 3 or np.any(tri < 0) or np.any(tri >= len(grad)):
        raise ValueError("cells deve avere shape (M,3) e indici validi.")

    alpha = float(params.monitor_strength)
    eye = np.broadcast_to(np.eye(2, dtype=np.float64), (len(tri), 2, 2)).copy()
    if params.monitor_type == "hessian":
        if nodes is None:
            raise ValueError("Il monitor Hessian-based richiede le coordinate nodali correnti.")
        hessian_node = recover_nodal_hessian(grad, nodes, tri, ring_depth=2)
        abs_hessian_node = np.empty_like(hessian_node)
        for i, h_i in enumerate(hessian_node):
            eigvals, eigvecs = np.linalg.eigh(0.5 * (h_i + h_i.T))
            abs_hessian_node[i] = (eigvecs * np.abs(eigvals)[None, :]) @ eigvecs.T
        metric = eye + alpha * abs_hessian_node[tri].mean(axis=1)
    else:
        grad_cell = grad[tri].mean(axis=1)
        g3 = eye + alpha * grad_cell[:, :, None] * grad_cell[:, None, :]
        if params.monitor_type == "arclength":
            metric = g3
        else:
            det_g3 = np.linalg.det(g3)
            if np.any(det_g3 <= 0.0) or not np.all(np.isfinite(det_g3)):
                raise FloatingPointError("Monitor G3 non SPD.")
            scalar = np.sqrt(det_g3)
            metric = scalar[:, None, None] * eye

    # Arithmetic SPD averaging on the dual cell-neighbour graph. Convex
    # combinations of SPD matrices remain SPD; eigenvalue clipping follows.
    neighbours = _cell_neighbours(tri)
    for _ in range(params.monitor_smoothing_passes):
        smoothed = metric.copy()
        for k, ids in enumerate(neighbours):
            local_mean = np.mean(metric[ids], axis=0)
            smoothed[k] = ((1.0 - params.smoothing_weight) * metric[k]
                           + params.smoothing_weight * local_mean)
        metric = smoothed

    projected = np.empty_like(metric)
    for k, mat in enumerate(metric):
        eigvals, eigvecs = np.linalg.eigh(0.5 * (mat + mat.T))
        eigvals = np.clip(eigvals, params.metric_eigenvalue_min, params.metric_eigenvalue_max)
        projected[k] = (eigvecs * eigvals[None, :]) @ eigvecs.T
    if not np.all(np.isfinite(projected)):
        raise FloatingPointError("Monitor tensor contiene valori non finiti.")
    return projected


class VariationalMMPDEController:
    """Stateless-per-episode mesh mover compatible with Stage-D's predict_raw API."""

    def __init__(
        self,
        reference_nodes: np.ndarray,
        reference_cells: np.ndarray,
        boundary_nodes: np.ndarray,
        params: MMPDEParameters,
    ) -> None:
        params.validate()
        self.params = params
        self.reference_nodes = np.ascontiguousarray(reference_nodes, dtype=np.float64)
        self.cells = np.ascontiguousarray(reference_cells, dtype=np.int64)
        self.boundary_nodes = np.ascontiguousarray(boundary_nodes, dtype=np.int64)
        if self.reference_nodes.ndim != 2 or self.reference_nodes.shape[1] != 2:
            raise ValueError("reference_nodes deve avere shape (N,2).")
        if self.cells.ndim != 2 or self.cells.shape[1] != 3:
            raise ValueError("reference_cells deve avere shape (M,3).")
        if np.any(self.cells < 0) or np.any(self.cells >= len(self.reference_nodes)):
            raise ValueError("reference_cells contiene indici non validi.")
        if np.any(_triangle_double_areas(self.reference_nodes, self.cells) <= 0.0):
            raise ValueError("La mesh computazionale di riferimento deve avere orientazione positiva.")
        self._prepare_reference_geometry()
        self.events: List[Dict[str, Any]] = []

    def _prepare_reference_geometry(self) -> None:
        r0 = self.reference_nodes[self.cells[:, 0]]
        r1 = self.reference_nodes[self.cells[:, 1]]
        r2 = self.reference_nodes[self.cells[:, 2]]
        dref = np.stack((r1 - r0, r2 - r0), axis=2)
        self.inv_dref = np.linalg.inv(dref)
        self.reference_areas = 0.5 * np.linalg.det(dref)
        self.edges = np.unique(np.sort(np.vstack((
            self.cells[:, [0, 1]], self.cells[:, [1, 2]], self.cells[:, [2, 0]]
        )), axis=1), axis=0)
        lumped = np.zeros(len(self.reference_nodes), dtype=np.float64)
        np.add.at(lumped, self.cells[:, 0], self.reference_areas / 3.0)
        np.add.at(lumped, self.cells[:, 1], self.reference_areas / 3.0)
        np.add.at(lumped, self.cells[:, 2], self.reference_areas / 3.0)
        if np.any(lumped <= 0.0):
            raise ValueError("La mesh di riferimento contiene nodi senza massa lumped positiva.")
        self.lumped_mass = lumped
        self.min_reference_edge = self._minimum_edge_length(self.reference_nodes)
        self._free_mask = np.ones(len(self.reference_nodes), dtype=bool)
        self._free_mask[self.boundary_nodes] = False

    def _minimum_edge_length(self, nodes: np.ndarray) -> float:
        lengths = np.linalg.norm(nodes[self.edges[:, 1]] - nodes[self.edges[:, 0]], axis=1)
        return float(np.min(lengths))

    def _element_energy_terms(
        self,
        nodes: np.ndarray,
        metric: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized element energy terms for n=q=2 and frozen SPD monitors."""
        x = np.ascontiguousarray(nodes, dtype=np.float64)
        g = np.ascontiguousarray(metric, dtype=np.float64)
        if x.shape != self.reference_nodes.shape or g.shape != (len(self.cells), 2, 2):
            raise ValueError("Shape di nodes o monitor non coerente con la mesh di riferimento.")
        dx = np.stack((x[self.cells[:, 1]] - x[self.cells[:, 0]],
                       x[self.cells[:, 2]] - x[self.cells[:, 0]]), axis=2)
        jac = np.matmul(dx, self.inv_dref)
        det_j = np.linalg.det(jac)
        if np.any(~np.isfinite(det_j)) or np.any(det_j <= 0.0):
            raise ValueError("Il funzionale variazionale richiede det(J_K)>0 per ogni elemento.")
        det_g = np.linalg.det(g)
        if np.any(~np.isfinite(det_g)) or np.any(det_g <= 0.0):
            raise ValueError("Il monitor deve essere SPD su ogni elemento.")
        inv_j = np.linalg.inv(jac)
        sqrt_g = np.sqrt(det_g)
        gbar_inverse = np.linalg.inv(g) * det_g[:, None, None] ** 0.25
        a = np.matmul(np.matmul(inv_j, gbar_inverse), inv_j.transpose(0, 2, 1))
        beta = np.trace(a, axis1=1, axis2=2)
        return jac, det_j, sqrt_g, gbar_inverse, beta

    def evaluate_energy(self, nodes: np.ndarray, metric: np.ndarray) -> float:
        """Evaluate Huang's q=2, n=2 functional with vectorized NumPy algebra."""
        _, det_j, sqrt_g, _, beta = self._element_energy_terms(nodes, metric)
        theta = float(self.params.theta)
        iso_term = theta * det_j * sqrt_g * beta**2
        eq_term = (1.0 - 2.0 * theta) * 4.0 / (det_j * sqrt_g)
        value = float(np.sum(self.reference_areas * (iso_term + eq_term)))
        if not np.isfinite(value):
            raise FloatingPointError("Il funzionale variazionale ha prodotto un valore non finito.")
        return value

    def _energy_gradient(self, nodes: np.ndarray, metric: np.ndarray) -> np.ndarray:
        """Analytic gradient of the frozen-monitor functional w.r.t. nodal coordinates.

        For S=J J^T, B=Gbar^{-1}, beta=tr(S^{-1}B),
        D_J beta = -2 S^{-1} B S^{-1} J. The element derivative is then
        mapped to the three physical vertices using J=(DX)(Dref)^{-1}.
        This avoids an autodiff dependency in the classical online controller.
        """
        jac, det_j, sqrt_g, gbar_inverse, beta = self._element_energy_terms(nodes, metric)
        inv_j = np.linalg.inv(jac)
        s = np.matmul(jac, jac.transpose(0, 2, 1))
        inv_s = np.linalg.inv(s)
        h = np.matmul(np.matmul(inv_s, gbar_inverse), inv_s)

        theta = float(self.params.theta)
        c1 = theta * sqrt_g
        c2 = (1.0 - 2.0 * theta) * 4.0 / sqrt_g
        grad_j = (
            (c1 * det_j * beta**2 - c2 / det_j)[:, None, None] * inv_j.transpose(0, 2, 1)
            - 4.0 * (c1 * det_j * beta)[:, None, None] * np.matmul(h, jac)
        )
        grad_dx = np.matmul(grad_j, self.inv_dref.transpose(0, 2, 1))
        grad_dx *= self.reference_areas[:, None, None]

        grad_nodes = np.zeros_like(self.reference_nodes)
        grad_v1 = grad_dx[:, :, 0]
        grad_v2 = grad_dx[:, :, 1]
        grad_v0 = -(grad_v1 + grad_v2)
        np.add.at(grad_nodes, self.cells[:, 0], grad_v0)
        np.add.at(grad_nodes, self.cells[:, 1], grad_v1)
        np.add.at(grad_nodes, self.cells[:, 2], grad_v2)
        if not np.all(np.isfinite(grad_nodes)):
            raise FloatingPointError("Il gradiente analitico del funzionale non è finito.")
        return grad_nodes

    def _mapping_determinants(self, nodes: np.ndarray) -> np.ndarray:
        x0 = nodes[self.cells[:, 0]]
        x1 = nodes[self.cells[:, 1]]
        x2 = nodes[self.cells[:, 2]]
        dx = np.stack((x1 - x0, x2 - x0), axis=2)
        return np.linalg.det(np.matmul(dx, self.inv_dref))

    def propose(self, nodes: np.ndarray, grad_u_nodal: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Integrate a monotone discrete gradient flow for one adaptation event."""
        started = time.perf_counter()
        x0 = np.ascontiguousarray(nodes, dtype=np.float64)
        if x0.shape != self.reference_nodes.shape or not np.all(np.isfinite(x0)):
            raise ValueError("nodes deve essere finito e avere la shape della mesh di riferimento.")
        if np.any(_triangle_double_areas(x0, self.cells) <= self.params.safety_epsilon):
            raise ValueError("La mesh corrente non soddisfa il vincolo geometrico iniziale.")
        grad_array = np.ascontiguousarray(grad_u_nodal, dtype=np.float64)
        metric_np = build_monitor_tensors(grad_array, self.cells, self.params, nodes=x0)
        metric_eigs = np.linalg.eigvalsh(metric_np)

        x = x0.copy()
        e_before = self.evaluate_energy(x, metric_np)
        e_current = e_before
        energy_history = [e_current]
        accepted = 0
        final_grad_norm = 0.0
        line_search_failures = 0
        termination_reason = "maximum_pseudo_iterations"
        epsilon = float(self.params.safety_epsilon)

        for _iteration in range(self.params.pseudo_iterations):
            grad = self._energy_gradient(x, metric_np)
            grad[~self._free_mask] = 0.0
            final_grad_norm = float(np.linalg.norm(grad[self._free_mask]))
            if not np.isfinite(final_grad_norm):
                termination_reason = "nonfinite_gradient"
                break
            if final_grad_norm <= 1.0e-14:
                termination_reason = "gradient_tolerance"
                break

            direction = -grad / self.lumped_mass[:, None]
            direction[~self._free_mask] = 0.0
            max_node_step = float(np.max(np.linalg.norm(direction, axis=1)))
            if not np.isfinite(max_node_step):
                termination_reason = "nonfinite_direction"
                break
            if max_node_step <= 1.0e-15:
                termination_reason = "zero_direction"
                break
            move_cap = self.params.max_move_fraction * self._minimum_edge_length(x)
            direction *= min(1.0, move_cap / max_node_step)
            slope = float(np.sum(grad * direction))
            if not np.isfinite(slope) or slope >= 0.0:
                termination_reason = "non_descent_direction"
                break

            accepted_this_iter = False
            alpha = 1.0
            for _ls in range(self.params.max_backtracking):
                candidate = x + alpha * direction
                candidate[~self._free_mask] = x0[~self._free_mask]
                signed_det = _triangle_double_areas(candidate, self.cells)
                mapping_det = self._mapping_determinants(candidate)
                if (np.all(np.isfinite(signed_det))
                        and np.all(signed_det > epsilon)
                        and np.all(np.isfinite(mapping_det))
                        and np.all(mapping_det > 1.0e-8)):
                    e_candidate = self.evaluate_energy(candidate, metric_np)
                    if np.isfinite(e_candidate) and e_candidate <= e_current + self.params.armijo * alpha * slope:
                        rel_drop = (e_current - e_candidate) / max(abs(e_current), 1.0e-14)
                        x = candidate
                        e_current = e_candidate
                        energy_history.append(e_current)
                        accepted += 1
                        accepted_this_iter = True
                        if rel_drop <= self.params.relative_energy_tolerance:
                            event = self._event_record(
                                started, e_before, e_current, energy_history,
                                accepted, final_grad_norm, line_search_failures,
                            )
                            event.update({
                                "termination_reason": "relative_energy_tolerance",
                                "pseudo_iterations_requested": int(self.params.pseudo_iterations),
                                "pseudo_iterations_exhausted": False,
                                "monitor_eigenvalue_min": float(np.min(metric_eigs)),
                                "monitor_eigenvalue_median": float(np.median(metric_eigs)),
                                "monitor_eigenvalue_max": float(np.max(metric_eigs)),
                                "monitor_gradient_l2_mean": float(np.mean(np.linalg.norm(grad_array, axis=1))),
                                "monitor_gradient_l2_max": float(np.max(np.linalg.norm(grad_array, axis=1))),
                            })
                            return x - x0, event
                        break
                alpha *= 0.5
            if not accepted_this_iter:
                line_search_failures += 1
                termination_reason = "line_search_failure"
                break

        event = self._event_record(
            started, e_before, e_current, energy_history,
            accepted, final_grad_norm, line_search_failures,
        )
        event.update({
            "termination_reason": termination_reason,
            "pseudo_iterations_requested": int(self.params.pseudo_iterations),
            "pseudo_iterations_exhausted": bool(termination_reason == "maximum_pseudo_iterations"),
            "monitor_eigenvalue_min": float(np.min(metric_eigs)),
            "monitor_eigenvalue_median": float(np.median(metric_eigs)),
            "monitor_eigenvalue_max": float(np.max(metric_eigs)),
            "monitor_gradient_l2_mean": float(np.mean(np.linalg.norm(grad_array, axis=1))),
            "monitor_gradient_l2_max": float(np.max(np.linalg.norm(grad_array, axis=1))),
        })
        return x - x0, event

    @staticmethod
    def _event_record(
        started: float,
        energy_before: float,
        energy_after: float,
        history: Sequence[float],
        accepted: int,
        grad_norm: float,
        line_search_failures: int,
    ) -> Dict[str, Any]:
        return {
            "energy_before": float(energy_before),
            "energy_after": float(energy_after),
            "relative_energy_decrease": float((energy_before - energy_after) / max(abs(energy_before), 1e-14)),
            "accepted_pseudo_steps": int(accepted),
            "final_gradient_l2": float(grad_norm),
            "line_search_failures": int(line_search_failures),
            "energy_monotone": bool(np.all(np.diff(np.asarray(history, dtype=np.float64)) <= 1e-12)),
            "elapsed_seconds": float(time.perf_counter() - started),
        }

    def predict_raw(
        self,
        mesh: Any,
        u: np.ndarray,
        grad_u: np.ndarray,
    ) -> Dict[str, Any]:
        """Stage-D-compatible inference API; returns an unfiltered displacement."""
        nodes = np.asarray(mesh.nodes, dtype=np.float64)
        cells = np.asarray(mesh.cells, dtype=np.int64)
        if nodes.shape != self.reference_nodes.shape or not np.array_equal(cells, self.cells):
            raise ValueError("La topologia/shape mesh non coincide con il riferimento r-adaptive.")
        field = np.ascontiguousarray(u, dtype=np.float64)
        if field.shape != (len(nodes),) or not np.all(np.isfinite(field)):
            raise ValueError("u non valida per il controller MMPDE.")
        disp, event = self.propose(nodes, grad_u)
        self.events.append(event)
        return {
            "raw_displacement": disp,
            "inference_time": float(event["elapsed_seconds"]),
            "mesh_functional_before": float(event["energy_before"]),
            "mesh_functional_after": float(event["energy_after"]),
            "accepted_pseudo_steps": int(event["accepted_pseudo_steps"]),
        }

    def reset_events(self) -> None:
        self.events.clear()
