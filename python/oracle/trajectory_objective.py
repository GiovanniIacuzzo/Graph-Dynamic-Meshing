"""Space-time mesh-movement objective for transient manufactured solutions.

The candidate endpoint mesh X_1 defines a linear path
    X(s) = X_0 + s (X_1 - X_0),  s in [0,1].
The objective integrates normalized P1 interpolation error and mesh quality
along the future adaptation window, plus a single endpoint displacement cost.
The existing deterministic safety filter is applied by TrajectoryMeshOptimizer
and certifies the whole linear path against the configured Jacobian floor.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np

from oracle.native_geometry import compute_native_mesh_metrics_arrays as compute_mesh_metrics
from oracle.objective import OracleObjective
from oracle.mms_field import MMSFieldAtTime


def _triangle_degree5_quadrature() -> Tuple[np.ndarray, np.ndarray]:
    """Degree-five Dunavant quadrature; weights sum to reference area 1/2."""
    a = 0.059715871789770
    b = 0.470142064105115
    c = 0.797426985353087
    d = 0.101286507323456
    bary = [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [a, b, b], [b, a, b], [b, b, a],
        [c, d, d], [d, c, d], [d, d, c],
    ]
    weights = [0.1125] + [0.066197076394253] * 3 + [0.062969590272414] * 3
    return np.asarray(bary, dtype=np.float64), np.asarray(weights, dtype=np.float64)


class TrajectoryOracleObjective:
    """Quadrature approximation of a trajectory-aware relocation objective.

    At each temporal quadrature point the interpolation error is normalized by
    its value on the current mesh at that time. This avoids allowing a time
    instant with a larger field amplitude to dominate solely due to scale.
    ``lambda_D`` penalizes the endpoint displacement once; ``lambda_Q`` acts
    on mesh quality throughout the path. Boundary nodes are fixed by the
    optimizer, and pathwise validity is enforced by the safety filter after
    optimization.
    """

    def __init__(
        self,
        pde: Any,
        initial_nodes: np.ndarray,
        cells: np.ndarray,
        boundary_nodes: np.ndarray,
        t_start: float,
        horizon: float,
        lambda_D: float = 0.01,
        lambda_Q: float = 0.001,
        epsilon: float = 1e-4,
        time_quadrature_order: int = 3,
    ):
        self.pde = pde
        self.initial_nodes = np.ascontiguousarray(initial_nodes, dtype=np.float64)
        self.cells = np.ascontiguousarray(cells, dtype=np.int64)
        self.boundary_nodes = np.ascontiguousarray(boundary_nodes, dtype=np.int64)
        self.t_start = float(t_start)
        self.horizon = float(horizon)
        self.lambda_D = float(lambda_D)
        self.lambda_Q = float(lambda_Q)
        self.epsilon = float(epsilon)
        self.time_quadrature_order = int(time_quadrature_order)

        if self.initial_nodes.ndim != 2 or self.initial_nodes.shape[1] != 2:
            raise ValueError("initial_nodes deve avere shape [N,2].")
        if self.cells.ndim != 2 or self.cells.shape[1] != 3:
            raise ValueError("cells deve avere shape [M,3].")
        if self.horizon <= 0.0 or not np.isfinite(self.horizon):
            raise ValueError("horizon deve essere finito e positivo.")
        if self.t_start < 0.0 or self.t_start + self.horizon > float(pde.T) + 1e-12:
            raise ValueError("La finestra temporale deve essere contenuta in [0, pde.T].")
        if self.time_quadrature_order not in (2, 3, 4, 5):
            raise ValueError("time_quadrature_order supportato: 2, 3, 4 o 5.")
        if self.lambda_D < 0.0 or self.lambda_Q < 0.0:
            raise ValueError("lambda_D e lambda_Q devono essere non negativi.")

        nodes_s, weights_s = np.polynomial.legendre.leggauss(self.time_quadrature_order)
        self.s_nodes = 0.5 * (nodes_s + 1.0)
        self.s_weights = 0.5 * weights_s
        self.time_nodes = self.t_start + self.horizon * self.s_nodes

        self.boundary_coords = self.initial_nodes[self.boundary_nodes].copy()
        self.num_nodes = self.initial_nodes.shape[0]
        self.num_cells = self.cells.shape[0]
        self._space_objectives = []
        quad_bary, quad_weights = _triangle_degree5_quadrature()
        base_metrics = compute_mesh_metrics(self.initial_nodes, self.cells)
        if np.any(base_metrics["jacobians"] < self.epsilon):
            raise ValueError("La mesh iniziale non soddisfa il floor Jacobiano epsilon.")

        for t_q in self.time_nodes:
            obj = OracleObjective(
                reference_field=MMSFieldAtTime(pde, float(t_q)),
                initial_nodes=self.initial_nodes,
                cells=self.cells,
                boundary_nodes=self.boundary_nodes,
                lambda_D=0.0,
                lambda_Q=self.lambda_Q,
                epsilon=self.epsilon,
            )
            # Use a degree-five spatial rule for the narrow Gaussian times sine
            # factors, and recompute its normalization with the same rule.
            obj.quad_bary = quad_bary.copy()
            obj.quad_weights = quad_weights.copy()
            e0 = obj._compute_interpolation_error(
                self.initial_nodes, base_metrics["jacobians"]
            )
            obj.norm_E = max(float(e0), 1e-12)
            self._space_objectives.append(obj)

    def value_and_gradient(self, endpoint_nodes: np.ndarray) -> Tuple[float, np.ndarray]:
        endpoint = np.asarray(endpoint_nodes, dtype=np.float64)
        if endpoint.shape != self.initial_nodes.shape or not np.all(np.isfinite(endpoint)):
            raise ValueError("endpoint_nodes deve avere shape [N,2] e valori finiti.")
        if self.boundary_nodes.size and not np.allclose(
            endpoint[self.boundary_nodes], self.boundary_coords, rtol=0.0, atol=1e-10
        ):
            raise ValueError("I nodi di contorno devono rimanere fissi.")

        displacement = endpoint - self.initial_nodes
        total_value = 0.0
        total_gradient = np.zeros_like(endpoint)
        for s, w, spatial_obj in zip(self.s_nodes, self.s_weights, self._space_objectives):
            path_nodes = self.initial_nodes + float(s) * displacement
            value_q, gradient_q = spatial_obj.value_and_gradient(path_nodes)
            total_value += float(w) * float(value_q)
            total_gradient += float(w) * float(s) * gradient_q

        n_safe = max(float(self.num_nodes), 1.0)
        displacement_cost = float(np.sum(displacement**2) / n_safe)
        total_value += self.lambda_D * displacement_cost
        total_gradient += (2.0 * self.lambda_D / n_safe) * displacement
        return float(total_value), total_gradient

    def evaluate(self, endpoint_nodes: np.ndarray) -> Dict[str, Any]:
        value, gradient = self.value_and_gradient(endpoint_nodes)
        displacement = np.asarray(endpoint_nodes, dtype=np.float64) - self.initial_nodes
        per_time = []
        for s, w, t, spatial_obj in zip(
            self.s_nodes, self.s_weights, self.time_nodes, self._space_objectives
        ):
            path_nodes = self.initial_nodes + float(s) * displacement
            v = spatial_obj.evaluate(path_nodes)["total_objective"]
            per_time.append({"time": float(t), "path_fraction": float(s), "weight": float(w), "objective": float(v)})
        return {
            "total_objective": float(value),
            "gradient_l2": float(np.linalg.norm(gradient)),
            "displacement_cost": float(np.sum(displacement**2) / max(float(self.num_nodes), 1.0)),
            "time_samples": per_time,
            "t_start": self.t_start,
            "horizon": self.horizon,
        }
