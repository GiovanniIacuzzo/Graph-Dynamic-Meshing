"""Offline optimizer for trajectory-aware mesh relocation labels."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from scipy.optimize import minimize, OptimizeResult

from oracle.native_geometry import (
    compute_native_mesh_metrics_arrays as compute_mesh_metrics,
    apply_native_safety_filter_tuple as apply_safety_filter,
)
from oracle.trajectory_objective import TrajectoryOracleObjective


class TrajectoryMeshOptimizer:
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
        max_iter: int = 200,
        ftol: float = 1e-7,
        gtol: float = 1e-5,
        time_quadrature_order: int = 3,
    ):
        self.initial_nodes = np.ascontiguousarray(initial_nodes, dtype=np.float64)
        self.cells = np.ascontiguousarray(cells, dtype=np.int64)
        self.boundary_nodes = np.ascontiguousarray(boundary_nodes, dtype=np.int64)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.ftol = float(ftol)
        self.gtol = float(gtol)
        self.is_boundary = np.zeros(self.initial_nodes.shape[0], dtype=bool)
        self.is_boundary[self.boundary_nodes] = True
        self.free_nodes_idx = np.flatnonzero(~self.is_boundary)
        self.objective = TrajectoryOracleObjective(
            pde=pde,
            initial_nodes=self.initial_nodes,
            cells=self.cells,
            boundary_nodes=self.boundary_nodes,
            t_start=t_start,
            horizon=horizon,
            lambda_D=lambda_D,
            lambda_Q=lambda_Q,
            epsilon=epsilon,
            time_quadrature_order=time_quadrature_order,
        )

    def _nodes_from_free(self, free_flat: np.ndarray) -> np.ndarray:
        nodes = self.initial_nodes.copy()
        nodes[self.free_nodes_idx] = np.asarray(free_flat, dtype=np.float64).reshape(-1, 2)
        # Exact preservation rather than tolerance-based boundary reconstruction.
        nodes[self.boundary_nodes] = self.initial_nodes[self.boundary_nodes]
        return nodes

    def _fun_grad(self, free_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        nodes = self._nodes_from_free(free_flat)
        value, gradient = self.objective.value_and_gradient(nodes)
        grad_free = gradient[self.free_nodes_idx].reshape(-1)
        return float(value), np.ascontiguousarray(grad_free, dtype=np.float64)

    def optimize(self) -> Dict[str, Any]:
        import time
        t0 = time.perf_counter()
        initial_objective, _ = self.objective.value_and_gradient(self.initial_nodes)
        x0 = self.initial_nodes[self.free_nodes_idx].reshape(-1).copy()
        result: OptimizeResult = minimize(
            fun=self._fun_grad,
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, 1.0)] * x0.size,
            options={
                "maxiter": self.max_iter,
                "maxfun": max(5000, 40 * self.max_iter),
                "maxls": 20,
                "ftol": self.ftol,
                "gtol": self.gtol,
            },
        )
        optimize_seconds = time.perf_counter() - t0
        candidate = self._nodes_from_free(result.x)
        raw_displacement = candidate - self.initial_nodes
        raw_candidate_objective, _ = self.objective.value_and_gradient(candidate)
        t_sf = time.perf_counter()
        alpha, safe_nodes, filter_accepted, min_j_filter = apply_safety_filter(
            nodes=self.initial_nodes,
            displacement=raw_displacement,
            cells=self.cells,
            epsilon=self.epsilon,
        )
        safety_seconds = time.perf_counter() - t_sf
        geom = compute_mesh_metrics(safe_nodes, self.cells)
        min_j = float(np.min(geom["jacobians"]))
        boundary_error = float(np.max(np.abs(
            safe_nodes[self.boundary_nodes] - self.initial_nodes[self.boundary_nodes]
        ))) if self.boundary_nodes.size else 0.0
        if min_j < self.epsilon - 1e-10 or boundary_error > 1e-10:
            raise RuntimeError(
                "Il risultato filtrato non soddisfa gli invarianti geometrici: "
                f"min_J={min_j:.6e}, boundary_error={boundary_error:.3e}."
            )
        summary = self.objective.evaluate(safe_nodes)
        summary.update({
            "initial_objective": float(initial_objective),
            "raw_candidate_objective": float(raw_candidate_objective),
            "safe_final_objective": float(summary["total_objective"]),
            "relative_objective_reduction_raw": float(
                (initial_objective - raw_candidate_objective) / max(abs(initial_objective), 1e-12)
            ),
            "min_jacobian": min_j,
            "mean_jacobian": float(np.mean(geom["jacobians"])),
            "min_escobar_quality": float(np.min(geom["escobar_quality"])),
            "mean_escobar_quality": float(np.mean(geom["escobar_quality"])),
            "num_inverted_cells": int(np.sum(geom["inverted"])),
            "safety_alpha_applied": float(alpha),
            "safety_filter_accepted_full_displacement": bool(filter_accepted),
            "safety_filter_min_jacobian": float(min_j_filter),
            "oracle_time_seconds": float(optimize_seconds),
            "safety_time_seconds": float(safety_seconds),
            "iterations": int(getattr(result, "nit", 0)),
            "nfev": int(getattr(result, "nfev", 0)),
            "njev": int(getattr(result, "njev", 0)),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
        })
        return {
            "optimal_nodes": safe_nodes,
            "target_displacement": safe_nodes - self.initial_nodes,
            "raw_candidate_nodes": candidate,
            "objective_history": [],
            "final_objective": float(summary["total_objective"]),
            "converged": bool(result.success and min_j >= self.epsilon and boundary_error <= 1e-10),
            "iterations": int(getattr(result, "nit", 0)),
            "final_metrics": summary,
        }
