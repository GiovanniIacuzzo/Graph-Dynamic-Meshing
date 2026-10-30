"""
oracle/optimizer.py
Ottimizzatore vincolato offline per generare una mesh di riferimento a budget finito.

Il solver cerca una riduzione del funzionale J(X; u_ref), ma un termine di budget
non costituisce una certificazione di ottimalità. La pipeline di dataset valuta
se la mesh risultante soddisfa il contratto esplicito di accettazione.
Vincoli geometrici:
    - J_e(X) >= epsilon   forall e in {1, ..., M}
    - X_i = X_i^{boundary} forall i in d_Omega

Funge da generatore del riferimento di apprendimento Delta X_ref = X_ref - X_current
per la supervisione offline della GNN; X_ref non è necessariamente un punto stazionario.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from scipy.optimize import minimize, OptimizeResult

from oracle.native_geometry import (
    compute_native_mesh_metrics_arrays as compute_mesh_metrics,
    apply_native_safety_filter_tuple as apply_safety_filter,
)
from oracle.field import GaussianBumpField
from oracle.objective import OracleObjective


class OfflineMeshOptimizer:
    """
    Generatore di riferimenti mesh r-adaptive a budget finito con diagnostica d'ottimizzazione.
    """

    def __init__(
        self,
        reference_field: GaussianBumpField,
        initial_nodes: np.ndarray,
        cells: np.ndarray,
        boundary_nodes: np.ndarray,
        lambda_D: float = 0.01,
        lambda_Q: float = 0.001,
        epsilon: float = 1e-4,
        max_iter: int = 250,
        ftol: float = 1e-7,
        gtol: float = 1e-5,
        quadrature_subdivision_level: int = 2,
    ):
        """
        Input:
            reference_field: istanza del campo analitico continuo u_ref
            initial_nodes  : coordinate nodali iniziali X_0, shape [N, 2], float64
            cells          : connettività topologica dei triangoli, shape [M, 3], int64
            boundary_nodes : indici dei vertici vincolati sul bordo d_Omega, shape [N_bnd]
            lambda_D       : peso del termine di penalizzazione dello spostamento
            lambda_Q       : peso del termine di regolarizzazione geometrica
            epsilon        : soglia minima ammissibile per il determinante dello Jacobiano
            max_iter       : numero massimo di iterazioni per l'ottimizzatore numerico
            ftol           : tolleranza relativa di arresto sul valore del funzionale
            gtol           : tolleranza sulla norma del gradiente proiettato
        """
        self.initial_nodes = np.ascontiguousarray(initial_nodes, dtype=np.float64)
        self.cells = np.ascontiguousarray(cells, dtype=np.int64)
        self.boundary_nodes = np.ascontiguousarray(boundary_nodes, dtype=np.int64)

        self.num_nodes: int = self.initial_nodes.shape[0]
        self.num_cells: int = self.cells.shape[0]
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.ftol = float(ftol)
        self.gtol = float(gtol)
        if (
            isinstance(quadrature_subdivision_level, bool)
            or int(quadrature_subdivision_level) != quadrature_subdivision_level
        ):
            raise ValueError("quadrature_subdivision_level must be an integer in [0, 4].")
        self.quadrature_subdivision_level = int(quadrature_subdivision_level)

        # Costruzione maschera booleana dei nodi interni liberi di muoversi
        self.is_boundary = np.zeros(self.num_nodes, dtype=bool)
        self.is_boundary[self.boundary_nodes] = True
        self.free_nodes_idx = np.where(~self.is_boundary)[0]
        self.num_free_nodes: int = len(self.free_nodes_idx)

        # Inizializzazione del funzionale obiettivo multiobiettivo
        self.objective = OracleObjective(
            reference_field=reference_field,
            initial_nodes=self.initial_nodes,
            cells=self.cells,
            boundary_nodes=self.boundary_nodes,
            lambda_D=lambda_D,
            lambda_Q=lambda_Q,
            epsilon=self.epsilon,
            quadrature_subdivision_level=self.quadrature_subdivision_level,
        )

        # Monitoraggio cronologico dell'ottimizzazione.  Cache the last
        # value/gradient pair so the SciPy callback does not recompute the
        # expensive quadrature objective at the same iterate.
        self.history: List[float] = []
        self._last_function_x: Optional[np.ndarray] = None
        self._last_function_value: Optional[float] = None

    def _reconstruct_full_nodes(self, x_free_flat: np.ndarray) -> np.ndarray:
        """
        Ricostruisce la matrice completa [N, 2] inserendo le coordinate interne
        e ripristinando rigorosamente le posizioni fisse di Dirichlet sul contorno.
        """
        x_full = self.initial_nodes.copy()
        x_full[self.free_nodes_idx] = x_free_flat.reshape((self.num_free_nodes, 2))
        return x_full

    def _objective_wrapper(self, x_free_flat: np.ndarray) -> float:
        """
        Funzione ponte scalare valutata unicamente sui gradi di libertà interni.
        """
        x_full = self._reconstruct_full_nodes(x_free_flat)
        res = self.objective.evaluate(x_full)
        val = float(res["total_objective"])
        self._last_function_x = np.array(x_free_flat, dtype=np.float64, copy=True)
        self._last_function_value = val
        return val

    def _callback(self, x_free_flat: np.ndarray) -> None:
        """Record objective history without repeating quadrature when possible."""
        if (
            self._last_function_x is not None
            and self._last_function_value is not None
            and np.array_equal(x_free_flat, self._last_function_x)
        ):
            self.history.append(float(self._last_function_value))
            return
        val = self._objective_wrapper(x_free_flat)
        self.history.append(float(val))
    
    def _objective_and_gradient_wrapper(
        self,
        x_free_flat: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Ricostruisce la mesh e restituisce il funzionale insieme
        al gradiente analitico rispetto ai soli nodi interni liberi.
        """
        x_full = self._reconstruct_full_nodes(x_free_flat)

        objective_value, grad_full = (
            self.objective.value_and_gradient(x_full)
        )
        self._last_function_x = np.array(x_free_flat, dtype=np.float64, copy=True)
        self._last_function_value = float(objective_value)

        grad_free = grad_full[self.free_nodes_idx].reshape(-1)

        return (
            float(objective_value),
            np.ascontiguousarray(grad_free, dtype=np.float64)
        )


    def optimize(
        self,
        initial_guess_nodes: Optional[np.ndarray] = None,
        max_iter_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Optimize from the reference mesh or continue from a prior candidate.

        ``initial_guess_nodes`` changes only the numerical starting point. The
        objective's reference mesh, displacement regularizer, and fixed boundary
        coordinates remain those passed to the constructor. This supports a
        controlled L-BFGS-B continuation when the iteration budget is reached.
        """
        import time

        t_start = time.perf_counter()
        self.history.clear()
        self._last_function_x = None
        self._last_function_value = None

        if initial_guess_nodes is None:
            x0_nodes = self.initial_nodes
        else:
            x0_nodes = np.array(initial_guess_nodes, dtype=np.float64, order="C", copy=True)
            if x0_nodes.shape != self.initial_nodes.shape:
                raise ValueError(
                    f"initial_guess_nodes deve avere shape {self.initial_nodes.shape}, ricevuta {x0_nodes.shape}."
                )
            if not np.all(np.isfinite(x0_nodes)):
                raise ValueError("initial_guess_nodes contiene valori non finiti.")
            # Boundary conditions are exact Dirichlet data, not optimization variables.
            x0_nodes[self.boundary_nodes] = self.objective.boundary_coords
        x0_free = x0_nodes[self.free_nodes_idx].flatten()

        initial_obj = self._objective_wrapper(x0_free)
        self.history.append(initial_obj)

        bounds = [(0.0, 1.0)] * (2 * self.num_free_nodes)

        opt_res: OptimizeResult = minimize(
            fun=self._objective_and_gradient_wrapper,
            x0=x0_free,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            callback=self._callback,
            options={
                "maxiter": int(max_iter_override if max_iter_override is not None else self.max_iter),
                "maxfun": max(15000, 50 * self.max_iter),
                "maxls": 20,
                "ftol": self.ftol,
                "gtol": self.gtol,
                "disp": False
            }
        )

        # Diagnose the termination independently of SciPy's success flag.
        # L-BFGS-B can stop either on relative objective reduction (ftol) or
        # projected-gradient tolerance (gtol); both criteria are retained in
        # the sample metadata instead of being conflated.
        final_fun_raw, final_grad_free = self._objective_and_gradient_wrapper(opt_res.x)
        projected_gradient = opt_res.x - np.clip(opt_res.x - final_grad_free, 0.0, 1.0)
        projected_gradient_inf = float(np.max(np.abs(projected_gradient))) if projected_gradient.size else 0.0
        projected_gradient_l2 = float(np.linalg.norm(projected_gradient))
        gradient_tolerance_met = bool(projected_gradient_inf <= self.gtol)
        termination_message = str(opt_res.message)
        termination_by_function_reduction = bool(
            "RELATIVE REDUCTION OF F" in termination_message.upper()
        )
        termination_by_projected_gradient = bool(
            "PROJECTED GRADIENT" in termination_message.upper()
        )

        # Ricostruzione del candidato non ancora filtrato.
        candidate_nodes = self._reconstruct_full_nodes(opt_res.x)
        raw_displacement = candidate_nodes - self.initial_nodes
        candidate_metrics = self.objective.evaluate(candidate_nodes)

        # Filtro geometrico di sicurezza.
        alpha, safe_nodes, is_safe, min_j = apply_safety_filter(
            nodes=self.initial_nodes,
            displacement=raw_displacement,
            cells=self.cells,
            epsilon=self.epsilon
        )

        optimal_nodes = safe_nodes
        target_displacement = optimal_nodes - self.initial_nodes

        # Metriche calcolate sulla mesh finale effettivamente restituita.
        final_metrics = self.objective.evaluate(optimal_nodes)
        geom_summary = compute_mesh_metrics(
            optimal_nodes,
            self.cells
        )
        # Include optimizer calls plus post-optimization diagnostics and safety
        # validation in this wall-clock metric. Constructor normalization is
        # intentionally excluded and generation.py also records end-to-end time.
        oracle_elapsed = time.perf_counter() - t_start

        final_metrics.update({
            "min_jacobian": float(
                np.min(geom_summary["jacobians"])
            ),
            "mean_jacobian": float(
                np.mean(geom_summary["jacobians"])
            ),
            "min_escobar_quality": float(
                np.min(geom_summary["escobar_quality"])
            ),
            "mean_escobar_quality": float(
                np.mean(geom_summary["escobar_quality"])
            ),
            "max_aspect_ratio": float(
                np.max(geom_summary["aspect_ratio"])
            ),
            "num_inverted_cells": int(
                np.sum(geom_summary["inverted"])
            ),
            "safety_alpha_applied": float(alpha),
            "safety_filter_accepted": bool(is_safe),
            "safety_filter_active": bool(alpha < 1.0 - 1.0e-12),
            "candidate_is_valid": bool(candidate_metrics["is_valid"]),
            "candidate_min_jacobian": float(candidate_metrics["min_jacobian"]),
            "projected_gradient_inf_norm": projected_gradient_inf,
            "projected_gradient_l2_norm": projected_gradient_l2,
            "gradient_tolerance": float(self.gtol),
            "gradient_tolerance_met": gradient_tolerance_met,
            "first_order_stationary": gradient_tolerance_met,
            "termination_by_function_reduction": termination_by_function_reduction,
            "termination_by_projected_gradient": termination_by_projected_gradient,
            "raw_final_objective": float(final_fun_raw),
            "oracle_time_seconds": float(oracle_elapsed),
            "time_seconds": float(oracle_elapsed),
            "nit": int(getattr(opt_res, "nit", 0)),
            "nfev": int(getattr(opt_res, "nfev", 0)),
            "njev": int(getattr(opt_res, "njev", 0)),
            "success": bool(opt_res.success),
            "status": int(opt_res.status),
            "message": termination_message,
            "gradient_method": "analytic",
            "quadrature_rule": self.objective.quadrature_rule,
            "quadrature_subdivision_level": int(self.objective.quadrature_subdivision_level),
            "quadrature_points_per_cell": int(self.objective.quad_bary.shape[0]),
        })

        return {
            "optimal_nodes": optimal_nodes,
            # Raw L-BFGS-B candidate is exposed only to support a continuation
            # pass. ``optimal_nodes`` remains the safety-filtered target.
            "candidate_nodes": np.ascontiguousarray(candidate_nodes, dtype=np.float64),
            "success": bool(opt_res.success),
            "status": int(opt_res.status),
            "message": termination_message,
            "projected_gradient_inf_norm": projected_gradient_inf,
            "projected_gradient_l2_norm": projected_gradient_l2,
            "gradient_tolerance_met": gradient_tolerance_met,
            "first_order_stationary": gradient_tolerance_met,
            "termination_by_function_reduction": termination_by_function_reduction,
            "termination_by_projected_gradient": termination_by_projected_gradient,
            "nfev": int(getattr(opt_res, "nfev", 0)),
            "njev": int(getattr(opt_res, "njev", 0)),
            "target_displacement": target_displacement,
            "objective_history": self.history,
            "starting_objective": float(initial_obj),
            "final_objective": float(
                final_metrics["total_objective"]
            ),
            # Distinguish SciPy termination success from stationarity of the
            # constrained objective. Function-reduction termination alone is
            # not enough to label this an oracle optimum.
            "converged": bool(
                opt_res.success
                and bool(candidate_metrics["is_valid"])
                and gradient_tolerance_met
            ),
            "iterations": int(opt_res.nit),
            "final_metrics": final_metrics
        }
