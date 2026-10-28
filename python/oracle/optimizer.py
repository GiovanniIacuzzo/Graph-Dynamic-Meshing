"""
oracle/optimizer.py
Ottimizzatore vincolato offline per la determinazione della mesh ottimale X* e del target Delta X*.

Risolve il problema variazionale continuo:
    X* = argmin_X J(X; u_ref)
soggetto a:
    - J_e(X) >= epsilon   forall e in {1, ..., M}
    - X_i = X_i^{boundary} forall i in d_Omega

Funge da generatore del target di apprendimento Delta X* = X* - X_current
per la supervisione offline della GNN.
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
    Risolutore numerico offline per l'adattività variazionale r-adaptive della mesh.
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
        gtol: float = 1e-5
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
            epsilon=self.epsilon
        )

        # Monitoraggio cronologico dell'ottimizzazione
        self.history: List[float] = []

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
        val = res["total_objective"]
        return val

    def _callback(self, x_free_flat: np.ndarray) -> None:
        """Callback di monitoraggio invocato ad ogni iterazione dell'algoritmo."""
        val = self._objective_wrapper(x_free_flat)
        self.history.append(val)
    
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

        grad_free = grad_full[self.free_nodes_idx].reshape(-1)

        return (
            float(objective_value),
            np.ascontiguousarray(grad_free, dtype=np.float64)
        )


    def optimize(self) -> Dict[str, Any]:
        """
        Esegue l'ottimizzazione Oracle utilizzando il gradiente analitico.

        La mesh ottimizzata viene successivamente verificata attraverso
        il safety filter. Le coordinate di bordo rimangono fisse.
        """
        import time

        t_start = time.perf_counter()
        self.history.clear()

        x0_free = self.initial_nodes[self.free_nodes_idx].flatten()

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
                "maxiter": self.max_iter,
                "maxfun": max(15000, 50 * self.max_iter),
                "maxls": 20,
                "ftol": self.ftol,
                "gtol": self.gtol,
                "disp": False
            }
        )

        oracle_elapsed = time.perf_counter() - t_start

        # Ricostruzione del candidato non ancora filtrato.
        candidate_nodes = self._reconstruct_full_nodes(opt_res.x)
        raw_displacement = candidate_nodes - self.initial_nodes

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
            "oracle_time_seconds": float(oracle_elapsed),
            "time_seconds": float(oracle_elapsed),
            "nit": int(getattr(opt_res, "nit", 0)),
            "nfev": int(getattr(opt_res, "nfev", 0)),
            "njev": int(getattr(opt_res, "njev", 0)),
            "success": bool(opt_res.success),
            "status": int(opt_res.status),
            "message": str(opt_res.message),
            "gradient_method": "analytic"
        })

        return {
            "optimal_nodes": optimal_nodes,
            "target_displacement": target_displacement,
            "objective_history": self.history,
            "final_objective": float(
                final_metrics["total_objective"]
            ),
            "converged": bool(
                opt_res.success and final_metrics["is_valid"]
            ),
            "iterations": int(opt_res.nit),
            "final_metrics": final_metrics
        }
