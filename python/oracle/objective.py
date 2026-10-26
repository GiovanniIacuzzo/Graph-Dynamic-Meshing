"""
oracle/objective.py
Funzione obiettivo variazionale multi-obiettivo per l'ottimizzatore offline dell'oracle.

Implementa:
    J(X) = E_tilde_interp(X) + lambda_D * D_tilde(X) + lambda_Q * Q_tilde(X)

Termini formulati e normalizzati:
    1. E_interp(X) = || u_ref - Pi_X u_ref ||_{L^2(Omega)}^2 (via quadratura di Gauss 2D)
    2. D(X)        = || X - X_current ||_F^2 / N
    3. Q(X)        = sum_e (1.0 - eta_e(X)) / M  (con eta_e metrica di Escobar-Knupp)

Vincoli verificati:
    - Non-inversione locale: min_e J_e(X) >= epsilon
    - Condizioni al contorno: X_i = X_i^{boundary} per i in d_Omega
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from geometry.geometry import compute_mesh_metrics
from oracle.field import GaussianBumpField


class OracleObjective:
    """
    Valutatore variazionale continuo per la ricerca della configurazione nodale ottimale X*.
    """

    def __init__(
        self,
        reference_field: GaussianBumpField,
        initial_nodes: np.ndarray,
        cells: np.ndarray,
        boundary_nodes: np.ndarray,
        lambda_D: float = 0.01,
        lambda_Q: float = 0.001,
        epsilon: float = 1e-4
    ):
        """
        Input:
            reference_field: istanza del campo continuo analitico u_ref(x, y)
            initial_nodes  : coordinate della mesh non adattata X_0, shape [N, 2], float64
            cells          : connettività topologica dei triangoli, shape [M, 3], int64
            boundary_nodes : indici 1D dei vertici vincolati sul contorno d_Omega, shape [N_bnd]
            lambda_D       : peso per la penalizzazione dello spostamento (D_tilde)
            lambda_Q       : peso per la qualità geometrica (Q_tilde)
            epsilon        : soglia minima ammissibile per il determinante dello Jacobiano
        """
        self.field = reference_field
        self.cells = np.ascontiguousarray(cells, dtype=np.int64)
        self.boundary_nodes = np.ascontiguousarray(boundary_nodes, dtype=np.int64)
        self.initial_nodes = np.ascontiguousarray(initial_nodes, dtype=np.float64)

        self.num_nodes: int = self.initial_nodes.shape[0]
        self.num_cells: int = self.cells.shape[0]

        self.lambda_D = float(lambda_D)
        self.lambda_Q = float(lambda_Q)
        self.epsilon = float(epsilon)
        self.optimization_jacobian_target = 3.0 * self.epsilon
        self.optimization_penalty_weight = 1e7

        # Coordinate di Dirichlet prefissate sul bordo
        self.boundary_coords = self.initial_nodes[self.boundary_nodes].copy()

        # Punti e pesi della regola di quadratura di Gauss simmetrica a 3 punti su simplesso
        # Coordinate baricentriche (zeta_1, zeta_2, zeta_3) con peso w_k = 1/3 (area simplesso rif. = 1/2)
        self.quad_bary = np.array([
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]
        ], dtype=np.float64)
        self.quad_weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float64)

        # Stima del fattore di normalizzazione iniziale E_0 per garantire scala unitaria
        metrics_0 = compute_mesh_metrics(self.initial_nodes, self.cells)
        assert np.all(metrics_0["jacobians"] >= self.epsilon), "La mesh iniziale viola la soglia epsilon"
        e_interp_0 = self._compute_interpolation_error(self.initial_nodes, metrics_0["jacobians"])
        self.norm_E = max(float(e_interp_0), 1e-12)

    def _compute_interpolation_error(self, nodes: np.ndarray, jacobians: np.ndarray) -> float:
        """
        Calcola l'integrale numerico dell'errore L2:
            || u_ref - Pi_X u_ref ||_{L^2(Omega)}^2 = sum_e int_e (u_ref - u_h)^2 dOmega
        utilizzando quadratura di Gauss su ciascun elemento.
        """
        # Coordinate dei vertici degli M triangoli: [M, 3, 2]
        tri_coords = nodes[self.cells]  # shape: [M, 3, 2]

        # Valutazione del campo nodale discreto u_h ai vertici (interpolante P1)
        u_nodes, _, _, _ = self.field.evaluate(nodes[:, 0], nodes[:, 1])
        u_tri_nodes = u_nodes[self.cells]  # [M, 3]

        total_err_sq = 0.0

        # Ciclo sui 3 punti di quadratura di Gauss
        for q_idx in range(3):
            bary = self.quad_bary[q_idx]  # [3]
            weight = self.quad_weights[q_idx]

            # Coordinate fisiche (x_q, y_q) del punto di quadratura: [M, 2]
            xq = (
                bary[0] * tri_coords[:, 0, 0]
                + bary[1] * tri_coords[:, 1, 0]
                + bary[2] * tri_coords[:, 2, 0]
            )
            yq = (
                bary[0] * tri_coords[:, 0, 1]
                + bary[1] * tri_coords[:, 1, 1]
                + bary[2] * tri_coords[:, 2, 1]
            )

            # Valore esatto continuo u_ref(x_q, y_q)
            u_exact, _, _, _ = self.field.evaluate(xq, yq)

            # Valore interpolato lineare P1: u_h(x_q, y_q) = sum_i zeta_i * u_i
            u_interp = (
                bary[0] * u_tri_nodes[:, 0]
                + bary[1] * u_tri_nodes[:, 1]
                + bary[2] * u_tri_nodes[:, 2]
            )

            # Integrazione locale: (u_exact - u_interp)^2 * det(J_e) * weight
            point_err = (u_exact - u_interp) ** 2
            total_err_sq += np.sum(point_err * np.abs(jacobians) * weight)

        return float(total_err_sq)


    @staticmethod
    def _triangle_jacobian_gradients(
        tri_coords: np.ndarray
    ) -> np.ndarray:
        """
        Calcola le derivate del determinante Jacobiano rispetto
        alle coordinate dei tre vertici di ciascun triangolo.

        Args:
            tri_coords: coordinate dei triangoli, shape [M, 3, 2].

        Returns:
            grad_j: shape [M, 3, 2].
                    grad_j[e, i, d] = dJ_e / dX_{i,d}.
        """
        p1 = tri_coords[:, 0, :]
        p2 = tri_coords[:, 1, :]
        p3 = tri_coords[:, 2, :]

        grad_j = np.empty(
            (tri_coords.shape[0], 3, 2),
            dtype=np.float64
        )

        # Derivate rispetto al primo vertice.
        grad_j[:, 0, 0] = p2[:, 1] - p3[:, 1]
        grad_j[:, 0, 1] = p3[:, 0] - p2[:, 0]

        # Derivate rispetto al secondo vertice.
        grad_j[:, 1, 0] = p3[:, 1] - p1[:, 1]
        grad_j[:, 1, 1] = p1[:, 0] - p3[:, 0]

        # Derivate rispetto al terzo vertice.
        grad_j[:, 2, 0] = p1[:, 1] - p2[:, 1]
        grad_j[:, 2, 1] = p2[:, 0] - p1[:, 0]

        return grad_j


    def _compute_interpolation_error_and_gradient(
        self,
        nodes: np.ndarray,
        jacobians: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Calcola l'errore L2 quadratico integrato e il gradiente analitico.

        Usa |J| per integrare anche configurazioni intermedie non ammissibili.
        Per J > 0 coincide con il funzionale originale.
        """
        tri_coords = nodes[self.cells]

        u_nodes, grad_u_nodes, _, _ = self.field.evaluate(
            nodes[:, 0], nodes[:, 1]
        )

        u_tri_nodes = u_nodes[self.cells]
        grad_u_tri_nodes = grad_u_nodes[self.cells]

        grad_jacobians = self._triangle_jacobian_gradients(tri_coords)

        jacobian_measure = np.abs(jacobians)
        jacobian_sign = np.where(jacobians >= 0.0, 1.0, -1.0)
        grad_measure_jacobians = (
            grad_jacobians * jacobian_sign[:, None, None]
        )

        total_err_sq = 0.0
        grad_local = np.zeros(
            (self.num_cells, 3, 2), dtype=np.float64
        )

        for q_idx in range(3):
            bary = self.quad_bary[q_idx]
            weight = self.quad_weights[q_idx]

            xq = (
                bary[0] * tri_coords[:, 0, 0]
                + bary[1] * tri_coords[:, 1, 0]
                + bary[2] * tri_coords[:, 2, 0]
            )
            yq = (
                bary[0] * tri_coords[:, 0, 1]
                + bary[1] * tri_coords[:, 1, 1]
                + bary[2] * tri_coords[:, 2, 1]
            )

            u_exact, grad_exact, _, _ = self.field.evaluate(xq, yq)

            u_interp = (
                bary[0] * u_tri_nodes[:, 0]
                + bary[1] * u_tri_nodes[:, 1]
                + bary[2] * u_tri_nodes[:, 2]
            )

            residual = u_exact - u_interp
            residual_sq = residual**2

            total_err_sq += np.sum(
                residual_sq * jacobian_measure * weight
            )

            for local_i in range(3):
                grad_residual = bary[local_i] * (
                    grad_exact - grad_u_tri_nodes[:, local_i, :]
                )

                grad_local[:, local_i, :] += weight * (
                    grad_measure_jacobians[:, local_i, :]
                    * residual_sq[:, None]
                    + 2.0
                    * jacobian_measure[:, None]
                    * residual[:, None]
                    * grad_residual
                )

        grad_nodes = np.zeros_like(nodes, dtype=np.float64)

        for local_i in range(3):
            np.add.at(
                grad_nodes,
                self.cells[:, local_i],
                grad_local[:, local_i, :]
            )

        return float(total_err_sq), grad_nodes

    def value_and_gradient(
        self,
        nodes: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Restituisce il funzionale Oracle e il gradiente analitico.

        Le configurazioni valide mantengono il funzionale originale.
        Per le configurazioni intermedie non ammissibili viene utilizzata
        una penalizzazione quadratica continua dei Jacobiani.
        """
        x_cand = np.asarray(nodes, dtype=np.float64)

        if x_cand.ndim == 1:
            if x_cand.size != 2 * self.num_nodes:
                raise ValueError(
                    "Il vettore nodale appiattito ha dimensione errata."
                )
            x_cand = x_cand.reshape(self.num_nodes, 2)

        if x_cand.shape != (self.num_nodes, 2):
            raise ValueError(
                f"Shape nodi non valida: {x_cand.shape}; "
                f"attesa {(self.num_nodes, 2)}."
            )

        if not np.all(np.isfinite(x_cand)):
            raise ValueError("Le coordinate nodali contengono NaN o Inf.")

        metrics = compute_mesh_metrics(x_cand, self.cells)
        jacobians = metrics["jacobians"]
        tri_coords = x_cand[self.cells]

        # 1. Errore d'interpolazione e gradiente.
        e_interp, grad_e = (
            self._compute_interpolation_error_and_gradient(
                x_cand, jacobians
            )
        )
        e_tilde = e_interp / self.norm_E
        grad_nodes = grad_e / self.norm_E

        # 2. Penalizzazione dello spostamento.
        displacement = x_cand - self.initial_nodes
        num_nodes_safe = max(float(self.num_nodes), 1.0)

        d_tilde = float(
            np.sum(displacement**2) / num_nodes_safe
        )

        grad_nodes += (
            2.0 * self.lambda_D / num_nodes_safe
        ) * displacement

        # 3. Penalizzazione della qualità di Escobar-Knupp.
        p1 = tri_coords[:, 0, :]
        p2 = tri_coords[:, 1, :]
        p3 = tri_coords[:, 2, :]

        sum_l_sq = (
            np.sum((p2 - p1)**2, axis=1)
            + np.sum((p3 - p2)**2, axis=1)
            + np.sum((p1 - p3)**2, axis=1)
        )
        safe_sum_l_sq = np.maximum(sum_l_sq, 1e-14)

        sqrt3_factor = 2.0 * np.sqrt(3.0)
        eta_raw = sqrt3_factor * jacobians / safe_sum_l_sq

        eta = metrics["escobar_quality"]
        q_tilde = float(np.mean(1.0 - eta))

        grad_jacobians = self._triangle_jacobian_gradients(
            tri_coords
        )

        grad_sum_l_sq = np.empty_like(grad_jacobians)
        grad_sum_l_sq[:, 0, :] = 2.0 * (
            2.0 * p1 - p2 - p3
        )
        grad_sum_l_sq[:, 1, :] = 2.0 * (
            2.0 * p2 - p1 - p3
        )
        grad_sum_l_sq[:, 2, :] = 2.0 * (
            2.0 * p3 - p1 - p2
        )

        grad_eta = sqrt3_factor * (
            grad_jacobians / safe_sum_l_sq[:, None, None]
            - jacobians[:, None, None]
            * grad_sum_l_sq
            / safe_sum_l_sq[:, None, None]**2
        )

        active_eta = (eta_raw > 0.0) & (eta_raw < 1.0)
        grad_eta *= active_eta[:, None, None]

        quality_factor = (
            -self.lambda_Q / max(float(self.num_cells), 1.0)
        )

        for local_i in range(3):
            np.add.at(
                grad_nodes,
                self.cells[:, local_i],
                quality_factor * grad_eta[:, local_i, :]
            )

        # 4. Penalizzazione quadratica continua delle violazioni.
        penalty_weight = self.optimization_penalty_weight

        violations = np.maximum(
            0.0,
            self.optimization_jacobian_target - jacobians
        )

        constraint_penalty = penalty_weight * float(
            np.sum(violations**2)
        )

        for local_i in range(3):
            local_grad = (
                -2.0
                * penalty_weight
                * violations[:, None]
                * grad_jacobians[:, local_i, :]
            )

            np.add.at(
                grad_nodes,
                self.cells[:, local_i],
                local_grad
            )

        # 5. Funzionale totale.
        total_objective = (
            e_tilde
            + self.lambda_D * d_tilde
            + self.lambda_Q * q_tilde
            + constraint_penalty
        )

        return float(total_objective), grad_nodes

    def evaluate(self, nodes: np.ndarray) -> Dict[str, Any]:
        """
        Valuta il funzionale Oracle usando la stessa penalizzazione
        geometrica di value_and_gradient().
        """
        x_cand = np.asarray(nodes, dtype=np.float64)

        if x_cand.ndim == 1:
            if x_cand.size != 2 * self.num_nodes:
                raise ValueError(
                    "Il vettore nodale appiattito ha dimensione errata."
                )
            x_cand = x_cand.reshape(self.num_nodes, 2)

        if x_cand.shape != (self.num_nodes, 2):
            raise ValueError(
                f"Shape nodi non valida: {x_cand.shape}; "
                f"attesa {(self.num_nodes, 2)}."
            )

        if not np.all(np.isfinite(x_cand)):
            raise ValueError("Le coordinate nodali contengono NaN o Inf.")

        if self.boundary_nodes.size:
            bnd_err = float(np.max(np.abs(
                x_cand[self.boundary_nodes] - self.boundary_coords
            )))
        else:
            bnd_err = 0.0

        boundary_preserved = bool(bnd_err < 1e-10)

        metrics = compute_mesh_metrics(x_cand, self.cells)
        jacobians = metrics["jacobians"]
        min_j = float(np.min(jacobians))

        # 1. Errore d'interpolazione normalizzato.
        e_interp = self._compute_interpolation_error(
            x_cand, jacobians
        )
        e_tilde = e_interp / self.norm_E

        # 2. Penalizzazione dello spostamento.
        displacement = x_cand - self.initial_nodes
        d_tilde = (
            float(np.sum(displacement**2))
            / max(float(self.num_nodes), 1.0)
        )

        # 3. Penalizzazione della qualità geometrica.
        q_tilde = float(np.mean(1.0 - metrics["escobar_quality"]))

        # 4. Penalizzazione rafforzata coerente con il gradiente.
        violations = np.maximum(
            0.0,
            self.optimization_jacobian_target - jacobians
        )
        constraint_penalty = self.optimization_penalty_weight * float(
            np.sum(violations**2)
        )

        total_obj = (
            e_tilde
            + self.lambda_D * d_tilde
            + self.lambda_Q * q_tilde
            + constraint_penalty
        )

        is_valid = bool(
            min_j >= self.epsilon and boundary_preserved
        )

        return {
            "total_objective": float(total_obj),
            "interpolation_error": float(e_interp),
            "movement_penalty": float(d_tilde),
            "quality_penalty": float(q_tilde),
            "constraint_penalty": float(constraint_penalty),
            "min_jacobian": min_j,
            "is_valid": is_valid,
        }