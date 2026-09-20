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
            total_err_sq += np.sum(point_err * jacobians * weight)

        return float(total_err_sq)

    def evaluate(self, nodes: np.ndarray) -> Dict[str, Any]:
        """
        Valuta la funzione obiettivo variazionale multi-obiettivo su una configurazione candidata X.

        Input:
            nodes: coordinate candidate X, shape [N, 2] oppure vettore appiattito [2*N], float64

        Output:
            Dizionario con metriche:
                - total_objective     : valore scalare J(X)
                - interpolation_error : errore L2 quadratico integrato E_interp
                - movement_penalty    : termine quadratico normalizzato D_tilde
                - quality_penalty     : metrica di distorsione geometrica normalizzata Q_tilde
                - min_jacobian        : valore minimo riscontrato di J_e
                - is_valid            : flag booleano (True se rispetta vincoli geometrici e di bordo)
        """
        if nodes.ndim == 1:
            x_cand = nodes.reshape((self.num_nodes, 2))
        else:
            x_cand = nodes

        # 1. Verifica consistenza nodi di contorno
        bnd_err = np.max(np.abs(x_cand[self.boundary_nodes] - self.boundary_coords))
        is_boundary_preserved = bool(bnd_err < 1e-10)

        # 2. Calcolo metriche geometriche
        metrics = compute_mesh_metrics(x_cand, self.cells)
        jacobians = metrics["jacobians"]
        min_j = float(np.min(jacobians))

        # Se sono presenti elementi invertiti o sub-critici (J_e < epsilon), la configurazione non è ammissibile
        if min_j < self.epsilon or not is_boundary_preserved:
            # Penalizzazione asintotica con barriera quadratica per guidare solutori vincolati
            barrier_penalty = 1e6 + 1e5 * np.sum(np.maximum(0.0, self.epsilon - jacobians) ** 2)
            return {
                "total_objective": float(barrier_penalty),
                "interpolation_error": np.inf,
                "movement_penalty": np.inf,
                "quality_penalty": np.inf,
                "min_jacobian": min_j,
                "is_valid": False,
            }

        # 3. Calcolo termine di accuratezza normalizzato E_tilde
        e_interp = self._compute_interpolation_error(x_cand, jacobians)
        e_tilde = e_interp / self.norm_E

        # 4. Calcolo termine cinematico di movimento normalizzato D_tilde
        # Normalizzato rispetto al numero di nodi interni liberi N
        disp = x_cand - self.initial_nodes
        d_raw = float(np.sum(disp**2))
        d_tilde = d_raw / max(float(self.num_nodes), 1.0)

        # 5. Calcolo termine geometrico normalizzato Q_tilde
        # Basato sulla metrica di Escobar-Knupp: eta_e in (0, 1]. eta_e = 1 indica triangolo ideale.
        # Penalizziamo la deviazione dall'ideale: (1.0 - eta_e)
        eta = metrics["escobar_quality"]
        q_tilde = float(np.mean(1.0 - eta))

        # 6. Aggregazione del funzionale complessivo
        total_obj = e_tilde + self.lambda_D * d_tilde + self.lambda_Q * q_tilde

        return {
            "total_objective": float(total_obj),
            "interpolation_error": float(e_interp),
            "movement_penalty": float(d_tilde),
            "quality_penalty": float(q_tilde),
            "min_jacobian": min_j,
            "is_valid": True,
        }

    def __call__(self, nodes_flat: np.ndarray) -> float:
        """
        Interfaccia scalare continua diretta per librerie di ottimizzazione (es. scipy.optimize.minimize).
        """
        result = self.evaluate(nodes_flat)
        return result["total_objective"]