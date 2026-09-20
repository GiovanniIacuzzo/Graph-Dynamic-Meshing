"""
numerics/ale.py
Integrazione differenziale Arbitrary Lagrangian-Eulerian (ALE) su domini a mesh mobile.

Risolve il bilancio di trasporto-diffusione sul dominio tempo-variante Omega(t):
    d/dt int_{Omega(t)} u dOmega + int_{Omega(t)} ((b - w) . grad(u)) dOmega
        + int_{Omega(t)} nu grad(u) . grad(v) dOmega = int_{Omega(t)} f v dOmega

dove:
    w = (X_{t+1} - X_t) / dt  e' la velocita' arbitraria della griglia.

Include:
    - Stabilizzazione SUPG sulla velocita' convettiva relativa c_rel = b - w.
    - Calcolo separato e rigoroso del residuo della Geometric Conservation Law (GCL).
"""

from typing import Tuple, Dict, Any, Optional
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from geometry.mesh import TriangularMesh2D
from numerics.pde import BasePDE


class ALESolver2D:
    """
    Risolutore numerico ALE continuo P1 per convezione-diffusione su mesh triangolari mobili.
    """

    def __init__(
        self,
        pde: BasePDE,
        theta: float = 1.0,
        use_supg: bool = True
    ):
        """
        Input:
            pde     : istanza di BasePDE con parametri fisici e forzante
            theta   : parametro di time-stepping (1.0 = Implicit Euler, 0.5 = Crank-Nicolson)
            use_supg: flag per abilitare la stabilizzazione SUPG sulla velocità relativa
        """
        self.pde = pde
        self.theta = float(theta)
        self.use_supg = bool(use_supg)

    @staticmethod
    def compute_mesh_velocity(
        mesh_t: TriangularMesh2D,
        mesh_t1: TriangularMesh2D,
        dt: float
    ) -> np.ndarray:
        """
        Calcola la velocità nodale della griglia:
            w = (X_{t+1} - X_t) / dt
        Returns:
            array [N, 2], float64
        """
        assert dt > 0.0, f"dt deve essere positivo, ottenuto: {dt}"
        assert mesh_t.num_nodes == mesh_t1.num_nodes, "Le due mesh devono avere lo stesso numero di nodi"
        w = (mesh_t1.nodes - mesh_t.nodes) / dt
        return np.ascontiguousarray(w, dtype=np.float64)

    def verify_gcl(
        self,
        mesh_t: TriangularMesh2D,
        mesh_t1: TriangularMesh2D,
        dt: float
    ) -> Tuple[float, np.ndarray]:
        """
        Verifica la Geometric Conservation Law (GCL) a livello discreto:
        Uno stato costante u == 1 deve soddisfare:
            int_{Omega_{t+1}} 1 dOmega - int_{Omega_t} 1 dOmega - int_t^{t+1} int_{Omega} div(w) dOmega dt = 0

        Returns:
            gcl_residual_norm : norma L2 normalizzata del residuo GCL su tutti i nodi
            gcl_nodal_residual: vettore dei residui per nodo [N]
        """
        # Coordinate e connettività
        cells = mesh_t.cells
        x_t, y_t = mesh_t.nodes[:, 0], mesh_t.nodes[:, 1]
        x_t1, y_t1 = mesh_t1.nodes[:, 0], mesh_t1.nodes[:, 1]

        # Calcolo aree triangolari a t e t+1
        j_t = (x_t[cells[:, 1]] - x_t[cells[:, 0]]) * (y_t[cells[:, 2]] - y_t[cells[:, 0]]) - \
              (x_t[cells[:, 2]] - x_t[cells[:, 0]]) * (y_t[cells[:, 1]] - y_t[cells[:, 0]])
        areas_t = 0.5 * j_t

        j_t1 = (x_t1[cells[:, 1]] - x_t1[cells[:, 0]]) * (y_t1[cells[:, 2]] - y_t1[cells[:, 0]]) - \
               (x_t1[cells[:, 2]] - x_t1[cells[:, 0]]) * (y_t1[cells[:, 1]] - y_t1[cells[:, 0]])
        areas_t1 = 0.5 * j_t1

        # Differenza geometrica di volume per elemento: Delta V_e = Area_{t+1} - Area_t
        delta_vol_elem = areas_t1 - areas_t

        # Divergenza del campo di velocità della mesh w integrata sulla cella:
        # int_e div(w) dOmega = int_e (dw_x/dx + dw_y/dy) dOmega
        w = self.compute_mesh_velocity(mesh_t, mesh_t1, dt)
        w1_x, w1_y = w[cells[:, 0], 0], w[cells[:, 0], 1]
        w2_x, w2_y = w[cells[:, 1], 0], w[cells[:, 1], 1]
        w3_x, w3_y = w[cells[:, 2], 0], w[cells[:, 2], 1]

        # Calcolo del gradiente su mesh media / mesh t1
        nodes_mid = 0.5 * (mesh_t.nodes + mesh_t1.nodes)
        x_m, y_m = nodes_mid[:, 0], nodes_mid[:, 1]
        j_m = (x_m[cells[:, 1]] - x_m[cells[:, 0]]) * (y_m[cells[:, 2]] - y_m[cells[:, 0]]) - \
              (x_m[cells[:, 2]] - x_m[cells[:, 0]]) * (y_m[cells[:, 1]] - y_m[cells[:, 0]])
        areas_m = 0.5 * j_m

        # Derivate spaziali delle funzioni di forma su configurazione media
        b1, c1 = y_m[cells[:, 1]] - y_m[cells[:, 2]], x_m[cells[:, 2]] - x_m[cells[:, 1]]
        b2, c2 = y_m[cells[:, 2]] - y_m[cells[:, 0]], x_m[cells[:, 0]] - x_m[cells[:, 2]]
        b3, c3 = y_m[cells[:, 0]] - y_m[cells[:, 1]], x_m[cells[:, 1]] - x_m[cells[:, 0]]

        inv_det_m = 1.0 / j_m
        div_w_elem = (
            (w1_x * b1 + w2_x * b2 + w3_x * b3) +
            (w1_y * c1 + w2_y * c2 + w3_y * c3)
        ) * inv_det_m  # costante per triangolo P1

        int_div_w = div_w_elem * areas_m * dt

        # Residuo locale GCL per elemento: Delta V_e - int_t^{t+1} int_e div(w)
        elem_gcl = delta_vol_elem - int_div_w

        # Assemblaggio nodale
        nodal_res = np.zeros(mesh_t.num_nodes, dtype=np.float64)
        np.add.at(nodal_res, cells.flatten(), np.repeat(elem_gcl / 3.0, 3))

        total_area = np.sum(areas_t1)
        res_norm = float(np.linalg.norm(nodal_res) / max(total_area, 1e-12))

        return res_norm, nodal_res

    def step(
        self,
        mesh_t: TriangularMesh2D,
        mesh_t1: TriangularMesh2D,
        state_t: np.ndarray,
        t_current: float,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], float]:
        """
        Esegue un singolo avanzamento temporale risolvendo la PDE accoppiata ALE.

        Input:
            mesh_t   : geometria al tempo t
            mesh_t1  : geometria al tempo t + dt (predetta e validata dal safety filter)
            state_t  : stato scalare u_t definito sui nodi di mesh_t, shape [N]
            t_current: tempo corrente
            dt       : passo temporale

        Output:
            state_t1      : soluzione calcolata u_{t+1} sulla mesh deformata mesh_t1, [N]
            mesh_velocity : campo di velocità nodale w [N, 2]
            solver_metrics: dizionario di telemetria e diagnostica numerica
            gcl_residual  : scalare dell'errore di conservazione geometrica
        """
        t0 = time.perf_counter()
        t_next = t_current + dt
        num_nodes = mesh_t.num_nodes
        cells = mesh_t.cells

        # 1. Calcolo cinematica della mesh: w = (X_{t+1} - X_t) / dt
        w = self.compute_mesh_velocity(mesh_t, mesh_t1, dt)

        # 2. Verifica del residuo GCL prima del solutore
        gcl_res, _ = self.verify_gcl(mesh_t, mesh_t1, dt)

        # 3. Configurazione media/valutazione per assemblaggio
        # Utilizziamo la configurazione a t^{n+theta}
        nodes_eval = (1.0 - self.theta) * mesh_t.nodes + self.theta * mesh_t1.nodes
        mesh_eval = TriangularMesh2D(nodes_eval, cells)

        # Estrazione gradienti funzioni di forma su mesh_eval
        x1, y1 = nodes_eval[cells[:, 0], 0], nodes_eval[cells[:, 0], 1]
        x2, y2 = nodes_eval[cells[:, 1], 0], nodes_eval[cells[:, 1], 1]
        x3, y3 = nodes_eval[cells[:, 2], 0], nodes_eval[cells[:, 2], 1]

        det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        areas = 0.5 * det_J
        assert np.all(det_J > 1e-15), "Rilevati elementi invertiti nella configurazione ALE intermedia"

        b1, c1 = y2 - y3, x3 - x2
        b2, c2 = y3 - y1, x1 - x3
        b3, c3 = y1 - y2, x2 - x1
        inv_det = 1.0 / det_J

        grad_phi = np.empty((mesh_t.num_cells, 3, 2), dtype=np.float64)
        grad_phi[:, 0, 0] = b1 * inv_det
        grad_phi[:, 0, 1] = c1 * inv_det
        grad_phi[:, 1, 0] = b2 * inv_det
        grad_phi[:, 1, 1] = c2 * inv_det
        grad_phi[:, 2, 0] = b3 * inv_det
        grad_phi[:, 2, 1] = c3 * inv_det

        # 4. Calcolo velocità relativa convettiva: c_rel = b - w
        b_phys = getattr(self.pde, "b", np.array([0.0, 0.0], dtype=np.float64))
        nu = float(getattr(self.pde, "nu", 1e-3))

        # Media baricentrica di w per ogni cella
        w_elem = np.mean(w[cells], axis=1)  # [M, 2]
        c_rel = b_phys[np.newaxis, :] - w_elem  # [M, 2]
        norm_c_rel = np.linalg.norm(c_rel, axis=1)

        # Parametro SUPG locale basato sulla velocità convettiva relativa c_rel
        h_k = np.sqrt(4.0 * areas / np.sqrt(3.0))
        if self.use_supg:
            pe_local = norm_c_rel * h_k / (2.0 * nu + 1e-14)
            tau_e = np.where(
                pe_local > 1.0,
                h_k / (2.0 * np.maximum(norm_c_rel, 1e-12)),
                (h_k**2) / (12.0 * nu + 1e-14)
            )
        else:
            tau_e = np.zeros(mesh_t.num_cells, dtype=np.float64)

        # 5. Assemblaggio delle matrici elementari ALE
        m_ref = np.array([
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0]
        ], dtype=np.float64) / 12.0
        elem_M = np.einsum("m,ij->mij", areas, m_ref)

        # Diffusione: nu * int_e grad(phi_i) . grad(phi_j)
        dot_grad = np.einsum("mih,mjh->mij", grad_phi, grad_phi)
        elem_K_diff = nu * np.einsum("m,mij->mij", areas, dot_grad)

        # Advezione relativa ALE: int_e ((b - w) . grad(phi_j)) phi_i
        c_dot_grad_j = np.einsum("mh,mjh->mj", c_rel, grad_phi)  # [M, 3]
        elem_K_adv = (1.0 / 3.0) * np.einsum("m,mj->mij", areas, c_dot_grad_j)

        # Stabilizzazione SUPG: int_e ((b - w) . grad(phi_i)) * tau_e * ((b - w) . grad(phi_j))
        c_dot_grad_i = np.einsum("mh,mih->mi", c_rel, grad_phi)  # [M, 3]
        elem_K_supg = np.einsum("m,mi,mj->mij", areas * tau_e, c_dot_grad_i, c_dot_grad_j)
        elem_M_supg = (1.0 / 3.0) * np.einsum("m,mi->mij", areas * tau_e, c_dot_grad_i)

        # Sistema algebrico accoppiato
        elem_LHS = (elem_M + elem_M_supg) / dt + self.theta * (elem_K_diff + elem_K_adv + elem_K_supg)
        elem_RHS = (elem_M + elem_M_supg) / dt - (1.0 - self.theta) * (elem_K_diff + elem_K_adv + elem_K_supg)

        # 6. Assemblaggio matrici globali sparse
        row_indices = np.repeat(cells[:, :, np.newaxis], 3, axis=2).flatten()
        col_indices = np.repeat(cells[:, np.newaxis, :], 3, axis=1).flatten()

        LHS_global = sp.coo_matrix(
            (elem_LHS.flatten(), (row_indices, col_indices)),
            shape=(num_nodes, num_nodes)
        ).tocsr()

        RHS_global = sp.coo_matrix(
            (elem_RHS.flatten(), (row_indices, col_indices)),
            shape=(num_nodes, num_nodes)
        ).tocsr()

        # Forzante fisica valutata al tempo intermedio
        t_eval = t_current + self.theta * dt
        cent_x = np.mean(nodes_eval[cells, 0], axis=1)
        cent_y = np.mean(nodes_eval[cells, 1], axis=1)
        f_vals = self.pde.evaluate_forcing(cent_x, cent_y, t_eval)

        f_galerkin = (1.0 / 3.0) * np.einsum("m,m->mi", areas, f_vals)
        f_supg = np.einsum("m,mi->mi", areas * tau_e * f_vals, c_dot_grad_i)
        f_elem = f_galerkin + f_supg

        f_global = np.zeros(num_nodes, dtype=np.float64)
        np.add.at(f_global, cells.flatten(), f_elem.flatten())

        # Vettore termine noto globale
        rhs_vec = RHS_global.dot(state_t) + f_global

        # 7. Condizioni al contorno di Dirichlet su mesh_t1 al tempo t_next
        bnd_nodes = mesh_t1.boundary_nodes
        is_bnd = np.zeros(num_nodes, dtype=bool)
        is_bnd[bnd_nodes] = True
        free_nodes = np.where(~is_bnd)[0]

        u_bnd_values = self.pde.evaluate_boundary(
            mesh_t1.nodes[bnd_nodes, 0],
            mesh_t1.nodes[bnd_nodes, 1],
            t_next
        )

        rhs_free = rhs_vec[free_nodes] - LHS_global[free_nodes, :][:, bnd_nodes].dot(u_bnd_values)
        A_free = LHS_global[free_nodes, :][:, free_nodes]

        # 8. Risoluzione sistema lineare ridotto
        u_free = spla.spsolve(A_free, rhs_free)

        # Ricostruzione stato su mesh_t1
        state_t1 = np.empty(num_nodes, dtype=np.float64)
        state_t1[free_nodes] = u_free
        state_t1[bnd_nodes] = u_bnd_values

        solve_time = time.perf_counter() - t0

        metrics = {
            "elapsed_time": solve_time,
            "max_mesh_velocity": float(np.max(np.linalg.norm(w, axis=1))),
            "mean_mesh_velocity": float(np.mean(np.linalg.norm(w, axis=1))),
            "gcl_residual": float(gcl_res),
            "cond_lhs_approx": float(A_free.diagonal().max() / max(A_free.diagonal().min(), 1e-12))
        }

        return state_t1, w, metrics, gcl_res