"""
numerics/fem.py
Solutore numerico classico agli Elementi Finiti (FEM) lineari continui P1.

Risolve il problema di convezione-diffusione 2D:
    du/dt + b * grad(u) - nu * Delta u = f(x, y, t)
su mesh triangolari non strutturate con formulazione SUPG
(Streamline-Upwind Petrov-Galerkin) e time-stepping implicito (theta-method).

Modulo puramente numerico: non ha alcuna dipendenza da GNN o ML.
"""

from typing import Dict, Any, Tuple, Optional
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from geometry.mesh import TriangularMesh2D
from numerics.pde import BasePDE


class ClassicalFEM2DSolver:
    """
    Solutore FEM P1 con stabilizzazione SUPG per problemi avvezione-diffusione.
    """

    def __init__(
        self,
        mesh: TriangularMesh2D,
        pde: BasePDE,
        theta: float = 1.0,
        use_supg: bool = True
    ):
        """
        Input:
            mesh     : istanza di TriangularMesh2D con coordinate e connettività attuali
            pde      : istanza di BasePDE per parametri fisici, forzante e condizioni al contorno
            theta    : parametro schema temporale (1.0 = Eulero Implicito, 0.5 = Crank-Nicolson)
            use_supg : flag booleano per abilitare la stabilizzazione SUPG
        """
        self.mesh = mesh
        self.pde = pde
        self.theta = float(theta)
        self.use_supg = bool(use_supg)

        self.num_nodes: int = self.mesh.num_nodes
        self.num_cells: int = self.mesh.num_cells
        self.cells = self.mesh.cells
        self.nodes = self.mesh.nodes
        self.boundary_nodes = self.mesh.boundary_nodes

        # Gradi di libertà interni liberi
        self.is_boundary = np.zeros(self.num_nodes, dtype=bool)
        self.is_boundary[self.boundary_nodes] = True
        self.free_nodes = np.where(~self.is_boundary)[0]

    def _compute_element_tensors(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcola i gradienti delle funzioni di forma lineari, le aree e il parametro SUPG tau_e
        vettorizzati per tutti gli M elementi triangolari.
        """
        nodes = self.nodes
        cells = self.cells

        # Vertici per cella: [M]
        x1, y1 = nodes[cells[:, 0], 0], nodes[cells[:, 0], 1]
        x2, y2 = nodes[cells[:, 1], 0], nodes[cells[:, 1], 1]
        x3, y3 = nodes[cells[:, 2], 0], nodes[cells[:, 2], 1]

        # Vettori lato
        u1, u2 = x2 - x1, y2 - y1
        v1, v2 = x3 - x1, y3 - y1

        # Determinante Jacobiano e area
        det_J = u1 * v2 - u2 * v1
        areas = 0.5 * det_J
        assert np.all(det_J > 1e-15), "Rilevati elementi invertiti o degeneri durante l'assemblaggio FEM!"

        # Gradienti spaziali delle 3 funzioni di forma standard P1 per ciascun triangolo
        # phi_1: b1 = y2 - y3, c1 = x3 - x2
        # phi_2: b2 = y3 - y1, c2 = x1 - x3
        # phi_3: b3 = y1 - y2, c3 = x2 - x1
        b1, c1 = y2 - y3, x3 - x2
        b2, c2 = y3 - y1, x1 - x3
        b3, c3 = y1 - y2, x2 - x1

        inv_det = 1.0 / det_J  # [M]
        grad_phi = np.empty((self.num_cells, 3, 2), dtype=np.float64)
        grad_phi[:, 0, 0] = b1 * inv_det
        grad_phi[:, 0, 1] = c1 * inv_det
        grad_phi[:, 1, 0] = b2 * inv_det
        grad_phi[:, 1, 1] = c2 * inv_det
        grad_phi[:, 2, 0] = b3 * inv_det
        grad_phi[:, 2, 1] = c3 * inv_det

        # Diametro caratteristico locale h_k e parametro di stabilizzazione SUPG tau_e
        # h_k approssimato tramite radice dell'area
        h_k = np.sqrt(4.0 * areas / np.sqrt(3.0))

        # Estrazione velocità convettiva b e diffusività nu se presenti nella PDE
        b_vec = getattr(self.pde, "b", np.array([0.0, 0.0], dtype=np.float64))
        nu = float(getattr(self.pde, "nu", 1e-3))
        norm_b = float(np.linalg.norm(b_vec))

        if self.use_supg and norm_b > 1e-12:
            # Formula classica standard: tau_e = ( (2*||b||/h)^2 + 9*(4*nu/h^2)^2 )^(-1/2)
            pe_local = norm_b * h_k / (2.0 * nu + 1e-14)
            tau_e = np.where(
                pe_local > 1.0,
                h_k / (2.0 * norm_b),
                (h_k**2) / (12.0 * nu + 1e-14)
            )
        else:
            tau_e = np.zeros(self.num_cells, dtype=np.float64)

        return areas, det_J, grad_phi, b_vec, nu, tau_e

    def assemble_system(
        self,
        t_current: float,
        dt: float
    ) -> Tuple[sp.csr_matrix, np.ndarray]:
        """
        Assembla la matrice globale tangenziale A e il vettore dei termini noti RHS.
        Formulazione debole convezione-diffusione + SUPG:
            M / dt + theta * (K_diff + K_adv + K_supg)
        """
        areas, det_J, grad_phi, b_vec, nu, tau_e = self._compute_element_tensors()
        M_nodes = 3

        # Matrice di massa locale di riferimento scalata con l'area
        # [2 1 1; 1 2 1; 1 1 2] / 12
        m_ref = np.array([
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0]
        ], dtype=np.float64) / 12.0

        # Allocazione matrici elementari: [M, 3, 3]
        elem_M = np.einsum("m,ij->mij", areas, m_ref)

        # 1. Diffusione: nu * int_e (grad phi_i . grad phi_j) dOmega
        # grad_phi: [M, 3, 2]
        dot_grad = np.einsum("mih,mjh->mij", grad_phi, grad_phi)
        elem_K_diff = nu * np.einsum("m,mij->mij", areas, dot_grad)

        # 2. Convezione standard Galerkin: int_e (b . grad phi_j) phi_i dOmega
        # b . grad phi_j: [M, 3]
        b_dot_grad = np.einsum("k,mjk->mj", b_vec, grad_phi)
        elem_K_adv = (1.0 / 3.0) * np.einsum("m,mj->mij", areas, b_dot_grad)

        # 3. Termine aggiuntivo SUPG: int_e (b . grad phi_i) * tau_e * (b . grad phi_j) dOmega
        b_dot_grad_i = np.einsum("k,mik->mi", b_vec, grad_phi)
        elem_K_supg = np.einsum("m,mi,mj->mij", areas * tau_e, b_dot_grad_i, b_dot_grad)

        # Termine di massa SUPG per derivata temporale: int_e (b . grad phi_i) * tau_e * phi_j dOmega
        elem_M_supg = (1.0 / 3.0) * np.einsum("m,mi->mij", areas * tau_e, b_dot_grad_i)

        # Matrici locali combinate
        # A_elem * u^{n+1} = B_elem * u^n + RHS_elem
        elem_LHS = (elem_M + elem_M_supg) / dt + self.theta * (elem_K_diff + elem_K_adv + elem_K_supg)
        elem_RHS_mat = (elem_M + elem_M_supg) / dt - (1.0 - self.theta) * (elem_K_diff + elem_K_adv + elem_K_supg)

        # Assemblaggio sparso COO
        row_indices = np.repeat(self.cells[:, :, np.newaxis], 3, axis=2).flatten()
        col_indices = np.repeat(self.cells[:, np.newaxis, :], 3, axis=1).flatten()

        LHS_global = sp.coo_matrix(
            (elem_LHS.flatten(), (row_indices, col_indices)),
            shape=(self.num_nodes, self.num_nodes)
        ).tocsr()

        RHS_mat_global = sp.coo_matrix(
            (elem_RHS_mat.flatten(), (row_indices, col_indices)),
            shape=(self.num_nodes, self.num_nodes)
        ).tocsr()

        # Forzante f valutata sui baricentri degli elementi a t^{n+theta}
        t_eval = t_current + self.theta * dt
        tri_x_cent = np.mean(self.nodes[self.cells, 0], axis=1)
        tri_y_cent = np.mean(self.nodes[self.cells, 1], axis=1)
        f_elem = self.pde.evaluate_forcing(tri_x_cent, tri_y_cent, t_eval)  # [M]

        # Contributo vettore di carico F: int_e f * (phi_i + tau_e * b . grad phi_i) dOmega
        f_galerkin = (1.0 / 3.0) * np.einsum("m,m->mi", areas, f_elem)
        f_supg = np.einsum("m,mi->mi", areas * tau_e * f_elem, b_dot_grad_i)
        f_local = f_galerkin + f_supg

        f_global = np.zeros(self.num_nodes, dtype=np.float64)
        np.add.at(f_global, self.cells.flatten(), f_local.flatten())

        return LHS_global, RHS_mat_global, f_global

    def step(
        self,
        u_current: np.ndarray,
        t_current: float,
        dt: float
    ) -> Tuple[np.ndarray, float, int, float]:
        """
        Esegue un singolo avanzamento temporale [t_current, t_current + dt].

        Input:
            u_current: soluzione nodale allo stato attuale t_current, shape [N]
            t_current: tempo corrente scalare
            dt       : ampiezza del passo temporale

        Output:
            u_next           : soluzione al tempo t_current + dt, shape [N]
            residual_norm    : norma L2 del residuo del sistema lineare
            linear_iterations: numero di iterazioni del solutore (1 per solutore diretto)
            elapsed_time     : tempo di calcolo impiegato in secondi
        """
        t0 = time.perf_counter()
        t_next = t_current + dt

        # 1. Assemblaggio matrici globali sparse
        LHS, RHS_mat, f_vec = self.assemble_system(t_current, dt)

        # 2. Vettore RHS grezzo: b_glob = RHS_mat * u^n + f_glob
        b_glob = RHS_mat.dot(u_current) + f_vec

        # 3. Imposizione condizioni al contorno di Dirichlet al tempo t_next
        x_bnd = self.nodes[self.boundary_nodes, 0]
        y_bnd = self.nodes[self.boundary_nodes, 1]
        u_bnd_values = self.pde.evaluate_boundary(x_bnd, y_bnd, t_next)

        # Modifica RHS per nodi interni (eliminazione Dirichlet)
        b_free = b_glob[self.free_nodes] - LHS[self.free_nodes, :][:, self.boundary_nodes].dot(u_bnd_values)
        A_free = LHS[self.free_nodes, :][:, self.free_nodes]

        # 4. Soluzione del sistema lineare ridotto
        # Utilizzo solutore diretto sparse LU super-stabile per mesh 2D
        u_free = spla.spsolve(A_free, b_free)

        # 5. Ricostruzione del vettore nodale completo u^{n+1}
        u_next = np.empty(self.num_nodes, dtype=np.float64)
        u_next[self.free_nodes] = u_free
        u_next[self.boundary_nodes] = u_bnd_values

        # 6. Calcolo residuo algebrico ||A * u^{n+1} - b||_2
        full_residual = LHS.dot(u_next) - b_glob
        full_residual[self.boundary_nodes] = 0.0
        res_norm = float(np.linalg.norm(full_residual))

        elapsed_time = time.perf_counter() - t0

        return u_next, res_norm, 1, elapsed_time