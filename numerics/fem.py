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

"""
numerics/fem.py
Solutore numerico ad Elementi Finiti P1 per Convezione-Diffusione transiente 2D.

Implementa:
    - Assemblaggio vettorializzato di matrici sparse: Massa (M), Rigidezza (K), Convezione (C)
    - Quadratura di Gauss di grado 5 su triangoli per il termine di carico F(t)
    - Integrazione temporale con Backward Euler (BDF1) e Dirichlet forte
    - Calcolo del gradiente nodale mediato per area (P1 area-weighted nodal recovery)
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from geometry.mesh import TriangularMesh2D
from numerics.pde import ManufacturedConvectionDiffusion2D


# =============================================================================
# REGOLA DI QUADRATURA SU TRIANGOLI DI RIFERIMENTO (DUNAVANT GRADO 5, 7 PUNTI)
# Coordinate baricentriche (lambda_1, lambda_2, lambda_3) e pesi normalizzati (somma = 0.5)
# =============================================================================
def _get_triangle_quadrature_degree_5() -> Tuple[np.ndarray, np.ndarray]:
    """
    Restituisce punti di quadratura (coordinate baricentriche xi, eta) e pesi
    per il triangolo di riferimento T_ref = {(xi, eta) : xi >= 0, eta >= 0, xi + eta <= 1}.
    Area del triangolo di riferimento = 0.5.
    """
    # Punto centrale
    p1 = [1.0 / 3.0, 1.0 / 3.0]
    w1 = 0.225 / 2.0  # peso scalato sull'area 0.5

    # Gruppo 1 (3 punti permutati)
    a = (6.0 - np.sqrt(15.0)) / 21.0
    b = (9.0 + 2.0 * np.sqrt(15.0)) / 21.0
    w_grp1 = ((155.0 - np.sqrt(15.0)) / 1200.0) / 2.0

    p_grp1 = [
        [a, a],
        [a, b],
        [b, a]
    ]

    # Gruppo 2 (3 punti permutati)
    c = (6.0 + np.sqrt(15.0)) / 21.0
    d = (9.0 - 2.0 * np.sqrt(15.0)) / 21.0
    w_grp2 = ((155.0 + np.sqrt(15.0)) / 1200.0) / 2.0

    p_grp2 = [
        [c, c],
        [c, d],
        [d, c]
    ]

    points = np.array([p1] + p_grp1 + p_grp2, dtype=np.float64)  # shape [7, 2] (xi, eta)
    weights = np.array([w1] + [w_grp1] * 3 + [w_grp2] * 3, dtype=np.float64)  # shape [7]
    return points, weights


class ConvectionDiffusionFEM2D:
    """
    Solutore FEM Galerkin continuo P1 per convezione-diffusione:
        M du/dt + C u + nu K u = F(t)
    """

    def __init__(
        self,
        mesh: TriangularMesh2D,
        pde: ManufacturedConvectionDiffusion2D
    ):
        self.mesh = mesh
        self.pde = pde
        self.num_nodes = mesh.num_nodes
        self.num_cells = mesh.num_cells
        self.nodes = mesh.nodes
        self.cells = mesh.cells
        self.boundary_nodes = mesh.boundary_nodes

        # Punti di quadratura grado 5
        self.qp_ref, self.qw_ref = _get_triangle_quadrature_degree_5()

        # Precomputazione geometrica locale P1
        self._compute_element_geometry()

        # Assemblaggio matrici statiche (indipendenti da t)
        self.M = self._assemble_mass_matrix()
        self.K = self._assemble_stiffness_matrix()
        self.C = self._assemble_convection_matrix()

    def _compute_element_geometry(self):
        """Precomputa Jacobiani, aree e gradienti delle funzioni di forma su ogni elemento."""
        x = self.nodes[:, 0]
        y = self.nodes[:, 1]
        c = self.cells

        x1, y1 = x[c[:, 0]], y[c[:, 0]]
        x2, y2 = x[c[:, 1]], y[c[:, 1]]
        x3, y3 = x[c[:, 2]], y[c[:, 2]]

        # Determinante Jacobiano del mapping affine J_e = 2 * Area(K_e)
        det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        self.det_J = det_J
        self.areas = 0.5 * np.abs(det_J)

        # Gradienti costanti delle 3 funzioni di forma locali:
        # phi_1 = 1 - xi - eta, phi_2 = xi, phi_3 = eta
        # grad(phi_i) = [dphi_i/dx, dphi_i/dy]
        inv_det = 1.0 / det_J
        self.dphi_dx = np.zeros((self.num_cells, 3), dtype=np.float64)
        self.dphi_dy = np.zeros((self.num_cells, 3), dtype=np.float64)

        self.dphi_dx[:, 0] = (y2 - y3) * inv_det
        self.dphi_dx[:, 1] = (y3 - y1) * inv_det
        self.dphi_dx[:, 2] = (y1 - y2) * inv_det

        self.dphi_dy[:, 0] = (x3 - x2) * inv_det
        self.dphi_dy[:, 1] = (x1 - x3) * inv_det
        self.dphi_dy[:, 2] = (x2 - x1) * inv_det

    def _assemble_mass_matrix(self) -> sp.csr_matrix:
        """Assembla la matrice di massa globale: M_ij = integral(phi_i * phi_j)."""
        # Su ogni triangolo P1: M_local = (Area / 12) * [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
        base_M = np.array([
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0]
        ], dtype=np.float64) / 12.0

        M_elem = self.areas[:, None, None] * base_M[None, :, :]  # [M, 3, 3]

        rows = np.repeat(self.cells[:, :, None], 3, axis=2).flatten()
        cols = np.repeat(self.cells[:, None, :], 3, axis=1).flatten()
        vals = M_elem.flatten()

        M_global = sp.coo_matrix((vals, (rows, cols)), shape=(self.num_nodes, self.num_nodes)).tocsr()
        return M_global

    def _assemble_stiffness_matrix(self) -> sp.csr_matrix:
        """Assembla la matrice di rigidezza: K_ij = integral(grad phi_i · grad phi_j)."""
        # grad phi_i e grad phi_j sono costanti su ciascun elemento
        # K_local_ij = Area * (dphi_dx_i * dphi_dx_j + dphi_dy_i * dphi_dy_j)
        gx = self.dphi_dx[:, :, None] * self.dphi_dx[:, None, :]  # [M, 3, 3]
        gy = self.dphi_dy[:, :, None] * self.dphi_dy[:, None, :]  # [M, 3, 3]
        K_elem = self.areas[:, None, None] * (gx + gy)

        rows = np.repeat(self.cells[:, :, None], 3, axis=2).flatten()
        cols = np.repeat(self.cells[:, None, :], 3, axis=1).flatten()
        vals = K_elem.flatten()

        K_global = sp.coo_matrix((vals, (rows, cols)), shape=(self.num_nodes, self.num_nodes)).tocsr()
        return K_global

    def _assemble_convection_matrix(self) -> sp.csr_matrix:
        """Assembla la matrice convettiva: C_ij = integral(phi_i * (b · grad phi_j))."""
        # b · grad phi_j è costante su ciascun triangolo: [M, 3]
        b_dot_grad = self.pde.bx * self.dphi_dx + self.pde.by * self.dphi_dy  # [M, 3]
        int_phi = (self.areas / 3.0)[:, None, None]  # [M, 1, 1]

        # int_phi viene replicato sulle 3 righe i, b_dot_grad sulle 3 colonne j -> [M, 3, 3]
        C_elem = int_phi * np.ones((1, 3, 1), dtype=np.float64) * b_dot_grad[:, None, :]

        rows = np.repeat(self.cells[:, :, None], 3, axis=2).flatten()
        cols = np.repeat(self.cells[:, None, :], 3, axis=1).flatten()
        vals = C_elem.flatten()

        C_global = sp.coo_matrix((vals, (rows, cols)), shape=(self.num_nodes, self.num_nodes)).tocsr()
        return C_global

    def assemble_rhs(self, t: float) -> np.ndarray:
        """
        Calcola il vettore di carico F_i = integral(phi_i * f(x, y, t))
        utilizzando la quadratura su triangolo di grado 5 (7 punti).
        """
        F = np.zeros(self.num_nodes, dtype=np.float64)
        c = self.cells
        x = self.nodes[:, 0]
        y = self.nodes[:, 1]

        x1, y1 = x[c[:, 0]], y[c[:, 0]]
        x2, y2 = x[c[:, 1]], y[c[:, 1]]
        x3, y3 = x[c[:, 2]], y[c[:, 2]]

        # Per ciascuno dei 7 punti di quadratura Dunavant
        for (xi, eta), w in zip(self.qp_ref, self.qw_ref):
            # Funzioni di forma nel punto di quadratura
            phi1 = 1.0 - xi - eta
            phi2 = xi
            phi3 = eta

            # Coordinate fisiche (x_q, y_q) su ciascun triangolo
            xq = phi1 * x1 + phi2 * x2 + phi3 * x3
            yq = phi1 * y1 + phi2 * y2 + phi3 * y3

            # Valutazione analitica esatta del forcing MMS f(xq, yq, t)
            f_vals = self.pde.evaluate_forcing(xq, yq, t)

            # Contributo integrato: w * f * det_J
            # Fattore 2 * w perché qw_ref somma a 0.5 (area riferimento), e det_J = 2 * Area
            factor = (2.0 * w) * self.areas * f_vals

            np.add.at(F, c[:, 0], factor * phi1)
            np.add.at(F, c[:, 1], factor * phi2)
            np.add.at(F, c[:, 2], factor * phi3)

        return F

    def step_backward_euler(
        self,
        u_n: np.ndarray,
        dt: float,
        t_next: float
    ) -> np.ndarray:
        """
        Avanza di un passo temporale risolvendo:
            (M/dt + C + nu*K) u^{n+1} = (M/dt) u^n + F^{n+1}
        con condizioni di Dirichlet omogenee u = 0 sui boundary nodes.
        """
        # Matrice di sistema: A = (1/dt) * M + C + nu * K
        inv_dt = 1.0 / dt
        A = inv_dt * self.M + self.C + self.pde.nu * self.K

        # Termine noto: b_vec = (1/dt) * M * u_n + F(t_next)
        F_next = self.assemble_rhs(t_next)
        b_vec = inv_dt * (self.M @ u_n) + F_next

        # Applicazione rigorosa delle condizioni al bordo di Dirichlet omogenee (u_bnd = 0)
        # Rimpiazzo righe corrispondenti al contorno con identità
        A = A.tolil()
        bnd = self.boundary_nodes
        A[bnd, :] = 0.0
        A[bnd, bnd] = 1.0
        b_vec[bnd] = 0.0

        A = A.tocsr()

        # Risoluzione diretta del sistema lineare sparso (SuperLU)
        u_next = spla.spsolve(A, b_vec)
        return u_next


def compute_nodal_gradient(
    nodes: np.ndarray,
    cells: np.ndarray,
    u: np.ndarray
) -> np.ndarray:
    """
    Calcola il gradiente nodale P1 continuo mediante media pesata per area.

    Formula:
        grad_u(X_i) = (sum_{e in T_i} Area_e * grad_u_e) / (sum_{e in T_i} Area_e)

    Returns:
        grad_u: [N, 2] gradiente sui nodi (du/dx, du/dy)
    """
    N = nodes.shape[0]
    x = nodes[:, 0]
    y = nodes[:, 1]

    x1, y1 = x[cells[:, 0]], y[cells[:, 0]]
    x2, y2 = x[cells[:, 1]], y[cells[:, 1]]
    x3, y3 = x[cells[:, 2]], y[cells[:, 2]]

    det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    areas = 0.5 * np.abs(det_J)
    inv_det = 1.0 / det_J

    # Gradienti delle funzioni di forma su ciascun elemento
    dphi_dx = np.column_stack([(y2 - y3) * inv_det, (y3 - y1) * inv_det, (y1 - y2) * inv_det])
    dphi_dy = np.column_stack([(x3 - x2) * inv_det, (x1 - x3) * inv_det, (x2 - x1) * inv_det])

    # Valori di u sui vertici degli elementi: [M, 3]
    u_elem = u[cells]

    # Gradiente P1 elementare costante: [M, 2]
    elem_grad_x = np.sum(u_elem * dphi_dx, axis=1)
    elem_grad_y = np.sum(u_elem * dphi_dy, axis=1)

    # Accumulo nodale pesato per area
    weighted_gx = areas * elem_grad_x
    weighted_gy = areas * elem_grad_y

    nodal_gx = np.zeros(N, dtype=np.float64)
    nodal_gy = np.zeros(N, dtype=np.float64)
    nodal_area_sum = np.zeros(N, dtype=np.float64)

    for i in range(3):
        node_indices = cells[:, i]
        np.add.at(nodal_gx, node_indices, weighted_gx)
        np.add.at(nodal_gy, node_indices, weighted_gy)
        np.add.at(nodal_area_sum, node_indices, areas)

    inv_total_area = 1.0 / np.maximum(nodal_area_sum, 1e-14)
    nodal_gx *= inv_total_area
    nodal_gy *= inv_total_area

    return np.column_stack([nodal_gx, nodal_gy])


def compute_field_gradients(
    mesh: TriangularMesh2D,
    u: np.ndarray,
    pde: Optional[ManufacturedConvectionDiffusion2D] = None,
    t: float = 0.0,
    gradient_mode: str = "fem"
) -> np.ndarray:
    """
    Funzione ponte per l'estrazione del gradiente a supporto della GNN:
        - "fem"  : gradiente ricavato numericamente dalla soluzione approssimata u^n
        - "exact": gradiente analitico esatto del campo MMS (ablation study per isolare l'errore)
    """
    if gradient_mode == "fem":
        return compute_nodal_gradient(mesh.nodes, mesh.cells, u)
    elif gradient_mode == "exact":
        assert pde is not None, "pde instance must be provided for exact gradient mode"
        _, ux, uy, _, _, _, _ = pde.evaluate_exact(mesh.nodes[:, 0], mesh.nodes[:, 1], t)
        return np.column_stack([ux, uy])
    else:
        raise ValueError(f"Modalità gradiente non riconosciuta: {gradient_mode}")