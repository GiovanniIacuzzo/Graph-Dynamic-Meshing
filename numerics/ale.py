"""
numerics/ale.py
Formulazione Arbitrary Lagrangian-Eulerian (ALE) FEM P1 conservativa conforme alla D-GCL.

Deriva dal Teorema del Trasporto di Reynolds sulla configurazione mobile Omega(t):
    d/dt int_{Omega(t)} u_h phi_i dx - int_{Omega(t)} div(w u_h) phi_i dx
    + int_{Omega(t)} (b . grad u_h) phi_i dx + nu int_{Omega(t)} grad u_h . grad phi_i dx
    = int_{Omega(t)} f phi_i dx

Discretizzazione temporale totalmente discreta (Backward Euler + Midpoint GCL):
    (1/dt) * (M^{n+1} U^{n+1} - M^n U^n)
    + C_{mesh}^{n+1/2} U^{n+1} + C_{phys}^{n+1} U^{n+1} + nu * K^{n+1} U^{n+1} = F^{n+1}

dove l'operatore cinematica di griglia valutato al punto medio X^{n+1/2} è:
    C_{mesh}^{n+1/2} = - C_{adv}^{w, n+1/2} - M_{div}^{w, n+1/2}
con:
    (C_{adv}^{w, n+1/2})_{ij} = int_{Omega^{n+1/2}} phi_i (w^{n+1/2} . grad phi_j) dx
    (M_{div}^{w, n+1/2})_{ij} = int_{Omega^{n+1/2}} (div w^{n+1/2}) phi_i phi_j dx
"""

from typing import Tuple, Dict, Any
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from geometry.mesh import TriangularMesh2D
from numerics.fem import _get_triangle_quadrature_degree_5


class ALESolver2D:
    def __init__(self, pde_model: Any, nu: float = 1e-3, bx: float = 0.4, by: float = 0.2):
        self.pde = pde_model
        self.nu = float(nu)
        self.b = np.array([bx, by], dtype=np.float64)

    @staticmethod
    def compute_mesh_velocity(nodes_n: np.ndarray, nodes_next: np.ndarray, dt: float) -> np.ndarray:
        return (nodes_next - nodes_n) / dt

    @staticmethod
    def assemble_mass_matrix(mesh: TriangularMesh2D) -> sp.csr_matrix:
        cells = mesh.cells
        nodes = mesh.nodes
        num_nodes = mesh.num_nodes

        x1, y1 = nodes[cells[:, 0], 0], nodes[cells[:, 0], 1]
        x2, y2 = nodes[cells[:, 1], 0], nodes[cells[:, 1], 1]
        x3, y3 = nodes[cells[:, 2], 0], nodes[cells[:, 2], 1]

        areas = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

        base_M = np.array([
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0]
        ], dtype=np.float64) / 12.0

        M_elem = areas[:, None, None] * base_M[None, :, :]
        rows = np.repeat(cells[:, :, None], 3, axis=2).flatten()
        cols = np.repeat(cells[:, None, :], 3, axis=1).flatten()
        return sp.coo_matrix((M_elem.flatten(), (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()

    def assemble_ale_system(
        self,
        mesh_n: TriangularMesh2D,
        mesh_next: TriangularMesh2D,
        dt: float,
        t_next: float,
        pure_gcl: bool = False
    ) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, Dict[str, Any]]:
        cells = mesh_next.cells
        num_nodes = mesh_next.num_nodes
        num_cells = cells.shape[0]

        nodes_n = mesh_n.nodes
        nodes_next = mesh_next.nodes
        w_nodes = self.compute_mesh_velocity(nodes_n, nodes_next, dt)
        nodes_mid = 0.5 * (nodes_n + nodes_next)

        # 1. Geometria al midpoint X^{n+1/2} per la cinematica della griglia
        x1_m, y1_m = nodes_mid[cells[:, 0], 0], nodes_mid[cells[:, 0], 1]
        x2_m, y2_m = nodes_mid[cells[:, 1], 0], nodes_mid[cells[:, 1], 1]
        x3_m, y3_m = nodes_mid[cells[:, 2], 0], nodes_mid[cells[:, 2], 1]

        det_J_mid = (x2_m - x1_m) * (y3_m - y1_m) - (x3_m - x1_m) * (y2_m - y1_m)
        areas_mid = 0.5 * np.abs(det_J_mid)
        inv_det_m = 1.0 / det_J_mid

        dphi_dx_m = np.column_stack([
            (y2_m - y3_m) * inv_det_m,
            (y3_m - y1_m) * inv_det_m,
            (y1_m - y2_m) * inv_det_m
        ])
        dphi_dy_m = np.column_stack([
            (x3_m - x2_m) * inv_det_m,
            (x1_m - x3_m) * inv_det_m,
            (x2_m - x1_m) * inv_det_m
        ])

        w_elem = w_nodes[cells]
        div_w = (
            w_elem[:, 0, 0] * dphi_dx_m[:, 0] + w_elem[:, 1, 0] * dphi_dx_m[:, 1] + w_elem[:, 2, 0] * dphi_dx_m[:, 2] +
            w_elem[:, 0, 1] * dphi_dy_m[:, 0] + w_elem[:, 1, 1] * dphi_dy_m[:, 1] + w_elem[:, 2, 1] * dphi_dy_m[:, 2]
        )

        # 2. Geometria su t^{n+1}
        x1_p, y1_p = nodes_next[cells[:, 0], 0], nodes_next[cells[:, 0], 1]
        x2_p, y2_p = nodes_next[cells[:, 1], 0], nodes_next[cells[:, 1], 1]
        x3_p, y3_p = nodes_next[cells[:, 2], 0], nodes_next[cells[:, 2], 1]

        det_J_p = (x2_p - x1_p) * (y3_p - y1_p) - (x3_p - x1_p) * (y2_p - y1_p)
        areas_next = 0.5 * np.abs(det_J_p)
        inv_det_p = 1.0 / det_J_p

        dphi_dx_p = np.column_stack([
            (y2_p - y3_p) * inv_det_p,
            (y3_p - y1_p) * inv_det_p,
            (y1_p - y2_p) * inv_det_p
        ])
        dphi_dy_p = np.column_stack([
            (x3_p - x2_p) * inv_det_p,
            (x1_p - x3_p) * inv_det_p,
            (x2_p - x1_p) * inv_det_p
        ])

        # 3. Geometria su t^n
        x1_n, y1_n = nodes_n[cells[:, 0], 0], nodes_n[cells[:, 0], 1]
        x2_n, y2_n = nodes_n[cells[:, 1], 0], nodes_n[cells[:, 1], 1]
        x3_n, y3_n = nodes_n[cells[:, 2], 0], nodes_n[cells[:, 2], 1]
        areas_n = 0.5 * np.abs((x2_n - x1_n) * (y3_n - y1_n) - (x3_n - x1_n) * (y2_n - y1_n))

        # Matrici di massa locali coerenti
        base_M = np.array([
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0]
        ], dtype=np.float64) / 12.0
        M_elem_next = areas_next[:, None, None] * base_M[None, :, :]

        M_next = self.assemble_mass_matrix(mesh_next)
        M_n = self.assemble_mass_matrix(mesh_n)

        # 4. Operatore Conservativo di Convezione di Griglia C_{mesh}^{n+1/2}(w)
        # C_{mesh, ij} = - int_{K_e^{n+1/2}} phi_i div(w phi_j) dx
        #              = - int_{K_e^{n+1/2}} phi_i (w . grad phi_j) dx - int_{K_e^{n+1/2}} (div w) phi_i phi_j dx
        qp_ref, qw_ref = _get_triangle_quadrature_degree_5()
        C_adv = np.zeros((num_cells, 3, 3), dtype=np.float64)

        for (xi, eta), weight in zip(qp_ref, qw_ref):
            phi = np.array([1.0 - xi - eta, xi, eta], dtype=np.float64)
            w_q = np.sum(w_elem * phi[None, :, None], axis=1)
            w_dot_grad = w_q[:, 0, None] * dphi_dx_m + w_q[:, 1, None] * dphi_dy_m
            term = phi[:, None] * w_dot_grad[:, None, :]
            C_adv += (2.0 * weight) * (areas_mid[:, None, None] * term)

        # Termine variazionale di dilatazione cinematica valutato sul midpoint X^{n+1/2}:
        # M_{div, e}^{w, n+1/2} = |K_e^{n+1/2}| * (div w^{n+1/2})_e * base_M
        # (Non usa M_elem_next - M_elem_n, rendendo la verifica D-GCL indipendente e genuina)
        M_div_w_elem = (areas_mid * div_w)[:, None, None] * base_M[None, :, :]
        C_mesh_elem = -C_adv - M_div_w_elem

        # 5. Convezione fisica e diffusione su t^{n+1}
        C_phys_elem = np.zeros((num_cells, 3, 3), dtype=np.float64)
        K_elem = np.zeros((num_cells, 3, 3), dtype=np.float64)
        F_elem = np.zeros((num_cells, 3), dtype=np.float64)

        if not pure_gcl:
            for (xi, eta), weight in zip(qp_ref, qw_ref):
                phi = np.array([1.0 - xi - eta, xi, eta], dtype=np.float64)
                b_grad = self.b[0] * dphi_dx_p + self.b[1] * dphi_dy_p
                C_phys_elem += (2.0 * weight) * (areas_next[:, None, None] * (phi[:, None] * b_grad[:, None, :]))

                if self.pde is not None:
                    xq = phi[0] * x1_p + phi[1] * x2_p + phi[2] * x3_p
                    yq = phi[0] * y1_p + phi[1] * y2_p + phi[2] * y3_p
                    _, _, _, _, _, _, f_val = self.pde.evaluate_exact(xq, yq, t_next)
                    F_elem += (2.0 * weight) * (areas_next[:, None] * f_val[:, None] * phi[None, :])

            if self.nu > 0.0:
                grad_dot = (
                    dphi_dx_p[:, :, None] * dphi_dx_p[:, None, :] +
                    dphi_dy_p[:, :, None] * dphi_dy_p[:, None, :]
                )
                K_elem = self.nu * (areas_next[:, None, None] * grad_dot)

        # 6. Assemblaggio matriciale globale: A = (1/dt)*M^{n+1} + C_mesh + C_phys + K
        rows = np.repeat(cells[:, :, None], 3, axis=2).flatten()
        cols = np.repeat(cells[:, None, :], 3, axis=1).flatten()

        A_elem = (1.0 / dt) * M_elem_next + C_mesh_elem + C_phys_elem + K_elem
        A_mat = sp.coo_matrix((A_elem.flatten(), (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()

        F_vec = np.zeros(num_nodes, dtype=np.float64)
        if not pure_gcl:
            np.add.at(F_vec, cells[:, 0], F_elem[:, 0])
            np.add.at(F_vec, cells[:, 1], F_elem[:, 1])
            np.add.at(F_vec, cells[:, 2], F_elem[:, 2])

        C_mesh_mat = sp.coo_matrix((C_mesh_elem.flatten(), (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
        C_adv_mat = sp.coo_matrix((C_adv.flatten(), (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
        M_div_w_mat = sp.coo_matrix((M_div_w_elem.flatten(), (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()

        r_elem = np.abs((areas_next - areas_n) / dt - areas_mid * div_w)

        diagnostics = {
            "max_elem_gcl_residual": float(np.max(r_elem)),
            "C_mesh_mat": C_mesh_mat,
            "C_adv_mat": C_adv_mat,
            "M_div_w_mat": M_div_w_mat,
            "M_next": M_next,
        }

        return A_mat, M_n, F_vec, diagnostics

    def step_ale(
        self,
        mesh_n: TriangularMesh2D,
        mesh_next: TriangularMesh2D,
        u_n: np.ndarray,
        dt: float,
        t_next: float,
        pure_gcl: bool = False,
        enforce_dirichlet: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        A_mat, M_n, F_vec, diag = self.assemble_ale_system(
            mesh_n, mesh_next, dt, t_next, pure_gcl=pure_gcl
        )

        # Termine temporale rigorosamente conservativo: (1/dt) * M^n @ u^n + F^{n+1}
        rhs = (1.0 / dt) * (M_n @ u_n) + F_vec

        if enforce_dirichlet:
            bnd_nodes = mesh_next.boundary_nodes
            if pure_gcl:
                u_bnd = np.ones(len(bnd_nodes), dtype=np.float64)
            else:
                u_bnd = self.pde.evaluate_solution(
                    mesh_next.nodes[bnd_nodes, 0],
                    mesh_next.nodes[bnd_nodes, 1],
                    t_next
                )

            # Condizioni di Dirichlet
            rhs = rhs - A_mat[:, bnd_nodes] @ u_bnd
            rhs[bnd_nodes] = u_bnd

            A_mod = A_mat.tolil()
            A_mod[bnd_nodes, :] = 0.0
            A_mod[:, bnd_nodes] = 0.0
            A_mod[bnd_nodes, bnd_nodes] = 1.0
            A_mod = A_mod.tocsr()
        else:
            A_mod = A_mat

        u_next = spla.spsolve(A_mod, rhs)
        return u_next, diag