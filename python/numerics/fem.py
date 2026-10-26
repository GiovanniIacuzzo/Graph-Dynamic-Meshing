"""
numerics/fem.py
Solutore agli Elementi Finiti P1 di Galerkin continuo per Convezione-Diffusione 2D
e ricostruzione dei gradienti nodali.
"""

from typing import Tuple, Optional, Any
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from geometry.mesh import TriangularMesh2D
from numerics.pde import ManufacturedConvectionDiffusion2D


def _get_triangle_quadrature_degree_5() -> Tuple[np.ndarray, np.ndarray]:
    """
    Restituisce punti di quadratura (coordinate baricentriche xi, eta) e pesi
    per il triangolo di riferimento T_ref = {(xi, eta) : xi >= 0, eta >= 0, xi + eta <= 1}.
    Area del triangolo di riferimento = 0.5.
    """
    p1 = [1.0 / 3.0, 1.0 / 3.0]
    w1 = 0.225 / 2.0

    a = (6.0 - np.sqrt(15.0)) / 21.0
    b = (9.0 + 2.0 * np.sqrt(15.0)) / 21.0
    w_grp1 = ((155.0 - np.sqrt(15.0)) / 1200.0) / 2.0
    p_grp1 = [[a, a], [a, b], [b, a]]

    c = (6.0 + np.sqrt(15.0)) / 21.0
    d = (9.0 - 2.0 * np.sqrt(15.0)) / 21.0
    w_grp2 = ((155.0 + np.sqrt(15.0)) / 1200.0) / 2.0
    p_grp2 = [[c, c], [c, d], [d, c]]

    points = np.array([p1] + p_grp1 + p_grp2, dtype=np.float64)
    weights = np.array([w1] + [w_grp1] * 3 + [w_grp2] * 3, dtype=np.float64)
    return points, weights


class ConvectionDiffusionFEM2D:
    """
    Solutore FEM Galerkin continuo P1 per convezione-diffusione:
        M du/dt + C u + nu K u = F(t)
    """

    def __init__(self, mesh: TriangularMesh2D, pde: ManufacturedConvectionDiffusion2D):
        self.mesh = mesh
        self.pde = pde
        self.num_nodes = mesh.num_nodes
        self.num_cells = mesh.num_cells
        self.nodes = mesh.nodes
        self.cells = mesh.cells
        self.boundary_nodes = mesh.boundary_nodes

        self.qp_ref, self.qw_ref = _get_triangle_quadrature_degree_5()
        self._compute_element_geometry()

        self.M = self._assemble_mass_matrix()
        self.K = self._assemble_stiffness_matrix()
        self.C = self._assemble_convection_matrix()

    def _compute_element_geometry(self) -> None:
        x = self.nodes[:, 0]
        y = self.nodes[:, 1]
        c = self.cells

        x1, y1 = x[c[:, 0]], y[c[:, 0]]
        x2, y2 = x[c[:, 1]], y[c[:, 1]]
        x3, y3 = x[c[:, 2]], y[c[:, 2]]

        det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        if np.any(det_J <= 1e-15):
            raise ValueError("Rilevati elementi invertiti o degeneri durante l'assemblaggio FEM.")
        self.det_J = det_J
        self.areas = 0.5 * np.abs(det_J)

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
        base_M = (
            np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float64)
            / 12.0
        )
        M_elem = self.areas[:, None, None] * base_M[None, :, :]
        rows = np.repeat(self.cells[:, :, None], 3, axis=2).flatten()
        cols = np.repeat(self.cells[:, None, :], 3, axis=1).flatten()
        return sp.coo_matrix(
            (M_elem.flatten(), (rows, cols)), shape=(self.num_nodes, self.num_nodes)
        ).tocsr()

    def _assemble_stiffness_matrix(self) -> sp.csr_matrix:
        gx = self.dphi_dx[:, :, None] * self.dphi_dx[:, None, :]
        gy = self.dphi_dy[:, :, None] * self.dphi_dy[:, None, :]
        K_elem = self.areas[:, None, None] * (gx + gy)
        rows = np.repeat(self.cells[:, :, None], 3, axis=2).flatten()
        cols = np.repeat(self.cells[:, None, :], 3, axis=1).flatten()
        return sp.coo_matrix(
            (K_elem.flatten(), (rows, cols)), shape=(self.num_nodes, self.num_nodes)
        ).tocsr()

    def _assemble_convection_matrix(self) -> sp.csr_matrix:
        b_dot_grad = self.pde.bx * self.dphi_dx + self.pde.by * self.dphi_dy
        int_phi = (self.areas / 3.0)[:, None, None]
        C_elem = int_phi * np.ones((1, 3, 1), dtype=np.float64) * b_dot_grad[:, None, :]
        rows = np.repeat(self.cells[:, :, None], 3, axis=2).flatten()
        cols = np.repeat(self.cells[:, None, :], 3, axis=1).flatten()
        return sp.coo_matrix(
            (C_elem.flatten(), (rows, cols)), shape=(self.num_nodes, self.num_nodes)
        ).tocsr()

    def assemble_rhs(self, t: float) -> np.ndarray:
        F = np.zeros(self.num_nodes, dtype=np.float64)
        c = self.cells
        x = self.nodes[:, 0]
        y = self.nodes[:, 1]

        x1, y1 = x[c[:, 0]], y[c[:, 0]]
        x2, y2 = x[c[:, 1]], y[c[:, 1]]
        x3, y3 = x[c[:, 2]], y[c[:, 2]]

        for (xi, eta), w in zip(self.qp_ref, self.qw_ref):
            phi1 = 1.0 - xi - eta
            phi2 = xi
            phi3 = eta

            xq = phi1 * x1 + phi2 * x2 + phi3 * x3
            yq = phi1 * y1 + phi2 * y2 + phi3 * y3
            f_vals = self.pde.evaluate_forcing(xq, yq, t)
            factor = (2.0 * w) * self.areas * f_vals

            np.add.at(F, c[:, 0], factor * phi1)
            np.add.at(F, c[:, 1], factor * phi2)
            np.add.at(F, c[:, 2], factor * phi3)

        return F

    def step_backward_euler(self, u_n: np.ndarray, dt: float, t_next: float) -> np.ndarray:
        inv_dt = 1.0 / dt
        A = inv_dt * self.M + self.C + self.pde.nu * self.K

        F_next = self.assemble_rhs(t_next)
        b_vec = inv_dt * (self.M @ u_n) + F_next

        A = A.tolil()
        bnd = self.boundary_nodes
        A[bnd, :] = 0.0
        A[bnd, bnd] = 1.0
        b_vec[bnd] = 0.0
        A = A.tocsr()

        return spla.spsolve(A, b_vec)


def compute_nodal_gradient(nodes: np.ndarray, cells: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Calcola il gradiente nodale P1 continuo mediante media pesata per area."""
    num_nodes = nodes.shape[0]
    x1, y1 = nodes[cells[:, 0], 0], nodes[cells[:, 0], 1]
    x2, y2 = nodes[cells[:, 1], 0], nodes[cells[:, 1], 1]
    x3, y3 = nodes[cells[:, 2], 0], nodes[cells[:, 2], 1]

    det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    areas = 0.5 * np.abs(det_J)
    inv_det = 1.0 / det_J

    dphi_dx = np.column_stack([(y2 - y3) * inv_det, (y3 - y1) * inv_det, (y1 - y2) * inv_det])
    dphi_dy = np.column_stack([(x3 - x2) * inv_det, (x1 - x3) * inv_det, (x2 - x1) * inv_det])

    u_elem = u[cells]
    gx_elem = np.sum(u_elem * dphi_dx, axis=1)
    gy_elem = np.sum(u_elem * dphi_dy, axis=1)

    grad_nodal = np.zeros((num_nodes, 2), dtype=np.float64)
    nodal_area = np.zeros(num_nodes, dtype=np.float64)

    for k in range(3):
        idx = cells[:, k]
        np.add.at(grad_nodal[:, 0], idx, areas * gx_elem)
        np.add.at(grad_nodal[:, 1], idx, areas * gy_elem)
        np.add.at(nodal_area, idx, areas)

    denom = np.maximum(nodal_area[:, None], 1e-14)
    return grad_nodal / denom


def compute_field_gradients(
    mesh: TriangularMesh2D,
    u: np.ndarray,
    pde: Optional[Any] = None,
    t: float = 0.0,
    gradient_mode: str = "fem",
) -> np.ndarray:
    """Restituisce il gradiente nodale ricostruito ('fem') o analitico ('exact')."""
    if gradient_mode == "exact":
        if pde is None:
            raise ValueError("pde richiesto per gradient_mode='exact'.")
        _, ux, uy, _, _, _, _ = pde.evaluate_exact(mesh.nodes[:, 0], mesh.nodes[:, 1], t)
        return np.column_stack([ux, uy])
    return compute_nodal_gradient(mesh.nodes, mesh.cells, u)