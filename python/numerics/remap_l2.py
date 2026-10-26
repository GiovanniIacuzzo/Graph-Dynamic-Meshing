"""
numerics/remap_l2.py
Operatore di trasferimento variazionale mediante proiezione L2 globale sulla nuova mesh:
    M_new * u_new = b_proj,   (b_proj)_i = int_{Omega} phi_i^{new}(x) u_old(x) dx
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from numerics.fem import _get_triangle_quadrature_degree_5
from numerics.remap import remap_p1_solution


def assemble_l2_projection_system(
    old_nodes: np.ndarray,
    old_cells: np.ndarray,
    old_u: np.ndarray,
    new_nodes: np.ndarray,
    new_cells: np.ndarray,
):
    """Assemble the new-mesh mass matrix and projection right-hand side.

    Returns ``(M_new, b_proj)`` for the variational problem
    ``M_new @ u_new = b_proj``. Keeping assembly separate makes the discrete
    conservation identity directly inspectable in scientific diagnostics.
    """
    old_nodes = np.asarray(old_nodes, dtype=np.float64)
    old_cells = np.asarray(old_cells, dtype=np.int64)
    old_u = np.asarray(old_u, dtype=np.float64)
    new_nodes = np.asarray(new_nodes, dtype=np.float64)
    new_cells = np.asarray(new_cells, dtype=np.int64)
    if old_nodes.ndim != 2 or old_nodes.shape[1] != 2 or new_nodes.ndim != 2 or new_nodes.shape[1] != 2:
        raise ValueError("old_nodes e new_nodes devono avere shape (N, 2).")
    if old_cells.ndim != 2 or old_cells.shape[1] != 3 or new_cells.ndim != 2 or new_cells.shape[1] != 3:
        raise ValueError("old_cells e new_cells devono avere shape (M, 3).")
    if old_u.shape != (old_nodes.shape[0],):
        raise ValueError(f"old_u deve avere shape ({old_nodes.shape[0]},).")
    if not all(np.all(np.isfinite(a)) for a in (old_nodes, new_nodes, old_u)):
        raise ValueError("Input del remapping L2 non finiti.")
    if np.any(old_cells < 0) or np.any(old_cells >= old_nodes.shape[0]):
        raise ValueError("old_cells contiene indici fuori dai limiti.")
    if np.any(new_cells < 0) or np.any(new_cells >= new_nodes.shape[0]):
        raise ValueError("new_cells contiene indici fuori dai limiti.")

    num_nodes = new_nodes.shape[0]
    tri = new_nodes[new_cells]
    x1, y1 = tri[:, 0, 0], tri[:, 0, 1]
    x2, y2 = tri[:, 1, 0], tri[:, 1, 1]
    x3, y3 = tri[:, 2, 0], tri[:, 2, 1]
    det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if np.any(det_J <= 0.0):
        raise ValueError("La mesh di proiezione deve avere triangoli non degeneri e orientati positivamente.")
    areas = 0.5 * det_J

    base_M = np.array(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float64
    ) / 12.0
    M_elem = areas[:, None, None] * base_M[None, :, :]
    rows = np.repeat(new_cells[:, :, None], 3, axis=2).ravel()
    cols = np.repeat(new_cells[:, None, :], 3, axis=1).ravel()
    M_new = sp.coo_matrix((M_elem.ravel(), (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()

    qp_ref, qw_ref = _get_triangle_quadrature_degree_5()
    rhs = np.zeros(num_nodes, dtype=np.float64)
    for (xi, eta), w in zip(qp_ref, qw_ref):
        phi1, phi2, phi3 = 1.0 - xi - eta, xi, eta
        xq = phi1 * x1 + phi2 * x2 + phi3 * x3
        yq = phi1 * y1 + phi2 * y2 + phi3 * y3
        u_old_q = remap_p1_solution(old_nodes, old_cells, old_u, np.column_stack((xq, yq)))
        factor = (2.0 * w) * areas * u_old_q
        np.add.at(rhs, new_cells[:, 0], factor * phi1)
        np.add.at(rhs, new_cells[:, 1], factor * phi2)
        np.add.at(rhs, new_cells[:, 2], factor * phi3)
    return M_new, rhs


def remap_l2_projection(
    old_nodes: np.ndarray,
    old_cells: np.ndarray,
    old_u: np.ndarray,
    new_nodes: np.ndarray,
    new_cells: np.ndarray,
) -> np.ndarray:
    """Project a source P1 field onto the P1 space defined on the new mesh."""
    M_new, rhs = assemble_l2_projection_system(
        old_nodes, old_cells, old_u, new_nodes, new_cells
    )
    return spla.spsolve(M_new, rhs)
