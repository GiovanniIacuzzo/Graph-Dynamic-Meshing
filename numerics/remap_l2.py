"""
numerics/remap_l2.py
Operatore di remapping continuo basato su L2-projection standard.

Risolve il sistema variazionale:
    int_Omega phi_i^{new} u_{new} dOmega = int_Omega phi_i^{new} u_{old} dOmega
ovvero:
    M^{new} u_{new} = b
con:
    b_i = sum_e int_{K_e^{new}} phi_i^{new}(x) * u_{old}(x) dOmega

u_{old}(x) viene valutato nei punti di quadratura di Dunavant (grado 5)
usando il localizzatore baricentrico su (old_nodes, old_cells).
"""

from typing import Tuple, Dict, Any
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from numerics.fem import _get_triangle_quadrature_degree_5
from numerics.remap import compute_barycentric_coordinates_2d


def assemble_l2_projection_system(
    old_nodes: np.ndarray,
    old_cells: np.ndarray,
    old_u: np.ndarray,
    new_nodes: np.ndarray,
    new_cells: np.ndarray,
    tol_barycentric: float = 1e-10
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Assembla la matrice di massa M^{new} e il termine noto b per la L2-projection.
    """
    num_new_nodes = new_nodes.shape[0]
    num_new_cells = new_cells.shape[0]
    num_old_cells = old_cells.shape[0]

    # Vertici per triangoli old
    ov1 = old_nodes[old_cells[:, 0]]
    ov2 = old_nodes[old_cells[:, 1]]
    ov3 = old_nodes[old_cells[:, 2]]

    # Bounding box dei triangoli old per pruning rapido
    min_ox = np.minimum(np.minimum(ov1[:, 0], ov2[:, 0]), ov3[:, 0]) - 1e-8
    max_ox = np.maximum(np.maximum(ov1[:, 0], ov2[:, 0]), ov3[:, 0]) + 1e-8
    min_oy = np.minimum(np.minimum(ov1[:, 1], ov2[:, 1]), ov3[:, 1]) - 1e-8
    max_oy = np.maximum(np.maximum(ov1[:, 1], ov2[:, 1]), ov3[:, 1]) + 1e-8

    # Geometria nuova mesh
    nx1, ny1 = new_nodes[new_cells[:, 0], 0], new_nodes[new_cells[:, 0], 1]
    nx2, ny2 = new_nodes[new_cells[:, 1], 0], new_nodes[new_cells[:, 1], 1]
    nx3, ny3 = new_nodes[new_cells[:, 2], 0], new_nodes[new_cells[:, 2], 1]

    det_J_new = (nx2 - nx1) * (ny3 - ny1) - (nx3 - nx1) * (ny2 - ny1)
    areas_new = 0.5 * np.abs(det_J_new)

    # 1. Matrice di massa globale M^{new}
    base_M = np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0]
    ], dtype=np.float64) / 12.0

    M_elem = areas_new[:, None, None] * base_M[None, :, :]
    rows_M = np.repeat(new_cells[:, :, None], 3, axis=2).flatten()
    cols_M = np.repeat(new_cells[:, None, :], 3, axis=1).flatten()
    M_new = sp.coo_matrix((M_elem.flatten(), (rows_M, cols_M)), shape=(num_new_nodes, num_new_nodes)).tocsr()

    # 2. Termine noto b via quadratura Dunavant grado 5 (7 punti per cella new)
    qp_ref, qw_ref = _get_triangle_quadrature_degree_5()
    b = np.zeros(num_new_nodes, dtype=np.float64)

    for (xi, eta), w in zip(qp_ref, qw_ref):
        phi1 = 1.0 - xi - eta
        phi2 = xi
        phi3 = eta

        # Coordinate fisiche dei punti di quadratura sulla nuova mesh: [num_new_cells]
        xq = phi1 * nx1 + phi2 * nx2 + phi3 * nx3
        yq = phi1 * ny1 + phi2 * ny2 + phi3 * ny3
        pts_q = np.column_stack([xq, yq])

        # Valutazione di u_old nei punti pts_q
        u_old_at_q = np.zeros(num_new_cells, dtype=np.float64)

        for e in range(num_new_cells):
            px, py = pts_q[e, 0], pts_q[e, 1]

            in_bbox = (px >= min_ox) & (px <= max_ox) & (py >= min_oy) & (py <= max_oy)
            cand = np.where(in_bbox)[0]
            if len(cand) == 0:
                cand = np.arange(num_old_cells)

            p_rep = np.tile(pts_q[e], (len(cand), 1))
            l1, l2, l3 = compute_barycentric_coordinates_2d(p_rep, ov1[cand], ov2[cand], ov3[cand])

            inside = (l1 >= -tol_barycentric) & (l2 >= -tol_barycentric) & (l3 >= -tol_barycentric)
            vld = np.where(inside)[0]

            if len(vld) == 0:
                # Scelta del triangolo con minima violazione se al bordo numerico
                min_v = np.maximum.reduce([np.maximum(0.0, -l1), np.maximum(0.0, -l2), np.maximum(0.0, -l3)])
                b_idx = np.argmin(min_v)
                c_idx = cand[b_idx]
                cl1, cl2, cl3 = max(0.0, l1[b_idx]), max(0.0, l2[b_idx]), max(0.0, l3[b_idx])
                s = cl1 + cl2 + cl3
                cl1, cl2, cl3 = cl1 / s, cl2 / s, cl3 / s
            else:
                c_idx = cand[vld[0]]
                cl1, cl2, cl3 = l1[vld[0]], l2[vld[0]], l3[vld[0]]

            old_tri = old_cells[c_idx]
            u_old_at_q[e] = cl1 * old_u[old_tri[0]] + cl2 * old_u[old_tri[1]] + cl3 * old_u[old_tri[2]]

        # Accumulo del termine noto b: w * (2 * Area) * u_old * phi_i
        factor = (2.0 * w) * areas_new * u_old_at_q
        np.add.at(b, new_cells[:, 0], factor * phi1)
        np.add.at(b, new_cells[:, 1], factor * phi2)
        np.add.at(b, new_cells[:, 2], factor * phi3)

    return M_new, b


def remap_l2_projection(
    old_nodes: np.ndarray,
    old_cells: np.ndarray,
    old_u: np.ndarray,
    new_nodes: np.ndarray,
    new_cells: np.ndarray,
    tol_barycentric: float = 1e-10
) -> np.ndarray:
    """
    Risolve la L2-projection M^{new} u_{new} = b con solver sparse diretto.
    """
    M_new, b = assemble_l2_projection_system(
        old_nodes, old_cells, old_u, new_nodes, new_cells, tol_barycentric
    )
    u_new = spla.spsolve(M_new, b)
    return u_new