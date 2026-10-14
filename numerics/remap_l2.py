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

def assemble_l2_projection_system(
    old_nodes: np.ndarray,
    old_cells: np.ndarray,
    old_u: np.ndarray,
    new_nodes: np.ndarray,
    new_cells: np.ndarray,
    tol_barycentric: float = 1e-10,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Assembla il sistema della proiezione L2:

        M_new @ u_new = b

    La matrice M_new è la matrice di massa consistente P1
    assemblata sulla nuova mesh.

    Il termine noto viene integrato sulle intersezioni fra triangoli
    vecchi e nuovi. Su ciascuna intersezione, il prodotto fra la
    funzione di forma nuova e il campo P1 vecchio è quadratico:
    una quadratura triangolare di grado 2 è quindi esatta, a meno
    degli errori numerici geometrici e floating point.

    L'indice spaziale e il filtro esatto sui bounding box riducono
    il numero di coppie di triangoli da sottoporre al clipping.

    Nota:
        Si assume che le mesh descrivano lo stesso dominio,
        siano conformi e abbiano elementi triangolari orientati
        positivamente.
    """

    # =============================================================
    # 1. Conversione e validazione degli input
    # =============================================================
    old_nodes = np.asarray(old_nodes, dtype=np.float64)
    old_cells = np.asarray(old_cells, dtype=np.int64)
    old_u = np.asarray(old_u, dtype=np.float64)

    new_nodes = np.asarray(new_nodes, dtype=np.float64)
    new_cells = np.asarray(new_cells, dtype=np.int64)

    if old_nodes.ndim != 2 or old_nodes.shape[1] != 2:
        raise ValueError("old_nodes deve avere shape (N_old, 2).")

    if new_nodes.ndim != 2 or new_nodes.shape[1] != 2:
        raise ValueError("new_nodes deve avere shape (N_new, 2).")

    if old_cells.ndim != 2 or old_cells.shape[1] != 3:
        raise ValueError("old_cells deve avere shape (M_old, 3).")

    if new_cells.ndim != 2 or new_cells.shape[1] != 3:
        raise ValueError("new_cells deve avere shape (M_new, 3).")

    if old_cells.shape[0] == 0 or new_cells.shape[0] == 0:
        raise ValueError("Le mesh devono contenere triangoli.")

    if old_u.shape != (old_nodes.shape[0],):
        raise ValueError(
            "old_u deve contenere un valore per nodo della vecchia mesh."
        )

    if not (
        np.all(np.isfinite(old_nodes))
        and np.all(np.isfinite(new_nodes))
        and np.all(np.isfinite(old_u))
    ):
        raise ValueError("Gli input contengono valori non finiti.")

    if tol_barycentric < 0.0:
        raise ValueError("tol_barycentric deve essere non negativa.")

    num_old_nodes = old_nodes.shape[0]
    num_new_nodes = new_nodes.shape[0]
    num_old_cells = old_cells.shape[0]
    num_new_cells = new_cells.shape[0]

    if np.any(old_cells < 0) or np.any(old_cells >= num_old_nodes):
        raise ValueError("old_cells contiene indici nodali non validi.")

    if np.any(new_cells < 0) or np.any(new_cells >= num_new_nodes):
        raise ValueError("new_cells contiene indici nodali non validi.")

    # =============================================================
    # 2. Geometria delle due mesh
    # =============================================================
    old_triangles = old_nodes[old_cells]
    new_triangles = new_nodes[new_cells]

    old_v1 = old_triangles[:, 0, :]
    old_v2 = old_triangles[:, 1, :]
    old_v3 = old_triangles[:, 2, :]

    new_v1 = new_triangles[:, 0, :]
    new_v2 = new_triangles[:, 1, :]
    new_v3 = new_triangles[:, 2, :]

    det_old = (
        (old_v2[:, 0] - old_v1[:, 0])
        * (old_v3[:, 1] - old_v1[:, 1])
        - (old_v3[:, 0] - old_v1[:, 0])
        * (old_v2[:, 1] - old_v1[:, 1])
    )

    det_new = (
        (new_v2[:, 0] - new_v1[:, 0])
        * (new_v3[:, 1] - new_v1[:, 1])
        - (new_v3[:, 0] - new_v1[:, 0])
        * (new_v2[:, 1] - new_v1[:, 1])
    )

    det_tol = 1e-14

    bad_old = np.flatnonzero(det_old <= det_tol)
    if bad_old.size:
        idx = int(bad_old[0])
        raise ValueError(
            "Vecchia mesh non valida: elemento invertito o degenere. "
            f"Indice={idx}, det_J={det_old[idx]:.3e}."
        )

    bad_new = np.flatnonzero(det_new <= det_tol)
    if bad_new.size:
        idx = int(bad_new[0])
        raise ValueError(
            "Nuova mesh non valida: elemento invertito o degenere. "
            f"Indice={idx}, det_J={det_new[idx]:.3e}."
        )

    areas_new = 0.5 * det_new

    all_nodes = np.vstack((old_nodes, new_nodes))
    extents = np.ptp(all_nodes, axis=0)
    geom_scale = max(float(np.max(extents)), 1.0)

    bbox_tol = 1e-12 * geom_scale
    clip_tol = 1e-13 * geom_scale**2
    point_tol = 1e-12 * geom_scale
    area_tol = 1e-16 * geom_scale**2

    # Bounding box reale di ogni triangolo.
    old_bbox_min = old_triangles.min(axis=1)
    old_bbox_max = old_triangles.max(axis=1)

    new_bbox_min = new_triangles.min(axis=1)
    new_bbox_max = new_triangles.max(axis=1)

    # =============================================================
    # 3. Operazioni geometriche
    # =============================================================
    def cross2(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def clean_polygon(points):
        cleaned = []

        for p in points:
            p = np.asarray(p, dtype=np.float64)

            if (
                not cleaned
                or np.linalg.norm(p - cleaned[-1]) > point_tol
            ):
                cleaned.append(p.copy())

        if (
            len(cleaned) > 1
            and np.linalg.norm(cleaned[0] - cleaned[-1]) <= point_tol
        ):
            cleaned.pop()

        return cleaned

    def clip_polygon_by_triangle(
        subject_triangle: np.ndarray,
        clip_triangle: np.ndarray,
    ) -> np.ndarray:
        """Clipping Sutherland-Hodgman fra due triangoli convessi."""
        polygon = [
            subject_triangle[0].copy(),
            subject_triangle[1].copy(),
            subject_triangle[2].copy(),
        ]

        for edge_idx in range(3):
            if not polygon:
                break

            a = clip_triangle[edge_idx]
            b = clip_triangle[(edge_idx + 1) % 3]
            edge = b - a

            input_polygon = polygon
            polygon = []

            previous = input_polygon[-1]
            d_previous = cross2(edge, previous - a)
            previous_inside = d_previous >= -clip_tol

            for current in input_polygon:
                d_current = cross2(edge, current - a)
                current_inside = d_current >= -clip_tol

                if current_inside != previous_inside:
                    denominator = d_previous - d_current

                    if abs(denominator) > 1e-30:
                        alpha = d_previous / denominator
                        alpha = float(np.clip(alpha, 0.0, 1.0))

                        intersection = (
                            previous + alpha * (current - previous)
                        )
                        polygon.append(intersection)

                if current_inside:
                    polygon.append(current.copy())

                previous = current
                d_previous = d_current
                previous_inside = current_inside

            polygon = clean_polygon(polygon)

        if len(polygon) < 3:
            return np.empty((0, 2), dtype=np.float64)

        return np.asarray(polygon, dtype=np.float64)

    def polygon_area(polygon: np.ndarray) -> float:
        if polygon.shape[0] < 3:
            return 0.0

        twice_area = 0.0

        for i in range(polygon.shape[0]):
            twice_area += cross2(
                polygon[i],
                polygon[(i + 1) % polygon.shape[0]],
            )

        return 0.5 * abs(twice_area)

    def barycentric_coordinates(
        points: np.ndarray,
        triangle: np.ndarray,
    ) -> np.ndarray:
        a, b, c = triangle

        ab = b - a
        ac = c - a
        det = cross2(ab, ac)

        if abs(det) <= det_tol:
            raise ValueError(
                "Coordinate baricentriche richieste "
                "per un triangolo degenere."
            )

        rel = points - a

        lambda_b = (
            rel[:, 0] * ac[1] - rel[:, 1] * ac[0]
        ) / det

        lambda_c = (
            ab[0] * rel[:, 1] - ab[1] * rel[:, 0]
        ) / det

        lambda_a = 1.0 - lambda_b - lambda_c

        return np.column_stack(
            (lambda_a, lambda_b, lambda_c)
        )

    # =============================================================
    # 4. Indice spaziale uniforme
    # =============================================================
    domain_min = old_nodes.min(axis=0)
    domain_max = old_nodes.max(axis=0)
    domain_extent = domain_max - domain_min

    if np.any(domain_extent <= 0.0):
        raise ValueError(
            "La vecchia mesh ha un'estensione del dominio nulla."
        )

    aspect = domain_extent[0] / domain_extent[1]
    target_bins = max(
        1.0,
        np.sqrt(max(num_old_cells / 2.0, 1.0)),
    )

    n_bins_x = max(
        1, int(round(target_bins * np.sqrt(aspect)))
    )
    n_bins_y = max(
        1, int(round(target_bins / np.sqrt(aspect)))
    )

    bin_width_x = domain_extent[0] / n_bins_x
    bin_width_y = domain_extent[1] / n_bins_y

    bins = [[] for _ in range(n_bins_x * n_bins_y)]

    def bin_range(
        box_min: np.ndarray,
        box_max: np.ndarray,
    ):
        ix0 = int(np.floor(
            (box_min[0] - domain_min[0] - bbox_tol)
            / bin_width_x
        ))
        ix1 = int(np.floor(
            (box_max[0] - domain_min[0] + bbox_tol)
            / bin_width_x
        ))

        iy0 = int(np.floor(
            (box_min[1] - domain_min[1] - bbox_tol)
            / bin_width_y
        ))
        iy1 = int(np.floor(
            (box_max[1] - domain_min[1] + bbox_tol)
            / bin_width_y
        ))

        ix0 = int(np.clip(ix0, 0, n_bins_x - 1))
        ix1 = int(np.clip(ix1, 0, n_bins_x - 1))
        iy0 = int(np.clip(iy0, 0, n_bins_y - 1))
        iy1 = int(np.clip(iy1, 0, n_bins_y - 1))

        return ix0, ix1, iy0, iy1

    for old_e in range(num_old_cells):
        ix0, ix1, iy0, iy1 = bin_range(
            old_bbox_min[old_e],
            old_bbox_max[old_e],
        )

        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                bins[iy * n_bins_x + ix].append(old_e)

    def candidate_old_cells(new_e: int) -> np.ndarray:
        """
        Restituisce solo i vecchi triangoli i cui bounding box
        intersecano realmente il bounding box del nuovo triangolo.

        Il filtro è necessario ma non sufficiente per l'intersezione
        dei triangoli: il clipping successivo resta indispensabile.
        """
        box_min = new_bbox_min[new_e]
        box_max = new_bbox_max[new_e]

        ix0, ix1, iy0, iy1 = bin_range(box_min, box_max)

        candidates = set()

        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                candidates.update(
                    bins[iy * n_bins_x + ix]
                )

        if not candidates:
            return np.empty(0, dtype=np.int64)

        candidate_ids = np.asarray(
            sorted(candidates), dtype=np.int64
        )

        # NUOVO FILTRO: esclude le coppie con bounding box disgiunti.
        overlaps_bbox = (
            (
                old_bbox_max[candidate_ids, 0]
                >= box_min[0] - bbox_tol
            )
            & (
                old_bbox_min[candidate_ids, 0]
                <= box_max[0] + bbox_tol
            )
            & (
                old_bbox_max[candidate_ids, 1]
                >= box_min[1] - bbox_tol
            )
            & (
                old_bbox_min[candidate_ids, 1]
                <= box_max[1] + bbox_tol
            )
        )

        return candidate_ids[overlaps_bbox]

    # =============================================================
    # 5. Matrice di massa della nuova mesh
    # =============================================================
    base_M = np.array(
        [
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    ) / 12.0

    M_elem = (
        areas_new[:, None, None]
        * base_M[None, :, :]
    )

    rows_M = np.repeat(
        new_cells[:, :, None], 3, axis=2
    ).flatten()

    cols_M = np.repeat(
        new_cells[:, None, :], 3, axis=1
    ).flatten()

    M_new = sp.coo_matrix(
        (
            M_elem.flatten(),
            (rows_M, cols_M),
        ),
        shape=(num_new_nodes, num_new_nodes),
    ).tocsr()

    # =============================================================
    # 6. Quadratura sulle intersezioni
    # =============================================================
    quad_bary = np.array(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=np.float64,
    )

    quad_weights = np.full(
        3, 1.0 / 6.0, dtype=np.float64
    )

    b = np.zeros(num_new_nodes, dtype=np.float64)

    for new_e in range(num_new_cells):

        new_tri = new_triangles[new_e]
        new_cell = new_cells[new_e]
        area_target = float(areas_new[new_e])

        candidates = candidate_old_cells(new_e)

        if candidates.size == 0:
            raise ValueError(
                "[RemapL2IntersectionError] Nessun triangolo "
                "vecchio candidato per la cella nuova "
                f"{new_e}."
            )

        covered_area = 0.0

        for old_e in candidates:

            old_tri = old_triangles[old_e]
            old_cell = old_cells[old_e]

            polygon = clip_polygon_by_triangle(
                new_tri, old_tri
            )

            if polygon.shape[0] < 3:
                continue

            overlap_area = polygon_area(polygon)

            if overlap_area <= area_tol:
                continue

            covered_area += overlap_area

            # Triangolazione a ventaglio del poligono di intersezione.
            p0 = polygon[0]

            for fan_idx in range(1, polygon.shape[0] - 1):

                sub_tri = np.asarray(
                    [
                        p0,
                        polygon[fan_idx],
                        polygon[fan_idx + 1],
                    ],
                    dtype=np.float64,
                )

                det_sub = cross2(
                    sub_tri[1] - sub_tri[0],
                    sub_tri[2] - sub_tri[0],
                )

                if abs(det_sub) <= 2.0 * area_tol:
                    continue

                if det_sub < 0.0:
                    sub_tri[[1, 2]] = sub_tri[[2, 1]]
                    det_sub = -det_sub

                q_points = quad_bary @ sub_tri

                lambda_new = barycentric_coordinates(
                    q_points, new_tri
                )

                lambda_old = barycentric_coordinates(
                    q_points, old_tri
                )

                u_old_q = lambda_old @ old_u[old_cell]

                weighted_u = (
                    det_sub * quad_weights * u_old_q
                )

                local_b = np.sum(
                    weighted_u[:, None] * lambda_new,
                    axis=0,
                )

                np.add.at(b, new_cell, local_b)

        # =========================================================
        # 7. Controllo della copertura di ciascun triangolo nuovo
        # =========================================================
        coverage_error = abs(
            covered_area - area_target
        )

        coverage_tol = max(
            1e-8 * area_target,
            1e-12 * geom_scale**2,
        )

        if coverage_error > coverage_tol:
            raise ValueError(
                "[RemapL2CoverageError] Le intersezioni con la "
                "vecchia mesh non ricoprono correttamente "
                f"la cella nuova {new_e}. "
                f"Area nuova={area_target:.16e}; "
                f"area intersezioni={covered_area:.16e}; "
                f"scarto={coverage_error:.3e}; "
                f"tolleranza={coverage_tol:.3e}. "
                "Verificare che le mesh coprano lo stesso dominio."
            )

    if not np.all(np.isfinite(b)):
        raise FloatingPointError(
            "Il termine noto della proiezione L2 contiene "
            "valori non finiti."
        )

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