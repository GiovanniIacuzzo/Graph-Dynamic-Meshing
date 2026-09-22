"""
numerics/remap.py
Operatore di remapping continuo P1 pointwise conforme per r-adaptation.

Garantisce il trasferimento rigoroso:
    u_new(x) = sum_{j=1}^3 lambda_j * u_old(x_j)
senza euristica di fallback (nearest-neighbor o inverse-distance) e senza
clamping artificiale di violazioni geometriche.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from geometry.mesh import TriangularMesh2D


def compute_barycentric_coordinates_2d(
    points: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola analiticamente le coordinate baricentriche (lambda_1, lambda_2, lambda_3)
    di un insieme di punti rispetto a un insieme di triangoli formati da (v1, v2, v3).
    """
    x, y = points[:, 0], points[:, 1]
    x1, y1 = v1[:, 0], v1[:, 1]
    x2, y2 = v2[:, 0], v2[:, 1]
    x3, y3 = v3[:, 0], v3[:, 1]

    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    inv_det = 1.0 / np.where(np.abs(det) < 1e-15, 1e-15, det)

    l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) * inv_det
    l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) * inv_det
    l3 = 1.0 - l1 - l2

    return l1, l2, l3


def compute_mesh_mass(mesh: TriangularMesh2D, u: np.ndarray) -> float:
    """
    Calcola la massa totale integrale su mesh P1:
        M = int_Omega u dOmega = sum_e (Area_e / 3) * (u_1 + u_2 + u_3)
    """
    cells = mesh.cells
    nodes = mesh.nodes

    x1, y1 = nodes[cells[:, 0], 0], nodes[cells[:, 0], 1]
    x2, y2 = nodes[cells[:, 1], 0], nodes[cells[:, 1], 1]
    x3, y3 = nodes[cells[:, 2], 0], nodes[cells[:, 2], 1]

    det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    areas = 0.5 * np.abs(det_J)
    elem_mean = np.mean(u[cells], axis=1)

    return float(np.sum(areas * elem_mean))


def remap_p1_solution(
    old_nodes: np.ndarray,
    old_cells: np.ndarray,
    old_u: np.ndarray,
    new_nodes: np.ndarray,
    tol_barycentric: float = 1e-10,
    return_diagnostics: bool = False
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    API primaria di remapping per lo Stage B.

    Per ciascun punto di new_nodes:
      1. Individua il triangolo in old_cells che lo contiene (lambda_i >= -tol)
      2. Interpola linearmente il campo con le coordinate baricentriche P1 esatte.
      3. Nessun clamping, nessun nearest-neighbor: solleva ValueError se un punto
         non appartiene geometricamente ad alcun elemento.
    """
    old_nodes = np.asarray(old_nodes, dtype=np.float64)
    new_nodes = np.asarray(new_nodes, dtype=np.float64)
    old_cells = np.asarray(old_cells, dtype=np.int64)
    old_u = np.asarray(old_u, dtype=np.float64)

    num_new = new_nodes.shape[0]
    num_cells = old_cells.shape[0]

    v1 = old_nodes[old_cells[:, 0]]
    v2 = old_nodes[old_cells[:, 1]]
    v3 = old_nodes[old_cells[:, 2]]

    # Bounding box dei triangoli per accelerare il pruning
    min_x = np.minimum(np.minimum(v1[:, 0], v2[:, 0]), v3[:, 0]) - 1e-8
    max_x = np.maximum(np.maximum(v1[:, 0], v2[:, 0]), v3[:, 0]) + 1e-8
    min_y = np.minimum(np.minimum(v1[:, 1], v2[:, 1]), v3[:, 1]) - 1e-8
    max_y = np.maximum(np.maximum(v1[:, 1], v2[:, 1]), v3[:, 1]) + 1e-8

    u_new = np.empty(num_new, dtype=np.float64)
    unlocalized_nodes = []
    max_boundary_violation = 0.0

    for i in range(num_new):
        px, py = new_nodes[i, 0], new_nodes[i, 1]

        # 1. Pruning tramite Bounding Box
        in_bbox = (px >= min_x) & (px <= max_x) & (py >= min_y) & (py <= max_y)
        candidates = np.where(in_bbox)[0]

        if len(candidates) == 0:
            candidates = np.arange(num_cells)

        # 2. Calcolo coordinate baricentriche
        pts_rep = np.tile(new_nodes[i], (len(candidates), 1))
        l1, l2, l3 = compute_barycentric_coordinates_2d(
            pts_rep, v1[candidates], v2[candidates], v3[candidates]
        )

        inside = (l1 >= -tol_barycentric) & (l2 >= -tol_barycentric) & (l3 >= -tol_barycentric)
        valid = np.where(inside)[0]

        if len(valid) == 0:
            # Calcolo diagnostico della violazione minima tra tutte le celle candidate
            min_viol = np.min(np.maximum.reduce([
                np.maximum(0.0, -l1),
                np.maximum(0.0, -l2),
                np.maximum(0.0, -l3)
            ]))
            max_boundary_violation = max(max_boundary_violation, float(min_viol))
            unlocalized_nodes.append((i, float(px), float(py), float(min_viol)))
            continue

        # Selezione primo elemento contenitore valido
        idx = candidates[valid[0]]
        b_l1 = float(l1[valid[0]])
        b_l2 = float(l2[valid[0]])
        b_l3 = float(l3[valid[0]])

        # Interpolazione affine pura P1 (nessuna alterazione o clamping artificiale)
        tri_v = old_cells[idx]
        u_new[i] = b_l1 * old_u[tri_v[0]] + b_l2 * old_u[tri_v[1]] + b_l3 * old_u[tri_v[2]]

    # 3. Gestione errori esplicita: nessun fallback tollerato
    if len(unlocalized_nodes) > 0:
        first_fail = unlocalized_nodes[0]
        raise ValueError(
            f"[RemapLocalizationError] Rilevati {len(unlocalized_nodes)} nodi non localizzabili! "
            f"Esempio: nodo {first_fail[0]} a coordinate ({first_fail[1]:.6f}, {first_fail[2]:.6f}), "
            f"violazione baricentrica minima = {first_fail[3]:.3e}. Tolleranza = {tol_barycentric:.1e}."
        )

    diagnostics = {
        "num_nodes_remapped": num_new,
        "max_boundary_violation": max_boundary_violation,
        "all_localized": True
    }

    if return_diagnostics:
        return u_new, diagnostics
    return u_new


class MeshStateRemapper:
    """
    Wrapper ad alto livello orientato a oggetti per il remapping di mesh conformi.
    Garantisce piena compatibilità con l'API unificata `remap_p1_solution`.
    """

    def __init__(self, tol_bary: float = 1e-10):
        self.tol_bary = tol_bary

    def remap(
        self,
        mesh_old: TriangularMesh2D,
        mesh_new: TriangularMesh2D,
        state_old: np.ndarray,
        mode: str = "standard"
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Esegue il remapping. 
        Modalità 'standard': pura interpolazione P1 pointwise.
        Modalità 'conservative': opzionale per studio ablativo (scalatura integrale di massa).
        """
        assert mode in ["standard", "conservative"], f"Modalità non supportata: {mode}"

        mass_old = compute_mesh_mass(mesh_old, state_old)

        # Chiamata alla funzione di riferimento senza fallback
        state_interp = remap_p1_solution(
            old_nodes=mesh_old.nodes,
            old_cells=mesh_old.cells,
            old_u=state_old,
            new_nodes=mesh_new.nodes,
            tol_barycentric=self.tol_bary
        )

        mass_interp = compute_mesh_mass(mesh_new, state_interp)

        if mode == "standard":
            state_new = state_interp
            mass_new = mass_interp
            scale_factor = 1.0
        else:
            # Scalatura globale opzionale per preservare la massa
            scale_factor = mass_old / mass_interp if abs(mass_interp) > 1e-14 else 1.0
            state_new = state_interp * scale_factor
            mass_new = compute_mesh_mass(mesh_new, state_new)

        cons_err = float(abs(mass_new - mass_old))
        diag = {
            "mass_old": mass_old,
            "mass_new": mass_new,
            "scale_factor": float(scale_factor),
            "mode": mode
        }

        return state_new, cons_err, diag