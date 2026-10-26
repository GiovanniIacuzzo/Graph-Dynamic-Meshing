"""
numerics/remap.py
Operatore di remapping pointwise P1 mediante localizzazione baricentrica sulla mesh sorgente.
"""

import numpy as np


def remap_p1_solution(
    old_nodes: np.ndarray,
    old_cells: np.ndarray,
    old_u: np.ndarray,
    new_nodes: np.ndarray,
    tol_bary: float = -1e-11,
    return_diagnostics: bool = False,
):
    """Evaluate a source P1 field at new nodes by barycentric localization.

    Points not located in any source element are rejected. The previous silent
    ``argmax(min_barycentric)`` extrapolation could conceal a domain mismatch.
    With ``return_diagnostics=True``, return ``(values, diagnostics)``.
    """
    old_nodes = np.ascontiguousarray(old_nodes, dtype=np.float64)
    old_cells = np.ascontiguousarray(old_cells, dtype=np.int64)
    old_u = np.ascontiguousarray(old_u, dtype=np.float64)
    new_nodes = np.ascontiguousarray(new_nodes, dtype=np.float64)
    if old_nodes.ndim != 2 or old_nodes.shape[1] != 2:
        raise ValueError("old_nodes deve avere shape (N, 2).")
    if old_cells.ndim != 2 or old_cells.shape[1] != 3 or old_cells.shape[0] == 0:
        raise ValueError("old_cells deve avere shape (M, 3), con M > 0.")
    if new_nodes.ndim != 2 or new_nodes.shape[1] != 2:
        raise ValueError("new_nodes deve avere shape (N_new, 2).")
    if old_u.shape != (old_nodes.shape[0],):
        raise ValueError(f"old_u deve avere shape ({old_nodes.shape[0]},).")
    if not np.all(np.isfinite(old_nodes)) or not np.all(np.isfinite(new_nodes)) or not np.all(np.isfinite(old_u)):
        raise ValueError("Coordinate e valori sorgente devono essere finiti.")
    if np.any(old_cells < 0) or np.any(old_cells >= old_nodes.shape[0]):
        raise ValueError("old_cells contiene indici fuori dai limiti di old_nodes.")
    if not np.isfinite(tol_bary) or tol_bary >= 0.0:
        raise ValueError("tol_bary deve essere un valore finito e negativo.")

    tri = old_nodes[old_cells]
    x1, y1 = tri[:, 0, 0], tri[:, 0, 1]
    x2, y2 = tri[:, 1, 0], tri[:, 1, 1]
    x3, y3 = tri[:, 2, 0], tri[:, 2, 1]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    scale = np.maximum(1.0, np.max(np.abs(tri), axis=(1, 2)))
    if np.any(det <= np.finfo(np.float64).eps * scale**2):
        raise ValueError("La mesh sorgente contiene triangoli degeneri o orientati non positivamente.")
    inv_det = 1.0 / det
    u1, u2, u3 = old_u[old_cells[:, 0]], old_u[old_cells[:, 1]], old_u[old_cells[:, 2]]

    remapped = np.empty(new_nodes.shape[0], dtype=np.float64)
    max_boundary_violation = 0.0
    for i, (px, py) in enumerate(new_nodes):
        l2 = ((px - x1) * (y3 - y1) - (x3 - x1) * (py - y1)) * inv_det
        l3 = ((x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)) * inv_det
        l1 = 1.0 - l2 - l3
        min_l = np.minimum(np.minimum(l1, l2), l3)
        inside = np.flatnonzero(min_l >= tol_bary)
        if inside.size == 0:
            best = int(np.argmax(min_l))
            violation = float(max(0.0, -min_l[best] + tol_bary))
            raise ValueError(
                f"Il nuovo nodo {i} non è localizzato nella mesh sorgente "
                f"(violazione baricentrica oltre tolleranza: {violation:.3e}). "
                "Il remapping P1 non esegue extrapolazione silenziosa."
            )
        k = int(inside[0])
        l_min_selected = float(min_l[k])
        if l_min_selected < 0.0 and l_min_selected < tol_bary:
            max_boundary_violation = max(max_boundary_violation, -l_min_selected + tol_bary)
        remapped[i] = l1[k] * u1[k] + l2[k] * u2[k] + l3[k] * u3[k]

    if not return_diagnostics:
        return remapped
    diagnostics = {
        "all_localized": True,
        "max_boundary_violation": float(max_boundary_violation),
        "num_points": int(new_nodes.shape[0]),
        "num_unlocalized": 0,
    }
    return remapped, diagnostics


def compute_mesh_mass(mesh_or_nodes, u: np.ndarray, cells: np.ndarray | None = None) -> float:
    """Integrate a continuous P1 nodal field over a positively oriented mesh.

    Accepts either ``compute_mesh_mass(mesh, u)`` or
    ``compute_mesh_mass(nodes, u, cells)``.
    """
    if cells is None:
        if not hasattr(mesh_or_nodes, "nodes") or not hasattr(mesh_or_nodes, "cells"):
            raise TypeError("Passare un oggetto mesh oppure nodes, u, cells.")
        nodes = np.asarray(mesh_or_nodes.nodes, dtype=np.float64)
        cell_array = np.asarray(mesh_or_nodes.cells, dtype=np.int64)
    else:
        nodes = np.asarray(mesh_or_nodes, dtype=np.float64)
        cell_array = np.asarray(cells, dtype=np.int64)
    values = np.asarray(u, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes deve avere shape (N, 2).")
    if cell_array.ndim != 2 or cell_array.shape[1] != 3 or cell_array.size == 0:
        raise ValueError("cells deve avere shape (M, 3), con M > 0.")
    if values.shape != (nodes.shape[0],):
        raise ValueError(f"u deve avere shape ({nodes.shape[0]},).")
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(values)):
        raise ValueError("nodes e u devono contenere valori finiti.")
    tri = nodes[cell_array]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    det = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
    if np.any(det <= 0.0):
        raise ValueError("compute_mesh_mass richiede triangoli non degeneri e orientati positivamente.")
    return float(np.sum(0.5 * det * np.mean(values[cell_array], axis=1)))

class MeshStateRemapper:
    """Small compatibility facade for mesh-integrated quantities used by metrics.

    Remapping operations remain explicit in ``remap_p1_solution`` and the L2
    projection module; this class currently provides the P1 total-mass integral.
    """

    @staticmethod
    def compute_total_mass(mesh, u: np.ndarray) -> float:
        nodes = np.asarray(mesh.nodes, dtype=np.float64)
        cells = np.asarray(mesh.cells, dtype=np.int64)
        values = np.asarray(u, dtype=np.float64)
        if values.shape != (nodes.shape[0],):
            raise ValueError(f"u deve avere shape ({nodes.shape[0]},), ricevuta {values.shape}.")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(nodes)):
            raise ValueError("Mesh e soluzione devono contenere soltanto valori finiti.")
        tri = nodes[cells]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        det = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
        if np.any(det <= 0.0):
            raise ValueError("La massa P1 richiede una mesh con tutti i triangoli non degeneri e orientati positivamente.")
        elem_mass = 0.5 * det * np.mean(values[cells], axis=1)
        return float(np.sum(elem_mass))
