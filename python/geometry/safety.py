"""
geometry/safety.py
Filtro deterministico globale O(M) contro inversioni e collassi geometrici della mesh.
"""

from typing import Tuple
import numpy as np


def apply_safety_filter(
    nodes: np.ndarray,
    displacement: np.ndarray,
    cells: np.ndarray,
    epsilon: float = 1e-4,
    tol_deg: float = 1e-12,
    safety_margin: float = 1e-6,
    verbose: bool = False,
) -> Tuple[float, np.ndarray, bool, float]:
    """Applica il filtro geometrico proiettivo deterministico O(M)."""
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    displacement = np.ascontiguousarray(displacement, dtype=np.float64)
    cells = np.ascontiguousarray(cells, dtype=np.int64)

    if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] < 3:
        raise ValueError(f"nodes deve avere forma [N, 2] con N >= 3, ricevuta {nodes.shape}.")
    if displacement.ndim != 2 or displacement.shape[1] != 2:
        raise ValueError(f"displacement deve avere forma [N, 2], ricevuta {displacement.shape}.")
    if cells.ndim != 2 or cells.shape[1] != 3 or cells.shape[0] < 1:
        raise ValueError(f"cells deve avere forma [M, 3] con M >= 1, ricevuta {cells.shape}.")
    if nodes.shape != displacement.shape:
        raise ValueError("nodes e displacement devono avere le stesse dimensioni.")
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(displacement)):
        raise ValueError("nodes e displacement devono contenere soltanto valori finiti.")
    if np.any(cells < 0) or np.any(cells >= nodes.shape[0]):
        raise ValueError("cells contiene indici nodali fuori intervallo.")
    if np.any((cells[:, 0] == cells[:, 1]) | (cells[:, 1] == cells[:, 2]) | (cells[:, 0] == cells[:, 2])):
        raise ValueError("cells contiene triangoli con indici nodali ripetuti.")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon deve essere finito e strettamente positivo, ricevuto {epsilon}.")
    if not np.isfinite(tol_deg) or tol_deg <= 0.0:
        raise ValueError(f"tol_deg deve essere finito e strettamente positivo, ricevuto {tol_deg}.")
    if not np.isfinite(safety_margin) or safety_margin < 0.0 or safety_margin >= 0.5:
        raise ValueError("safety_margin deve essere finito e appartenere a [0, 0.5).")

    num_cells = cells.shape[0]
    eps = float(epsilon)

    x1, y1 = nodes[cells[:, 0], 0], nodes[cells[:, 0], 1]
    x2, y2 = nodes[cells[:, 1], 0], nodes[cells[:, 1], 1]
    x3, y3 = nodes[cells[:, 2], 0], nodes[cells[:, 2], 1]

    u1, u2 = x2 - x1, y2 - y1
    v1, v2 = x3 - x1, y3 - y1

    C = u1 * v2 - u2 * v1
    min_c = float(np.min(C))

    roundoff_tol = max(1e-12, 1e-9 * abs(eps))
    if min_c < eps - roundoff_tol:
        raise ValueError(
            "Mesh iniziale non ammissibile per il safety filter: "
            f"min(J_0)={min_c:.16e} < epsilon={eps:.16e} "
            f"(tolleranza roundoff={roundoff_tol:.3e})."
        )

    if np.max(np.abs(displacement)) == 0.0:
        return 1.0, nodes.copy(), True, min_c

    dx1, dy1 = displacement[cells[:, 0], 0], displacement[cells[:, 0], 1]
    dx2, dy2 = displacement[cells[:, 1], 0], displacement[cells[:, 1], 1]
    dx3, dy3 = displacement[cells[:, 2], 0], displacement[cells[:, 2], 1]

    du1, du2 = dx2 - dx1, dy2 - dy1
    dv1, dv2 = dx3 - dx1, dy3 - dy1

    A = du1 * dv2 - du2 * dv1
    B = (u1 * dv2 + du1 * v2) - (u2 * dv1 + du2 * v1)

    C_tilde = np.maximum(0.0, C - eps)
    alpha_e = np.ones(num_cells, dtype=np.float64)

    on_boundary = C_tilde <= 0.0
    immediate_violation = on_boundary & (
        (B < 0.0) | ((np.abs(B) <= tol_deg) & (A < 0.0))
    )
    alpha_e[immediate_violation] = 0.0

    boundary_arch = on_boundary & (~immediate_violation) & (B > 0.0) & (A < 0.0)
    if np.any(boundary_arch):
        B_b = B[boundary_arch]
        A_b = A[boundary_arch]
        crosses_before_one = B_b < -A_b
        r_arch = np.ones_like(B_b)
        r_arch[crosses_before_one] = -B_b[crosses_before_one] / A_b[crosses_before_one]
        alpha_e[boundary_arch] = r_arch

    strictly_inside = ~on_boundary

    is_linear = strictly_inside & (np.abs(A) < tol_deg)
    mask_lin_down = is_linear & (B < 0.0) & ((C_tilde + B) <= 0.0)
    if np.any(mask_lin_down):
        r_lin = -C_tilde[mask_lin_down] / B[mask_lin_down]
        alpha_e[mask_lin_down] = np.clip(r_lin, 0.0, 1.0)

    is_quad = strictly_inside & (np.abs(A) >= tol_deg)
    Delta = B**2 - 4.0 * A * C_tilde
    scale_delta = (B**2 + np.abs(4.0 * A * C_tilde)) * tol_deg

    mask_near_tangent = is_quad & (Delta < 0.0) & (Delta >= -scale_delta) & (A > 0.0) & (B < 0.0)
    if np.any(mask_near_tangent):
        r_vert = -B[mask_near_tangent] / (2.0 * A[mask_near_tangent])
        alpha_e[mask_near_tangent] = np.where((r_vert > 0.0) & (r_vert <= 1.0), r_vert, 1.0)

    mask_real = is_quad & (Delta >= 0.0)
    if np.any(mask_real):
        A_q = A[mask_real]
        B_q = B[mask_real]
        C_q = C_tilde[mask_real]
        sqrt_Delta = np.sqrt(np.maximum(0.0, Delta[mask_real]))

        sgn = np.where(B_q >= 0.0, 1.0, -1.0)
        q = -0.5 * (B_q + sgn * sqrt_Delta)
        q = np.where(np.abs(B_q) < tol_deg, -0.5 * sqrt_Delta, q)

        safe_q = np.where(np.abs(q) > 0.0, q, 1.0)
        r1 = np.where(np.abs(q) > 0.0, q / A_q, 2.0)
        r2 = np.where(np.abs(q) > 0.0, C_q / safe_q, 2.0)

        r1_valid = np.where((r1 > 0.0) & (r1 <= 1.0), r1, 2.0)
        r2_valid = np.where((r2 > 0.0) & (r2 <= 1.0), r2, 2.0)
        r_first = np.minimum(r1_valid, r2_valid)

        alpha_e[mask_real] = np.where(r_first <= 1.0, r_first, 1.0)

    alpha_raw = float(np.min(alpha_e))

    if alpha_raw >= 1.0:
        alpha_global = 1.0
        is_safe = True
    else:
        margin = float(np.clip(safety_margin, 0.0, 0.5))
        alpha_global = max(0.0, (1.0 - margin) * alpha_raw)
        is_safe = False

    def _compute_min_j(alpha_val: float) -> Tuple[np.ndarray, float]:
        n_safe = nodes + alpha_val * displacement
        x1_s, y1_s = n_safe[cells[:, 0], 0], n_safe[cells[:, 0], 1]
        x2_s, y2_s = n_safe[cells[:, 1], 0], n_safe[cells[:, 1], 1]
        x3_s, y3_s = n_safe[cells[:, 2], 0], n_safe[cells[:, 2], 1]
        j_arr = (x2_s - x1_s) * (y3_s - y1_s) - (x3_s - x1_s) * (y2_s - y1_s)
        return n_safe, float(np.min(j_arr))

    nodes_safe, min_j_after = _compute_min_j(alpha_global)

    if min_j_after < eps - roundoff_tol:
        is_safe = False
        if alpha_global == 1.0:
            alpha_global = 1.0 - max(float(safety_margin), 1e-6)
            nodes_safe, min_j_after = _compute_min_j(alpha_global)
        while min_j_after < eps - roundoff_tol and alpha_global > 1e-16:
            alpha_global *= 0.5
            nodes_safe, min_j_after = _compute_min_j(alpha_global)
        if min_j_after < eps - roundoff_tol:
            alpha_global = 0.0
            nodes_safe = nodes.copy()
            min_j_after = min_c

    if verbose:
        print(f"[safety] alpha={alpha_global:.6e}  gap_in={min_c - eps:.3e}  min_J_out={min_j_after:.12e}")

    return alpha_global, nodes_safe, is_safe, min_j_after