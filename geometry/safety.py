"""
geometry/safety.py
Filtro deterministico globale O(M) contro inversioni e collassi geometrici della mesh.

Risolve in forma chiusa per ciascun elemento triangolare lineare il vincolo continuo:
    J_e(alpha) >= epsilon  per alpha in [0, alpha_e^*]
lungo la traiettoria:
    X(alpha) = X_t + alpha * Delta X

Determina:
    alpha_global = min_e alpha_e^*
garantendo in modo esatto e verificabile:
    min_e J_e(X_safe) >= epsilon
senza ottimizzazioni iterative a runtime.
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
    verbose: bool = False
) -> Tuple[float, np.ndarray, bool, float]:
    """
    Applica il filtro geometrico proiettivo deterministico O(M).

    Input:
        nodes        : coordinate nodali al passo t, shape [N, 2], dtype float64
        displacement : campo di spostamento predetto Delta X, shape [N, 2], dtype float64
        cells        : connettività triangolare, shape [M, 3], dtype int64
        epsilon      : soglia minima ammissibile per il determinante Jacobiano (J_e >= epsilon)
        tol_deg      : tolleranza numerica per rilevare regimi lineari (|A_e| < tol_deg)
        safety_margin: margine conservativo di arretramento delta_margin (default 1e-6, Eq. 60)
        verbose      : se True, stampa il log diagnostico del filtro

    Output:
        alpha_global : scalare in [0.0, 1.0] applicabile a tutta la mesh
        nodes_safe   : coordinate nodali aggiornate X_safe = X_t + alpha_global * Delta X, [N, 2]
        is_safe      : True se alpha_global == 1.0 (spostamento pienamente ammissibile)
        min_j_after  : minimo determinante Jacobiano effettivo a posteriori
    """
    assert nodes.ndim == 2 and nodes.shape[1] == 2, f"nodes deve avere forma [N, 2], trovata {nodes.shape}"
    assert displacement.ndim == 2 and displacement.shape[1] == 2, (
        f"displacement deve avere forma [N, 2], trovata {displacement.shape}"
    )
    assert cells.ndim == 2 and cells.shape[1] == 3, f"cells deve avere forma [M, 3], trovata {cells.shape}"
    assert nodes.shape == displacement.shape, "nodes e displacement devono avere le stesse dimensioni"
    assert epsilon > 0.0, f"La soglia geometrica epsilon deve essere strettamente positiva, ricevuta: {epsilon}"

    num_cells = cells.shape[0]
    eps = float(epsilon)

    # 1. Estrazione vettorizzata dei vertici iniziali e verifica di ammissibilità a t = 0
    x1, y1 = nodes[cells[:, 0], 0], nodes[cells[:, 0], 1]
    x2, y2 = nodes[cells[:, 1], 0], nodes[cells[:, 1], 1]
    x3, y3 = nodes[cells[:, 2], 0], nodes[cells[:, 2], 1]

    u1, u2 = x2 - x1, y2 - y1
    v1, v2 = x3 - x1, y3 - y1

    # C_e = J_e(0): determinante Jacobiano nella configurazione iniziale
    C = u1 * v2 - u2 * v1
    min_c = float(np.min(C))

    # Tolleranza stretta di arrotondamento float64 sulla configurazione iniziale
    roundoff_tol = max(1e-12, 1e-9 * abs(eps))
    if min_c < eps - roundoff_tol:
        raise ValueError(
            "Mesh iniziale non ammissibile per il safety filter: "
            f"min(J_0)={min_c:.16e} < epsilon={eps:.16e} "
            f"(tolleranza roundoff={roundoff_tol:.3e})."
        )

    # Controllo rapido DOPO aver certificato la mesh iniziale: se lo spostamento è nullo, uscita immediata
    if np.max(np.abs(displacement)) < tol_deg:
        return 1.0, nodes.copy(), True, min_c

    # 2. Coefficienti del polinomio quadratico J_e(alpha) = A_e * alpha^2 + B_e * alpha + C_e
    dx1, dy1 = displacement[cells[:, 0], 0], displacement[cells[:, 0], 1]
    dx2, dy2 = displacement[cells[:, 1], 0], displacement[cells[:, 1], 1]
    dx3, dy3 = displacement[cells[:, 2], 0], displacement[cells[:, 2], 1]

    du1, du2 = dx2 - dx1, dy2 - dy1
    dv1, dv2 = dx3 - dx1, dy3 - dy1

    A = du1 * dv2 - du2 * dv1
    B = (u1 * dv2 + du1 * v2) - (u2 * dv1 + du2 * v1)

    # Termine noto traslato rispetto alla soglia nominale esatta epsilon:
    # q_e(alpha) = A_e * alpha^2 + B_e * alpha + C_tilde >= 0
    C_tilde = np.maximum(0.0, C - eps)
    alpha_e = np.ones(num_cells, dtype=np.float64)

    # -------------------------------------------------------------------------
    # 3. Elementi già sulla soglia iniziale (C_tilde == 0)
    # -------------------------------------------------------------------------
    on_boundary = C_tilde <= 0.0
    # Se la derivata prima B < 0 (o B == 0 e A < 0), q_e(alpha) scende subito sotto zero per alpha > 0
    immediate_violation = on_boundary & ((B < -tol_deg) | ((np.abs(B) <= tol_deg) & (A < -tol_deg)))
    alpha_e[immediate_violation] = 0.0

    # Se C_tilde == 0, B > 0 ma A < 0, la parabola sale e poi riattraversa zero in alpha = -B / A
    boundary_arch = on_boundary & (B > tol_deg) & (A < -tol_deg)
    if np.any(boundary_arch):
        r_arch = -B[boundary_arch] / A[boundary_arch]
        alpha_e[boundary_arch] = np.where((r_arch > 0.0) & (r_arch <= 1.0), r_arch, 1.0)

    # -------------------------------------------------------------------------
    # 4. Elementi strettamente interni (C_tilde > 0)
    # -------------------------------------------------------------------------
    strictly_inside = ~on_boundary

    # CASO 4.1: Regime lineare (|A| < tol_deg)
    is_linear = strictly_inside & (np.abs(A) < tol_deg)
    mask_lin_down = is_linear & (B < -tol_deg)
    if np.any(mask_lin_down):
        r_lin = -C_tilde[mask_lin_down] / B[mask_lin_down]
        # Poiché C_tilde > 0 e B < 0, r_lin è strettamente positiva (anche se molto piccola)
        alpha_e[mask_lin_down] = np.where(r_lin <= 1.0, r_lin, 1.0)

    # CASO 4.2: Regime quadratico genuino (|A| >= tol_deg)
    is_quad = strictly_inside & (np.abs(A) >= tol_deg)
    Delta = B ** 2 - 4.0 * A * C_tilde

    # Scala di grandezza per rilevare tangenze quasi-degeneri (Delta ~ 0 negativo per roundoff)
    scale_delta = (B ** 2 + np.abs(4.0 * A * C_tilde) + 1.0) * tol_deg

    # Sotto-caso 4.2a: Quasi-tangenza con A > 0 e B < 0 (vertice della parabola in -B / (2A))
    mask_near_tangent = is_quad & (Delta < 0.0) & (Delta >= -scale_delta) & (A > 0.0) & (B < 0.0)
    if np.any(mask_near_tangent):
        r_vert = -B[mask_near_tangent] / (2.0 * A[mask_near_tangent])
        alpha_e[mask_near_tangent] = np.where((r_vert > 0.0) & (r_vert <= 1.0), r_vert, 1.0)

    # Sotto-caso 4.2b: Radici reali (Delta >= 0)
    mask_real = is_quad & (Delta >= 0.0)
    if np.any(mask_real):
        A_q = A[mask_real]
        B_q = B[mask_real]
        C_q = C_tilde[mask_real]
        sqrt_Delta = np.sqrt(np.maximum(0.0, Delta[mask_real]))

        # Formula stabile di Citardauq (Eq. 58 del paper)
        sgn = np.where(B_q >= 0.0, 1.0, -1.0)
        q = -0.5 * (B_q + sgn * sqrt_Delta)
        q = np.where(np.abs(B_q) < tol_deg, -0.5 * sqrt_Delta, q)

        safe_q = np.where(np.abs(q) > 0.0, q, 1.0)
        r1 = np.where(np.abs(q) > 0.0, q / A_q, 2.0)
        r2 = np.where(np.abs(q) > 0.0, C_q / safe_q, 2.0)

        # Poiché C_q > 0, qualsiasi radice reale positiva r > 0 (anche < 1e-15) è un'intersezione genuina
        r1_valid = np.where((r1 > 0.0) & (r1 <= 1.0), r1, 2.0)
        r2_valid = np.where((r2 > 0.0) & (r2 <= 1.0), r2, 2.0)
        r_first = np.minimum(r1_valid, r2_valid)

        alpha_e[mask_real] = np.where(r_first <= 1.0, r_first, 1.0)

    # -------------------------------------------------------------------------
    # 5. Determinazione del fattore globale (Eq. 60 del paper) e verifica a posteriori
    # -------------------------------------------------------------------------
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

    # Salvaguardia esatta a posteriori contro roundoff float64 in casi estremi
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