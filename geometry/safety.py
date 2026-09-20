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
    safety_margin: float = 1e-6
) -> Tuple[float, np.ndarray, bool, float]:
    """
    Applica il filtro geometrico proiettivo deterministico.

    Input:
        nodes        : coordinate nodali al passo t, shape [N, 2], dtype float64
        displacement : campo di spostamento predetto Delta X, shape [N, 2], dtype float64
        cells        : connettività triangolare, shape [M, 3], dtype int64
        epsilon      : soglia minima ammissibile per il determinante Jacobiano (J_e >= epsilon)
        tol_deg      : tolleranza numerica per rilevare regimi lineari (|A_e| < tol_deg)
        safety_margin: margine conservativo di arretramento per approssimazione float64

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

    num_cells = cells.shape[0]

    # Controllo rapido: se lo spostamento è numericamente nullo (||Delta X||_max ~ 0), nessuna correzione necessaria
    if np.max(np.abs(displacement)) < tol_deg:
        x1 = nodes[cells[:, 0], 0]
        y1 = nodes[cells[:, 0], 1]
        x2 = nodes[cells[:, 1], 0]
        y2 = nodes[cells[:, 1], 1]
        x3 = nodes[cells[:, 2], 0]
        y3 = nodes[cells[:, 2], 1]
        j_init = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        min_j = float(np.min(j_init))
        return 1.0, nodes.copy(), True, min_j

    # 1. Estrazione vettorizzata dei vertici iniziali e dei vettori spostamento per ogni cella
    x1, y1 = nodes[cells[:, 0], 0], nodes[cells[:, 0], 1]
    x2, y2 = nodes[cells[:, 1], 0], nodes[cells[:, 1], 1]
    x3, y3 = nodes[cells[:, 2], 0], nodes[cells[:, 2], 1]

    dx1, dy1 = displacement[cells[:, 0], 0], displacement[cells[:, 0], 1]
    dx2, dy2 = displacement[cells[:, 1], 0], displacement[cells[:, 1], 1]
    dx3, dy3 = displacement[cells[:, 2], 0], displacement[cells[:, 2], 1]

    # Vettori lato configurazione t: u = P2 - P1, v = P3 - P1
    u1, u2 = x2 - x1, y2 - y1
    v1, v2 = x3 - x1, y3 - y1

    # Vettori lato differenziali: du = dP2 - dP1, dv = dP3 - dP1
    du1, du2 = dx2 - dx1, dy2 - dy1
    dv1, dv2 = dx3 - dx1, dy3 - dy1

    # 2. Coefficienti del polinomio quadratico J_e(alpha) = A_e * alpha^2 + B_e * alpha + C_e
    C = u1 * v2 - u2 * v1
    A = du1 * dv2 - du2 * dv1
    B = (u1 * dv2 + du1 * v2) - (u2 * dv1 + du2 * v1)

    min_c = np.min(C)
    assert min_c >= epsilon, (
        f"Mesh iniziale non valida per il filtro: min(J) = {min_c:.4e} < epsilon = {epsilon:.4e}"
    )

    # Traslazione per trovare le intersezioni con la soglia critica:
    # P_e(alpha) = A_e * alpha^2 + B_e * alpha + C_tilde_e = 0, con C_tilde_e = C_e - epsilon > 0
    C_tilde = C - epsilon
    alpha_e = np.ones(num_cells, dtype=np.float64)

    # -------------------------------------------------------------------------
    # 3. Risoluzione robusta caso per caso
    # -------------------------------------------------------------------------

    # CASO 1: Comportamento degenere / lineare (|A| < tol_deg)
    # L'equazione si riduce a: B * alpha + C_tilde = 0
    is_linear = np.abs(A) < tol_deg

    # Se B >= 0, la funzione è non-decrescente partendo da C_tilde > 0, quindi J_e(alpha) >= epsilon per ogni alpha >= 0.
    # Se B < -tol_deg, la retta decresce e interseca la soglia a r = -C_tilde / B > 0.
    mask_lin_down = is_linear & (B < -tol_deg)
    r_lin = -C_tilde[mask_lin_down] / B[mask_lin_down]
    valid_lin = (r_lin > 0.0) & (r_lin < 1.0)
    alpha_e[mask_lin_down] = np.where(valid_lin, r_lin, 1.0)

    # CASO 2: Comportamento quadratico genuino (|A| >= tol_deg)
    is_quad = ~is_linear
    Delta = B**2 - 4.0 * A * C_tilde

    # Sotto-caso 2.1: Radici reali (Delta >= 0).
    # Se Delta < 0, il polinomio non interseca mai la soglia. Essendo C_tilde > 0,
    # significa che J_e(alpha) > epsilon per ogni alpha reale (alpha_e = 1.0).
    mask_real = is_quad & (Delta >= 0.0)

    if np.any(mask_real):
        A_q = A[mask_real]
        B_q = B[mask_real]
        C_q = C_tilde[mask_real]
        sqrt_Delta = np.sqrt(np.maximum(0.0, Delta[mask_real]))

        # Formulazione di Citardauq / formula stabile per evitare cancellazione numerica se B_q ha modulo grande
        sgn = np.where(B_q >= 0.0, 1.0, -1.0)
        q = -0.5 * (B_q + sgn * sqrt_Delta)

        # Se B_q == 0, q = -0.5 * sqrt_Delta
        q = np.where(np.abs(B_q) < tol_deg, -0.5 * sqrt_Delta, q)

        # Calcolo radici r1, r2
        safe_A = np.where(np.abs(A_q) > tol_deg, A_q, 1.0)
        safe_q = np.where(np.abs(q) > tol_deg, q, 1.0)

        r1 = np.where(np.abs(A_q) > tol_deg, q / safe_A, 2.0)
        r2 = np.where(np.abs(q) > tol_deg, C_q / safe_q, 2.0)

        # Cerchiamo la prima intersezione positiva nell'intervallo ammissibile (0, 1]
        # Le radici <= 0 o > 1.0 non vincolano l'intervallo [0, 1] partendo da C_tilde > 0
        r1_valid = np.where((r1 > 1e-15) & (r1 <= 1.0), r1, 2.0)
        r2_valid = np.where((r2 > 1e-15) & (r2 <= 1.0), r2, 2.0)
        r_first = np.minimum(r1_valid, r2_valid)

        # Se esiste una radice valida in (0, 1], essa rappresenta il limite superiore per quell'elemento
        has_crossing = r_first <= 1.0
        alpha_quad_res = np.ones_like(A_q)
        alpha_quad_res[has_crossing] = r_first[has_crossing]

        alpha_e[mask_real] = alpha_quad_res

    # -------------------------------------------------------------------------
    # 4. Determinazione del minimo globale con arretramento precauzionale
    # -------------------------------------------------------------------------
    alpha_raw = float(np.min(alpha_e))

    if alpha_raw >= 1.0:
        alpha_global = 1.0
        is_safe = True
    else:
        # Applica il margine di sicurezza numerico per compensare approssimazioni di arrotondamento float64
        alpha_global = max(0.0, alpha_raw * (1.0 - safety_margin))
        is_safe = False

    # 5. Generazione coordinate corrette
    nodes_safe = nodes + alpha_global * displacement

    # 6. Verifica a posteriori del minimo determinante Jacobiano
    x1_s = nodes_safe[cells[:, 0], 0]
    y1_s = nodes_safe[cells[:, 0], 1]
    x2_s = nodes_safe[cells[:, 1], 0]
    y2_s = nodes_safe[cells[:, 1], 1]
    x3_s = nodes_safe[cells[:, 2], 0]
    y3_s = nodes_safe[cells[:, 2], 1]

    j_after = (x2_s - x1_s) * (y3_s - y1_s) - (x3_s - x1_s) * (y2_s - y1_s)
    min_j_after = float(np.min(j_after))

    return alpha_global, nodes_safe, is_safe, min_j_after