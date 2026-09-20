"""
evaluation/metrics.py
Calcolo e aggregazione delle metriche per la validazione scientifica di Graph-Dynamic Meshing.

Fornisce la quantificazione disaccoppiata e formale di:
    - Metriche di Apprendimento (GNN vs Oracle target)
    - Metriche Geometriche (qualita' ed integrita' topologica della mesh)
    - Metriche Fisiche (errori funzionali L2/H1, conservazione integrale, GCL)
    - Metriche Computazionali (profiling disaccoppiato dei tempi di esecuzione)
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from geometry.mesh import TriangularMesh2D
from geometry.geometry import compute_mesh_metrics
from numerics.remap import MeshStateRemapper


# =============================================================================
# 1. METRICHE DI APPRENDIMENTO (Learning Metrics)
# =============================================================================

def compute_displacement_metrics(
    pred_disp: np.ndarray,
    target_disp: np.ndarray,
    eps: float = 1e-12
) -> Dict[str, float]:
    """
    Misura la fedeltà con cui la GNN approssima il target variazionale dell'oracle.

    Input:
        pred_disp  : spostamento nodale predetto dalla rete [N, 2], float64
        target_disp: spostamento nodale calcolato dall'oracle [N, 2], float64

    Output:
        displacement_l2_error       : ||Delta X_pred - Delta X*||_F
        displacement_relative_error: ||Delta X_pred - Delta X*||_F / max(||Delta X*||_F, eps)
        displacement_max_error      : max_i ||Delta X_pred,i - Delta X*,i||_2
        displacement_cosine_sim     : similarità cosinusoidale media sui vettori nodali non nulli
    """
    assert pred_disp.shape == target_disp.shape, (
        f"Shape mismatch: {pred_disp.shape} vs {target_disp.shape}"
    )

    diff = pred_disp - target_disp
    diff_norm_f = float(np.linalg.norm(diff))
    target_norm_f = float(np.linalg.norm(target_disp))
    rel_error = diff_norm_f / max(target_norm_f, eps)

    nodal_err = np.linalg.norm(diff, axis=1)
    max_err = float(np.max(nodal_err))

    # Calcolo della similarità cosinusoidale vettoriale nodo per nodo
    p_norm = np.linalg.norm(pred_disp, axis=1)
    t_norm = np.linalg.norm(target_disp, axis=1)
    valid = (p_norm > eps) & (t_norm > eps)
    if np.any(valid):
        cos_sim = np.sum(pred_disp[valid] * target_disp[valid], axis=1) / (p_norm[valid] * t_norm[valid])
        mean_cos = float(np.mean(cos_sim))
    else:
        mean_cos = 1.0

    return {
        "displacement_l2_error": diff_norm_f,
        "displacement_relative_error": rel_error,
        "displacement_max_error": max_err,
        "displacement_cosine_sim": mean_cos
    }


# =============================================================================
# 2. METRICHE GEOMETRICHE (Geometric Metrics)
# =============================================================================

def compute_mesh_geometric_metrics(mesh: TriangularMesh2D) -> Dict[str, float]:
    """
    Estrae le statistiche essenziali di qualità e validità topologica della griglia.

    Input:
        mesh: istanza di TriangularMesh2D

    Output:
        min_jacobian       : minimo determinante Jacobiano min_e J_e
        mean_jacobian      : valore medio dei determinanti Jacobiani
        inverted_cells     : numero di elementi con J_e <= 0 (deve essere 0 per stabilita')
        min_escobar_quality: peggior valore di qualita' adimensionale di Escobar-Knupp
        mean_mesh_quality  : qualita' media di Escobar-Knupp in (0, 1]
        max_aspect_ratio   : rapporto d'aspetto massimo tra gli elementi
        mean_aspect_ratio  : aspect ratio medio
    """
    raw_metrics = compute_mesh_metrics(mesh.nodes, mesh.cells)
    jacobians = raw_metrics["jacobians"]
    quality = raw_metrics["escobar_quality"]
    ar = raw_metrics["aspect_ratio"]
    inverted = raw_metrics["inverted"]

    return {
        "min_jacobian": float(np.min(jacobians)),
        "mean_jacobian": float(np.mean(jacobians)),
        "inverted_cells": int(np.sum(inverted)),
        "min_escobar_quality": float(np.min(quality)),
        "mean_mesh_quality": float(np.mean(quality)),
        "max_aspect_ratio": float(np.max(ar)),
        "mean_aspect_ratio": float(np.mean(ar))
    }


# =============================================================================
# 3. METRICHE FISICHE (Physical Metrics)
# =============================================================================

def compute_functional_pde_errors(
    mesh: TriangularMesh2D,
    u_approx: np.ndarray,
    exact_evaluator: Any,
    t: float
) -> Dict[str, float]:
    """
    Calcola gli errori di campo L2 e semi-norma H1 integrati nel dominio tramite
    regola di quadratura di Gauss a 3 punti per elementi triangolari P1.

    Input:
        mesh           : mesh corrente con coordinate deformate
        u_approx       : soluzione numerica nodale calcolata dal solutore [N], float64
        exact_evaluator: callable o oggetto PDE con metodo evaluate_exact(x, y, t)
        t              : istante temporale in cui calcolare l'errore

    Output:
        solution_l2_error         : ||u_exact - u_approx||_{L^2(Omega)}
        solution_l2_relative_error: ||u_exact - u_approx||_L2 / ||u_exact||_L2
        solution_h1_error         : |u_exact - u_approx|_{H^1(Omega)} (semi-norma sui gradienti)
    """
    nodes = mesh.nodes
    cells = mesh.cells
    num_cells = mesh.num_cells

    # Calcolo coordinate dei vertici: [M, 3, 2]
    tri_pts = nodes[cells]
    x1, y1 = tri_pts[:, 0, 0], tri_pts[:, 0, 1]
    x2, y2 = tri_pts[:, 1, 0], tri_pts[:, 1, 1]
    x3, y3 = tri_pts[:, 2, 0], tri_pts[:, 2, 1]

    det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    inv_det = 1.0 / np.maximum(det_J, 1e-15)

    # Gradienti costanti per elemento delle funzioni di forma P1: [M, 3, 2]
    grad_phi = np.empty((num_cells, 3, 2), dtype=np.float64)
    grad_phi[:, 0, 0] = (y2 - y3) * inv_det
    grad_phi[:, 0, 1] = (x3 - x2) * inv_det
    grad_phi[:, 1, 0] = (y3 - y1) * inv_det
    grad_phi[:, 1, 1] = (x1 - x3) * inv_det
    grad_phi[:, 2, 0] = (y1 - y2) * inv_det
    grad_phi[:, 2, 1] = (x2 - x1) * inv_det

    # Gradiente della soluzione numerica approssimata per ciascun triangolo: [M, 2]
    u_elem_nodes = u_approx[cells]  # [M, 3]
    grad_u_approx = np.einsum("mi,mih->mh", u_elem_nodes, grad_phi)

    # Punti e pesi della regola di Gauss a 3 punti
    quad_bary = np.array([
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]
    ], dtype=np.float64)
    quad_weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float64)

    l2_err_sq = 0.0
    l2_ref_sq = 0.0
    h1_err_sq = 0.0

    for q in range(3):
        bary = quad_bary[q]
        w = quad_weights[q]

        xq = bary[0] * x1 + bary[1] * x2 + bary[2] * x3
        yq = bary[0] * y1 + bary[1] * y2 + bary[2] * y3

        # Soluzione esatta analitica al punto di quadratura
        u_exact, grad_exact, _ = exact_evaluator.evaluate_exact(xq, yq, t)

        # Soluzione approssimata interpolata linearmente
        u_h_q = bary[0] * u_elem_nodes[:, 0] + bary[1] * u_elem_nodes[:, 1] + bary[2] * u_elem_nodes[:, 2]

        # Accumulo quadratura L2
        diff_u = u_exact - u_h_q
        l2_err_sq += np.sum(diff_u**2 * det_J * w)
        l2_ref_sq += np.sum(u_exact**2 * det_J * w)

        # Accumulo quadratura semi-norma H1 (differenza tra vettori gradiente)
        diff_grad = grad_exact - grad_u_approx
        diff_grad_norm_sq = diff_grad[:, 0]**2 + diff_grad[:, 1]**2
        h1_err_sq += np.sum(diff_grad_norm_sq * det_J * w)

    l2_err = float(np.sqrt(max(0.0, l2_err_sq)))
    l2_ref = float(np.sqrt(max(0.0, l2_ref_sq)))
    rel_l2 = l2_err / max(l2_ref, 1e-12)
    h1_err = float(np.sqrt(max(0.0, h1_err_sq)))

    return {
        "solution_l2_error": l2_err,
        "solution_l2_relative_error": rel_l2,
        "solution_h1_error": h1_err
    }


def compute_physical_conservation_metrics(
    mesh_old: TriangularMesh2D,
    mesh_new: TriangularMesh2D,
    state_old: np.ndarray,
    state_new: np.ndarray,
    gcl_residual: float = 0.0
) -> Dict[str, float]:
    """
    Verifica i bilanci integrali globali sul dominio e l'integrità geometrica ALE.

    Input:
        mesh_old    : griglia al passo precedente
        mesh_new    : griglia adattata al passo corrente
        state_old   : vettore di campo nodale u_t
        state_new   : vettore di campo nodale u_{t+1}
        gcl_residual: residuo discreto GCL fornito dal modulo ALE

    Output:
        mass_old          : massa integrata int_Omega u_t dOmega
        mass_new          : massa integrata int_Omega u_{t+1} dOmega
        conservation_error: differenza assoluta |mass_new - mass_old|
        gcl_error         : residuo normalizzato della Geometric Conservation Law
    """
    m_old = MeshStateRemapper.compute_total_mass(mesh_old, state_old)
    m_new = MeshStateRemapper.compute_total_mass(mesh_new, state_new)
    cons_err = float(np.abs(m_new - m_old))

    return {
        "mass_old": m_old,
        "mass_new": m_new,
        "conservation_error": cons_err,
        "gcl_error": float(gcl_residual)
    }


# =============================================================================
# 4. METRICHE COMPUTAZIONALI E AGGREGAZIONE GLOBALE
# =============================================================================

def format_computational_timings(
    gnn_time: float = 0.0,
    safety_time: float = 0.0,
    remap_time: float = 0.0,
    fem_time: float = 0.0
) -> Dict[str, float]:
    """
    Formatta il bilancio cronometrico disaccoppiato in millisecondi e secondi totali.
    """
    tot = gnn_time + safety_time + remap_time + fem_time
    return {
        "gnn_time": float(gnn_time),
        "safety_time": float(safety_time),
        "remap_time": float(remap_time),
        "fem_time": float(fem_time),
        "total_time": float(tot)
    }


def aggregate_scientific_report(
    displacement_metrics: Optional[Dict[str, float]] = None,
    geometric_metrics: Optional[Dict[str, float]] = None,
    physical_metrics: Optional[Dict[str, float]] = None,
    computational_metrics: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Raggruppa tutte le metriche in un dizionario unificato, pronto per la
    generazione di tabelle LaTeX, serializzazione JSON e salvataggio nei manifest.
    """
    report: Dict[str, Any] = {}
    if displacement_metrics:
        report.update(displacement_metrics)
    if geometric_metrics:
        report.update(geometric_metrics)
    if physical_metrics:
        report.update(physical_metrics)
    if computational_metrics:
        report.update(computational_metrics)
    return report