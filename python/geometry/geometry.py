"""
geometry/geometry.py
Calcolo delle metriche geometriche, qualità e validità di elementi triangolari 2D.

Fornisce primitive analitiche indipendenti dalla GNN per:
    - Calcolo del determinante Jacobiano J_e di trasformazione
    - Area euclidea degli elementi
    - Jacobiano normalizzato (rapporto rispetto all'area ideale equilatera)
    - Aspect ratio esatto (rapporto tra lato massimo e inraggio)
    - Angoli interni minimi e massimi
    - Metrica algebrica adimensionale di Escobar-Knupp (eta_e in [0, 1])
    - Rilevamento deterministico di elementi invertiti (J_e <= 0) o degeneri
"""

from typing import Dict, Tuple
import numpy as np


def compute_mesh_metrics(
    nodes: np.ndarray,
    cells: np.ndarray,
    tol_degenerate: float = 1e-14
) -> Dict[str, np.ndarray]:
    """
    Calcola le metriche geometriche fondamentali per tutti gli elementi triangolari.

    Input:
        nodes: coordinate nodali della mesh, shape [N, 2], dtype float64
        cells: matrice di connettività triangolare, shape [M, 3], dtype int64
        tol_degenerate: tolleranza numerica per prevenire divisioni per zero

    Output:
        Dizionario contenente array NumPy 1D di dimensione [M]:
            - 'jacobians': determinante dello Jacobiano di ciascun triangolo J_e
            - 'areas': area geometrica dell'elemento (0.5 * J_e)
            - 'normalized_jacobians': rapporto J_e / J_ideal a parità di perimetro
            - 'aspect_ratio': rapporto di allungamento normalizzato (1.0 = equilatero)
            - 'min_angle': angolo interno minimo del triangolo in radianti
            - 'max_angle': angolo interno massimo del triangolo in radianti
            - 'escobar_quality': metrica di forma adimensionale eta_e in [0, 1]
            - 'inverted': maschera booleana (True se J_e <= 0)
            - 'degenerate': maschera booleana (True se area <= tol_degenerate o lato nullo)
    """
    assert nodes.ndim == 2 and nodes.shape[1] == 2, f"nodes deve avere forma [N, 2], trovata {nodes.shape}"
    assert cells.ndim == 2 and cells.shape[1] == 3, f"cells deve avere forma [M, 3], trovata {cells.shape}"

    # Estrazione vettorizzata delle coordinate dei tre vertici per ogni triangolo
    x1, y1 = nodes[cells[:, 0], 0], nodes[cells[:, 0], 1]
    x2, y2 = nodes[cells[:, 1], 0], nodes[cells[:, 1], 1]
    x3, y3 = nodes[cells[:, 2], 0], nodes[cells[:, 2], 1]

    # Vettori lato orientati:
    # v12 = P2 - P1, v23 = P3 - P2, v31 = P1 - P3
    v12_x, v12_y = x2 - x1, y2 - y1
    v23_x, v23_y = x3 - x2, y3 - y2
    v31_x, v31_y = x1 - x3, y1 - y3

    # Vettore opposto a v31 per calcolo vertice 1: v13 = P3 - P1
    v13_x, v13_y = x3 - x1, y3 - y1

    # 1. Determinante dello Jacobiano di trasformazione rispetto al simplesso unitario:
    # J_e = det [[x2-x1, x3-x1], [y2-y1, y3-y1]] = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    jacobians = v12_x * v13_y - v13_x * v12_y
    areas = 0.5 * jacobians
    inverted = jacobians <= 0.0

    # 2. Lunghezze dei lati al quadrato e norme euclidee
    l12_sq = v12_x**2 + v12_y**2
    l23_sq = v23_x**2 + v23_y**2
    l31_sq = v31_x**2 + v31_y**2

    l12 = np.sqrt(np.maximum(l12_sq, tol_degenerate))
    l23 = np.sqrt(np.maximum(l23_sq, tol_degenerate))
    l31 = np.sqrt(np.maximum(l31_sq, tol_degenerate))

    # 3. Metrica di qualità algebrica di Escobar/Knupp (eta_e)
    # Per triangoli 2D: eta_e = 4 * sqrt(3) * Area / (l12^2 + l23^2 + l31^2)
    # eta_e = 1.0 per triangolo equilatero perfetto; eta_e <= 0 per elementi collassati/invertiti
    sum_l_sq = l12_sq + l23_sq + l31_sq
    safe_denom = np.maximum(sum_l_sq, tol_degenerate)
    escobar_quality = (2.0 * np.sqrt(3.0) * jacobians) / safe_denom
    escobar_quality = np.clip(escobar_quality, 0.0, 1.0)

    # 4. Aspect Ratio normalizzato: AR = l_max / (2 * sqrt(3) * r_in)
    # r_in = Area / semi_perimeter
    semi_perimeter = 0.5 * (l12 + l23 + l31)
    safe_areas = np.maximum(areas, tol_degenerate)
    r_in = safe_areas / np.maximum(semi_perimeter, tol_degenerate)
    l_max = np.maximum(np.maximum(l12, l23), l31)
    aspect_ratio = l_max / (2.0 * np.sqrt(3.0) * np.maximum(r_in, tol_degenerate))

    # 5. Jacobiano Normalizzato
    # Rapporto rispetto all'area del triangolo equilatero con perimetro equivalente:
    # Area_equi = (Perimetro^2) / (12 * sqrt(3)) -> J_ideal = (2 * s)^2 / (6 * sqrt(3))
    j_ideal = (4.0 * semi_perimeter**2) / (6.0 * np.sqrt(3.0))
    normalized_jacobians = jacobians / np.maximum(j_ideal, tol_degenerate)

    # 6. Angoli interni (in radianti) tramite teorema del coseno vettorializzato
    # Vertice 1: lati v12 e v13
    cos_theta1 = (v12_x * v13_x + v12_y * v13_y) / np.maximum(l12 * l31, tol_degenerate)
    # Vertice 2: lati -v12 e v23
    cos_theta2 = (-v12_x * v23_x - v12_y * v23_y) / np.maximum(l12 * l23, tol_degenerate)
    # Vertice 3: lati -v23 e -v13 (ovvero v31 e v32)
    cos_theta3 = (-v13_x * (-v23_x) - v13_y * (-v23_y)) / np.maximum(l31 * l23, tol_degenerate)

    cos_theta1 = np.clip(cos_theta1, -1.0, 1.0)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)

    angles = np.stack([
        np.arccos(cos_theta1),
        np.arccos(cos_theta2),
        np.arccos(cos_theta3)
    ], axis=1)

    min_angle = np.min(angles, axis=1)
    max_angle = np.max(angles, axis=1)

    # 7. Rilevamento celle degeneri
    degenerate = (areas <= tol_degenerate) | (l12 <= tol_degenerate) | (l23 <= tol_degenerate) | (l31 <= tol_degenerate)

    return {
        "jacobians": jacobians,
        "areas": areas,
        "normalized_jacobians": normalized_jacobians,
        "aspect_ratio": aspect_ratio,
        "min_angle": min_angle,
        "max_angle": max_angle,
        "escobar_quality": escobar_quality,
        "inverted": inverted,
        "degenerate": degenerate,
    }


def compute_mesh_quality_summary(metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Estrae una sintesi scalare aggregata delle metriche per logging e checkpoint.

    Input:
        metrics: output di compute_mesh_metrics

    Output:
        Dizionario con minimi, medie, massimi e conteggi critici di distorsione.
    """
    return {
        "min_jacobian": float(np.min(metrics["jacobians"])),
        "mean_jacobian": float(np.mean(metrics["jacobians"])),
        "min_escobar_quality": float(np.min(metrics["escobar_quality"])),
        "mean_escobar_quality": float(np.mean(metrics["escobar_quality"])),
        "max_aspect_ratio": float(np.max(metrics["aspect_ratio"])),
        "mean_aspect_ratio": float(np.mean(metrics["aspect_ratio"])),
        "min_angle_deg": float(np.rad2deg(np.min(metrics["min_angle"]))),
        "num_inverted_cells": int(np.sum(metrics["inverted"])),
        "num_degenerate_cells": int(np.sum(metrics["degenerate"]))
    }