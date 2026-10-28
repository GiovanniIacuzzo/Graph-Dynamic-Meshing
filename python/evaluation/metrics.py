"""Metriche di valutazione GDM con kernel geometrici/fisici nativi.

Python aggrega i risultati per report, ma non implementa mesh metrics, norme
FEM o integrazione della massa. Questi kernel vengono invocati attraverso
``gdm._gdm_core`` (C++20/deal.II).
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np


def compute_displacement_metrics(
    pred_disp: np.ndarray,
    target_disp: np.ndarray,
    eps: float = 1.0e-12,
) -> Dict[str, float]:
    """Metriche vettoriali di errore tra spostamento GNN e target oracle."""
    pred = np.asarray(pred_disp, dtype=np.float64)
    target = np.asarray(target_disp, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError(f"pred_disp e target_disp devono avere la stessa shape [N,2]: {pred.shape}, {target.shape}.")
    if pred.shape[0] == 0 or not np.all(np.isfinite(pred)) or not np.all(np.isfinite(target)):
        raise ValueError("Gli spostamenti devono contenere almeno un nodo e valori finiti.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps deve essere finito e positivo.")

    diff = pred - target
    diff_norm_f = float(np.linalg.norm(diff))
    target_norm_f = float(np.linalg.norm(target))
    nodal_err = np.linalg.norm(diff, axis=1)
    p_norm = np.linalg.norm(pred, axis=1)
    t_norm = np.linalg.norm(target, axis=1)
    valid = (p_norm > eps) & (t_norm > eps)
    if np.any(valid):
        cos_sim = np.sum(pred[valid] * target[valid], axis=1) / (p_norm[valid] * t_norm[valid])
        mean_cos = float(np.mean(cos_sim))
    else:
        mean_cos = 1.0
    return {
        "displacement_l2_error": diff_norm_f,
        "displacement_relative_error": diff_norm_f / max(target_norm_f, eps),
        "displacement_max_error": float(np.max(nodal_err)),
        "displacement_cosine_sim": mean_cos,
    }


def compute_mesh_geometric_metrics(mesh: Any) -> Dict[str, float]:
    """Estrae statistiche geometriche usando esclusivamente il core C++."""
    from gdm.native_api import NativeMeshAdapter, compute_native_mesh_metrics

    if not isinstance(mesh, NativeMeshAdapter):
        raise TypeError("mesh deve essere un NativeMeshAdapter prodotto da gdm._gdm_core.")
    raw = compute_native_mesh_metrics(mesh)
    jacobians = np.asarray(raw["jacobians"], dtype=np.float64)
    quality = np.asarray(raw["escobar_quality"], dtype=np.float64)
    aspect_ratio = np.asarray(raw["aspect_ratio"], dtype=np.float64)
    inverted = np.asarray(raw["inverted"], dtype=bool)
    if jacobians.size == 0 or quality.size == 0 or aspect_ratio.size == 0:
        raise RuntimeError("Il core nativo ha restituito metriche geometriche vuote.")
    return {
        "min_jacobian": float(np.min(jacobians)),
        "mean_jacobian": float(np.mean(jacobians)),
        "inverted_cells": int(np.sum(inverted)),
        "min_escobar_quality": float(np.min(quality)),
        "mean_mesh_quality": float(np.mean(quality)),
        "max_aspect_ratio": float(np.max(aspect_ratio)),
        "mean_aspect_ratio": float(np.mean(aspect_ratio)),
    }


def compute_functional_pde_errors(
    mesh: Any,
    u_approx: np.ndarray,
    exact_evaluator: Any,
    t: float,
) -> Dict[str, float]:
    """Calcola gli errori continui FEM integrati dal kernel nativo.

    ``exact_evaluator`` deve essere il modello PDE nativo restituito da
    ``gdm.create_pde_model``. Non viene usato un evaluator o una quadratura
    alternativa in Python.
    """
    from gdm.native_api import (
        NativeMeshAdapter,
        compute_native_continuous_error_details,
    )

    if not isinstance(mesh, NativeMeshAdapter):
        raise TypeError("mesh deve essere un NativeMeshAdapter prodotto da gdm._gdm_core.")
    details = compute_native_continuous_error_details(mesh, u_approx, exact_evaluator, t)
    return {
        "solution_l2_error": float(details["abs_l2_error"]),
        "solution_l2_relative_error": float(details["rel_l2_error"]),
        "solution_h1_error": float(details["abs_h1_seminorm_error"]),
        "solution_h1_relative_error": float(details["rel_h1_seminorm_error"]),
    }


def compute_physical_conservation_metrics(
    mesh_old: Any,
    mesh_new: Any,
    state_old: np.ndarray,
    state_new: np.ndarray,
    gcl_residual: float = 0.0,
) -> Dict[str, float]:
    """Calcola la massa su entrambe le mesh tramite il kernel nativo C++."""
    from gdm.native_api import NativeMeshAdapter, compute_native_discrete_mass

    if not isinstance(mesh_old, NativeMeshAdapter) or not isinstance(mesh_new, NativeMeshAdapter):
        raise TypeError("mesh_old e mesh_new devono essere NativeMeshAdapter nativi.")
    if not np.isfinite(gcl_residual):
        raise ValueError("gcl_residual deve essere finito.")
    mass_old = compute_native_discrete_mass(mesh_old, state_old)
    mass_new = compute_native_discrete_mass(mesh_new, state_new)
    return {
        "mass_old": mass_old,
        "mass_new": mass_new,
        "conservation_error": float(abs(mass_new - mass_old)),
        "gcl_error": float(gcl_residual),
    }


def format_computational_timings(
    gnn_time: float = 0.0,
    safety_time: float = 0.0,
    remap_time: float = 0.0,
    fem_time: float = 0.0,
) -> Dict[str, float]:
    """Normalizza il riepilogo dei tempi, mantenendoli in secondi."""
    values = [float(gnn_time), float(safety_time), float(remap_time), float(fem_time)]
    if not all(np.isfinite(v) and v >= 0.0 for v in values):
        raise ValueError("Tutti i tempi devono essere finiti e non negativi.")
    total = sum(values)
    return {
        "gnn_time": values[0],
        "safety_time": values[1],
        "remap_time": values[2],
        "fem_time": values[3],
        "total_time": float(total),
    }


def aggregate_scientific_report(
    displacement_metrics: Optional[Dict[str, float]] = None,
    geometric_metrics: Optional[Dict[str, float]] = None,
    physical_metrics: Optional[Dict[str, float]] = None,
    computational_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Combina gruppi di metriche già calcolate per JSON, tabelle e manifest."""
    report: Dict[str, Any] = {}
    for group in (displacement_metrics, geometric_metrics, physical_metrics, computational_metrics):
        if group:
            report.update(group)
    return report
