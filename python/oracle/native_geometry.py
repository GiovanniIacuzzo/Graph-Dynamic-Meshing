"""Adapters for native mesh geometry and admissibility kernels.

This module contains no Python implementation of mesh metrics or safety checks.
Those operations are delegated to the compiled ``gdm._gdm_core`` extension. The
SciPy oracle may still define its scalar optimization functional and analytic
coordinate gradient in Python, but it cannot make a mesh-validity decision itself.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np


def compute_native_mesh_metrics_arrays(
    nodes: np.ndarray,
    cells: np.ndarray,
    tol_degenerate: float = 1.0e-14,
) -> Dict[str, Any]:
    """Calculate per-cell and aggregate mesh-quality metrics in C++."""
    from gdm.native_api import compute_native_mesh_metrics_arrays as _compute

    return _compute(nodes, cells, tol_degenerate=tol_degenerate)


def apply_native_safety_filter_arrays(
    nodes: np.ndarray,
    displacement: np.ndarray,
    cells: np.ndarray,
    boundary_nodes: Optional[np.ndarray] = None,
    epsilon: float = 1.0e-4,
    tol_deg: float = 1.0e-12,
    safety_margin: float = 1.0e-6,
    lock_boundary_nodes: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Apply the canonical C++ safety filter to a NumPy-defined base mesh."""
    from gdm.native_api import apply_native_safety_filter_arrays as _apply

    return _apply(
        nodes=nodes,
        displacement=displacement,
        cells=cells,
        boundary_nodes=boundary_nodes,
        epsilon=epsilon,
        safety_margin=safety_margin,
        tol_deg=tol_deg,
        lock_boundary_nodes=lock_boundary_nodes,
        verbose=verbose,
    )


def apply_native_safety_filter_tuple(
    nodes: np.ndarray,
    displacement: np.ndarray,
    cells: np.ndarray,
    epsilon: float = 1.0e-4,
    tol_deg: float = 1.0e-12,
    safety_margin: float = 1.0e-6,
    verbose: bool = False,
    boundary_nodes: Optional[np.ndarray] = None,
    lock_boundary_nodes: bool = True,
) -> Tuple[float, np.ndarray, bool, float]:
    """Compatibility tuple for optimizer callers; every result is native-derived."""
    result = apply_native_safety_filter_arrays(
        nodes=nodes,
        displacement=displacement,
        cells=cells,
        boundary_nodes=boundary_nodes,
        epsilon=epsilon,
        tol_deg=tol_deg,
        safety_margin=safety_margin,
        lock_boundary_nodes=lock_boundary_nodes,
        verbose=verbose,
    )
    return (
        float(result["alpha_global"]),
        np.asarray(result["safe_nodes"], dtype=np.float64),
        bool(result["is_safe"]),
        float(result["min_jacobian_after"]),
    )
