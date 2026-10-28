"""Adapter che espone lo stato numerico nativo come input della GNN Python."""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from .native_api import (
    NativeMeshAdapter,
    _gdm_core,
    compute_native_field_gradients,
)


def build_gnn_inference_inputs(
    mesh: NativeMeshAdapter,
    u_h: np.ndarray,
    pde: Optional[_gdm_core.ManufacturedConvectionDiffusion2D] = None,
    t: float = 0.0,
    exact_gradient: bool = False,
) -> Tuple[NativeMeshAdapter, np.ndarray, np.ndarray]:
    """Return ``(native_mesh, u, grad_u)`` without reconstructing a Python mesh.

    Connectivity, boundary nodes and gradients originate from ``gdm._gdm_core``;
    this function only validates arrays and packages the inputs for the GNN.
    """
    u_arr = np.ascontiguousarray(u_h, dtype=np.float64)
    if u_arr.ndim != 1 or u_arr.shape[0] != mesh.num_nodes:
        raise ValueError(f"u_h deve avere shape ({mesh.num_nodes},), ricevuta {u_arr.shape}.")
    if not np.all(np.isfinite(u_arr)):
        raise ValueError("u_h contiene valori non finiti.")
    grad_u_arr = compute_native_field_gradients(
        mesh, u_arr, pde, float(t), bool(exact_gradient)
    )
    return mesh, u_arr, grad_u_arr
