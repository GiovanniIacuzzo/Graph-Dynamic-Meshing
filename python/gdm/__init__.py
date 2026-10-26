"""
Interfaccia Python di raccordo verso il nucleo nativo C++20 / deal.II (_gdm_core).
Mantiene invariati i moduli PyTorch della GNN e fornisce un'API stabile per Stage A, B e C.
"""

from gdm.native_api import (
    NativeMeshAdapter,
    create_pde_model,
    create_fem_solver,
    create_ale_solver,
    compute_native_mesh_metrics,
    apply_native_safety_filter,
    compute_native_continuous_errors,
    remap_p1_native,
    remap_l2_native,
)
from gdm.feature_adapter import build_gnn_inference_inputs
from gdm.runtime import NativeDynamicRuntime

__all__ = [
    "NativeMeshAdapter",
    "create_pde_model",
    "create_fem_solver",
    "create_ale_solver",
    "compute_native_mesh_metrics",
    "apply_native_safety_filter",
    "compute_native_continuous_errors",
    "remap_p1_native",
    "remap_l2_native",
    "build_gnn_inference_inputs",
    "NativeDynamicRuntime",
]