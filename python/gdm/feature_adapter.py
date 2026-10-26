"""
gdm/feature_adapter.py
Adattatore esplicito fra lo stato C++ (`NativeMeshAdapter`, `u_h`) e il contratto Python
richiesto da `MeshMoverInferenceEngine.predict(mesh, u, grad_u)` in `inference.py`.
"""

from typing import Tuple, Optional
import numpy as np

from geometry.mesh import TriangularMesh2D
from . import _gdm_core
from .native_api import NativeMeshAdapter


def build_gnn_inference_inputs(
    mesh: NativeMeshAdapter,
    u_h: np.ndarray,
    pde: Optional[_gdm_core.ManufacturedConvectionDiffusion2D] = None,
    t: float = 0.0,
    exact_gradient: bool = False,
) -> Tuple[TriangularMesh2D, np.ndarray, np.ndarray]:
    """
    Costruisce la tupla `(py_mesh, u_arr, grad_u_arr)` nel formato esatto atteso dalla GNN Python
    senza alterare pesi, normalizzazioni o dimensioni delle feature.
    """
    u_arr = np.ascontiguousarray(u_h, dtype=np.float64)
    if u_arr.ndim != 1 or u_arr.shape[0] != mesh.num_nodes:
        raise ValueError(f"u_h deve avere shape ({mesh.num_nodes},), ricevuta {u_arr.shape}.")
    if not np.all(np.isfinite(u_arr)):
        raise ValueError("u_h contiene valori non finiti.")

    grad_u_arr = np.asarray(
        _gdm_core.compute_field_gradients(
            mesh.core, u_arr, pde, float(t), bool(exact_gradient)
        ),
        dtype=np.float64,
    )
    if grad_u_arr.shape != (mesh.num_nodes, 2) or not np.all(np.isfinite(grad_u_arr)):
        raise RuntimeError("Il core nativo ha restituito un gradiente nodale malformato o non finito.")

    py_mesh = TriangularMesh2D(
        nodes=mesh.nodes.copy(),
        cells=mesh.cells.copy(),
        boundary_nodes=mesh.boundary_nodes.copy(),
    )
    if not np.array_equal(py_mesh.boundary_nodes, mesh.boundary_nodes):
        raise RuntimeError("La maschera di bordo Python non coincide con quella della mesh nativa.")
    if not np.array_equal(py_mesh.cells, mesh.cells):
        raise RuntimeError("L'adapter Python ha alterato la connettività della mesh nativa.")
    return py_mesh, u_arr, grad_u_arr