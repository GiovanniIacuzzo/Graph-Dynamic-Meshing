"""Typed Python interface to the compiled ``gdm._gdm_core`` extension."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    from . import _gdm_core
except ImportError as exc:
    raise ImportError(
        "Impossibile importare l'estensione nativa 'gdm._gdm_core'. "
        "Dopo aver installato l'ambiente Python, configura deal.II e compila il core "
        "con: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build"
    ) from exc


def _points_array(values: np.ndarray, name: str = "nodes") -> np.ndarray:
    arr = np.ascontiguousarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} deve avere forma (N, 2), ricevuta {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene coordinate non finite.")
    return arr


def _cells_array(values: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(values, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"cells deve avere forma (M, 3), ricevuta {arr.shape}.")
    return arr


class NativeMeshAdapter:
    """Adattatore di ``_gdm_core.MeshState`` compatibile con le API di ``TriangularMesh2D``."""

    def __init__(self, core_mesh: "_gdm_core.MeshState") -> None:
        self._core = core_mesh

    @classmethod
    def from_arrays(
        cls,
        nodes: np.ndarray,
        cells: np.ndarray,
        boundary_nodes: Optional[np.ndarray] = None,
    ) -> "NativeMeshAdapter":
        nodes_arr = _points_array(nodes)
        cells_arr = _cells_array(cells)
        bnd = [] if boundary_nodes is None else np.asarray(boundary_nodes, dtype=np.int64).tolist()
        return cls(_gdm_core.MeshState(nodes_arr, cells_arr, bnd))

    @classmethod
    def from_domain(
        cls, domain_cfg: Dict[str, Any], nx: int = 32, ny: int = 32
    ) -> "NativeMeshAdapter":
        dom = _gdm_core.DomainParameters()
        dom.x_min = float(domain_cfg.get("x_min", 0.0))
        dom.x_max = float(domain_cfg.get("x_max", 1.0))
        dom.y_min = float(domain_cfg.get("y_min", 0.0))
        dom.y_max = float(domain_cfg.get("y_max", 1.0))
        dom.nx = int(nx)
        dom.ny = int(ny)
        dom.validate()
        return cls(_gdm_core.MeshState.create_structured_mesh(dom))

    @property
    def core(self) -> "_gdm_core.MeshState":
        return self._core

    @property
    def num_nodes(self) -> int:
        return int(self._core.num_nodes)

    @property
    def num_cells(self) -> int:
        return int(self._core.num_cells)

    @property
    def nodes(self) -> np.ndarray:
        return np.asarray(self._core.nodes, dtype=np.float64)

    @property
    def cells(self) -> np.ndarray:
        return np.asarray(self._core.cells, dtype=np.int64)

    @property
    def boundary_nodes(self) -> np.ndarray:
        return np.asarray(self._core.boundary_nodes, dtype=np.int64).copy()

    @property
    def boundary_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_nodes, dtype=bool)
        mask[self.boundary_nodes] = True
        return mask

    @property
    def interior_mask(self) -> np.ndarray:
        return ~self.boundary_mask

    @property
    def directed_edges(self) -> np.ndarray:
        return np.asarray(self._core.directed_edges, dtype=np.int64).reshape(-1, 2)

    @property
    def undirected_edges(self) -> np.ndarray:
        return np.asarray(self._core.undirected_edges, dtype=np.int64).reshape(-1, 2)

    @property
    def edges(self) -> np.ndarray:
        """Archi orientati bidirezionali, come ``TriangularMesh2D.edges``."""
        return self.directed_edges

    @property
    def num_edges(self) -> int:
        return int(self.directed_edges.shape[0])

    def update_coordinates(self, new_nodes: np.ndarray, enforce_fixed_boundary: bool = False) -> "NativeMeshAdapter":
        coords = _points_array(new_nodes, "new_nodes")
        if coords.shape != self.nodes.shape:
            raise ValueError(f"new_nodes ha shape {coords.shape}; attesa {self.nodes.shape}.")
        updated = self._core.cloned_with_coordinates(coords, bool(enforce_fixed_boundary))
        return NativeMeshAdapter(updated)


def create_pde_model(
    bx: float = 0.4,
    by: float = 0.2,
    nu: float = 1.0e-3,
    sigma: float = 0.06,
    beta: float = 0.5,
    T: float = 0.5,
) -> "_gdm_core.ManufacturedConvectionDiffusion2D":
    params = _gdm_core.PDEParameters()
    params.bx = float(bx)
    params.by = float(by)
    params.nu = float(nu)
    params.sigma = float(sigma)
    params.beta = float(beta)
    params.T = float(T)
    params.validate()
    return _gdm_core.ManufacturedConvectionDiffusion2D(params)


def create_fem_solver(
    mesh: NativeMeshAdapter,
    pde: "_gdm_core.ManufacturedConvectionDiffusion2D",
) -> "_gdm_core.ConvectionDiffusionFEM2D":
    return _gdm_core.ConvectionDiffusionFEM2D(mesh.core, pde)


def create_ale_solver(
    pde: Optional["_gdm_core.ManufacturedConvectionDiffusion2D"] = None,
    nu: float = 1.0e-3,
    bx: float = 0.4,
    by: float = 0.2,
) -> "_gdm_core.ALESolver2D":
    return _gdm_core.ALESolver2D(pde, float(nu), float(bx), float(by))


def compute_native_mesh_metrics(
    mesh: NativeMeshAdapter, tol_degenerate: float = 1.0e-14
) -> Dict[str, Any]:
    if not np.isfinite(tol_degenerate) or tol_degenerate <= 0.0:
        raise ValueError("tol_degenerate deve essere finito e strettamente positivo.")
    return dict(_gdm_core.compute_mesh_metrics(mesh.core, float(tol_degenerate)))


def apply_native_safety_filter(
    mesh: NativeMeshAdapter,
    displacement: np.ndarray,
    epsilon: float = 1.0e-4,
    safety_margin: float = 1.0e-6,
    tol_deg: float = 1.0e-12,
    lock_boundary_nodes: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Applica il filtro C++ e valida forma e valori finiti del risultato."""
    disp = _points_array(displacement, "displacement")
    if disp.shape != mesh.nodes.shape:
        raise ValueError(f"displacement ha shape {disp.shape}; attesa {mesh.nodes.shape}.")
    params = _gdm_core.SafetyParameters()
    params.epsilon = float(epsilon)
    params.safety_margin = float(safety_margin)
    params.tol_deg = float(tol_deg)
    params.lock_boundary_nodes = bool(lock_boundary_nodes)
    params.verbose = bool(verbose)
    params.validate()
    result = dict(_gdm_core.apply_safety_filter(mesh.core, disp, params))
    safe_nodes = np.asarray(result.get("safe_nodes"), dtype=np.float64)
    if safe_nodes.shape != mesh.nodes.shape or not np.all(np.isfinite(safe_nodes)):
        raise RuntimeError("Il safety filter C++ ha restituito safe_nodes malformato o non finito.")
    alpha = float(result.get("alpha_global", np.nan))
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0 + 1.0e-12:
        raise RuntimeError(f"Il safety filter C++ ha restituito alpha_global non valido: {alpha}.")
    result["safe_nodes"] = safe_nodes
    result["safe_displacement"] = safe_nodes - mesh.nodes
    result["alpha_global"] = min(1.0, max(0.0, alpha))
    result["is_safe"] = bool(result.get("is_safe", False))
    return result


def compute_native_continuous_errors(
    mesh: NativeMeshAdapter,
    u_h: np.ndarray,
    pde: "_gdm_core.ManufacturedConvectionDiffusion2D",
    t: float,
) -> Tuple[float, float]:
    field = np.ascontiguousarray(u_h, dtype=np.float64)
    if field.ndim != 1 or field.shape[0] != mesh.num_nodes:
        raise ValueError(f"u_h deve avere shape ({mesh.num_nodes},), ricevuta {field.shape}.")
    if not np.all(np.isfinite(field)):
        raise ValueError("u_h contiene valori non finiti.")
    err_l2, err_h1 = _gdm_core.compute_continuous_errors(mesh.core, field, pde, float(t))
    return float(err_l2), float(err_h1)


def remap_p1_native(
    old_mesh: NativeMeshAdapter, old_u: np.ndarray, new_mesh: NativeMeshAdapter
) -> Dict[str, Any]:
    field = np.ascontiguousarray(old_u, dtype=np.float64)
    if field.shape != (old_mesh.num_nodes,):
        raise ValueError(f"old_u deve avere shape ({old_mesh.num_nodes},), ricevuta {field.shape}.")
    return dict(_gdm_core.remap_p1_solution(old_mesh.core, field, new_mesh.core))


def remap_l2_native(
    old_mesh: NativeMeshAdapter, old_u: np.ndarray, new_mesh: NativeMeshAdapter
) -> Dict[str, Any]:
    field = np.ascontiguousarray(old_u, dtype=np.float64)
    if field.shape != (old_mesh.num_nodes,):
        raise ValueError(f"old_u deve avere shape ({old_mesh.num_nodes},), ricevuta {field.shape}.")
    return dict(_gdm_core.remap_l2_projection(old_mesh.core, field, new_mesh.core))
