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
    """Adapter leggero per lo stato mesh di ``gdm._gdm_core``; nessuna mesh Python parallela."""

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
        """Archi orientati bidirezionali esposti dal core nativo per la GNN."""
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


def create_linear_solver_parameters(config: Optional[Dict[str, Any]] = None) -> "_gdm_core.LinearSolverParameters":
    """Create validated native linear-solver settings from a plain dictionary."""
    cfg = dict(config or {})
    params = _gdm_core.LinearSolverParameters()
    for name in ("rel_tol", "abs_tol", "max_iterations", "gmres_krylov_dim", "ssor_omega",
                 "use_direct_solver", "right_preconditioning", "ale_rhs_norm_floor"):
        if name in cfg:
            setattr(params, name, cfg[name])
    params.validate()
    return params


def create_stabilization_parameters(
    mode: str = "galerkin", config: Optional[Dict[str, Any]] = None
) -> "_gdm_core.StabilizationParameters":
    """Create native Galerkin/SUPG settings; Galerkin remains the default."""
    normalized = str(mode).strip().lower()
    if normalized not in {"galerkin", "supg"}:
        raise ValueError("stabilization mode deve essere 'galerkin' oppure 'supg'.")
    cfg = dict(config or {})
    params = _gdm_core.StabilizationParameters()
    params.use_supg = normalized == "supg"
    for name in ("c_time", "c_advection", "c_diffusion"):
        if name in cfg:
            setattr(params, name, float(cfg[name]))
    params.validate()
    return params


def create_fem_solver(
    mesh: NativeMeshAdapter,
    pde: "_gdm_core.ManufacturedConvectionDiffusion2D",
    stabilization_mode: str = "galerkin",
    stabilization_config: Optional[Dict[str, Any]] = None,
    linear_solver_config: Optional[Dict[str, Any]] = None,
) -> "_gdm_core.ConvectionDiffusionFEM2D":
    solver_params = create_linear_solver_parameters(linear_solver_config)
    stabilization_params = create_stabilization_parameters(stabilization_mode, stabilization_config)
    return _gdm_core.ConvectionDiffusionFEM2D(mesh.core, pde, solver_params, stabilization_params)


def create_ale_solver(
    pde: Optional["_gdm_core.ManufacturedConvectionDiffusion2D"] = None,
    nu: float = 1.0e-3,
    bx: float = 0.4,
    by: float = 0.2,
    stabilization_mode: str = "galerkin",
    stabilization_config: Optional[Dict[str, Any]] = None,
    linear_solver_config: Optional[Dict[str, Any]] = None,
) -> "_gdm_core.ALESolver2D":
    solver_params = create_linear_solver_parameters(linear_solver_config)
    stabilization_params = create_stabilization_parameters(stabilization_mode, stabilization_config)
    return _gdm_core.ALESolver2D(
        pde, float(nu), float(bx), float(by), solver_params, stabilization_params
    )


def compute_native_mesh_metrics(
    mesh: NativeMeshAdapter, tol_degenerate: float = 1.0e-14
) -> Dict[str, Any]:
    if not np.isfinite(tol_degenerate) or tol_degenerate <= 0.0:
        raise ValueError("tol_degenerate deve essere finito e strettamente positivo.")
    return dict(_gdm_core.compute_mesh_metrics(mesh.core, float(tol_degenerate)))


def compute_native_mesh_metrics_arrays(
    nodes: np.ndarray,
    cells: np.ndarray,
    tol_degenerate: float = 1.0e-14,
) -> Dict[str, Any]:
    """Calcola le metriche della mesh direttamente nel core C++ da array NumPy.

    Non costruisce un ``MeshState``: è adatto a diagnosticare proposte GNN/oracle
    che possono contenere celle invertite e quindi non sono ancora ammissibili
    come triangolazione FEM. Tutta la geometria è valutata dal binding nativo.
    """
    nodes_arr = _points_array(nodes, "nodes")
    cells_arr = _cells_array(cells)
    if nodes_arr.shape[0] < 3 or cells_arr.shape[0] < 1:
        raise ValueError("La mesh deve contenere almeno 3 nodi e 1 triangolo.")
    if np.any(cells_arr < 0) or np.any(cells_arr >= nodes_arr.shape[0]):
        raise ValueError("cells contiene indici nodali fuori intervallo.")
    if np.any(
        (cells_arr[:, 0] == cells_arr[:, 1])
        | (cells_arr[:, 1] == cells_arr[:, 2])
        | (cells_arr[:, 0] == cells_arr[:, 2])
    ):
        raise ValueError("cells contiene triangoli con indici nodali ripetuti.")
    if not np.isfinite(tol_degenerate) or tol_degenerate <= 0.0:
        raise ValueError("tol_degenerate deve essere finito e strettamente positivo.")
    return dict(
        _gdm_core.compute_mesh_metrics_arrays(
            nodes_arr, cells_arr, float(tol_degenerate)
        )
    )


def compute_native_field_gradients(
    mesh: NativeMeshAdapter,
    u: np.ndarray,
    pde: Optional["_gdm_core.ManufacturedConvectionDiffusion2D"] = None,
    t: float = 0.0,
    exact_gradient: bool = False,
) -> np.ndarray:
    """Calcola i gradienti nodali tramite il core nativo, senza FEM Python."""
    field = np.ascontiguousarray(u, dtype=np.float64)
    if field.shape != (mesh.num_nodes,) or not np.all(np.isfinite(field)):
        raise ValueError(f"u deve essere un vettore finito di shape ({mesh.num_nodes},).")
    if not np.isfinite(t):
        raise ValueError("t deve essere finito.")
    grad = np.asarray(
        _gdm_core.compute_field_gradients(
            mesh.core, field, pde, float(t), bool(exact_gradient)
        ),
        dtype=np.float64,
    )
    if grad.shape != (mesh.num_nodes, 2) or not np.all(np.isfinite(grad)):
        raise RuntimeError("Il core nativo ha restituito gradienti non validi.")
    return grad


def apply_native_safety_filter_arrays(
    nodes: np.ndarray,
    displacement: np.ndarray,
    cells: np.ndarray,
    boundary_nodes: Optional[np.ndarray] = None,
    epsilon: float = 1.0e-4,
    safety_margin: float = 1.0e-6,
    tol_deg: float = 1.0e-12,
    lock_boundary_nodes: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Applica il filtro di sicurezza C++ a coordinate/connettività NumPy.

    La mesh di riferimento deve essere valida; ``displacement`` può invece
    rappresentare una proposta non ammissibile. Le decisioni di sicurezza e la
    contrazione lungo il segmento sono sempre calcolate dal core nativo.
    """
    mesh = NativeMeshAdapter.from_arrays(nodes, cells, boundary_nodes)
    return apply_native_safety_filter(
        mesh,
        displacement,
        epsilon=epsilon,
        safety_margin=safety_margin,
        tol_deg=tol_deg,
        lock_boundary_nodes=lock_boundary_nodes,
        verbose=verbose,
    )


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

def compute_native_continuous_error_details(
    mesh: NativeMeshAdapter,
    u_h: np.ndarray,
    pde: "_gdm_core.ManufacturedConvectionDiffusion2D",
    t: float,
) -> Dict[str, float]:
    """Restituisce norme L2/H1 assolute e relative integrate da deal.II/C++."""
    field = np.ascontiguousarray(u_h, dtype=np.float64)
    if field.shape != (mesh.num_nodes,) or not np.all(np.isfinite(field)):
        raise ValueError(f"u_h deve essere un vettore finito di shape ({mesh.num_nodes},).")
    if not np.isfinite(t):
        raise ValueError("t deve essere finito.")
    values = dict(
        _gdm_core.compute_continuous_error_details(
            mesh.core, field, pde, float(t)
        )
    )
    return {str(k): float(v) for k, v in values.items()}

def compute_native_discrete_mass(
    mesh: NativeMeshAdapter, u: np.ndarray
) -> float:
    """Integra il campo nodale sulla mesh mediante il kernel C++ nativo."""
    field = np.ascontiguousarray(u, dtype=np.float64)
    if field.shape != (mesh.num_nodes,) or not np.all(np.isfinite(field)):
        raise ValueError(f"u deve essere un vettore finito di shape ({mesh.num_nodes},).")
    result = float(_gdm_core.compute_discrete_mass(mesh.core, field))
    if not np.isfinite(result):
        raise RuntimeError("Il core nativo ha restituito una massa non finita.")
    return result

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
