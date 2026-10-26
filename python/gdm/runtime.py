"""Runtime dinamico C++/deal.II con inferenza GNN Python e filtro di sicurezza nativo."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from . import _gdm_core
from .native_api import (
    NativeMeshAdapter,
    apply_native_safety_filter,
    compute_native_continuous_errors,
    compute_native_mesh_metrics,
    create_ale_solver,
    create_fem_solver,
    create_pde_model,
    remap_l2_native,
    remap_p1_native,
)
from .feature_adapter import build_gnn_inference_inputs


class NativeDynamicRuntime:
    """Orchestra i solver C++ e la GNN, preservando il protocollo temporale Stage C."""

    METHODS = {"fixed", "gdm_p1", "gdm_l2", "gdm_ale", "oracle_ale"}

    def __init__(
        self,
        domain_cfg: Dict[str, Any],
        nx: int = 32,
        ny: int = 32,
        bx: float = 0.4,
        by: float = 0.2,
        nu: float = 1.0e-3,
        sigma: float = 0.06,
        beta: float = 0.5,
        T: float = 0.5,
        dt: float = 0.005,
        adapt_interval: int = 10,
        inference_engine: Optional[Any] = None,
        oracle_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.domain_cfg = dict(domain_cfg)
        self.nx = int(nx)
        self.ny = int(ny)
        self.bx = float(bx)
        self.by = float(by)
        self.nu = float(nu)
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.T = float(T)
        self.dt = float(dt)
        self.adapt_interval = int(adapt_interval)
        self.engine = inference_engine
        self.oracle_cfg = dict(oracle_cfg or {})

        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx e ny devono essere almeno 2.")
        if not np.isfinite(self.T) or self.T <= 0.0:
            raise ValueError("T deve essere finito e strettamente positivo.")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt deve essere finito e strettamente positivo.")
        if self.adapt_interval < 1:
            raise ValueError("adapt_interval deve essere almeno 1.")
        ratio = self.T / self.dt
        self.n_steps = int(round(ratio))
        if abs(self.n_steps * self.dt - self.T) > 1.0e-10 * self.T + 1.0e-12:
            raise ValueError("T deve essere un multiplo intero di dt.")

        self.pde = create_pde_model(
            bx=self.bx, by=self.by, nu=self.nu, sigma=self.sigma, beta=self.beta, T=self.T
        )
        cfg = getattr(self.engine, "config", {}) if self.engine is not None else {}
        if isinstance(cfg, dict):
            safety_cfg = cfg.get("safety", cfg.get("geometry", {}).get("safety", {}))
        else:
            safety_cfg = {}
        self.epsilon = float(safety_cfg.get("epsilon", 1.0e-4))
        self.safety_margin = float(safety_cfg.get("safety_margin", 1.0e-6))
        self.tol_deg = float(safety_cfg.get("tol_deg", 1.0e-12))

        objective_cfg = self.oracle_cfg.get("objective", self.oracle_cfg)
        self.oracle_lambda_D = float(objective_cfg.get("lambda_D", 0.01))
        self.oracle_lambda_Q = float(objective_cfg.get("lambda_Q", 0.001))
        self.oracle_epsilon = float(self.oracle_cfg.get("epsilon", self.epsilon))
        self.oracle_max_iter = int(self.oracle_cfg.get("max_iter", 3))
        if self.oracle_max_iter < 1:
            raise ValueError("oracle max_iter deve essere almeno 1.")

    def _native_metrics(self, mesh: NativeMeshAdapter) -> Dict[str, Any]:
        return compute_native_mesh_metrics(mesh)

    def _gnn_target(self, mesh: NativeMeshAdapter, u: np.ndarray, t: float) -> tuple[np.ndarray, Dict[str, Any]]:
        if self.engine is None:
            raise RuntimeError("Per le modalità GDM è necessario configurare inference_engine.")
        if not hasattr(self.engine, "predict_raw"):
            raise AttributeError(
                "inference_engine deve esporre predict_raw(mesh, u, grad_u); "
                "non è consentito usare predict(), che applica il filtro Python."
            )

        t_adapt_start = time.perf_counter()
        py_mesh, u_in, grad_in = build_gnn_inference_inputs(
            mesh, u, self.pde, float(t), exact_gradient=False
        )
        raw_result = self.engine.predict_raw(py_mesh, u_in, grad_in)
        adapt_elapsed = time.perf_counter() - t_adapt_start
        raw_disp = np.asarray(raw_result["raw_displacement"], dtype=np.float64)
        if raw_disp.shape != mesh.nodes.shape or not np.all(np.isfinite(raw_disp)):
            raise RuntimeError("predict_raw ha restituito uno spostamento di forma non valida o non finito.")

        t_filter = time.perf_counter()
        filter_result = apply_native_safety_filter(
            mesh=mesh,
            displacement=raw_disp,
            epsilon=self.epsilon,
            safety_margin=self.safety_margin,
            tol_deg=self.tol_deg,
            lock_boundary_nodes=False,  # predict_raw azzera già lo spostamento sul bordo
        )
        filter_elapsed = time.perf_counter() - t_filter
        target_nodes = np.asarray(filter_result["safe_nodes"], dtype=np.float64)
        detail = {
            "raw_displacement": raw_disp,
            "safe_displacement": target_nodes - mesh.nodes,
            "alpha": float(filter_result["alpha_global"]),
            "is_safe": bool(filter_result["is_safe"]),
            "min_j_before": float(filter_result["min_jacobian_before"]),
            "min_j_after": float(filter_result["min_jacobian_after"]),
            "inference_time_s": float(raw_result.get("inference_time", 0.0)),
            "adapt_time_s": float(adapt_elapsed),
            "safety_time_s": float(filter_elapsed),
        }
        return target_nodes, detail

    def _oracle_target(self, mesh: NativeMeshAdapter, t: float) -> tuple[np.ndarray, Dict[str, Any]]:
        # L'oracolo Stage C ottimizza un campo GaussianBumpField, distinto dalla
        # soluzione manufactured usata per integrare la PDE.
        from oracle.field import GaussianBumpField
        from oracle.optimizer import OfflineMeshOptimizer

        center = self.pde.get_center(float(t))
        field = GaussianBumpField(
            mu_x=float(center[0]),
            mu_y=float(center[1]),
            sigma=self.sigma,
            amplitude=float(np.exp(-self.beta * t)),
        )
        timer = time.perf_counter()
        optimizer = OfflineMeshOptimizer(
            reference_field=field,
            initial_nodes=mesh.nodes,
            cells=mesh.cells,
            boundary_nodes=mesh.boundary_nodes,
            lambda_D=self.oracle_lambda_D,
            lambda_Q=self.oracle_lambda_Q,
            epsilon=self.oracle_epsilon,
            max_iter=self.oracle_max_iter,
        )
        result = optimizer.optimize()
        elapsed = time.perf_counter() - timer
        target = np.asarray(result["optimal_nodes"], dtype=np.float64)
        if target.shape != mesh.nodes.shape or not np.all(np.isfinite(target)):
            raise RuntimeError("L'oracle ha restituito coordinate target non valide.")
        return target, {
            "alpha": float(result["final_metrics"].get("safety_alpha_applied", 1.0)),
            "oracle_time_s": float(elapsed),
        }

    def run_episode(
        self,
        method: str = "gdm_ale",
        replay_targets: Optional[List[np.ndarray]] = None,
        compute_diagnostics: bool = True,
        record_targets: bool = False,
    ) -> Dict[str, Any]:
        """Esegue una simulazione e restituisce errori, geometria e tempi separati."""
        if method not in self.METHODS:
            raise ValueError(f"Metodo {method!r} non supportato; metodi disponibili: {sorted(self.METHODS)}")
        if method == "oracle_ale" and replay_targets is not None:
            raise ValueError("oracle_ale non accetta replay_targets; il target oracle viene ricalcolato a ogni finestra.")
        if method == "fixed" and replay_targets is not None:
            raise ValueError("fixed non accetta replay_targets perché non esegue adattamento della mesh.")
        if replay_targets is not None and method != "fixed":
            expected_windows = (self.n_steps - 1) // self.adapt_interval + 1
            if len(replay_targets) != expected_windows:
                raise ValueError(
                    f"replay_targets deve contenere esattamente {expected_windows} target per questo protocollo; "
                    f"ricevuti {len(replay_targets)}."
                )

        t_wall_start = time.perf_counter()
        init_mesh = NativeMeshAdapter.from_domain(self.domain_cfg, nx=self.nx, ny=self.ny)
        curr_mesh = init_mesh
        u_curr = np.asarray(self.pde.evaluate_solution(curr_mesh.nodes, 0.0), dtype=np.float64)

        fem_setup_start = time.perf_counter()
        fem_solver = create_fem_solver(curr_mesh, self.pde) if method in {"fixed", "gdm_p1", "gdm_l2"} else None
        fem_setup_elapsed = time.perf_counter() - fem_setup_start if fem_solver is not None else 0.0
        ale_solver = create_ale_solver(self.pde) if method in {"gdm_ale", "oracle_ale"} else None

        base_nodes = curr_mesh.nodes.copy()
        target_nodes = curr_mesh.nodes.copy()
        recorded_targets: List[np.ndarray] = []
        adaptation_steps: List[int] = []
        adaptation_times: List[float] = []
        time_arr: List[float] = [0.0] if compute_diagnostics else []
        l2_series: List[float] = []
        h1_series: List[float] = []
        min_j_series: List[float] = []
        alpha_series: List[float] = [1.0] if compute_diagnostics else []
        snapshots: List[tuple[float, np.ndarray, np.ndarray]] = (
            [(0.0, curr_mesh.nodes.copy(), u_curr.copy())] if compute_diagnostics else []
        )

        if compute_diagnostics:
            err_l2_0, err_h1_0 = compute_native_continuous_errors(curr_mesh, u_curr, self.pde, 0.0)
            l2_series.append(err_l2_0)
            h1_series.append(err_h1_0)
            min_j_series.append(float(self._native_metrics(curr_mesh)["min_jacobian"]))

        fem_time_s = fem_setup_elapsed
        adapt_time_s = 0.0
        safety_time_s = 0.0
        remap_time_s = 0.0
        alpha_current_window = 1.0
        window_idx = 0

        for step in range(1, self.n_steps + 1):
            t_curr = (step - 1) * self.dt
            t_next = step * self.dt

            # Stessa convenzione di Stage C: target all'inizio di ogni finestra.
            if method != "fixed" and (step - 1) % self.adapt_interval == 0:
                detail: Dict[str, Any] = {}

                if replay_targets is not None:
                    if window_idx >= len(replay_targets):
                        raise ValueError(
                            f"replay_targets contiene {len(replay_targets)} target, ma ne serve almeno {window_idx + 1}."
                        )
                    target_nodes = np.asarray(replay_targets[window_idx], dtype=np.float64).copy()
                    if target_nodes.shape != curr_mesh.nodes.shape or not np.all(np.isfinite(target_nodes)):
                        raise ValueError(f"replay target {window_idx} ha shape o coordinate non valide.")
                elif method == "oracle_ale":
                    target_nodes, detail = self._oracle_target(curr_mesh, t_curr)
                    adapt_time_s += float(detail.get("oracle_time_s", 0.0))
                else:
                    target_nodes, detail = self._gnn_target(curr_mesh, u_curr, t_curr)
                    adapt_time_s += float(detail.get("adapt_time_s", detail.get("inference_time_s", 0.0)))
                    safety_time_s += float(detail.get("safety_time_s", 0.0))

                alpha_current_window = float(detail.get("alpha", 1.0))
                if record_targets:
                    recorded_targets.append(target_nodes.copy())
                    adaptation_steps.append(step)
                    adaptation_times.append(t_curr)

                base_nodes = curr_mesh.nodes.copy()
                window_idx += 1

                if method in {"gdm_p1", "gdm_l2"}:
                    t_remap = time.perf_counter()
                    new_mesh = init_mesh.update_coordinates(target_nodes)
                    if method == "gdm_p1":
                        remap_result = remap_p1_native(curr_mesh, u_curr, new_mesh)
                    else:
                        remap_result = remap_l2_native(curr_mesh, u_curr, new_mesh)
                    u_curr = np.asarray(remap_result["remapped_field"], dtype=np.float64)
                    boundary = new_mesh.boundary_nodes
                    u_exact = np.asarray(self.pde.evaluate_solution(new_mesh.nodes, t_curr), dtype=np.float64)
                    u_curr[boundary] = u_exact[boundary]
                    curr_mesh = new_mesh
                    remap_time_s += time.perf_counter() - t_remap
                    t_fem_setup = time.perf_counter()
                    fem_solver.rebuild_for_mesh(curr_mesh.core)
                    fem_time_s += time.perf_counter() - t_fem_setup

            # Avanzamento temporale.
            if method in {"fixed", "gdm_p1", "gdm_l2"}:
                t_fem = time.perf_counter()
                u_curr = np.asarray(fem_solver.step_backward_euler(u_curr, self.dt, t_next), dtype=np.float64)
                fem_time_s += time.perf_counter() - t_fem
            else:
                k_step = ((step - 1) % self.adapt_interval) + 1
                interp_nodes = base_nodes + (k_step / self.adapt_interval) * (target_nodes - base_nodes)
                next_mesh = init_mesh.update_coordinates(interp_nodes)
                t_ale = time.perf_counter()
                u_next, _diagnostics = ale_solver.step_ale(
                    curr_mesh.core, next_mesh.core, u_curr, self.dt, t_next, False, True
                )
                u_curr = np.asarray(u_next, dtype=np.float64)
                curr_mesh = next_mesh
                fem_time_s += time.perf_counter() - t_ale

            if compute_diagnostics:
                time_arr.append(t_next)
                err_l2, err_h1 = compute_native_continuous_errors(curr_mesh, u_curr, self.pde, t_next)
                l2_series.append(err_l2)
                h1_series.append(err_h1)
                min_j_series.append(float(self._native_metrics(curr_mesh)["min_jacobian"]))
                alpha_series.append(alpha_current_window)
                if step in {
                    int(round(self.n_steps / 3)),
                    int(round(2 * self.n_steps / 3)),
                    self.n_steps,
                }:
                    snapshots.append((t_next, curr_mesh.nodes.copy(), u_curr.copy()))

        wall_time_s = time.perf_counter() - t_wall_start
        component_time = fem_time_s + adapt_time_s + safety_time_s + remap_time_s
        result: Dict[str, Any] = {
            "fem_time_s": float(fem_time_s),
            "adapt_time_s": float(adapt_time_s),
            "safety_time_s": float(safety_time_s),
            "remap_time_s": float(remap_time_s),
            "sum_components_s": float(component_time),
            "other_overhead_s": float(max(0.0, wall_time_s - component_time)),
            "total_wall_time_s": float(wall_time_s),
            "wall_clock_s": float(wall_time_s),
            "recorded_targets": recorded_targets,
            "adaptation_steps": adaptation_steps,
            "adaptation_times": adaptation_times,
        }
        if compute_diagnostics:
            result.update({
                "time": time_arr,
                "rel_l2_error": l2_series,
                "rel_h1_error": h1_series,
                "final_l2": float(l2_series[-1]),
                "final_h1": float(h1_series[-1]),
                "time_int_l2": float(np.trapz(l2_series, time_arr) / self.T),
                "time_int_h1": float(np.trapz(h1_series, time_arr) / self.T),
                "min_jacobian": float(np.min(min_j_series)),
                "min_j_series": min_j_series,
                "alpha_series": alpha_series,
                "snapshots": snapshots,
                "final_mesh": curr_mesh,
                "final_u": u_curr,
                # Alias compatibili con il runtime precedente.
                "final_rel_l2": float(l2_series[-1]),
                "mean_rel_l2": float(np.mean(l2_series)),
                "error_history": l2_series,
                "min_j_history": min_j_series,
                "alpha_history": alpha_series,
            })
        return result
