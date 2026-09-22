"""
numerics/dynamic_solver.py
Motore di simulazione transiente per PDE Convezione-Diffusione con r-adattivitá GDM.

Supporta:
    - 'fixed'                 : Griglia fissa (B0)
    - 'gnn_fem' (remap_method): GDM con gradiente nodale FEM runtime (B1: p1, B2: l2)
    - 'oracle'  (remap_method): Oracolo variazionale (B3)
"""

from typing import Dict, Any, List, Optional
import time
import numpy as np

from geometry.mesh import TriangularMesh2D
from geometry.geometry import compute_mesh_metrics
from geometry.safety import apply_safety_filter
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.fem import ConvectionDiffusionFEM2D, compute_field_gradients, _get_triangle_quadrature_degree_5
from numerics.remap import remap_p1_solution
from numerics.remap_l2 import remap_l2_projection
from oracle.field import GaussianBumpField
from oracle.optimizer import OfflineMeshOptimizer
from inference import MeshMoverInferenceEngine


def compute_continuous_errors(
    mesh: TriangularMesh2D,
    u_h: np.ndarray,
    pde: ManufacturedConvectionDiffusion2D,
    t: float
) -> tuple[float, float]:
    """Calcola l'errore continuo relativo L2 e la seminorma H1 con quadratura di Dunavant grado 5."""
    qp_ref, qw_ref = _get_triangle_quadrature_degree_5()
    cells = mesh.cells
    nodes = mesh.nodes

    x1, y1 = nodes[cells[:, 0], 0], nodes[cells[:, 0], 1]
    x2, y2 = nodes[cells[:, 1], 0], nodes[cells[:, 1], 1]
    x3, y3 = nodes[cells[:, 2], 0], nodes[cells[:, 2], 1]

    det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    areas = 0.5 * np.abs(det_J)
    inv_det = 1.0 / det_J

    dphi_dx = np.column_stack([(y2 - y3) * inv_det, (y3 - y1) * inv_det, (y1 - y2) * inv_det])
    dphi_dy = np.column_stack([(x3 - x2) * inv_det, (x1 - x3) * inv_det, (x2 - x1) * inv_det])

    u_elem = u_h[cells]
    duh_dx = np.sum(u_elem * dphi_dx, axis=1)
    duh_dy = np.sum(u_elem * dphi_dy, axis=1)

    l2_err_sq, h1_err_sq = 0.0, 0.0
    l2_ref_sq, h1_ref_sq = 0.0, 0.0

    for (xi, eta), w in zip(qp_ref, qw_ref):
        phi1 = 1.0 - xi - eta
        phi2 = xi
        phi3 = eta

        xq = phi1 * x1 + phi2 * x2 + phi3 * x3
        yq = phi1 * y1 + phi2 * y2 + phi3 * y3

        u_ex, ux_ex, uy_ex, _, _, _, _ = pde.evaluate_exact(xq, yq, t)
        u_fem_q = phi1 * u_elem[:, 0] + phi2 * u_elem[:, 1] + phi3 * u_elem[:, 2]

        diff_u = u_fem_q - u_ex
        diff_gx = duh_dx - ux_ex
        diff_gy = duh_dy - uy_ex

        factor = (2.0 * w) * areas
        l2_err_sq += np.sum(factor * (diff_u ** 2))
        h1_err_sq += np.sum(factor * (diff_gx ** 2 + diff_gy ** 2))

        l2_ref_sq += np.sum(factor * (u_ex ** 2))
        h1_ref_sq += np.sum(factor * (ux_ex ** 2 + uy_ex ** 2))

    rel_l2 = float(np.sqrt(max(l2_err_sq, 0.0)) / np.sqrt(max(l2_ref_sq, 1e-14)))
    rel_h1 = float(np.sqrt(max(h1_err_sq, 0.0)) / np.sqrt(max(h1_ref_sq, 1e-14)))
    return rel_l2, rel_h1


class DynamicPDESimulator:
    def __init__(
        self,
        initial_mesh: TriangularMesh2D,
        pde: ManufacturedConvectionDiffusion2D,
        config: Dict[str, Any],
        inference_engine: Optional[MeshMoverInferenceEngine] = None,
        epsilon: float = 1e-4
    ):
        self.initial_mesh = initial_mesh
        self.pde = pde
        self.config = config
        self.engine = inference_engine
        self.epsilon = float(epsilon)

    def run_simulation(
        self,
        mode: str = "gnn_fem",
        remap_method: str = "p1",
        T: float = 0.5,
        dt: float = 0.005,
        adapt_interval: int = 10
    ) -> Dict[str, Any]:
        assert mode in ["fixed", "gnn_fem", "gnn_exact", "oracle"], f"Modalità non valida: {mode}"
        assert remap_method in ["p1", "l2"], f"Remap method non valido: {remap_method}"

        current_mesh = self.initial_mesh
        fem_solver = ConvectionDiffusionFEM2D(current_mesh, self.pde)
        u_current = self.pde.evaluate_solution(current_mesh.nodes[:, 0], current_mesh.nodes[:, 1], 0.0)

        n_steps = int(round(T / dt))
        current_time = 0.0

        history = {
            "time": [0.0],
            "rel_l2_error": [],
            "rel_h1_error": [],
            "min_jacobian": [],
            "mean_quality": [],
            "inverted_cells": [],
            "alpha_global": [],
            "safety_activations": 0,
            "remap_error_l2": [],
            "timing_fem_ms": [],
            "timing_gnn_ms": [],
            "timing_safety_ms": [],
            "timing_remap_ms": [],
            "timing_total_step_ms": [],
            "mesh_snapshots": [(0.0, current_mesh.nodes.copy(), u_current.copy())]
        }

        l2_init, h1_init = compute_continuous_errors(current_mesh, u_current, self.pde, 0.0)
        history["rel_l2_error"].append(l2_init)
        history["rel_h1_error"].append(h1_init)

        geom_init = compute_mesh_metrics(current_mesh.nodes, current_mesh.cells)
        history["min_jacobian"].append(float(np.min(geom_init["jacobians"])))
        history["mean_quality"].append(float(np.mean(geom_init["escobar_quality"])))
        history["inverted_cells"].append(int(np.sum(geom_init["inverted"])))
        history["alpha_global"].append(1.0)

        t_sim_start = time.perf_counter()

        for step in range(1, n_steps + 1):
            t_step_start = time.perf_counter()
            t_next = step * dt
            time_fem = 0.0
            time_gnn = 0.0
            time_safety = 0.0
            time_remap = 0.0

            alpha_step = 1.0
            remap_err_step = 0.0

            if mode != "fixed" and (step % adapt_interval == 0):
                grad_mode = "exact" if mode == "gnn_exact" else "fem"
                grad_u = compute_field_gradients(
                    mesh=current_mesh,
                    u=u_current,
                    pde=self.pde,
                    t=current_time,
                    gradient_mode=grad_mode
                )

                if mode in ["gnn_fem", "gnn_exact"]:
                    pred_res = self.engine.predict(current_mesh, u_current, grad_u)
                    new_mesh = pred_res["adapted_mesh"]
                    alpha_step = float(pred_res.get("safety_alpha", pred_res.get("alpha", 1.0)))
                    time_gnn = pred_res["timing"]["gnn_time"] * 1000.0
                    time_safety = pred_res["timing"]["safety_time"] * 1000.0

                elif mode == "oracle":
                    t_or_start = time.perf_counter()
                    cx, cy = self.pde.get_center(current_time)
                    field_oracle = GaussianBumpField(
                        mu_x=cx, mu_y=cy, sigma=self.pde.sigma, amplitude=np.exp(-self.pde.beta * current_time)
                    )
                    opt = OfflineMeshOptimizer(
                        reference_field=field_oracle,
                        initial_nodes=current_mesh.nodes,
                        cells=current_mesh.cells,
                        boundary_nodes=current_mesh.boundary_nodes,
                        lambda_D=self.config["oracle"]["objective"]["lambda_D"],
                        lambda_Q=self.config["oracle"]["objective"]["lambda_Q"],
                        epsilon=self.epsilon,
                        max_iter=3
                    )
                    opt_res = opt.optimize()
                    time_gnn = (time.perf_counter() - t_or_start) * 1000.0
                    alpha_step = opt_res["final_metrics"]["safety_alpha_applied"]
                    new_mesh = current_mesh.update_coordinates(opt_res["optimal_nodes"])

                if alpha_step < 1.0:
                    history["safety_activations"] += 1

                # Re-proiezione di stato: P1 vs L2
                t_rem_start = time.perf_counter()
                if remap_method == "p1":
                    u_remapped = remap_p1_solution(
                        old_nodes=current_mesh.nodes,
                        old_cells=current_mesh.cells,
                        old_u=u_current,
                        new_nodes=new_mesh.nodes
                    )
                else:
                    u_remapped = remap_l2_projection(
                        old_nodes=current_mesh.nodes,
                        old_cells=current_mesh.cells,
                        old_u=u_current,
                        new_nodes=new_mesh.nodes,
                        new_cells=new_mesh.cells
                    )
                time_remap = (time.perf_counter() - t_rem_start) * 1000.0

                u_exact_new_nodes = self.pde.evaluate_solution(new_mesh.nodes[:, 0], new_mesh.nodes[:, 1], current_time)
                remap_err_step = float(np.linalg.norm(u_remapped - u_exact_new_nodes) / np.linalg.norm(u_exact_new_nodes))

                current_mesh = new_mesh
                u_current = u_remapped
                fem_solver = ConvectionDiffusionFEM2D(current_mesh, self.pde)

            t_fem_start = time.perf_counter()
            u_next = fem_solver.step_backward_euler(u_n=u_current, dt=dt, t_next=t_next)
            time_fem = (time.perf_counter() - t_fem_start) * 1000.0

            u_current = u_next
            current_time = t_next

            l2_err, h1_err = compute_continuous_errors(current_mesh, u_current, self.pde, current_time)
            geom_m = compute_mesh_metrics(current_mesh.nodes, current_mesh.cells)
            time_total_step = (time.perf_counter() - t_step_start) * 1000.0

            history["time"].append(current_time)
            history["rel_l2_error"].append(l2_err)
            history["rel_h1_error"].append(h1_err)
            history["min_jacobian"].append(float(np.min(geom_m["jacobians"])))
            history["mean_quality"].append(float(np.mean(geom_m["escobar_quality"])))
            history["inverted_cells"].append(int(np.sum(geom_m["inverted"])))
            history["alpha_global"].append(alpha_step)
            history["remap_error_l2"].append(remap_err_step)
            history["timing_fem_ms"].append(time_fem)
            history["timing_gnn_ms"].append(time_gnn)
            history["timing_safety_ms"].append(time_safety)
            history["timing_remap_ms"].append(time_remap)
            history["timing_total_step_ms"].append(time_total_step)

            if step in [int(round(n_steps / 3)), int(round(2 * n_steps / 3)), n_steps]:
                history["mesh_snapshots"].append((current_time, current_mesh.nodes.copy(), u_current.copy()))

        history["total_elapsed_simulation_s"] = time.perf_counter() - t_sim_start
        history["final_mesh"] = current_mesh
        history["final_solution"] = u_current
        return history