"""
tests/test_dynamic_one_step.py
Test di validazione unitaria dell'accoppiamento dinamico su singolo time-step.

Verifica la pipeline completa:
    u^n (MMS) -> FEM gradient -> GNN (frozen) -> Safety Filter -> Remap P1 -> Step FEM (t^{n+1})
Isolando quantitativamente ciascun contributo di errore.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import load_config, ensure_dir
from generate_dataset import create_initial_mesh_2d
from geometry.mesh import TriangularMesh2D
from geometry.geometry import compute_mesh_metrics
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.fem import ConvectionDiffusionFEM2D, compute_nodal_gradient, _get_triangle_quadrature_degree_5
from numerics.remap import remap_p1_solution
from inference import MeshMoverInferenceEngine


def compute_continuous_l2_error(
    mesh: TriangularMesh2D,
    u_h: np.ndarray,
    pde: ManufacturedConvectionDiffusion2D,
    t: float
) -> float:
    """Calcola l'errore continuo reale L2 integrando su ciascun triangolo con Dunavant grado 5."""
    qp_ref, qw_ref = _get_triangle_quadrature_degree_5()
    cells = mesh.cells
    nodes = mesh.nodes

    x1, y1 = nodes[cells[:, 0], 0], nodes[cells[:, 0], 1]
    x2, y2 = nodes[cells[:, 1], 0], nodes[cells[:, 1], 1]
    x3, y3 = nodes[cells[:, 2], 0], nodes[cells[:, 2], 1]

    det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    areas = 0.5 * np.abs(det_J)
    u_elem = u_h[cells]

    l2_sq = 0.0
    for (xi, eta), w in zip(qp_ref, qw_ref):
        phi1 = 1.0 - xi - eta
        phi2 = xi
        phi3 = eta

        xq = phi1 * x1 + phi2 * x2 + phi3 * x3
        yq = phi1 * y1 + phi2 * y2 + phi3 * y3

        u_ex = pde.evaluate_solution(xq, yq, t)
        u_fem_q = phi1 * u_elem[:, 0] + phi2 * u_elem[:, 1] + phi3 * u_elem[:, 2]

        factor = (2.0 * w) * areas
        l2_sq += np.sum(factor * ((u_fem_q - u_ex) ** 2))

    return float(np.sqrt(max(l2_sq, 0.0)))


def test_dynamic_one_step_coupling():
    cfg = load_config("config/config.yaml")
    checkpoint_path = Path("results/stage_a/checkpoints/best_model.pt")
    assert checkpoint_path.exists(), f"Checkpoint GNN non trovato: {checkpoint_path}"

    pde = ManufacturedConvectionDiffusion2D(
        bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=0.5
    )
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    mesh_0 = create_initial_mesh_2d(domain, nx=32, ny=32)

    t_n = 0.10
    dt = 0.005
    t_next = t_n + dt

    # 1. Stato iniziale discreto al tempo t_n
    u_n_nodes = pde.evaluate_solution(mesh_0.nodes[:, 0], mesh_0.nodes[:, 1], t_n)

    # 2. Calcolo gradiente nodale FEM numerico
    grad_u_fem = compute_nodal_gradient(mesh_0.nodes, mesh_0.cells, u_n_nodes)

    # 3. Inferenza GNN (Frozen) + Safety Filter integrato
    engine = MeshMoverInferenceEngine(checkpoint_path=checkpoint_path, config=cfg, device="cpu")
    pred_res = engine.predict(mesh_0, u_n_nodes, grad_u_fem)

    raw_disp = pred_res["raw_displacement"]
    mesh_new = pred_res["adapted_mesh"]
    alpha_safety = pred_res.get("safety_alpha", pred_res.get("alpha", 1.0))

    # 4. Verifica geometrica della nuova mesh
    geom_m = compute_mesh_metrics(mesh_new.nodes, mesh_new.cells)
    min_j = float(np.min(geom_m["jacobians"]))
    num_inv = int(np.sum(geom_m["inverted"]))

    assert num_inv == 0, f"Rilevate {num_inv} celle invertite dopo l'inferenza GNN e il Safety Filter!"
    assert min_j >= float(cfg["geometry"]["safety"]["epsilon"]) * 0.999

    # 5. Remap P1 dello stato u_n sulla nuova griglia
    u_n_remapped = remap_p1_solution(mesh_0.nodes, mesh_0.cells, u_n_nodes, mesh_new.nodes)
    assert not np.isnan(u_n_remapped).any()

    # Calcolo dell'errore puro di remapping rispetto alla soluzione esatta valutata sui nuovi nodi al tempo t_n
    u_exact_new_tn = pde.evaluate_solution(mesh_new.nodes[:, 0], mesh_new.nodes[:, 1], t_n)
    remap_l2_nodal = float(np.linalg.norm(u_n_remapped - u_exact_new_tn) / np.linalg.norm(u_exact_new_tn))

    # 6. Passo di avanzamento FEM Backward Euler sulla nuova mesh
    fem_dynamic = ConvectionDiffusionFEM2D(mesh_new, pde)
    u_next_dynamic = fem_dynamic.step_backward_euler(u_n=u_n_remapped, dt=dt, t_next=t_next)

    # Errore continuo PDE dopo il passo dinamico
    pde_error_dynamic = compute_continuous_l2_error(mesh_new, u_next_dynamic, pde, t_next)

    # 7. Baseline di controllo: Passo FEM eseguito su mesh FISSA (senza adattamento)
    fem_fixed = ConvectionDiffusionFEM2D(mesh_0, pde)
    u_next_fixed = fem_fixed.step_backward_euler(u_n=u_n_nodes, dt=dt, t_next=t_next)
    pde_error_fixed = compute_continuous_l2_error(mesh_0, u_next_fixed, pde, t_next)

    metrics = {
        "t_n": t_n,
        "dt": dt,
        "alpha_safety": alpha_safety,
        "min_jacobian": min_j,
        "inverted_cells": num_inv,
        "gnn_disp_norm": float(np.linalg.norm(raw_disp)),
        "gnn_disp_max": float(np.max(np.linalg.norm(raw_disp, axis=1))),
        "remap_l2_error_relative": remap_l2_nodal,
        "final_pde_l2_error_dynamic": pde_error_dynamic,
        "final_pde_l2_error_fixed": pde_error_fixed
    }

    output_dir = ensure_dir("results/stage_b/tests")
    with open(output_dir / "dynamic_one_step.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Asserzioni qualitative: il coupling non introduce instabilità o esplosioni dell'errore
    assert pde_error_dynamic < 0.05, f"Errore PDE dopo il passo dinamico troppo alto: {pde_error_dynamic:.3e}"
    assert remap_l2_nodal < 0.02, f"Errore remapping nodale troppo alto: {remap_l2_nodal:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])