"""
tests/test_ale_gcl.py
Test C1 (Mesh Motion Verification) e Test C2 (Discrete GCL e Free-Stream Preservation).
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from generate_dataset import create_initial_mesh_2d
from geometry.mesh import TriangularMesh2D
from numerics.ale import ALESolver2D


def analytical_mesh_motion(nodes_0: np.ndarray, t: float, T: float = 0.5, A: float = 0.02) -> np.ndarray:
    x = nodes_0[:, 0]
    y = nodes_0[:, 1]
    spatial = np.sin(np.pi * x) * np.sin(np.pi * y)
    dx = A * spatial * np.sin(2.0 * np.pi * t / T)
    dy = A * spatial * np.cos(2.0 * np.pi * t / T)
    nodes_t = nodes_0.copy()
    nodes_t[:, 0] += dx
    nodes_t[:, 1] += dy
    return nodes_t


def analytical_mesh_velocity(nodes_0: np.ndarray, t: float, T: float = 0.5, A: float = 0.02) -> np.ndarray:
    x = nodes_0[:, 0]
    y = nodes_0[:, 1]
    spatial = np.sin(np.pi * x) * np.sin(np.pi * y)
    omega = 2.0 * np.pi / T
    wx = A * spatial * omega * np.cos(omega * t)
    wy = -A * spatial * omega * np.sin(omega * t)
    return np.column_stack([wx, wy])


def test_c1_mesh_motion_verification():
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    mesh = create_initial_mesh_2d(domain, nx=32, ny=32)
    nodes_0 = mesh.nodes.copy()

    dt = 0.005
    T = 0.5
    t_n = 0.10
    t_next = t_n + dt

    nodes_n = analytical_mesh_motion(nodes_0, t_n, T=T, A=0.02)
    nodes_next = analytical_mesh_motion(nodes_0, t_next, T=T, A=0.02)

    w_discrete = (nodes_next - nodes_n) / dt
    w_exact = analytical_mesh_velocity(nodes_0, t_n + 0.5 * dt, T=T, A=0.02)

    err_l2 = float(np.linalg.norm(w_discrete - w_exact) / np.linalg.norm(w_exact))
    err_linf = float(np.max(np.abs(w_discrete - w_exact)))

    print("\n" + "=" * 65)
    print("TEST C1 — MESH MOTION VERIFICATION")
    print("=" * 65)
    print(f"Errore Relativo L2 Velocita' : {err_l2:.4e}")
    print(f"Errore Max Linf Velocita'     : {err_linf:.4e}")

    assert err_l2 < 5e-3


def test_c2_discrete_gcl():
    """
    Test C2: Discrete GCL rigorosa e Free-Stream Preservation:
      - Residuo geometrico element-wise R_e
      - Residuo algebrico ((M_next - M_n) @ 1)/dt + C(-w) @ 1 = 0
      - Conservazione nel tempo di u = 1 su 100 time-step
    """
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    mesh_0 = create_initial_mesh_2d(domain, nx=32, ny=32)
    nodes_0 = mesh_0.nodes.copy()

    ale_solver = ALESolver2D(pde_model=None, nu=0.0, bx=0.0, by=0.0)

    T = 0.5
    dt = 0.005
    n_steps = 100

    u_current = np.ones(mesh_0.num_nodes, dtype=np.float64)
    current_mesh = mesh_0

    max_elem_residual = 0.0
    max_algebraic_residual = 0.0
    max_free_stream_error = 0.0

    for step in range(1, n_steps + 1):
        t_next = step * dt
        nodes_next = analytical_mesh_motion(nodes_0, t_next, T=T, A=0.02)
        next_mesh = mesh_0.update_coordinates(nodes_next)

        # 1. Verifica algebrica dell'identita' D-GCL su u = 1
        _, M_n, _, diag = ale_solver.assemble_ale_system(current_mesh, next_mesh, dt, t_next, pure_gcl=True)
        M_next = ale_solver.assemble_mass_matrix(next_mesh)
        C_mesh = diag["C_mesh_mat"]

        ones_vec = np.ones(next_mesh.num_nodes, dtype=np.float64)
        alg_res_vec = ((M_next - M_n) @ ones_vec) / dt + C_mesh @ ones_vec
        step_alg_res = float(np.max(np.abs(alg_res_vec)))

        max_algebraic_residual = max(max_algebraic_residual, step_alg_res)
        max_elem_residual = max(max_elem_residual, diag["max_elem_gcl_residual"])

        # 2. Risoluzione temporale effettiva del passo conservativo
        u_next, _ = ale_solver.step_ale(current_mesh, next_mesh, u_current, dt=dt, t_next=t_next, pure_gcl=True)
        step_err = float(np.max(np.abs(u_next - 1.0)))
        max_free_stream_error = max(max_free_stream_error, step_err)

        u_current = u_next
        current_mesh = next_mesh

    print("\n" + "=" * 65)
    print("TEST C2 — DISCRETE GCL & FREE-STREAM PRESERVATION")
    print("=" * 65)
    print(f"Max Element-wise GCL Residual  : {max_elem_residual:.4e}")
    print(f"Max Algebraic GCL Identity Res : {max_algebraic_residual:.4e}")
    print(f"Max Free-Stream Error (u = 1)  : {max_free_stream_error:.4e}")

    # L'identita' deve essere rispettata e u = 1 deve essere conservata
    assert max_elem_residual < 1e-12, f"Residuo geometrico non nullo: {max_elem_residual:.3e}"
    assert max_free_stream_error < 1e-10, f"Free-stream non preservato: {max_free_stream_error:.3e}"