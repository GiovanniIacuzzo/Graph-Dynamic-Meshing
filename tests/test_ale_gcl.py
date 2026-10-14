"""
tests/test_ale_gcl.py
Test C1 (Mesh Motion Verification), Test C2 (Discrete GCL e Free-Stream Preservation)
e Test C2b (Discrete Mass Conservation su campi non uniformi).
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from generate_dataset import create_initial_mesh_2d
from geometry.mesh import TriangularMesh2D
from numerics.ale import ALESolver2D


def analytical_mesh_motion(
    nodes_0: np.ndarray,
    t: float,
    T: float = 0.5,
    A: float = 0.02,
) -> np.ndarray:
    """
    Legge di moto analitica regolare con deformazione identicamente nulla a t = 0:
        X(0) = X_0,   X(t)|_{dOmega} = X_0|_{dOmega}.
    """
    x = nodes_0[:, 0]
    y = nodes_0[:, 1]

    spatial = np.sin(np.pi * x) * np.sin(np.pi * y)
    omega = 2.0 * np.pi / T

    dx = A * spatial * np.sin(omega * t)
    dy = 0.5 * A * spatial * (np.cos(omega * t) - 1.0)

    nodes_t = nodes_0.copy()
    nodes_t[:, 0] += dx
    nodes_t[:, 1] += dy

    return nodes_t


def analytical_mesh_velocity(
    nodes_0: np.ndarray,
    t: float,
    T: float = 0.5,
    A: float = 0.02,
) -> np.ndarray:
    """Derivata temporale esatta dX/dt della legge analytical_mesh_motion."""
    x = nodes_0[:, 0]
    y = nodes_0[:, 1]

    spatial = np.sin(np.pi * x) * np.sin(np.pi * y)
    omega = 2.0 * np.pi / T

    wx = A * spatial * omega * np.cos(omega * t)
    wy = -0.5 * A * spatial * omega * np.sin(omega * t)

    return np.column_stack([wx, wy])


def test_c1_mesh_motion_verification():
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    mesh = create_initial_mesh_2d(domain, nx=32, ny=32)
    nodes_0 = mesh.nodes.copy()

    # Verifica che a t=0 la mesh coincida esattamente con quella iniziale
    nodes_at_zero = analytical_mesh_motion(nodes_0, 0.0, T=0.5, A=0.02)
    np.testing.assert_allclose(nodes_at_zero, nodes_0, atol=1e-15)

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
    print(f"Errore Max Linf Velocita'    : {err_linf:.4e}")

    assert err_l2 < 5e-3


def test_c2_discrete_gcl():
    """
    Test C2: Discrete GCL e preservazione del campo costante.

    Verifica indipendentemente:
      1. Residuo geometrico elemento per elemento: |(Area^{n+1} - Area^n)/dt - Area^{n+1/2} div(w^{n+1/2})|.
      2. Identità algebrica discreta D-GCL non-tautologica: ||((M^{n+1} - M^n)/dt + C_mesh^{n+1/2}) 1||_inf.
      3. Preservazione end-to-end del free-stream u = 1 durante 100 passi deformanti.
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

        # 1. Assemblaggio e verifica genuina della D-GCL discreta
        _, M_n, _, diag = ale_solver.assemble_ale_system(
            current_mesh, next_mesh, dt, t_next, pure_gcl=True
        )
        M_next = diag["M_next"]
        C_mesh = diag["C_mesh_mat"]

        ones_vec = np.ones(next_mesh.num_nodes, dtype=np.float64)

        # C_mesh è assemblato usando solo div(w^{n+1/2}) sul punto medio,
        # quindi il confronto con (M^{n+1} - M^n)/dt è un test indipendente:
        algebraic_residual_vec = ((M_next - M_n) @ ones_vec) / dt + C_mesh @ ones_vec
        step_algebraic_residual = float(np.linalg.norm(algebraic_residual_vec, ord=np.inf))

        max_algebraic_residual = max(max_algebraic_residual, step_algebraic_residual)
        max_elem_residual = max(max_elem_residual, float(diag["max_elem_gcl_residual"]))

        # 2. Test end-to-end su u = 1
        u_next, _ = ale_solver.step_ale(
            current_mesh, next_mesh, u_current, dt=dt, t_next=t_next, pure_gcl=True
        )

        if not np.all(np.isfinite(u_next)):
            raise AssertionError(f"Il solver ALE ha prodotto valori non finiti al passo {step}.")

        step_free_stream_error = float(np.max(np.abs(u_next - 1.0)))
        max_free_stream_error = max(max_free_stream_error, step_free_stream_error)

        u_current = u_next
        current_mesh = next_mesh

    print("\n" + "=" * 65)
    print("TEST C2 — DISCRETE GCL & FREE-STREAM PRESERVATION")
    print("=" * 65)
    print(f"Max Element-wise GCL Residual : {max_elem_residual:.4e}")
    print(f"Max Algebraic GCL Residual    : {max_algebraic_residual:.4e}")
    print(f"Max Free-Stream Error (u = 1) : {max_free_stream_error:.4e}")

    assert max_elem_residual < 1e-12, f"Residuo geometrico non nullo: {max_elem_residual:.3e}"
    assert max_algebraic_residual < 1e-12, f"Identità algebrica D-GCL non rispettata: {max_algebraic_residual:.3e}"
    assert max_free_stream_error < 1e-12, f"Free-stream non preservato: {max_free_stream_error:.3e}"


def test_c2b_discrete_mass_conservation():
    """
    Test C2b: Conservazione globale della massa discreta m_h^n = 1^T M^n U^n
    su un campo non uniforme sotto deformazione ALE della griglia (b=0, nu=0, f=0).

    Verifica:
      1. Identità duale di conservazione dell'operatore di mesh: ||1^T C_mesh^{n+1/2}||_inf ~ 0.
      2. Invarianza dell'integrale discreto m_h^n = int_{Omega(t_n)} u_h^n dx lungo 100 passi.
    """
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    mesh_0 = create_initial_mesh_2d(domain, nx=32, ny=32)
    nodes_0 = mesh_0.nodes.copy()

    ale_solver = ALESolver2D(pde_model=None, nu=0.0, bx=0.0, by=0.0)

    # Impulso gaussiano non uniforme centrato nel dominio
    x0, y0 = nodes_0[:, 0], nodes_0[:, 1]
    u_current = np.exp(-((x0 - 0.5) ** 2 + (y0 - 0.5) ** 2) / (2.0 * 0.08 ** 2))

    M_0 = ale_solver.assemble_mass_matrix(mesh_0)
    ones_vec = np.ones(mesh_0.num_nodes, dtype=np.float64)
    initial_mass = float(ones_vec @ (M_0 @ u_current))

    T = 0.5
    dt = 0.005
    n_steps = 100
    current_mesh = mesh_0

    max_dual_op_residual = 0.0
    max_mass_drift = 0.0

    for step in range(1, n_steps + 1):
        t_next = step * dt
        nodes_next = analytical_mesh_motion(nodes_0, t_next, T=T, A=0.02)
        next_mesh = mesh_0.update_coordinates(nodes_next)

        # Evoluzione conservativa pura senza forzatura di Dirichlet al bordo (w|_dOmega = 0)
        u_next, diag = ale_solver.step_ale(
            current_mesh,
            next_mesh,
            u_current,
            dt=dt,
            t_next=t_next,
            pure_gcl=True,
            enforce_dirichlet=False
        )

        C_mesh = diag["C_mesh_mat"]
        M_next = diag["M_next"]

        # 1. Verifica che 1^T C_mesh = 0^T (conservazione per qualunque campo U^{n+1})
        dual_res = float(np.max(np.abs(ones_vec @ C_mesh)))
        max_dual_op_residual = max(max_dual_op_residual, dual_res)

        # 2. Verifica che m_h^{n+1} = 1^T M^{n+1} U^{n+1} == m_h^0
        current_mass = float(ones_vec @ (M_next @ u_next))
        mass_drift = abs(current_mass - initial_mass) / abs(initial_mass)
        max_mass_drift = max(max_mass_drift, mass_drift)

        u_current = u_next
        current_mesh = next_mesh

    print("\n" + "=" * 65)
    print("TEST C2b — DISCRETE MASS CONSERVATION (NON-UNIFORM FIELD)")
    print("=" * 65)
    print(f"Initial Discrete Mass m_h^0     : {initial_mass:.12e}")
    print(f"Max Dual Operator ||1^T C_mesh||: {max_dual_op_residual:.4e}")
    print(f"Max Relative Mass Drift         : {max_mass_drift:.4e}")

    assert max_dual_op_residual < 1e-13, f"Operatore C_mesh non conservativo: {max_dual_op_residual:.3e}"
    assert max_mass_drift < 1e-12, f"Violazione conservazione della massa: {max_mass_drift:.3e}"