"""
tests/test_remap.py
Test di validazione rigorosa per l'operatore di remapping baricentrico P1.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from generate_dataset import create_initial_mesh_2d
from geometry.safety import apply_safety_filter
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.remap import remap_p1_solution


def _get_free_nodes_indices(mesh) -> np.ndarray:
    """Helper locale per isolare gli indici dei nodi interni liberi."""
    is_bnd = np.zeros(mesh.num_nodes, dtype=bool)
    is_bnd[mesh.boundary_nodes] = True
    return np.where(~is_bnd)[0]


def test_remap_r1_constant_preservation():
    """
    Test R1: Qualsiasi mesh deformata ammissibile DEVE preservare
    rigorosamente un campo costante u = c (qui c = 1.0).
    """
    mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=16, ny=16)
    N = mesh.num_nodes

    rng = np.random.default_rng(seed=42)
    raw_disp = np.zeros((N, 2), dtype=np.float64)
    free_idx = _get_free_nodes_indices(mesh)
    raw_disp[free_idx] = rng.uniform(-0.015, 0.015, size=(len(free_idx), 2))

    _, safe_nodes, is_safe, _ = apply_safety_filter(mesh.nodes, raw_disp, mesh.cells, epsilon=1e-4)
    assert is_safe

    old_u = np.ones(N, dtype=np.float64)
    u_remapped = remap_p1_solution(mesh.nodes, mesh.cells, old_u, safe_nodes)

    np.testing.assert_allclose(u_remapped, 1.0, atol=1e-13)


def test_remap_r2_linear_exactness():
    """
    Test R2: Il remapping P1 conforme su coordinate baricentriche
    DEVE essere esatto a precisione macchina su campi lineari u = x + 2y.
    """
    mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=16, ny=16)
    N = mesh.num_nodes

    rng = np.random.default_rng(seed=123)
    raw_disp = np.zeros((N, 2), dtype=np.float64)
    free_idx = _get_free_nodes_indices(mesh)
    raw_disp[free_idx] = rng.uniform(-0.02, 0.02, size=(len(free_idx), 2))

    _, safe_nodes, is_safe, _ = apply_safety_filter(mesh.nodes, raw_disp, mesh.cells, epsilon=1e-4)
    assert is_safe

    old_u = mesh.nodes[:, 0] + 2.0 * mesh.nodes[:, 1]
    u_remapped = remap_p1_solution(mesh.nodes, mesh.cells, old_u, safe_nodes)

    expected_new_u = safe_nodes[:, 0] + 2.0 * safe_nodes[:, 1]
    np.testing.assert_allclose(u_remapped, expected_new_u, atol=1e-12)


def test_remap_r3_manufactured_solution_multi_scale():
    """
    Test R3: Valuta l'errore del remapping sul campo nonlineare Manufactured
    a scale di spostamento crescente (0.25, 0.50, 1.00).
    Verifica:
      1. Monotonia dell'errore al crescere della deformazione della mesh
      2. Ordine di grandezza limitato da O(h^2 * ||nabla^2 u||) (rel_l2 < 0.05)
    """
    mesh = create_initial_mesh_2d({"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}, nx=32, ny=32)
    pde = ManufacturedConvectionDiffusion2D()
    t_fixed = 0.2
    u_exact_old = pde.evaluate_solution(mesh.nodes[:, 0], mesh.nodes[:, 1], t=t_fixed)

    _, ux, uy, _, _, _, _ = pde.evaluate_exact(mesh.nodes[:, 0], mesh.nodes[:, 1], t=t_fixed)
    grad_norm = np.maximum(np.sqrt(ux**2 + uy**2), 1e-6)
    base_disp = np.zeros_like(mesh.nodes)
    free_idx = _get_free_nodes_indices(mesh)
    base_disp[free_idx, 0] = 0.008 * (ux[free_idx] / grad_norm[free_idx])
    base_disp[free_idx, 1] = 0.008 * (uy[free_idx] / grad_norm[free_idx])

    errors_l2 = []
    max_errors = []

    for scale in [0.25, 0.5, 1.0]:
        disp = scale * base_disp
        _, safe_nodes, is_safe, _ = apply_safety_filter(mesh.nodes, disp, mesh.cells, epsilon=1e-4)
        assert is_safe

        u_remapped = remap_p1_solution(mesh.nodes, mesh.cells, u_exact_old, safe_nodes)
        u_exact_new = pde.evaluate_solution(safe_nodes[:, 0], safe_nodes[:, 1], t=t_fixed)

        rel_l2 = float(np.linalg.norm(u_remapped - u_exact_new) / np.linalg.norm(u_exact_new))
        max_err = float(np.max(np.abs(u_remapped - u_exact_new)))

        errors_l2.append(rel_l2)
        max_errors.append(max_err)

        # Su mesh 32x32 e sigma=0.06, l'errore del remapping P1 rimane rigorosamente sotto il 5%
        assert rel_l2 < 0.05, f"Errore L2 relativo anomalo alla scala {scale}: {rel_l2:.3e}"
        assert max_err < 0.08, f"Errore puntuale massimo anomalo alla scala {scale}: {max_err:.3e}"

    # Verifica della monotonia: deformazioni maggiori producono errori di interpolazione crescenti
    assert errors_l2[0] < errors_l2[1] < errors_l2[2], (
        f"L'errore di remapping non cresce monotonicamente con lo spostamento: {errors_l2}"
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])