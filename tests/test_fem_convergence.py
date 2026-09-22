"""
tests/test_fem_convergence.py
Studio formale e disaccoppiato della convergenza spaziale e temporale del solutore FEM P1.

Metodologia numerica:
    A) Convergenza Spaziale (h-refinement):
       - Nx = Ny in {8, 16, 32, 64}
       - Valutazione dell'errore continuo reale L2 e della seminorma H1
         tramite quadratura di Dunavant (grado 5, 7 punti per triangolo)
       - Bilanciamento parabolico dt = C * h^2 su orizzonte breve
       - Ordini asintotici attesi:
             p_L2 ~ 2.0,  p_H1 ~ 1.0

    B) Convergenza Temporale (dt-refinement via Self-Convergence):
       - Mesh fissata a Nx = Ny = 32
       - Soluzione di riferimento calcolata con dt_ref = 2.5e-4
       - Valutazione su dt in {8*dt_ref, 4*dt_ref, 2*dt_ref}
       - Ordine asintotico atteso per Backward Euler (BDF1):
             p_dt ~ 1.0
"""

import sys
import json
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from generate_dataset import create_initial_mesh_2d
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.fem import ConvectionDiffusionFEM2D, _get_triangle_quadrature_degree_5
from utils.utils import ensure_dir


def compute_continuous_fem_errors(
    fem: ConvectionDiffusionFEM2D,
    u_h: np.ndarray,
    pde: ManufacturedConvectionDiffusion2D,
    t: float
) -> tuple[float, float]:
    """
    Calcola l'errore continuo ||u_h - u_exact||_L2 e |u_h - u_exact|_H1
    integrando analiticamente su ogni triangolo con quadratura di Dunavant grado 5 (7 punti).
    """
    qp_ref, qw_ref = _get_triangle_quadrature_degree_5()
    cells = fem.cells
    nodes = fem.nodes
    areas = fem.areas

    x1, y1 = nodes[cells[:, 0], 0], nodes[cells[:, 0], 1]
    x2, y2 = nodes[cells[:, 1], 0], nodes[cells[:, 1], 1]
    x3, y3 = nodes[cells[:, 2], 0], nodes[cells[:, 2], 1]

    # Valori della soluzione FEM sui vertici [M, 3]
    u_elem = u_h[cells]
    # Gradienti costanti di u_h su ogni elemento: [M]
    duh_dx = np.sum(u_elem * fem.dphi_dx, axis=1)
    duh_dy = np.sum(u_elem * fem.dphi_dy, axis=1)

    l2_error_sq = 0.0
    h1_error_sq = 0.0

    for (xi, eta), w in zip(qp_ref, qw_ref):
        phi1 = 1.0 - xi - eta
        phi2 = xi
        phi3 = eta

        # Coordinate fisiche dei punti di quadratura su tutti i triangoli [M]
        xq = phi1 * x1 + phi2 * x2 + phi3 * x3
        yq = phi1 * y1 + phi2 * y2 + phi3 * y3

        # Soluzione esatta e sue derivate nei punti di quadratura
        u_ex, ux_ex, uy_ex, _, _, _, _ = pde.evaluate_exact(xq, yq, t)

        # Soluzione FEM P1 nel punto di quadratura
        u_fem_q = phi1 * u_elem[:, 0] + phi2 * u_elem[:, 1] + phi3 * u_elem[:, 2]

        # Differenze
        diff_u = u_fem_q - u_ex
        diff_gx = duh_dx - ux_ex
        diff_gy = duh_dy - uy_ex

        # Integrale: factor = 2 * w * Area
        factor = (2.0 * w) * areas
        l2_error_sq += np.sum(factor * (diff_u ** 2))
        h1_error_sq += np.sum(factor * (diff_gx ** 2 + diff_gy ** 2))

    return float(np.sqrt(max(l2_error_sq, 0.0))), float(np.sqrt(max(h1_error_sq, 0.0)))


def compute_convergence_rates(steps: list[float], errors: list[float]) -> list[float]:
    """Calcola i tassi empirici p = log(E1 / E2) / log(s1 / s2)."""
    rates = []
    for i in range(len(errors) - 1):
        rate = np.log(errors[i] / errors[i + 1]) / np.log(steps[i] / steps[i + 1])
        rates.append(float(rate))
    return rates


def test_fem_spatial_convergence():
    """
    Verifica l'ordine di convergenza spaziale h per elementi triangolari P1.
    """
    pde = ManufacturedConvectionDiffusion2D(
        bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=0.5
    )
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}

    resolutions = [8, 16, 32, 64]
    h_values = [1.0 / n for n in resolutions]
    t_end = 0.005

    errors_l2 = []
    errors_h1 = []

    for n in resolutions:
        h = 1.0 / n
        dt = 0.05 * (h ** 2)
        n_steps = max(int(round(t_end / dt)), 1)
        dt = t_end / n_steps

        mesh = create_initial_mesh_2d(domain, nx=n, ny=n)
        fem = ConvectionDiffusionFEM2D(mesh, pde)

        u_num = pde.evaluate_solution(mesh.nodes[:, 0], mesh.nodes[:, 1], t=0.0)

        current_t = 0.0
        for _ in range(n_steps):
            current_t += dt
            u_num = fem.step_backward_euler(u_n=u_num, dt=dt, t_next=current_t)

        l2_err, h1_err = compute_continuous_fem_errors(fem, u_num, pde, t=t_end)
        errors_l2.append(l2_err)
        errors_h1.append(h1_err)

    rates_l2 = compute_convergence_rates(h_values, errors_l2)
    rates_h1 = compute_convergence_rates(h_values, errors_h1)

    output_dir = ensure_dir("results/stage_b/convergence")
    with open(output_dir / "spatial_convergence.json", "w", encoding="utf-8") as f:
        json.dump({
            "resolutions": resolutions,
            "h_values": h_values,
            "errors_l2": errors_l2,
            "errors_h1": errors_h1,
            "rates_l2": rates_l2,
            "rates_h1": rates_h1
        }, f, indent=2)

    # Media asintotica sui due raffinamenti più fini (16->32, 32->64)
    mean_rate_l2 = float(np.mean(rates_l2[-2:]))
    mean_rate_h1 = float(np.mean(rates_h1[-2:]))

    # Criteri di accettazione teorici P1: L2 -> 2.0 (soglia 1.80), H1 -> 1.0 (soglia 0.90)
    assert mean_rate_l2 >= 1.80, f"Tasso spaziale L2 insufficiente: {mean_rate_l2:.2f} (atteso ~2.0, tassi: {rates_l2})"
    assert mean_rate_h1 >= 0.90, f"Tasso spaziale H1 insufficiente: {mean_rate_h1:.2f} (atteso ~1.0, tassi: {rates_h1})"


def test_fem_temporal_convergence():
    """
    Verifica l'ordine di convergenza temporale dt per BDF1 (Backward Euler)
    mediante analisi di Self-Convergence sulla stessa griglia spaziale.
    """
    pde = ManufacturedConvectionDiffusion2D(
        bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=0.5
    )
    domain = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}

    mesh = create_initial_mesh_2d(domain, nx=32, ny=32)
    fem = ConvectionDiffusionFEM2D(mesh, pde)

    t_end = 0.04
    dt_ref = 0.00025
    n_ref = int(round(t_end / dt_ref))

    # 1. Riferimento fine
    u_ref = pde.evaluate_solution(mesh.nodes[:, 0], mesh.nodes[:, 1], t=0.0)
    t_curr = 0.0
    for _ in range(n_ref):
        t_curr += dt_ref
        u_ref = fem.step_backward_euler(u_n=u_ref, dt=dt_ref, t_next=t_curr)

    # 2. Multipli
    dt_list = [0.008, 0.004, 0.002, 0.001]
    errors_dt = []

    for dt in dt_list:
        n_steps = int(round(t_end / dt))
        u_dt = pde.evaluate_solution(mesh.nodes[:, 0], mesh.nodes[:, 1], t=0.0)
        t_curr = 0.0
        for _ in range(n_steps):
            t_curr += dt
            u_dt = fem.step_backward_euler(u_n=u_dt, dt=dt, t_next=t_curr)

        diff = u_dt - u_ref
        err_l2 = float(np.sqrt(max(diff @ fem.M @ diff, 0.0)))
        errors_dt.append(err_l2)

    rates_dt = compute_convergence_rates(dt_list, errors_dt)

    output_dir = ensure_dir("results/stage_b/convergence")
    with open(output_dir / "temporal_convergence.json", "w", encoding="utf-8") as f:
        json.dump({
            "dt_list": dt_list,
            "errors_dt": errors_dt,
            "rates_dt": rates_dt
        }, f, indent=2)

    mean_rate_dt = float(np.mean(rates_dt[-2:]))

    # Criterio di accettazione BDF1: p -> 1.0 (tolleranza >= 0.85)
    assert mean_rate_dt >= 0.85, f"Tasso temporale BDF1 insufficiente: {mean_rate_dt:.2f} (atteso ~1.0, tassi: {rates_dt})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])