#!/usr/bin/env python3
"""
Diagnostica del trasferimento Stage A -> Stage C.

Confronta la mesh iniziale e la mesh predetta dalla GNN sulla
soluzione analitica della PDE MMS, senza avanzamento temporale.

Per ogni istante vengono confrontati:
    - errore di interpolazione P1 sulla mesh fissa;
    - errore di interpolazione P1 sulla mesh GNN;
    - spostamento nodale effettivo dopo il safety filter;
    - alpha globale di sicurezza;
    - Jacobiano minimo della mesh adattata.

Due varianti:
    1. Gradiente FEM, corrispondente alla pipeline Stage C.
    2. Gradiente analitico, come controllo diagnostico.

Il test non modifica checkpoint o dataset e non esegue training.
"""

import sys
from pathlib import Path

import numpy as np

# Consente l'esecuzione diretta dalla radice del repository.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.utils import load_config
from generate_dataset import create_initial_mesh_2d
from numerics.pde import ManufacturedConvectionDiffusion2D
from numerics.fem import compute_field_gradients
from numerics.dynamic_solver import compute_continuous_errors
from geometry.geometry import compute_mesh_metrics
from inference import MeshMoverInferenceEngine


def main():
    # ---------------------------------------------------------
    # 1. Configurazione coerente con Stage C
    # ---------------------------------------------------------
    config_path = ROOT / "config" / "config.yaml"
    checkpoint_path = (
        ROOT / "results" / "stage_a" / "checkpoints" / "best_model.pt"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configurazione non trovata: {config_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint non trovato: {checkpoint_path}"
        )

    config = load_config(str(config_path))

    domain = config["geometry"]["domain"]
    grid_cfg = config["geometry"]["initial_grid"]

    nx = int(grid_cfg["nx"])
    ny = int(grid_cfg["ny"])
    h = max(1.0 / nx, 1.0 / ny)

    # Stessi parametri della PDE definiti in stage_c_campaign.py.
    pde = ManufacturedConvectionDiffusion2D(
        bx=0.4,
        by=0.2,
        nu=1e-3,
        sigma=0.06,
        beta=0.5,
        T=0.5,
    )

    # La campagna Stage C originale usa CPU per l'inferenza.
    engine = MeshMoverInferenceEngine(
        checkpoint_path=checkpoint_path,
        config=config,
        device="cpu",
    )

    times = [0.0, 0.25, 0.5]

    print("=" * 100)
    print("DIAGNOSTICA DEL TRASFERIMENTO GNN STAGE A -> STAGE C")
    print("=" * 100)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mesh iniziale: {nx} x {ny}")
    print(f"Tempi analizzati: {times}")
    print()

    # ---------------------------------------------------------
    # 2. Mesh di riferimento uniforme
    # ---------------------------------------------------------
    reference_mesh = create_initial_mesh_2d(
        domain,
        nx=nx,
        ny=ny,
    )

    # ---------------------------------------------------------
    # 3. Diagnostica indipendente a ogni istante
    # ---------------------------------------------------------
    for t in times:

        print("-" * 100)
        print(f"TEMPO t = {t:.3f}")
        print("-" * 100)

        # Valori analitici ai nodi della mesh iniziale.
        u_fixed = pde.evaluate_solution(
            reference_mesh.nodes[:, 0],
            reference_mesh.nodes[:, 1],
            t,
        )

        # Errore della semplice interpolazione P1 sulla mesh fissa.
        err_fixed, _ = compute_continuous_errors(
            reference_mesh,
            u_fixed,
            pde,
            t,
        )

        # Gradiente ottenuto con lo stesso metodo della campagna C.
        grad_fem = compute_field_gradients(
            mesh=reference_mesh,
            u=u_fixed,
            pde=pde,
            t=t,
            gradient_mode="fem",
        )

        # Gradiente analitico, usato soltanto come confronto.
        _, ux_exact, uy_exact, _, _, _, _ = pde.evaluate_exact(
            reference_mesh.nodes[:, 0],
            reference_mesh.nodes[:, 1],
            t,
        )

        grad_exact = np.column_stack(
            [ux_exact, uy_exact]
        )

        cases = [
            ("FEM gradient", grad_fem),
            ("Exact gradient", grad_exact),
        ]

        print(
            f"{'Input gradient':<18}"
            f"{'E fixed [%]':>13}"
            f"{'E adapted [%]':>16}"
            f"{'Change [%]':>13}"
            f"{'alpha':>10}"
            f"{'RMS disp/h':>13}"
            f"{'min J':>13}"
        )

        for label, grad_u in cases:

            # Predizione GNN + safety filter.
            prediction = engine.predict(
                mesh=reference_mesh,
                u=u_fixed,
                grad_u=grad_u,
            )

            adapted_mesh = prediction["adapted_mesh"]

            # Campo esatto valutato ai nuovi nodi.
            u_adapted_exact = pde.evaluate_solution(
                adapted_mesh.nodes[:, 0],
                adapted_mesh.nodes[:, 1],
                t,
            )

            # Errore P1 continuo sulla mesh adattata.
            err_adapted, _ = compute_continuous_errors(
                adapted_mesh,
                u_adapted_exact,
                pde,
                t,
            )

            # Riduzione positiva = miglioramento.
            relative_change = (
                100.0
                * (err_fixed - err_adapted)
                / max(err_fixed, 1e-14)
            )

            # Spostamento realmente applicato dopo il safety filter.
            actual_disp = (
                adapted_mesh.nodes - reference_mesh.nodes
            )

            rms_displacement = float(
                np.sqrt(
                    np.mean(
                        np.sum(actual_disp**2, axis=1)
                    )
                )
            )

            displacement_over_h = rms_displacement / h

            geom = compute_mesh_metrics(
                adapted_mesh.nodes,
                adapted_mesh.cells,
            )

            min_j = float(np.min(geom["jacobians"]))
            alpha = float(prediction["alpha_global"])

            print(
                f"{label:<18}"
                f"{100.0 * err_fixed:>13.4f}"
                f"{100.0 * err_adapted:>16.4f}"
                f"{relative_change:>13.2f}"
                f"{alpha:>10.4f}"
                f"{displacement_over_h:>13.4e}"
                f"{min_j:>13.4e}"
            )

        print()


if __name__ == "__main__":
    main()