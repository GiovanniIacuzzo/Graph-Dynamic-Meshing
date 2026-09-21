"""
tests/test_safety_stress.py
Stress test per la validazione scientifica del Geometric Safety Filter analitico.

Dimostra l'attivazione del filtro scalando artificialmente le predizioni della GNN
    Delta X(k) = k * Delta X_GNN,   k in [1, 2, 3, 5, 10]
e verificando che alpha_global intervenga prevenendo inversioni (J_e >= epsilon).
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import load_config
from inference import MeshMoverInferenceEngine
from generate_dataset import create_initial_mesh_2d
from oracle.field import GaussianBumpField
from geometry.safety import apply_safety_filter
from geometry.geometry import compute_mesh_metrics


def run_safety_stress_test():
    cfg = load_config("config/config.yaml")
    checkpoint_path = Path("results/stage_a/checkpoints/best_model.pt")
    assert checkpoint_path.exists(), f"Checkpoint non trovato: {checkpoint_path}"

    engine = MeshMoverInferenceEngine(checkpoint_path=checkpoint_path, config=cfg)
    mesh = create_initial_mesh_2d(cfg["geometry"]["domain"], nx=32, ny=32)

    # Istanziazione di un campo con forte gradiente localizzato
    field = GaussianBumpField(mu_x=0.5, mu_y=0.5, sigma=0.035, amplitude=1.2)
    u, grad_u, _, _ = field.evaluate(mesh.nodes[:, 0], mesh.nodes[:, 1])

    # Predizione base della GNN
    pred_res = engine.predict(mesh, u, grad_u)
    base_disp = pred_res["predicted_displacement"]

    epsilon = float(cfg["geometry"]["safety"]["epsilon"])
    multipliers = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    print("\n" + "=" * 80)
    print(f"{'SAFETY FILTER STRESS TEST (Stage A)':^80}")
    print("=" * 80)
    header = f"{'Scale (k)':<10} | {'Raw J_min':<14} | {'Raw Inverted':<14} | {'alpha_global':<14} | {'Safe J_min':<14} | {'Filter Active':<12}"
    print(header)
    print("-" * len(header))

    for k in multipliers:
        disp_k = k * base_disp

        # Metriche senza filtro di sicurezza (Raw)
        nodes_raw = mesh.nodes + disp_k
        metrics_raw = compute_mesh_metrics(nodes_raw, mesh.cells)
        raw_min_j = float(np.min(metrics_raw["jacobians"]))
        raw_inverted = int(np.sum(metrics_raw["inverted"]))

        # Applicazione analitica del filtro O(M)
        alpha, nodes_safe, is_safe, safe_min_j = apply_safety_filter(
            nodes=mesh.nodes,
            displacement=disp_k,
            cells=mesh.cells,
            epsilon=epsilon
        )

        filter_active = "YES" if alpha < 1.0 else "NO"
        print(f"{k:<10.1f} | {raw_min_j:<+14.4e} | {raw_inverted:<14d} | {alpha:<14.6f} | {safe_min_j:<14.4e} | {filter_active:<12}")

        # Asserzioni matematiche rigorose
        assert safe_min_j >= epsilon * (1.0 - 1e-5), f"Violazione epsilon al fattore {k}: {safe_min_j} < {epsilon}"
        if raw_min_j < epsilon:
            assert alpha < 1.0, f"Il filtro non è intervenuto nonostante raw_min_j={raw_min_j} < epsilon"

    print("=" * len(header))
    print("Test completato con successo: Nessuna cella invertita e J_min >= epsilon preservato in ogni scenario.\n")


if __name__ == "__main__":
    run_safety_stress_test()