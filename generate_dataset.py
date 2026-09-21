"""
generate_dataset.py
Script di generazione offline del dataset per Graph-Dynamic Meshing (Stage A).

Pipeline di esecuzione:
    1. Lettura configurazione globale config/config.yaml
    2. Creazione della mesh iniziale triangolare non strutturata/uniforme
    3. Campionamento dei parametri fisici per le 4 partizioni:
       - train
       - validation
       - test_interp (interpolazione parametrica)
       - test_extrap (estrapolazione out-of-distribution)
    4. Valutazione del campo analitico e delle sue derivate
    5. Ottimizzazione offline variazionale dell'oracle -> determinazione di X* e Delta X*
    6. Validazione geometrica di ammissibilità (min J_e >= epsilon)
    7. Serializzazione dei campioni in data/raw/<sample_id>.pkl
    8. Creazione del manifest formale data/manifest.json
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Inclusione root nel path di sistema per esecuzione diretta
sys.path.append(str(Path(__file__).resolve().parent))

from geometry.mesh import TriangularMesh2D
from geometry.geometry import compute_mesh_metrics
from oracle.field import GaussianBumpField
from oracle.optimizer import OfflineMeshOptimizer
from utils.utils import load_config, ensure_dir, save_pickle, set_seed, setup_logger, Timer


def create_initial_mesh_2d(
    domain: Dict[str, float],
    nx: int = 32,
    ny: int = 32
) -> TriangularMesh2D:
    """
    Costruisce una triangolazione 2D conforme regolare sul dominio rettangolare.
    Suddivide ciascun rettangolo della griglia strutturata in due triangoli.
    """
    x = np.linspace(domain["x_min"], domain["x_max"], nx + 1, dtype=np.float64)
    y = np.linspace(domain["y_min"], domain["y_max"], ny + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    
    nodes = np.column_stack([xx.flatten(), yy.flatten()])
    
    cells = []
    for j in range(ny):
        for i in range(nx):
            # Indici dei 4 vertici della cella rettangolare
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = (j + 1) * (nx + 1) + i
            n3 = n2 + 1
            
            # Due triangoli orientati in senso antiorario
            cells.append([n0, n1, n3])
            cells.append([n0, n3, n2])
            
    cells = np.array(cells, dtype=np.int64)
    return TriangularMesh2D(nodes, cells)


def sample_parameters(config: Dict[str, Any], seed: int = 42) -> Dict[str, List[Dict[str, float]]]:
    """
    Genera i parametri fisici della Gaussian bump garantendo che i test set
    rispettino rigorosamente i requisiti di interpolazione ed estrapolazione.
    """
    rng = np.random.default_rng(seed)
    f_cfg = config["field"]["sampling_ranges"]
    splits_cfg = config["dataset"]["splits"]
    extrap_cfg = config["dataset"]["extrapolation_ranges"]

    sampled: Dict[str, List[Dict[str, float]]] = {
        "train": [],
        "validation": [],
        "test_interp": [],
        "test_extrap": []
    }

    # 1. Train, Validation e Test-Interp campionati all'interno del dominio standard
    for split_name in ["train", "validation", "test_interp"]:
        n_samples = splits_cfg[f"num_{split_name}"]
        for _ in range(n_samples):
            sampled[split_name].append({
                "mu_x": float(rng.uniform(f_cfg["mu_x_min"], f_cfg["mu_x_max"])),
                "mu_y": float(rng.uniform(f_cfg["mu_y_min"], f_cfg["mu_y_max"])),
                "sigma": float(rng.uniform(f_cfg["sigma_min"], f_cfg["sigma_max"])),
                "amplitude": float(rng.uniform(f_cfg["amplitude_min"], f_cfg["amplitude_max"]))
            })

    # 2. Test-Extrap: parametri campionati al di fuori del range (campi molto ripidi o molto dispersi)
    n_extrap = splits_cfg["num_test_extrap"]
    for i in range(n_extrap):
        if i % 2 == 0:
            # Regime a gradiente estremo (sigma molto piccolo)
            s_min, s_max = extrap_cfg["sigma_extrap_low"]
        else:
            # Regime diffuso (sigma molto grande)
            s_min, s_max = extrap_cfg["sigma_extrap_high"]

        sampled["test_extrap"].append({
            "mu_x": float(rng.uniform(f_cfg["mu_x_min"], f_cfg["mu_x_max"])),
            "mu_y": float(rng.uniform(f_cfg["mu_y_min"], f_cfg["mu_y_max"])),
            "sigma": float(rng.uniform(s_min, s_max)),
            "amplitude": float(rng.uniform(f_cfg["amplitude_min"], f_cfg["amplitude_max"]))
        })

    return sampled


def main():
    # 1. Caricamento configurazione e logger
    config_path = Path("config/config.yaml")
    assert config_path.exists(), f"File di configurazione mancante: {config_path.resolve()}"
    config = load_config(config_path)

    set_seed(config["experiment"]["seed"])
    logger = setup_logger(
        name="DatasetGenerator",
        log_file=Path(config["experiment"]["output_dir"]) / "dataset_generation.log"
    )
    logger.info("=== Avvio Pipeline Offline di Generazione Dataset ===")

    raw_dir = ensure_dir(config["dataset"]["raw_dir"])
    manifest_path = Path(config["dataset"]["manifest_path"])
    ensure_dir(manifest_path.parent)

    # 2. Creazione della mesh iniziale
    dom = config["geometry"]["domain"]
    grid_cfg = config["geometry"]["initial_grid"]
    logger.info(f"Generazione mesh iniziale 2D ({grid_cfg['nx']}x{grid_cfg['ny']})...")
    base_mesh = create_initial_mesh_2d(dom, nx=grid_cfg["nx"], ny=grid_cfg["ny"])
    logger.info(f"Mesh inizializzata: {base_mesh.num_nodes} nodi, {base_mesh.num_cells} celle triangolari.")

    # 3. Campionamento parametri per tutti gli split
    parameters_by_split = sample_parameters(config, seed=config["experiment"]["seed"])
    manifest: Dict[str, List[str]] = {
        "train": [],
        "validation": [],
        "test_interp": [],
        "test_extrap": []
    }

    sample_counter = 0
    total_samples = sum(len(samples) for samples in parameters_by_split.values())
    logger.info(f"Pianificata la generazione di {total_samples} campioni totali.")

    # Parametri operativi dell'oracle
    oracle_cfg = config["oracle"]
    geom_cfg = config["geometry"]["safety"]

    # 4. Loop di ottimizzazione offline e generazione dati
    with Timer() as total_timer:
        for split_name, params_list in parameters_by_split.items():
            logger.info(f"--- Inizio elaborazione split: '{split_name}' ({len(params_list)} campioni) ---")

            for p_idx, params in enumerate(params_list):
                sample_id = f"sample_{sample_counter:05d}_{split_name}"
                sample_counter += 1

                # Istanziazione campo continuo analitico
                field = GaussianBumpField(
                    mu_x=params["mu_x"],
                    mu_y=params["mu_y"],
                    sigma=params["sigma"],
                    amplitude=params["amplitude"]
                )

                # Valutazione campo e gradiente nodale iniziale
                u_init, grad_u_init, _, _ = field.evaluate(
                    base_mesh.nodes[:, 0],
                    base_mesh.nodes[:, 1]
                )

                # Ottimizzazione offline variazionale per il calcolo di X*
                optimizer = OfflineMeshOptimizer(
                    reference_field=field,
                    initial_nodes=base_mesh.nodes,
                    cells=base_mesh.cells,
                    boundary_nodes=base_mesh.boundary_nodes,
                    lambda_D=oracle_cfg["objective"]["lambda_D"],
                    lambda_Q=oracle_cfg["objective"]["lambda_Q"],
                    epsilon=geom_cfg["epsilon"],
                    max_iter=oracle_cfg["optimizer"]["max_iter"],
                    ftol=oracle_cfg["optimizer"]["ftol"],
                    gtol=oracle_cfg["optimizer"]["gtol"]
                )

                opt_result = optimizer.optimize()

                optimal_nodes = opt_result["optimal_nodes"]
                target_displacement = opt_result["target_displacement"]
                final_metrics = opt_result["final_metrics"]

                # Verifica formale di ammissibilità geometrica
                assert final_metrics["min_jacobian"] >= geom_cfg["epsilon"], (
                    f"Errore critico su {sample_id}: Jacobiano minimo "
                    f"{final_metrics['min_jacobian']:.3e} < epsilon"
                )
                assert final_metrics["num_inverted_cells"] == 0, (
                    f"Errore critico su {sample_id}: trovate celle invertite!"
                )

                # Composizione dizionario sample secondo il contratto formale
                sample_data = {
                    "mesh": {
                        "nodes": base_mesh.nodes,
                        "cells": base_mesh.cells
                    },
                    "state": {
                        "u": u_init,
                        "grad_u": grad_u_init
                    },
                    "target": {
                        "optimal_nodes": optimal_nodes,
                        "displacement": target_displacement
                    },
                    "metadata": {
                        "sample_id": sample_id,
                        "split": split_name,
                        "parameters": params,
                        "converged": opt_result["converged"],
                        "iterations": opt_result["iterations"],
                        "oracle_metrics": final_metrics
                    }
                }

                # Salvataggio binario del campione
                sample_file = raw_dir / f"{sample_id}.pkl"
                save_pickle(sample_data, sample_file)
                manifest[split_name].append(sample_id)

                if (p_idx + 1) % 10 == 0 or (p_idx + 1) == len(params_list):
                    logger.info(
                        f"[{split_name}] Progresso: {p_idx + 1}/{len(params_list)} | "
                        f"Ultimo J_min: {final_metrics['min_jacobian']:.4e} | "
                        f"Iter: {opt_result['iterations']}"
                    )

    # 5. Salvataggio del manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest registrato con successo in: {manifest_path.resolve()}")
    logger.info(
        f"=== Generazione Dataset Completata con successo in {total_timer.elapsed_seconds:.2f} s ==="
    )


if __name__ == "__main__":
    main()


"""
(gdm_env) giovanni02@MacBook-Air-del-Professore Graph-Dynamic-Meshing % python generate_dataset.py
[2026-09-20 11:25:39] [INFO] [DatasetGenerator] === Avvio Pipeline Offline di Generazione Dataset ===
[2026-09-20 11:25:39] [INFO] [DatasetGenerator] Generazione mesh iniziale 2D (32x32)...
[2026-09-20 11:25:39] [INFO] [DatasetGenerator] Mesh inizializzata: 1089 nodi, 2048 celle triangolari.
[2026-09-20 11:25:39] [INFO] [DatasetGenerator] Pianificata la generazione di 210 campioni totali.
[2026-09-20 11:25:39] [INFO] [DatasetGenerator] --- Inizio elaborazione split: 'train' (120 campioni) ---
[2026-09-20 11:26:43] [INFO] [DatasetGenerator] [train] Progresso: 10/120 | Ultimo J_min: 4.0913e-04 | Iter: 3
[2026-09-20 11:27:53] [INFO] [DatasetGenerator] [train] Progresso: 20/120 | Ultimo J_min: 4.1983e-04 | Iter: 3
[2026-09-20 11:29:03] [INFO] [DatasetGenerator] [train] Progresso: 30/120 | Ultimo J_min: 4.2427e-04 | Iter: 3
[2026-09-20 11:30:11] [INFO] [DatasetGenerator] [train] Progresso: 40/120 | Ultimo J_min: 4.7529e-04 | Iter: 3
[2026-09-20 11:31:21] [INFO] [DatasetGenerator] [train] Progresso: 50/120 | Ultimo J_min: 3.7334e-04 | Iter: 3
[2026-09-20 11:32:37] [INFO] [DatasetGenerator] [train] Progresso: 60/120 | Ultimo J_min: 3.0242e-04 | Iter: 3
[2026-09-20 11:33:54] [INFO] [DatasetGenerator] [train] Progresso: 70/120 | Ultimo J_min: 4.4781e-04 | Iter: 3
[2026-09-20 11:35:12] [INFO] [DatasetGenerator] [train] Progresso: 80/120 | Ultimo J_min: 4.3862e-04 | Iter: 3
[2026-09-20 11:36:33] [INFO] [DatasetGenerator] [train] Progresso: 90/120 | Ultimo J_min: 4.1290e-04 | Iter: 3
[2026-09-20 11:38:02] [INFO] [DatasetGenerator] [train] Progresso: 100/120 | Ultimo J_min: 4.1839e-04 | Iter: 3
[2026-09-20 11:39:25] [INFO] [DatasetGenerator] [train] Progresso: 110/120 | Ultimo J_min: 4.1362e-04 | Iter: 3
[2026-09-20 11:40:43] [INFO] [DatasetGenerator] [train] Progresso: 120/120 | Ultimo J_min: 4.7417e-04 | Iter: 3
[2026-09-20 11:40:43] [INFO] [DatasetGenerator] --- Inizio elaborazione split: 'validation' (30 campioni) ---
[2026-09-20 11:42:02] [INFO] [DatasetGenerator] [validation] Progresso: 10/30 | Ultimo J_min: 2.8279e-04 | Iter: 3
[2026-09-20 11:43:20] [INFO] [DatasetGenerator] [validation] Progresso: 20/30 | Ultimo J_min: 4.4668e-04 | Iter: 3
[2026-09-20 11:44:39] [INFO] [DatasetGenerator] [validation] Progresso: 30/30 | Ultimo J_min: 5.0995e-04 | Iter: 3
[2026-09-20 11:44:39] [INFO] [DatasetGenerator] --- Inizio elaborazione split: 'test_interp' (30 campioni) ---
[2026-09-20 11:45:58] [INFO] [DatasetGenerator] [test_interp] Progresso: 10/30 | Ultimo J_min: 5.1686e-04 | Iter: 3
[2026-09-20 11:47:18] [INFO] [DatasetGenerator] [test_interp] Progresso: 20/30 | Ultimo J_min: 3.6207e-04 | Iter: 3
[2026-09-20 11:48:38] [INFO] [DatasetGenerator] [test_interp] Progresso: 30/30 | Ultimo J_min: 4.4228e-04 | Iter: 3
[2026-09-20 11:48:38] [INFO] [DatasetGenerator] --- Inizio elaborazione split: 'test_extrap' (30 campioni) ---
[2026-09-20 11:50:08] [INFO] [DatasetGenerator] [test_extrap] Progresso: 10/30 | Ultimo J_min: 4.7344e-04 | Iter: 3
[2026-09-20 11:51:40] [INFO] [DatasetGenerator] [test_extrap] Progresso: 20/30 | Ultimo J_min: 2.6682e-04 | Iter: 1
[2026-09-20 11:53:05] [INFO] [DatasetGenerator] [test_extrap] Progresso: 30/30 | Ultimo J_min: 4.6220e-04 | Iter: 3
[2026-09-20 11:53:05] [INFO] [DatasetGenerator] Manifest registrato con successo in: /Users/giovanni02/Desktop/Progetti/Graph-Dynamic-Meshing/data/manifest.json
[2026-09-20 11:53:05] [INFO] [DatasetGenerator] === Generazione Dataset Completata con successo in 1646.08 s ===
(gdm_env) giovanni02@MacBook-Air-del-Professore Graph-Dynamic-Meshing %
"""