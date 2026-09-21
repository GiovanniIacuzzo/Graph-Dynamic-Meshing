"""
train.py
Entry point per l'addestramento supervisionato della GNN MeshMover in Graph-Dynamic Meshing.

Pipeline implementata:
    1. Caricamento parametri da config/config.yaml e fissaggio dei semi
    2. Creazione dei DataLoader per train e validation
    3. Inizializzazione del modello MeshMoverGNN, ottimizzatore AdamW e LR Scheduler
    4. Calcolo della loss ibrida:
       - L_target : errore quadratico medio (MSE) rispetto a Delta X* dell'oracle
       - L_smooth : regolarizzazione Laplaciana / energia di Dirichlet dello spostamento
       - L_barrier: barriera geometrica differenziabile sul determinante Jacobiano J_e
    5. Monitoraggio metriche di validazione, early stopping e salvataggio dei checkpoint
    6. Archiviazione della configurazione esatta per la riproducibilità scientifica
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Inclusione della radice nel path per esecuzione diretta
sys.path.append(str(Path(__file__).resolve().parent))

from model.model import MeshMoverGNN
from dataset.dataset import GraphDynamicMeshingDataset, collate_graph_batch
from evaluation.metrics import compute_displacement_metrics
from utils.utils import load_config, save_config, ensure_dir, set_seed, setup_logger, Timer


# =============================================================================
# DEFINIZIONE DELLE LOSS FUNCTIONS DIFFERENZIABILI
# =============================================================================

def compute_dirichlet_smoothness_loss(
    displacement: torch.Tensor,
    edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Penalizza gradienti spaziali eccessivi dello spostamento nodale (Laplacian smoothing):
        L_smooth = sum_{(i,j) in E} ||Delta X_i - Delta X_j||^2 / |E|
    """
    src, dst = edge_index[0], edge_index[1]
    diff = displacement[src] - displacement[dst]
    return torch.mean(torch.sum(diff**2, dim=-1))


def compute_adimensional_barrier_loss(
    nodes: torch.Tensor,
    displacement: torch.Tensor,
    cells: torch.Tensor,
    epsilon: float = 1e-4,
    delta_rel: float = 0.0,
    penalty_weight: float = 50.0
) -> torch.Tensor:
    """
    Penalità geometrica differenziabile per evitare configurazioni
    prossime all'inversione del mesh.

    La penalità è nulla quando:
        m_bar >= delta_rel

    e cresce quadraticamente quando:
        m_bar < delta_rel

    con:

        m_bar = (J_e(X + Delta X) - epsilon) / J_{e,0}

    In questo modo la loss geometrica agisce solo come vincolo
    di sicurezza e non incentiva il modello a mantenere il mesh
    quanto più possibile vicino alla configurazione iniziale.
    """

    # Determinante non deformato J_{e,0}
    x0_1, y0_1 = nodes[cells[:, 0], 0], nodes[cells[:, 0], 1]
    x0_2, y0_2 = nodes[cells[:, 1], 0], nodes[cells[:, 1], 1]
    x0_3, y0_3 = nodes[cells[:, 2], 0], nodes[cells[:, 2], 1]

    j0 = (
        (x0_2 - x0_1) * (y0_3 - y0_1)
        - (x0_3 - x0_1) * (y0_2 - y0_1)
    )

    j0 = torch.clamp(j0, min=1e-12)

    # Determinante deformato J_e
    x_new = nodes + displacement

    x1, y1 = x_new[cells[:, 0], 0], x_new[cells[:, 0], 1]
    x2, y2 = x_new[cells[:, 1], 0], x_new[cells[:, 1], 1]
    x3, y3 = x_new[cells[:, 2], 0], x_new[cells[:, 2], 1]

    jacobians = (
        (x2 - x1) * (y3 - y1)
        - (x3 - x1) * (y2 - y1)
    )

    # Margine geometrico adimensionale
    m_bar = (jacobians - epsilon) / j0

    # Penalizza esclusivamente le violazioni
    violation = torch.relu(delta_rel - m_bar)

    # Zero per mesh sicuri, quadratica per mesh non sicuri
    loss = penalty_weight * torch.mean(violation ** 2)

    return loss

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: Dict[str, float],
    epsilon: float,
    effective_w_barrier: float
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    l_target_accum = 0.0
    l_smooth_accum = 0.0
    l_barrier_accum = 0.0
    norm_ratio_accum = 0.0
    num_batches = len(loader)

    w_target = loss_weights.get("w_target", 1.0)
    w_smooth = loss_weights.get("w_smooth", 0.01)
    w_barrier = effective_w_barrier

    for batch in loader:
        node_feats = batch["node_features"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_feats = batch["edge_features"].to(device)
        target_disp = batch["target_displacement"].to(device)

        optimizer.zero_grad()
        pred_disp = model(node_feats, edge_index, edge_feats)

        # Mascheramento Dirichlet: forza a zero lo spostamento sui nodi di bordo
        is_bnd = node_feats[:, 5:6]
        pred_disp = pred_disp * (1.0 - is_bnd)

        l_target = torch.mean((pred_disp - target_disp) ** 2)
        l_smooth = compute_dirichlet_smoothness_loss(pred_disp, edge_index)

        l_barrier = torch.tensor(0.0, device=device)
        if w_barrier > 0.0:
            barrier_terms = []
            node_offset = 0
            for item in batch["raw_samples"]:
                n_count = item["nodes"].size(0)
                nodes_i = item["nodes"].to(device)
                cells_i = item["cells"].to(device)
                disp_i = pred_disp[node_offset : node_offset + n_count]
                barrier_terms.append(
                    compute_adimensional_barrier_loss(nodes_i, disp_i, cells_i, epsilon=epsilon)
                )
                node_offset += n_count
            l_barrier = torch.mean(torch.stack(barrier_terms))

        loss = w_target * l_target + w_smooth * l_smooth + w_barrier * l_barrier
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            ratio = (torch.norm(pred_disp) / torch.clamp(torch.norm(target_disp), min=1e-12)).item()

        total_loss += loss.item()
        l_target_accum += l_target.item()
        l_smooth_accum += l_smooth.item()
        l_barrier_accum += l_barrier.item()
        norm_ratio_accum += ratio

    return {
        "train_loss": total_loss / max(num_batches, 1),
        "train_l_target": l_target_accum / max(num_batches, 1),
        "train_l_smooth": l_smooth_accum / max(num_batches, 1),
        "train_l_barrier": l_barrier_accum / max(num_batches, 1),
        "train_norm_ratio": norm_ratio_accum / max(num_batches, 1)
    }


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_weights: Dict[str, float],
    epsilon: float,
    effective_w_barrier: float
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    l_target_accum = 0.0
    l_smooth_accum = 0.0
    l_barrier_accum = 0.0
    rel_errors = []
    l2_errors = []
    norm_ratios = []
    num_batches = len(loader)

    w_target = loss_weights.get("w_target", 1.0)
    w_smooth = loss_weights.get("w_smooth", 0.01)
    w_barrier = effective_w_barrier

    with torch.no_grad():
        for batch in loader:
            node_feats = batch["node_features"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_feats = batch["edge_features"].to(device)
            target_disp = batch["target_displacement"].to(device)

            pred_disp = model(node_feats, edge_index, edge_feats)
            is_bnd = node_feats[:, 5:6]
            pred_disp = pred_disp * (1.0 - is_bnd)

            l_target = torch.mean((pred_disp - target_disp) ** 2)
            l_smooth = compute_dirichlet_smoothness_loss(pred_disp, edge_index)

            l_barrier = torch.tensor(0.0, device=device)
            if w_barrier > 0.0:
                barrier_terms = []
                node_offset = 0
                for item in batch["raw_samples"]:
                    n_count = item["nodes"].size(0)
                    nodes_i = item["nodes"].to(device)
                    cells_i = item["cells"].to(device)
                    disp_i = pred_disp[node_offset : node_offset + n_count]
                    barrier_terms.append(
                        compute_adimensional_barrier_loss(nodes_i, disp_i, cells_i, epsilon=epsilon)
                    )
                    node_offset += n_count
                l_barrier = torch.mean(torch.stack(barrier_terms))

            loss = w_target * l_target + w_smooth * l_smooth + w_barrier * l_barrier
            total_loss += loss.item()
            l_target_accum += l_target.item()
            l_smooth_accum += l_smooth.item()
            l_barrier_accum += l_barrier.item()

            pred_np = pred_disp.cpu().numpy()
            target_np = target_disp.cpu().numpy()
            disp_m = compute_displacement_metrics(pred_np, target_np)
            rel_errors.append(disp_m["displacement_relative_error"])
            l2_errors.append(disp_m["displacement_l2_error"])

            ratio = np.linalg.norm(pred_np) / max(np.linalg.norm(target_np), 1e-12)
            norm_ratios.append(ratio)

    return {
        "val_loss": total_loss / max(num_batches, 1),
        "val_l_target": l_target_accum / max(num_batches, 1),
        "val_l_smooth": l_smooth_accum / max(num_batches, 1),
        "val_l_barrier": l_barrier_accum / max(num_batches, 1),
        "val_disp_rel_error": float(np.mean(rel_errors)),
        "val_disp_l2_error": float(np.mean(l2_errors)),
        "val_norm_ratio": float(np.mean(norm_ratios))
    }

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    config_path = Path("config/config.yaml")
    assert config_path.exists(), f"File di configurazione non trovato: {config_path.resolve()}"
    config = load_config(config_path)

    exp_cfg = config["experiment"]
    set_seed(exp_cfg["seed"], deterministic_torch=exp_cfg["deterministic"])

    output_dir = ensure_dir(exp_cfg["output_dir"])
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")

    logger = setup_logger(
        name="Trainer",
        log_file=output_dir / "training.log",
        level=exp_cfg.get("log_level", "INFO")
    )
    logger.info("=== Inizializzazione Pipeline di Training GNN (Stage A) ===")

    # Salvataggio tracciabile della configurazione esatta per l'esperimento
    save_config(config, output_dir / "reproduced_config.yaml")

    # 1. Selezione del Device
    device_name = exp_cfg.get("device", "cpu")
    if device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Dispositivo di calcolo attivo: Apple Silicon GPU (MPS)")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Dispositivo di calcolo attivo: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.info("Dispositivo di calcolo attivo: CPU")

    # 2. Inizializzazione Datasets e DataLoaders
    data_cfg = config["dataset"]
    train_dataset = GraphDynamicMeshingDataset(
        data_dir=data_cfg["raw_dir"],
        split="train"
    )
    val_dataset = GraphDynamicMeshingDataset(
        data_dir=data_cfg["raw_dir"],
        split="validation"
    )

    t_cfg = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=t_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_graph_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=t_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_graph_batch
    )

    logger.info(f"Dataset caricati: Train={len(train_dataset)} campioni, Val={len(val_dataset)} campioni.")

    # 3. Istanziazione del Modello
    m_cfg = config["model"]
    model = MeshMoverGNN(
        node_in_dim=m_cfg["node_in_dim"],
        edge_in_dim=m_cfg["edge_in_dim"],
        out_dim=m_cfg["out_dim"],
        latent_dim=m_cfg["latent_dim"],
        num_message_passing_steps=m_cfg["num_message_passing_steps"],
        mlp_num_layers=m_cfg["mlp_num_layers"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Modello {m_cfg['name']} inizializzato con {total_params:,} parametri addestrabili.")

    # 4. Ottimizzatore e Learning Rate Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(t_cfg["learning_rate"]),
        weight_decay=float(t_cfg.get("weight_decay", 1e-5))
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_cfg["epochs"],
        eta_min=float(t_cfg.get("lr_scheduler", {}).get("eta_min", 1e-6))
    )

    # 5. Loop di Addestramento
    best_val_loss = float("inf")
    patience_counter = 0
    patience = t_cfg.get("early_stopping_patience", 25)
    epsilon = config["geometry"]["safety"]["epsilon"]
    loss_weights = t_cfg.get("loss_weights", {})
    warmup_epochs = int(t_cfg.get("barrier_warmup_epochs", 20))
    ramp_epochs = int(t_cfg.get("barrier_ramp_epochs", 20))
    base_w_barrier = float(loss_weights.get("w_barrier", 0.0))

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_disp_rel_error": [],
        "lr": []
    }

    logger.info(f"Inizio ciclo di addestramento per {t_cfg['epochs']} epoche...")

    with Timer() as total_train_timer:
        for epoch in range(1, t_cfg["epochs"] + 1):
            t_epoch_start = time.perf_counter()

            if epoch <= warmup_epochs:
                effective_w_barrier = 0.0
            else:
                ramp_progress = min(
                    (epoch - warmup_epochs) / max(ramp_epochs, 1),
                    1.0
                )
                effective_w_barrier = base_w_barrier * ramp_progress

            # Train pass
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                loss_weights=loss_weights,
                epsilon=epsilon,
                effective_w_barrier=effective_w_barrier
            )

            # Validation pass
            val_metrics = validate_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                loss_weights=loss_weights,
                epsilon=epsilon,
                effective_w_barrier=effective_w_barrier
            )

            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            t_epoch = time.perf_counter() - t_epoch_start

            # Registrazione storia
            history["epochs"].append(epoch)
            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_disp_rel_error"].append(val_metrics["val_disp_rel_error"])
            history["lr"].append(current_lr)

            # Logging periodico
            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoca [{epoch:03d}/{t_cfg['epochs']:03d}] ({t_epoch:.1f}s) | "
                    f"L_tot: {train_metrics['train_loss']:.3e} "
                    f"(Tgt: {train_metrics['train_l_target']:.3e}, "
                    f"Sm: {train_metrics['train_l_smooth']:.3e}, "
                    f"Bar: {train_metrics['train_l_barrier']:.3e}, "
                    f"BarW: {effective_w_barrier:.2e}) | "
                    f"Val Tgt: {val_metrics['val_l_target']:.3e} | "
                    f"Val RelErr: {val_metrics['val_disp_rel_error']:.2%} | "
                    f"||dX||/||dX*||: {val_metrics['val_norm_ratio']:.2f}"
                )

            # Checkpoint del modello migliore
            val_loss = val_metrics["val_l_target"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                    "config": config
                }
                torch.save(best_checkpoint, checkpoints_dir / "best_model.pt")
                logger.info(f"--> [✓] Nuovo miglior checkpoint salvato all'epoca {epoch} (Val Loss: {val_loss:.4e})")
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping attivato all'epoca {epoch}: nessun miglioramento da {patience} epoche.")
                break

    # Salvataggio checkpoint finale e telemetria complessiva
    final_checkpoint = {
        "epoch": len(history["epochs"]),
        "model_state_dict": model.state_dict(),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "config": config
    }
    torch.save(final_checkpoint, checkpoints_dir / "final_model.pt")

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info(f"=== Addestramento concluso in {total_train_timer.elapsed_seconds:.2f} s ===")
    logger.info(f"Miglior modello salvato in: {(checkpoints_dir / 'best_model.pt').resolve()}")


if __name__ == "__main__":
    main()

"""
(gdm_env) giovanni02@MacBook-Air-del-Professore Graph-Dynamic-Meshing % python train.py
[2026-09-20 14:35:51] [INFO] [Trainer] === Inizializzazione Pipeline di Training GNN (Stage A) ===
[2026-09-20 14:35:51] [INFO] [Trainer] Dispositivo di calcolo attivo: Apple Silicon GPU (MPS)
[2026-09-20 14:35:51] [INFO] [Trainer] Dataset caricati: Train=120 campioni, Val=30 campioni.
[2026-09-20 14:35:51] [INFO] [Trainer] Modello MeshMoverGNN inizializzato con 749,314 parametri addestrabili.
[2026-09-20 14:35:52] [INFO] [Trainer] Inizio ciclo di addestramento per 150 epoche...
[2026-09-20 14:36:01] [INFO] [Trainer] Epoca [001/150] (9.8s) | L_tot: 3.010e+00 (Tgt: 9.126e-02, Sm: 1.876e-02, Bar: 2.918e+03) | Val RelErr: 2044.28% | ||dX||/||dX*||: 20.41
[2026-09-20 14:36:02] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 1 (Val Loss: 1.5055e-03)
[2026-09-20 14:36:14] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 3 (Val Loss: 5.5131e-04)
[2026-09-20 14:36:21] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 4 (Val Loss: 1.5501e-04)
[2026-09-20 14:36:27] [INFO] [Trainer] Epoca [005/150] (6.3s) | L_tot: 1.633e-04 (Tgt: 4.866e-05, Sm: 1.289e-05, Bar: 1.146e-01) | Val RelErr: 328.48% | ||dX||/||dX*||: 3.13
[2026-09-20 14:36:27] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 5 (Val Loss: 1.3814e-04)
[2026-09-20 14:36:34] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 6 (Val Loss: 1.3585e-04)
[2026-09-20 14:36:40] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 7 (Val Loss: 1.3247e-04)
[2026-09-20 14:36:47] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 8 (Val Loss: 1.3020e-04)
[2026-09-20 14:36:53] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 9 (Val Loss: 1.2851e-04)
[2026-09-20 14:36:59] [INFO] [Trainer] Epoca [010/150] (6.3s) | L_tot: 1.276e-04 (Tgt: 1.605e-05, Sm: 5.473e-06, Bar: 1.115e-01) | Val RelErr: 259.32% | ||dX||/||dX*||: 2.40
[2026-09-20 14:36:59] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 10 (Val Loss: 1.2692e-04)
[2026-09-20 14:37:17] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 11 (Val Loss: 1.2547e-04)
[2026-09-20 14:37:33] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 12 (Val Loss: 1.2416e-04)
[2026-09-20 14:37:45] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 13 (Val Loss: 1.2303e-04)
[2026-09-20 14:37:53] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 14 (Val Loss: 1.2189e-04)
[2026-09-20 14:38:02] [INFO] [Trainer] Epoca [015/150] (8.4s) | L_tot: 1.212e-04 (Tgt: 1.090e-05, Sm: 3.522e-06, Bar: 1.103e-01) | Val RelErr: 214.70% | ||dX||/||dX*||: 1.91
[2026-09-20 14:38:02] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 15 (Val Loss: 1.2090e-04)
[2026-09-20 14:38:10] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 16 (Val Loss: 1.2000e-04)
[2026-09-20 14:38:17] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 17 (Val Loss: 1.1922e-04)
[2026-09-20 14:38:24] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 18 (Val Loss: 1.1846e-04)
[2026-09-20 14:38:31] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 19 (Val Loss: 1.1782e-04)
[2026-09-20 14:38:38] [INFO] [Trainer] Epoca [020/150] (6.8s) | L_tot: 1.173e-04 (Tgt: 7.742e-06, Sm: 2.325e-06, Bar: 1.095e-01) | Val RelErr: 182.11% | ||dX||/||dX*||: 1.55
[2026-09-20 14:38:38] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 20 (Val Loss: 1.1717e-04)
[2026-09-20 14:38:45] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 21 (Val Loss: 1.1662e-04)
[2026-09-20 14:38:52] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 22 (Val Loss: 1.1611e-04)
[2026-09-20 14:39:00] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 23 (Val Loss: 1.1562e-04)
[2026-09-20 14:39:07] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 24 (Val Loss: 1.1521e-04)
[2026-09-20 14:39:13] [INFO] [Trainer] Epoca [025/150] (6.9s) | L_tot: 1.148e-04 (Tgt: 5.768e-06, Sm: 1.583e-06, Bar: 1.091e-01) | Val RelErr: 158.01% | ||dX||/||dX*||: 1.27
[2026-09-20 14:39:14] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 25 (Val Loss: 1.1481e-04)
[2026-09-20 14:39:21] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 26 (Val Loss: 1.1447e-04)
[2026-09-20 14:39:28] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 27 (Val Loss: 1.1416e-04)
[2026-09-20 14:39:35] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 28 (Val Loss: 1.1384e-04)
[2026-09-20 14:39:42] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 29 (Val Loss: 1.1359e-04)
[2026-09-20 14:39:49] [INFO] [Trainer] Epoca [030/150] (7.0s) | L_tot: 1.133e-04 (Tgt: 4.531e-06, Sm: 1.123e-06, Bar: 1.088e-01) | Val RelErr: 140.86% | ||dX||/||dX*||: 1.06
[2026-09-20 14:39:49] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 30 (Val Loss: 1.1332e-04)
[2026-09-20 14:39:56] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 31 (Val Loss: 1.1311e-04)
[2026-09-20 14:40:04] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 32 (Val Loss: 1.1290e-04)
[2026-09-20 14:40:11] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 33 (Val Loss: 1.1268e-04)
[2026-09-20 14:40:18] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 34 (Val Loss: 1.1251e-04)
[2026-09-20 14:40:25] [INFO] [Trainer] Epoca [035/150] (7.3s) | L_tot: 1.123e-04 (Tgt: 3.708e-06, Sm: 8.260e-07, Bar: 1.086e-01) | Val RelErr: 128.27% | ||dX||/||dX*||: 0.89
[2026-09-20 14:40:25] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 35 (Val Loss: 1.1235e-04)
[2026-09-20 14:40:33] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 36 (Val Loss: 1.1222e-04)
[2026-09-20 14:40:40] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 37 (Val Loss: 1.1208e-04)
[2026-09-20 14:40:47] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 38 (Val Loss: 1.1194e-04)
[2026-09-20 14:40:54] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 39 (Val Loss: 1.1182e-04)
[2026-09-20 14:41:02] [INFO] [Trainer] Epoca [040/150] (7.8s) | L_tot: 1.116e-04 (Tgt: 3.185e-06, Sm: 6.378e-07, Bar: 1.084e-01) | Val RelErr: 119.56% | ||dX||/||dX*||: 0.77
[2026-09-20 14:41:02] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 40 (Val Loss: 1.1173e-04)
[2026-09-20 14:41:10] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 41 (Val Loss: 1.1161e-04)
[2026-09-20 14:41:17] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 42 (Val Loss: 1.1154e-04)
[2026-09-20 14:41:24] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 43 (Val Loss: 1.1144e-04)
[2026-09-20 14:41:32] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 44 (Val Loss: 1.1137e-04)
[2026-09-20 14:41:40] [INFO] [Trainer] Epoca [045/150] (7.8s) | L_tot: 1.112e-04 (Tgt: 2.840e-06, Sm: 5.140e-07, Bar: 1.083e-01) | Val RelErr: 113.60% | ||dX||/||dX*||: 0.69
[2026-09-20 14:41:40] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 45 (Val Loss: 1.1132e-04)
[2026-09-20 14:41:48] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 46 (Val Loss: 1.1123e-04)
[2026-09-20 14:41:55] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 47 (Val Loss: 1.1117e-04)
[2026-09-20 14:42:03] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 48 (Val Loss: 1.1112e-04)
[2026-09-20 14:42:11] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 49 (Val Loss: 1.1107e-04)
[2026-09-20 14:42:18] [INFO] [Trainer] Epoca [050/150] (7.5s) | L_tot: 1.109e-04 (Tgt: 2.602e-06, Sm: 4.288e-07, Bar: 1.083e-01) | Val RelErr: 108.65% | ||dX||/||dX*||: 0.61
[2026-09-20 14:42:18] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 50 (Val Loss: 1.1102e-04)
[2026-09-20 14:42:26] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 51 (Val Loss: 1.1096e-04)
[2026-09-20 14:42:33] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 52 (Val Loss: 1.1092e-04)
[2026-09-20 14:42:41] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 53 (Val Loss: 1.1088e-04)
[2026-09-20 14:42:49] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 54 (Val Loss: 1.1085e-04)
[2026-09-20 14:42:58] [INFO] [Trainer] Epoca [055/150] (8.4s) | L_tot: 1.107e-04 (Tgt: 2.435e-06, Sm: 3.703e-07, Bar: 1.082e-01) | Val RelErr: 105.21% | ||dX||/||dX*||: 0.56
[2026-09-20 14:42:58] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 55 (Val Loss: 1.1081e-04)
[2026-09-20 14:43:06] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 56 (Val Loss: 1.1079e-04)
[2026-09-20 14:43:15] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 57 (Val Loss: 1.1075e-04)
[2026-09-20 14:43:23] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 58 (Val Loss: 1.1072e-04)
[2026-09-20 14:43:31] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 59 (Val Loss: 1.1070e-04)
[2026-09-20 14:43:39] [INFO] [Trainer] Epoca [060/150] (8.0s) | L_tot: 1.105e-04 (Tgt: 2.325e-06, Sm: 3.292e-07, Bar: 1.082e-01) | Val RelErr: 102.95% | ||dX||/||dX*||: 0.52
[2026-09-20 14:43:39] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 60 (Val Loss: 1.1067e-04)
[2026-09-20 14:43:47] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 61 (Val Loss: 1.1065e-04)
[2026-09-20 14:43:55] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 62 (Val Loss: 1.1064e-04)
[2026-09-20 14:44:03] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 63 (Val Loss: 1.1061e-04)
[2026-09-20 14:44:11] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 64 (Val Loss: 1.1059e-04)
[2026-09-20 14:44:19] [INFO] [Trainer] Epoca [065/150] (8.0s) | L_tot: 1.104e-04 (Tgt: 2.243e-06, Sm: 2.984e-07, Bar: 1.082e-01) | Val RelErr: 101.56% | ||dX||/||dX*||: 0.49
[2026-09-20 14:44:19] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 65 (Val Loss: 1.1058e-04)
[2026-09-20 14:44:27] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 66 (Val Loss: 1.1056e-04)
[2026-09-20 14:44:35] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 67 (Val Loss: 1.1054e-04)
[2026-09-20 14:44:43] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 68 (Val Loss: 1.1053e-04)
[2026-09-20 14:44:51] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 69 (Val Loss: 1.1053e-04)
[2026-09-20 14:44:59] [INFO] [Trainer] Epoca [070/150] (7.7s) | L_tot: 1.104e-04 (Tgt: 2.186e-06, Sm: 2.752e-07, Bar: 1.082e-01) | Val RelErr: 100.05% | ||dX||/||dX*||: 0.46
[2026-09-20 14:44:59] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 70 (Val Loss: 1.1050e-04)
[2026-09-20 14:45:07] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 71 (Val Loss: 1.1049e-04)
[2026-09-20 14:45:15] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 72 (Val Loss: 1.1048e-04)
[2026-09-20 14:45:32] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 74 (Val Loss: 1.1046e-04)
[2026-09-20 14:45:40] [INFO] [Trainer] Epoca [075/150] (8.2s) | L_tot: 1.103e-04 (Tgt: 2.144e-06, Sm: 2.578e-07, Bar: 1.082e-01) | Val RelErr: 99.14% | ||dX||/||dX*||: 0.45
[2026-09-20 14:45:40] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 75 (Val Loss: 1.1045e-04)
[2026-09-20 14:45:49] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 76 (Val Loss: 1.1044e-04)
[2026-09-20 14:45:57] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 77 (Val Loss: 1.1044e-04)
[2026-09-20 14:46:05] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 78 (Val Loss: 1.1043e-04)
[2026-09-20 14:46:13] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 79 (Val Loss: 1.1042e-04)
[2026-09-20 14:46:20] [INFO] [Trainer] Epoca [080/150] (7.8s) | L_tot: 1.103e-04 (Tgt: 2.112e-06, Sm: 2.437e-07, Bar: 1.082e-01) | Val RelErr: 98.57% | ||dX||/||dX*||: 0.43
[2026-09-20 14:46:21] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 80 (Val Loss: 1.1042e-04)
[2026-09-20 14:46:28] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 81 (Val Loss: 1.1040e-04)
[2026-09-20 14:46:36] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 82 (Val Loss: 1.1040e-04)
[2026-09-20 14:46:44] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 83 (Val Loss: 1.1039e-04)
[2026-09-20 14:46:52] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 84 (Val Loss: 1.1039e-04)
[2026-09-20 14:47:00] [INFO] [Trainer] Epoca [085/150] (8.2s) | L_tot: 1.102e-04 (Tgt: 2.089e-06, Sm: 2.329e-07, Bar: 1.082e-01) | Val RelErr: 97.99% | ||dX||/||dX*||: 0.42
[2026-09-20 14:47:00] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 85 (Val Loss: 1.1038e-04)
[2026-09-20 14:47:16] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 87 (Val Loss: 1.1038e-04)
[2026-09-20 14:47:24] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 88 (Val Loss: 1.1037e-04)
[2026-09-20 14:47:32] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 89 (Val Loss: 1.1037e-04)
[2026-09-20 14:47:40] [INFO] [Trainer] Epoca [090/150] (7.7s) | L_tot: 1.102e-04 (Tgt: 2.073e-06, Sm: 2.243e-07, Bar: 1.082e-01) | Val RelErr: 97.73% | ||dX||/||dX*||: 0.42
[2026-09-20 14:47:40] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 90 (Val Loss: 1.1037e-04)
[2026-09-20 14:47:48] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 91 (Val Loss: 1.1036e-04)
[2026-09-20 14:47:56] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 92 (Val Loss: 1.1036e-04)
[2026-09-20 14:48:04] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 93 (Val Loss: 1.1035e-04)
[2026-09-20 14:48:12] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 94 (Val Loss: 1.1035e-04)
[2026-09-20 14:48:19] [INFO] [Trainer] Epoca [095/150] (7.8s) | L_tot: 1.102e-04 (Tgt: 2.058e-06, Sm: 2.172e-07, Bar: 1.081e-01) | Val RelErr: 97.31% | ||dX||/||dX*||: 0.41
[2026-09-20 14:48:20] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 95 (Val Loss: 1.1034e-04)
[2026-09-20 14:48:28] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 96 (Val Loss: 1.1034e-04)
[2026-09-20 14:48:51] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 99 (Val Loss: 1.1033e-04)
[2026-09-20 14:48:59] [INFO] [Trainer] Epoca [100/150] (7.6s) | L_tot: 1.102e-04 (Tgt: 2.052e-06, Sm: 2.120e-07, Bar: 1.081e-01) | Val RelErr: 97.12% | ||dX||/||dX*||: 0.40
[2026-09-20 14:48:59] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 100 (Val Loss: 1.1033e-04)
[2026-09-20 14:49:07] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 101 (Val Loss: 1.1033e-04)
[2026-09-20 14:49:22] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 103 (Val Loss: 1.1032e-04)
[2026-09-20 14:49:30] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 104 (Val Loss: 1.1032e-04)
[2026-09-20 14:49:38] [INFO] [Trainer] Epoca [105/150] (7.7s) | L_tot: 1.102e-04 (Tgt: 2.041e-06, Sm: 2.069e-07, Bar: 1.081e-01) | Val RelErr: 96.91% | ||dX||/||dX*||: 0.40
[2026-09-20 14:49:38] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 105 (Val Loss: 1.1032e-04)
[2026-09-20 14:50:01] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 108 (Val Loss: 1.1032e-04)
[2026-09-20 14:50:09] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 109 (Val Loss: 1.1032e-04)
[2026-09-20 14:50:17] [INFO] [Trainer] Epoca [110/150] (7.8s) | L_tot: 1.102e-04 (Tgt: 2.036e-06, Sm: 2.035e-07, Bar: 1.081e-01) | Val RelErr: 96.78% | ||dX||/||dX*||: 0.39
[2026-09-20 14:50:17] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 110 (Val Loss: 1.1031e-04)
[2026-09-20 14:50:40] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 113 (Val Loss: 1.1031e-04)
[2026-09-20 14:50:48] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 114 (Val Loss: 1.1031e-04)
[2026-09-20 14:50:56] [INFO] [Trainer] Epoca [115/150] (7.9s) | L_tot: 1.102e-04 (Tgt: 2.031e-06, Sm: 2.005e-07, Bar: 1.081e-01) | Val RelErr: 96.68% | ||dX||/||dX*||: 0.39
[2026-09-20 14:50:56] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 115 (Val Loss: 1.1031e-04)
[2026-09-20 14:51:04] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 116 (Val Loss: 1.1031e-04)
[2026-09-20 14:51:12] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 117 (Val Loss: 1.1031e-04)
[2026-09-20 14:51:20] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 118 (Val Loss: 1.1030e-04)
[2026-09-20 14:51:28] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 119 (Val Loss: 1.1030e-04)
[2026-09-20 14:51:36] [INFO] [Trainer] Epoca [120/150] (8.0s) | L_tot: 1.102e-04 (Tgt: 2.029e-06, Sm: 1.987e-07, Bar: 1.081e-01) | Val RelErr: 96.60% | ||dX||/||dX*||: 0.39
[2026-09-20 14:51:36] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 120 (Val Loss: 1.1030e-04)
[2026-09-20 14:51:44] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 121 (Val Loss: 1.1030e-04)
[2026-09-20 14:52:00] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 123 (Val Loss: 1.1030e-04)
[2026-09-20 14:52:16] [INFO] [Trainer] Epoca [125/150] (7.9s) | L_tot: 1.102e-04 (Tgt: 2.027e-06, Sm: 1.968e-07, Bar: 1.081e-01) | Val RelErr: 96.58% | ||dX||/||dX*||: 0.39
[2026-09-20 14:52:24] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 126 (Val Loss: 1.1030e-04)
[2026-09-20 14:52:32] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 127 (Val Loss: 1.1030e-04)
[2026-09-20 14:52:40] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 128 (Val Loss: 1.1030e-04)
[2026-09-20 14:52:48] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 129 (Val Loss: 1.1030e-04)
[2026-09-20 14:52:56] [INFO] [Trainer] Epoca [130/150] (7.8s) | L_tot: 1.102e-04 (Tgt: 2.023e-06, Sm: 1.956e-07, Bar: 1.081e-01) | Val RelErr: 96.52% | ||dX||/||dX*||: 0.39
[2026-09-20 14:52:56] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 130 (Val Loss: 1.1030e-04)
[2026-09-20 14:53:04] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 131 (Val Loss: 1.1030e-04)
[2026-09-20 14:53:12] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 132 (Val Loss: 1.1030e-04)
[2026-09-20 14:53:28] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 134 (Val Loss: 1.1030e-04)
[2026-09-20 14:53:36] [INFO] [Trainer] Epoca [135/150] (7.8s) | L_tot: 1.102e-04 (Tgt: 2.023e-06, Sm: 1.949e-07, Bar: 1.081e-01) | Val RelErr: 96.49% | ||dX||/||dX*||: 0.39
[2026-09-20 14:53:36] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 135 (Val Loss: 1.1030e-04)
[2026-09-20 14:53:44] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 136 (Val Loss: 1.1030e-04)
[2026-09-20 14:53:52] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 137 (Val Loss: 1.1030e-04)
[2026-09-20 14:54:00] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 138 (Val Loss: 1.1029e-04)
[2026-09-20 14:54:08] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 139 (Val Loss: 1.1029e-04)
[2026-09-20 14:54:16] [INFO] [Trainer] Epoca [140/150] (7.9s) | L_tot: 1.102e-04 (Tgt: 2.022e-06, Sm: 1.944e-07, Bar: 1.081e-01) | Val RelErr: 96.47% | ||dX||/||dX*||: 0.39
[2026-09-20 14:54:16] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 140 (Val Loss: 1.1029e-04)
[2026-09-20 14:54:24] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 141 (Val Loss: 1.1029e-04)
[2026-09-20 14:54:32] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 142 (Val Loss: 1.1029e-04)
[2026-09-20 14:54:40] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 143 (Val Loss: 1.1029e-04)
[2026-09-20 14:54:48] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 144 (Val Loss: 1.1029e-04)
[2026-09-20 14:54:56] [INFO] [Trainer] Epoca [145/150] (7.8s) | L_tot: 1.102e-04 (Tgt: 2.021e-06, Sm: 1.942e-07, Bar: 1.081e-01) | Val RelErr: 96.47% | ||dX||/||dX*||: 0.39
[2026-09-20 14:54:56] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 145 (Val Loss: 1.1029e-04)
[2026-09-20 14:55:04] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 146 (Val Loss: 1.1029e-04)
[2026-09-20 14:55:11] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 147 (Val Loss: 1.1029e-04)
[2026-09-20 14:55:19] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 148 (Val Loss: 1.1029e-04)
[2026-09-20 14:55:27] [INFO] [Trainer] --> [✓] Nuovo miglior checkpoint salvato all'epoca 149 (Val Loss: 1.1029e-04)
[2026-09-20 14:55:35] [INFO] [Trainer] Epoca [150/150] (7.8s) | L_tot: 1.102e-04 (Tgt: 2.021e-06, Sm: 1.942e-07, Bar: 1.081e-01) | Val RelErr: 96.46% | ||dX||/||dX*||: 0.39
[2026-09-20 14:55:35] [INFO] [Trainer] === Addestramento concluso in 1183.11 s ===
[2026-09-20 14:55:35] [INFO] [Trainer] Miglior modello salvato in: /Users/giovanni02/Desktop/Progetti/Graph-Dynamic-Meshing/results/stage_a/checkpoints/best_model.pt
(gdm_env) giovanni02@MacBook-Air-del-Professore Graph-Dynamic-Meshing % 
"""