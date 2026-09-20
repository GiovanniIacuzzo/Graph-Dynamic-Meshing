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


def compute_differentiable_jacobian_barrier_loss(
    nodes: torch.Tensor,
    displacement: torch.Tensor,
    cells: torch.Tensor,
    epsilon: float = 1e-4
) -> torch.Tensor:
    """
    Calcola la barriera geometrica differenziabile durante l'addestramento
    per favorire configurazioni ammissibili ed evitare collassi elementari:
        J_e(X + Delta X) > epsilon
    Usa la penalità:
        psi(J_e) = -ln((J_e - epsilon) / J_{e,0}) per J_e > epsilon,
        con penalizzazione asintotica quadratica se J_e <= epsilon.
    """
    # Coordinate traslate: X_new = X + Delta X
    x_new = nodes + displacement

    x1, y1 = x_new[cells[:, 0], 0], x_new[cells[:, 0], 1]
    x2, y2 = x_new[cells[:, 1], 0], x_new[cells[:, 1], 1]
    x3, y3 = x_new[cells[:, 2], 0], x_new[cells[:, 2], 1]

    # Determinante Jacobiano differenziabile: (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    jacobians = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

    # Margine di sicurezza rispetto a epsilon
    margin = jacobians - epsilon

    # Zona sicura: -log(margin / J_scale)
    # Zona critica/inversione: penalizzazione asintotica (C1 continua)
    safe_threshold = 1e-3
    is_safe = margin > safe_threshold

    loss_safe = -torch.log(torch.clamp(margin[is_safe], min=1e-12))
    # Ramo quadratico per margini inferiori alla soglia critica
    violation = safe_threshold - margin[~is_safe]
    loss_violation = -np.log(safe_threshold) + 1e3 * (violation**2)

    total_elements = jacobians.size(0)
    loss = (torch.sum(loss_safe) + torch.sum(loss_violation)) / max(total_elements, 1)
    return loss


# =============================================================================
# ROUTINE DI TRAINING E VALIDATION PER SINGOLA EPOCA
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: Dict[str, float],
    epsilon: float
) -> Dict[str, float]:
    """Esegue un'epoca di ottimizzazione con backpropagation."""
    model.train()
    total_loss = 0.0
    loss_target_accum = 0.0
    loss_smooth_accum = 0.0
    loss_barrier_accum = 0.0
    num_batches = len(loader)

    w_target = loss_weights.get("w_target", 1.0)
    w_smooth = loss_weights.get("w_smooth", 0.01)
    w_barrier = loss_weights.get("w_barrier", 0.001)

    for batch in loader:
        node_feats = batch["node_features"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_feats = batch["edge_features"].to(device)
        target_disp = batch["target_displacement"].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_disp = model(node_feats, edge_index, edge_feats)

        # 1. Termine di supervisione diretta (MSE rispetto al target dell'oracle)
        l_target = torch.mean((pred_disp - target_disp)**2)

        # 2. Termine di regolarità (smoothness)
        l_smooth = compute_dirichlet_smoothness_loss(pred_disp, edge_index)

        # 3. Termine di barriera geometrica differenziabile
        # Calcolato estraendo i campioni grezzi del batch
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
                    compute_differentiable_jacobian_barrier_loss(nodes_i, disp_i, cells_i, epsilon=epsilon)
                )
                node_offset += n_count
            l_barrier = torch.mean(torch.stack(barrier_terms))

        # Loss totale combinata
        loss = w_target * l_target + w_smooth * l_smooth + w_barrier * l_barrier

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        loss_target_accum += l_target.item()
        loss_smooth_accum += l_smooth.item()
        loss_barrier_accum += l_barrier.item()

    return {
        "train_loss": total_loss / max(num_batches, 1),
        "train_l_target": loss_target_accum / max(num_batches, 1),
        "train_l_smooth": loss_smooth_accum / max(num_batches, 1),
        "train_l_barrier": loss_barrier_accum / max(num_batches, 1),
    }


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_weights: Dict[str, float],
    epsilon: float
) -> Dict[str, float]:
    """Valuta il modello sul set di validazione calcolando metriche e loss."""
    model.eval()
    total_loss = 0.0
    rel_errors = []
    l2_errors = []
    num_batches = len(loader)

    w_target = loss_weights.get("w_target", 1.0)
    w_smooth = loss_weights.get("w_smooth", 0.01)
    w_barrier = loss_weights.get("w_barrier", 0.001)

    with torch.no_grad():
        for batch in loader:
            node_feats = batch["node_features"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_feats = batch["edge_features"].to(device)
            target_disp = batch["target_displacement"].to(device)

            pred_disp = model(node_feats, edge_index, edge_feats)

            l_target = torch.mean((pred_disp - target_disp)**2)
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
                        compute_differentiable_jacobian_barrier_loss(nodes_i, disp_i, cells_i, epsilon=epsilon)
                    )
                    node_offset += n_count
                l_barrier = torch.mean(torch.stack(barrier_terms))

            loss = w_target * l_target + w_smooth * l_smooth + w_barrier * l_barrier
            total_loss += loss.item()

            # Calcolo metriche formali
            pred_np = pred_disp.cpu().numpy()
            target_np = target_disp.cpu().numpy()
            disp_m = compute_displacement_metrics(pred_np, target_np)
            rel_errors.append(disp_m["displacement_relative_error"])
            l2_errors.append(disp_m["displacement_l2_error"])

    return {
        "val_loss": total_loss / max(num_batches, 1),
        "val_disp_rel_error": float(np.mean(rel_errors)),
        "val_disp_l2_error": float(np.mean(l2_errors))
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

            # Train pass
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                loss_weights=loss_weights,
                epsilon=epsilon
            )

            # Validation pass
            val_metrics = validate_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                loss_weights=loss_weights,
                epsilon=epsilon
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
                    f"Epoca [{epoch:03d}/{t_cfg['epochs']:03d}] ({t_epoch:.2f}s) | "
                    f"Train Loss: {train_metrics['train_loss']:.4e} | "
                    f"Val Loss: {val_metrics['val_loss']:.4e} | "
                    f"Val Rel Err: {val_metrics['val_disp_rel_error']:.2%} | "
                    f"LR: {current_lr:.2e}"
                )

            # Checkpoint del modello migliore
            val_loss = val_metrics["val_loss"]
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