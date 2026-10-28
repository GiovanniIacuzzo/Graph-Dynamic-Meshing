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

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Percorsi espliciti per esecuzione diretta senza dipendere da installazione editable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "python"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

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
    parser = argparse.ArgumentParser(description="Train GDM MeshMoverGNN")
    parser.add_argument("--config", default="config/config.yaml", help="Percorso del file YAML di configurazione")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"File di configurazione non trovato: {config_path.resolve()}")
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
