"""Train/evaluate Stage-D GDM with a differentiable trajectory-aware objective.

This trainer deliberately does not use the legacy L-BFGS displacement labels:
all 88 audited oracle runs hit their iteration limit. It learns from the known
transient MMS objective directly, while the deterministic filter is used during
validation for deployment-aligned selection.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "python"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dataset.dataset import GraphDynamicMeshingDataset, collate_graph_batch
from model.model import build_model_from_config
from utils.utils import set_seed, ensure_dir
from plotting.training import plot_selfsupervised_history
from oracle.native_geometry import (
    apply_native_safety_filter_tuple as apply_safety_filter,
    compute_native_mesh_metrics_arrays as compute_mesh_metrics,
)
from oracle.torch_trajectory_objective import (
    trajectory_objective_torch,
    compute_baseline_interpolation_errors,
    fixed_mesh_objective,
    pathwise_jacobian_barrier,
    _signed_jacobians,
)


def _objective_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    obj = cfg["training"]["objective"]
    return {
        "lambda_D": float(obj.get("lambda_D", 0.01)),
        "lambda_Q": float(obj.get("lambda_Q", 0.001)),
        "epsilon": float(cfg["geometry"]["safety"].get("epsilon", 1e-4)),
        "time_quadrature_order": int(obj.get("time_quadrature_order", 3)),
        "barrier_weight": float(obj.get("barrier_weight", 1.0)),
        "relative_jacobian_floor": float(obj.get("relative_jacobian_floor", 0.15)),
    }


def _sample_obj(
    item: Dict[str, Any],
    disp: torch.Tensor,
    cfg: Dict[str, Any],
    device: torch.device,
    baseline_cache: Dict[str, torch.Tensor] | None = None,
):
    nodes = item["nodes"].to(device=device, dtype=disp.dtype)
    cells = item["cells"].to(device=device, dtype=torch.long)
    params = item["metadata"]["parameters"]
    kwargs = _objective_kwargs(cfg)
    baseline = None if baseline_cache is None else baseline_cache.get(str(item["sample_id"]))
    return trajectory_objective_torch(
        initial_nodes=nodes,
        displacement=disp,
        cells=cells,
        params=params,
        t_start=float(params["t_start"]),
        horizon=float(params["adapt_horizon"]),
        baseline_errors=baseline,
        **kwargs,
    )


def _prepare_baseline_cache(dataset, cfg: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """Precompute exact fixed-mesh quadrature denominators once per sample."""
    order = int(cfg["training"]["objective"].get("time_quadrature_order", 3))
    cache: Dict[str, torch.Tensor] = {}
    started = time.perf_counter()
    for index in range(len(dataset)):
        item = dataset[index]
        nodes = item["nodes"].to(device=device, dtype=torch.float32)
        cells = item["cells"].to(device=device, dtype=torch.long)
        params = item["metadata"]["parameters"]
        cache[str(item["sample_id"])] = compute_baseline_interpolation_errors(
            initial_nodes=nodes,
            cells=cells,
            params=params,
            t_start=float(params["t_start"]),
            horizon=float(params["adapt_horizon"]),
            time_quadrature_order=order,
        )
    print(f"Baseline quadrature precompute: {len(cache)} samples in {time.perf_counter()-started:.2f}s")
    return cache


def _smoothness(displacement: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    diff = displacement[src] - displacement[dst]
    return torch.mean(torch.sum(diff.square(), dim=-1))


def _train_epoch(model, loader, optimizer, device, cfg, baseline_cache):
    model.train()
    totals = {k: [] for k in (
        "loss", "objective", "interp", "quality", "barrier", "smoothness",
        "objective_reduction", "path_min_jacobian", "path_min_normalized_jacobian",
        "path_floor_violation_fraction",
    )}
    w_smooth = float(cfg["training"]["objective"].get("w_smooth", 1e-4))
    grad_clip = float(cfg["training"].get("grad_clip_norm", 1.0))
    for batch in loader:
        x = batch["node_features"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_features"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, edge_index, edge_attr)
        # Feature index 5 is the fixed-boundary mask; enforce exactly zero motion.
        pred = pred * (1.0 - x[:, 5:6])
        offset = 0
        sample_losses, sample_obj, sample_interp, sample_quality = [], [], [], []
        sample_barrier, sample_smooth, reductions = [], [], []
        sample_path_min, sample_path_min_norm, sample_path_violations = [], [], []
        for item in batch["raw_samples"]:
            n = int(item["nodes"].shape[0])
            disp = pred[offset:offset+n]
            result = _sample_obj(item, disp, cfg, device, baseline_cache)
            smooth = _smoothness(disp, item["edge_index"].to(device))
            sample_losses.append(result["loss"] + w_smooth * smooth)
            sample_obj.append(result["objective"])
            sample_interp.append(result["normalized_interpolation_error"])
            sample_quality.append(result["mesh_quality_penalty"])
            sample_barrier.append(result["jacobian_barrier"])
            sample_smooth.append(smooth)
            sample_path_min.append(result["path_min_jacobian"])
            sample_path_min_norm.append(result["path_min_normalized_jacobian"])
            sample_path_violations.append(result["path_floor_violation_fraction"])
            with torch.no_grad():
                nodes_item = item["nodes"].to(device=device, dtype=disp.dtype)
                cells_item = item["cells"].to(device=device, dtype=torch.long)
                fixed = fixed_mesh_objective(
                    nodes_item, cells_item,
                    lambda_Q=float(cfg["training"]["objective"].get("lambda_Q", 0.001)),
                )["objective"]
                reductions.append((fixed - result["objective"]) / torch.clamp(torch.abs(fixed), min=1e-12))
            offset += n
        loss = torch.stack(sample_losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Loss non finita durante Stage-D self-supervised training.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        with torch.no_grad():
            for key, vals in [
                ("loss", sample_losses), ("objective", sample_obj), ("interp", sample_interp),
                ("quality", sample_quality), ("barrier", sample_barrier), ("smoothness", sample_smooth),
                ("objective_reduction", reductions), ("path_min_jacobian", sample_path_min),
                ("path_min_normalized_jacobian", sample_path_min_norm),
                ("path_floor_violation_fraction", sample_path_violations),
            ]:
                totals[key].extend(float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for v in vals)
    return {k: float(np.mean(v)) for k, v in totals.items()}


@torch.no_grad()
def _validate(model, loader, device, cfg, baseline_cache):
    model.eval()
    safety = cfg["geometry"]["safety"]
    rows = []
    for batch in loader:
        x = batch["node_features"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_features"].to(device)
        pred = model(x, edge_index, edge_attr)
        pred = pred * (1.0 - x[:, 5:6])
        offset = 0
        for item in batch["raw_samples"]:
            n = int(item["nodes"].shape[0])
            disp_raw = pred[offset:offset+n]
            nodes_t = item["nodes"].to(device=device, dtype=disp_raw.dtype)
            cells_t = item["cells"].to(device=device, dtype=torch.long)
            nodes_np = item["nodes"].cpu().numpy().astype(np.float64)
            cells_np = item["cells"].cpu().numpy().astype(np.int64)
            disp_np = disp_raw.cpu().numpy().astype(np.float64)
            # Diagnose the raw proposal on the entire closed segment, before
            # any safety-filter contraction. This must not be inferred from
            # endpoint-only Jacobians or interior time-quadrature samples.
            eps = float(safety.get("epsilon", 1e-4))
            rel_floor = float(cfg["training"]["objective"].get("relative_jacobian_floor", 0.15))
            raw_barrier, raw_path_j_min_cells, raw_target_floor = pathwise_jacobian_barrier(
                nodes_t, disp_raw, cells_t, epsilon=eps, relative_jacobian_floor=rel_floor
            )
            raw_path_bad_fraction = torch.mean(
                (raw_path_j_min_cells < raw_target_floor).to(dtype=disp_raw.dtype)
            )
            raw_path_min = torch.min(raw_path_j_min_cells)
            raw_initial_j = _signed_jacobians(nodes_t, cells_t).detach()
            raw_endpoint_min = torch.min(_signed_jacobians(nodes_t + disp_raw, cells_t))
            alpha, safe_nodes, full_safe, min_j_filter = apply_safety_filter(
                nodes=nodes_np,
                displacement=disp_np,
                cells=cells_np,
                epsilon=float(safety.get("epsilon", 1e-4)),
                tol_deg=float(safety.get("tol_deg", 1e-12)),
                safety_margin=float(safety.get("safety_margin", 1e-6)),
            )
            safe_disp = torch.as_tensor(safe_nodes - nodes_np, dtype=disp_raw.dtype, device=device)
            result = _sample_obj(item, safe_disp, cfg, device, baseline_cache)
            fixed = fixed_mesh_objective(
                item["nodes"].to(device=device, dtype=safe_disp.dtype),
                item["cells"].to(device=device, dtype=torch.long),
                lambda_Q=float(cfg["training"]["objective"].get("lambda_Q", 0.001)),
            )["objective"]
            geom = compute_mesh_metrics(safe_nodes, cells_np)
            rows.append({
                "sample_id": str(item["sample_id"]),
                "split": str(item["metadata"].get("split", "unknown")),
                "sigma": float(item["metadata"]["parameters"]["sigma"]),
                "beta": float(item["metadata"]["parameters"]["beta"]),
                "t_start": float(item["metadata"]["parameters"]["t_start"]),
                "fixed_objective": float(fixed.cpu()),
                "safe_objective": float(result["objective"].cpu()),
                "normalized_interpolation_error": float(result["normalized_interpolation_error"].cpu()),
                "mesh_quality_penalty": float(result["mesh_quality_penalty"].cpu()),
                "displacement_cost": float(result["displacement_cost"].cpu()),
                "objective_reduction_fraction": float(((fixed-result["objective"])/torch.clamp(torch.abs(fixed), min=1e-12)).cpu()),
                "safety_alpha": float(alpha),
                "safety_intervened": bool(alpha < 1.0-1e-12),
                "min_jacobian": float(np.min(geom["jacobians"])),
                "min_normalized_jacobian": float(np.min(geom["normalized_jacobians"])),
                "inverted_cells": int(np.sum(geom["inverted"])),
                "degenerate_cells": int(np.sum(geom["degenerate"])),
                "raw_min_jacobian": float(raw_endpoint_min.detach().cpu()),
                "raw_path_min_jacobian": float(raw_path_min.detach().cpu()),
                "raw_path_min_normalized_jacobian": float(torch.min(
                    raw_path_j_min_cells / raw_initial_j.abs().clamp_min(1e-14)
                ).detach().cpu()),
                "raw_path_floor_violation_fraction": float(raw_path_bad_fraction.detach().cpu()),
                "raw_path_jacobian_barrier": float(raw_barrier.detach().cpu()),
                "filter_min_jacobian": float(min_j_filter),
            })
            offset += n
    if not rows:
        raise RuntimeError("Validation split vuoto.")
    keys = [
        "fixed_objective", "safe_objective", "normalized_interpolation_error",
        "mesh_quality_penalty", "displacement_cost", "objective_reduction_fraction",
        "safety_alpha", "min_jacobian", "min_normalized_jacobian",
        "raw_min_jacobian", "raw_path_min_jacobian",
        "raw_path_min_normalized_jacobian", "raw_path_floor_violation_fraction",
        "raw_path_jacobian_barrier",
    ]
    summary = {key: {"mean": float(np.mean([r[key] for r in rows])), "median": float(np.median([r[key] for r in rows]))} for key in keys}
    summary["safety_intervention_fraction"] = float(np.mean([r["safety_intervened"] for r in rows]))
    summary["raw_path_violation_sample_fraction"] = float(np.mean([r["raw_path_floor_violation_fraction"] > 0.0 for r in rows]))
    summary["samples_with_inverted_cells_after_filter"] = int(sum(r["inverted_cells"] > 0 for r in rows))
    summary["samples_with_degenerate_cells_after_filter"] = int(sum(r["degenerate_cells"] > 0 for r in rows))
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/stage_d.yaml")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--epochs", type=int, default=None, help="Override il numero totale di epoche previsto dalla configurazione.")
    parser.add_argument("--resume", default=None, help="Checkpoint latest_checkpoint.pt da cui riprendere l'addestramento.")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg.get("self_supervised", cfg)
    if args.dataset_dir:
        cfg["dataset"]["raw_dir"] = args.dataset_dir
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs deve essere >= 1.")
        cfg["training"]["epochs"] = int(args.epochs)
        cfg.setdefault("training", {}).setdefault("lr_scheduler", {})["T_max"] = int(args.epochs)
    set_seed(int(cfg["experiment"]["seed"]), deterministic_torch=bool(cfg["experiment"].get("deterministic", False)))
    cpu_threads = max(1, int(cfg["training"].get("cpu_num_threads", 1)))
    torch.set_num_threads(cpu_threads)

    requested = args.device or cfg["experiment"].get("device", "cpu")
    if requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if requested in ("mps", "cuda"):
            print(f"Dispositivo {requested} non disponibile; uso CPU.")
    outdir = ensure_dir(Path(cfg["experiment"]["output_dir"]))
    ckpt_dir = ensure_dir(outdir / "checkpoints")
    norm_fields = bool(cfg.get("features", {}).get("normalize_field_scale", False))
    raw_dir = cfg["dataset"]["raw_dir"]
    train_ds = GraphDynamicMeshingDataset(raw_dir, "train", normalize_field_scale=norm_fields)
    val_ds = GraphDynamicMeshingDataset(raw_dir, "validation", normalize_field_scale=norm_fields)
    t_cfg = cfg["training"]
    loader_kw = {"batch_size": int(t_cfg.get("batch_size", 1)), "collate_fn": collate_graph_batch, "num_workers": 0}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    baseline_cache_train = _prepare_baseline_cache(train_ds, cfg, device)
    baseline_cache_val = _prepare_baseline_cache(val_ds, cfg, device)

    model = build_model_from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(t_cfg["learning_rate"]), weight_decay=float(t_cfg.get("weight_decay", 1e-5)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(t_cfg.get("lr_scheduler", {}).get("T_max", t_cfg["epochs"])),
        eta_min=float(t_cfg.get("lr_scheduler", {}).get("eta_min", 1e-6)),
    )
    best_val = float("inf")  # composite deployment-aligned selection score
    best_safe_objective = float("inf")
    patience = int(t_cfg.get("early_stopping_patience", 20))
    bad_epochs = 0
    history_path = outdir / "training_history.json"
    history: List[Dict[str, Any]] = []
    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint di ripresa non trovato: {resume_path}")
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        # If a different total epoch count was requested, preserve the current
        # scheduler position while updating the final horizon.
        scheduler.T_max = int(t_cfg["epochs"])
        prior_method = payload.get("training_method", "")
        if prior_method != "trajectory_objective_self_supervised_pathwise_safety_v3":
            raise ValueError(
                "Il checkpoint di ripresa non usa l'obiettivo pathwise Stage-D.1.2. "
                "Avvia una nuova run in una directory output separata, senza riutilizzare il checkpoint precedente."
            )
        best_val = float(payload.get("best_validation_selection_score", float("inf")))
        best_safe_objective = float(payload.get("best_validation_safe_objective", float("inf")))
        bad_epochs = int(payload.get("bad_epochs", 0))
        start_epoch = int(payload.get("epoch", 0)) + 1
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"Ripresa da {resume_path}, epoca iniziale {start_epoch}; best_val={best_val:.6f}")

    if start_epoch > int(t_cfg["epochs"]):
        raise ValueError(f"Il checkpoint è già all'epoca {start_epoch-1}, >= totale richiesto {t_cfg['epochs']}; usare --epochs con un valore maggiore.")

    training_start = time.perf_counter()
    best_checkpoint_path = ckpt_dir / "best_model.pt"
    latest_checkpoint_path = ckpt_dir / "latest_checkpoint.pt"
    for epoch in range(start_epoch, int(t_cfg["epochs"]) + 1):
        train_metrics = _train_epoch(model, train_loader, optimizer, device, cfg, baseline_cache_train)
        val_summary, val_rows = _validate(model, val_loader, device, cfg, baseline_cache_val)
        val_obj = val_summary["safe_objective"]["mean"]
        selection_cfg = t_cfg.get("selection", {})
        filter_intervention_weight = float(selection_cfg.get("filter_intervention_weight", 0.05))
        path_barrier_weight = float(selection_cfg.get("path_barrier_weight", t_cfg["objective"].get("barrier_weight", 1.0)))
        val_selection_score = (
            val_obj
            + path_barrier_weight * val_summary["raw_path_jacobian_barrier"]["mean"]
            + filter_intervention_weight * (1.0 - val_summary["safety_alpha"]["mean"])
        )
        lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "lr": lr, "validation_selection_score": val_selection_score,
                        "train": train_metrics, "validation": val_summary})
        scheduler.step()

        improved = val_selection_score < best_val - 1e-8
        if improved:
            best_val = val_selection_score
            best_safe_objective = val_obj
            bad_epochs = 0
            best_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_validation_safe_objective": best_safe_objective,
                "best_validation_selection_score": best_val,
                "validation_metrics": val_summary,
                "config": cfg,
                "training_method": "trajectory_objective_self_supervised_pathwise_safety_v3",
            }
            torch.save(best_payload, best_checkpoint_path)
            (outdir / "best_validation_samples.json").write_text(json.dumps(val_rows, indent=2), encoding="utf-8")
        else:
            bad_epochs += 1

        # Persist progress after each epoch to support long MPS/CPU runs.
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_safe_objective": best_safe_objective,
            "best_validation_selection_score": best_val,
            "bad_epochs": bad_epochs,
            "config": cfg,
            "training_method": "trajectory_objective_self_supervised_pathwise_safety_v3",
        }, latest_checkpoint_path)
        if epoch == start_epoch or epoch % 5 == 0:
            print(
                f"epoch={epoch:03d}/{t_cfg['epochs']} train_obj={train_metrics['objective']:.6f} "
                f"train_path_barrier={train_metrics['barrier']:.4e} "
                f"val_safe_obj={val_obj:.6f} selection={val_selection_score:.6f} "
                f"path_barrier={val_summary['raw_path_jacobian_barrier']['mean']:.4e} "
                f"raw_path_bad={val_summary['raw_path_violation_sample_fraction']:.1%} "
                f"alpha={val_summary['safety_alpha']['median']:.5f}", flush=True
            )
        if bad_epochs >= patience:
            print(f"early stopping: nessun miglioramento per {patience} epoche.", flush=True)
            break

    elapsed = time.perf_counter() - training_start
    summary = {
        "experiment": "stage_d_selfsupervised_training",
        "config": str(config_path), "dataset_dir": str(Path(raw_dir).resolve()),
        "device": str(device), "cpu_num_threads": cpu_threads,
        "elapsed_seconds_this_run": elapsed, "epochs_completed_total": len(history),
        "best_validation_safe_objective": best_safe_objective,
        "best_validation_selection_score": best_val,
        "selection_formula": "safe_objective_mean + path_barrier_weight*raw_path_jacobian_barrier_mean + filter_intervention_weight*(1-mean_alpha)",
        "selection_weights": {
            "path_barrier_weight": float(t_cfg.get("selection", {}).get("path_barrier_weight", t_cfg["objective"].get("barrier_weight", 1.0))),
            "filter_intervention_weight": float(t_cfg.get("selection", {}).get("filter_intervention_weight", 0.05)),
        },
        "best_checkpoint": str(best_checkpoint_path.resolve()),
        "latest_checkpoint": str(latest_checkpoint_path.resolve()),
        "history": str(history_path.resolve()),
        "note": "Stage-D.1.2 self-supervised objective with exact pathwise quadratic-Jacobian barrier; legacy oracle displacement targets are not used. Fresh output directory recommended.",
    }
    (outdir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure_paths = plot_selfsupervised_history(history, outdir / "figures")
    print("Training figures:", {key: [str(p) for p in value] for key, value in figure_paths.items()})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
