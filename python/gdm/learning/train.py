"""Paired training and validation for trajectory ranking, independent of the ALE backend."""
import csv
import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import TARGETS, TrajectoryDataset, collate_samples, file_hash
from .model import TrajectoryPolicy, TrajectoryRanker


SCHEMA = 2


def _seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _objective(predicted, costs, ranking_weight, tie_tolerance):
    optimum = costs.min(dim=1, keepdim=True).values
    regret = (costs - optimum) / optimum.clamp_min(1e-8)
    target = torch.log1p(regret)
    delta = target[:, :, None] - target[:, None, :]
    score_delta = predicted[:, :, None] - predicted[:, None, :]
    ordered = delta > tie_tolerance
    ranking = (F.softplus(-score_delta) * ordered).sum() / ordered.sum().clamp_min(1)
    return F.smooth_l1_loss(predicted, target) + ranking_weight * ranking


def _run_epoch(model, loader, actions, eligible, optimizer, device, options):
    model.train(optimizer is not None)
    total_loss, records = 0.0, []
    for graph, scores, cases, windows in loader:
        graph, scores = graph.to(device), scores.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            predicted = model(graph, actions)
            loss = _objective(predicted, scores, options["ranking_weight"], options["tie_tolerance"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite ranking loss.")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"],
                                               error_if_nonfinite=True)
                optimizer.step()
        total_loss += loss.item() * len(cases)
        if optimizer is None:
            choice = predicted.masked_fill(~eligible, torch.inf).argmin(dim=1)
            optimum = scores.min(dim=1).values
            value = scores.gather(1, choice[:, None])[:, 0]
            regret = (value - optimum) / optimum.clamp_min(1e-8)
            for case, window, action, score, best, gap in zip(
                cases, windows, choice.tolist(), value.tolist(), optimum.tolist(), regret.tolist()
            ):
                records.append({"case_id": case, "window": window, "action": action,
                                "chosen_cost": score, "best_cost": best, "relative_regret": gap})
    return total_loss / len(loader.dataset), records


def train(directory, dataset, *, protocol, actions, eligible, options, target, device="cpu", seed=None):
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    epochs, batch_size = int(options["epochs"]), int(options["batch_size"])
    if min(epochs, batch_size, int(options["threads"]), int(options["latent_dim"])) < 1:
        raise ValueError("Epochs, batch size, threads and latent dimension must be positive.")
    seed = int(options["seeds"][0] if seed is None else seed)
    _seed(seed, bool(options["deterministic"]))
    torch.set_num_threads(int(options["threads"]))
    tensor_actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    mask = torch.as_tensor(eligible, dtype=torch.bool, device=device)
    if (tensor_actions.ndim != 2 or not torch.isfinite(tensor_actions).all()
            or mask.shape != (len(actions),) or not mask.any()):
        raise ValueError("Invalid action catalogue or eligibility mask.")
    training = TrajectoryDataset(dataset, protocol, actions, "train", target)
    validation = TrajectoryDataset(dataset, protocol, actions, "validation", target)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), collate_fn=collate_samples)
    val_loader = DataLoader(validation, batch_size=batch_size, collate_fn=collate_samples)
    architecture = {"latent_dim": int(options["latent_dim"]), "action_dimension": tensor_actions.size(1),
                    "message_passing_steps": int(options["message_passing_steps"])}
    model = TrajectoryRanker(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"],
                                  weight_decay=options["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema": SCHEMA, "protocol": protocol, "target": target, "seed": seed,
        "manifest_sha256": file_hash(training.manifest_path), "architecture": architecture,
        "actions": np.asarray(actions).tolist(), "eligible": np.asarray(eligible, dtype=bool).tolist(),
    }
    history, best, best_records = [], (float("inf"), float("inf")), []
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        training_loss, _ = _run_epoch(model, train_loader, tensor_actions, mask, optimizer, device, options)
        validation_loss, records = _run_epoch(model, val_loader, tensor_actions, mask, None, device, options)
        regret = float(np.mean([item["relative_regret"] for item in records]))
        accuracy = float(np.mean([item["relative_regret"] <= options["tie_tolerance"] for item in records]))
        history.append({"epoch": epoch, "train_loss": training_loss, "validation_loss": validation_loss,
                        "validation_regret": regret, "validation_near_optimal_rate": accuracy, "learning_rate": lr})
        if (regret, validation_loss) < best:
            best, best_records, best_epoch = (regret, validation_loss), records, epoch
            temp = directory / "best.pt.tmp"
            torch.save({**metadata, "epoch": epoch, "model": model.state_dict()}, temp)
            temp.replace(directory / "best.pt")
        scheduler.step()
        print(f"{target} seed={seed} epoch={epoch} val_regret={regret:.4g}", flush=True)
    for name, rows in (("history.csv", history), ("validation_actions.csv", best_records)):
        with (directory / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {**metadata, "status": "complete", "selected_epoch": best_epoch,
              "options": options, "device": str(device), "torch_version": str(torch.__version__),
              "training_seconds": perf_counter() - started}
    (directory / "training.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def load_policy(path, *, protocol, actions, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (checkpoint.get("schema") != SCHEMA or checkpoint["protocol"] != protocol
            or not np.array_equal(checkpoint["actions"], actions)):
        raise ValueError("Checkpoint schema, numerical protocol or action catalogue mismatch.")
    model = TrajectoryRanker(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model"])
    policy = TrajectoryPolicy(model, actions, checkpoint["eligible"])
    return policy, {key: checkpoint[key] for key in ("target", "seed", "manifest_sha256")}
