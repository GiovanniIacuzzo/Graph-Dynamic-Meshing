"""Publication figures for supervised and self-supervised GDM training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import matplotlib.pyplot as plt

from .style import PALETTE, save_figure, set_publication_style


def plot_training_history(history: Mapping[str, Any] | str | Path, output_dir: str | Path) -> dict[str, list[Path]]:
    """Plot the supervised Stage-A loss, relative displacement error, and LR."""
    if isinstance(history, (str, Path)):
        history = json.loads(Path(history).read_text(encoding="utf-8"))
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()
    epochs = np.asarray(history.get("epochs", []), dtype=float)
    if epochs.size == 0:
        raise ValueError("Training history does not contain a non-empty 'epochs' field.")
    paths: dict[str, list[Path]] = {}

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for key, label, colour in (("train_loss", "Training loss", PALETTE["raw"]), ("val_loss", "Validation loss", PALETTE["safe"])):
        values = np.asarray(history.get(key, []), dtype=float)
        if values.shape == epochs.shape and np.any(np.isfinite(values)):
            positive = values[np.isfinite(values) & (values > 0)]
            if positive.size:
                ax.plot(epochs, values, label=label, color=colour)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Supervised training and validation losses")
    ax.legend()
    ax.grid(True, which="both", axis="y")
    fig.tight_layout()
    paths["loss"] = save_figure(fig, output_dir / "training_loss")

    err = np.asarray(history.get("val_disp_rel_error", []), dtype=float)
    if err.shape == epochs.shape and np.any(np.isfinite(err)):
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        ax.plot(epochs, 100.0 * err, color=PALETTE["target"], label="Validation relative displacement error")
        if np.any(np.isfinite(err)):
            best_i = int(np.nanargmin(err))
            ax.scatter([epochs[best_i]], [100.0 * err[best_i]], color=PALETTE["warning"], zorder=3, label=f"Best: epoch {int(epochs[best_i])}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Relative displacement error [%]")
        ax.set_title("Validation displacement prediction")
        ax.legend()
        ax.grid(True, axis="y")
        fig.tight_layout()
        paths["relative_error"] = save_figure(fig, output_dir / "validation_displacement_error")

    lr = np.asarray(history.get("lr", []), dtype=float)
    if lr.shape == epochs.shape and np.any(np.isfinite(lr)):
        fig, ax = plt.subplots(figsize=(5.4, 2.8))
        ax.plot(epochs, lr, color=PALETTE["fixed"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("Learning-rate schedule")
        ax.grid(True, axis="y")
        fig.tight_layout()
        paths["learning_rate"] = save_figure(fig, output_dir / "learning_rate")
    return paths


def plot_selfsupervised_history(history: Sequence[Mapping[str, Any]] | str | Path, output_dir: str | Path) -> dict[str, list[Path]]:
    """Plot Stage-D objective and deployment/safety diagnostics by epoch."""
    if isinstance(history, (str, Path)):
        history = json.loads(Path(history).read_text(encoding="utf-8"))
    rows = list(history)
    if not rows:
        return {}
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()
    epochs = np.asarray([r.get("epoch", i + 1) for i, r in enumerate(rows)], dtype=float)
    paths: dict[str, list[Path]] = {}

    def nested(path: tuple[str, ...]) -> np.ndarray:
        out = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value.get(key, {}) if isinstance(value, Mapping) else {}
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.asarray(out, dtype=float)

    curves = [
        (("train", "objective"), "Training objective", PALETTE["raw"]),
        (("validation", "safe_objective", "mean"), "Validation safe objective", PALETTE["safe"]),
        (("validation", "raw_path_jacobian_barrier", "mean"), "Validation path barrier", PALETTE["target"]),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for path, label, colour in curves:
        vals = nested(path)
        if np.any(np.isfinite(vals)):
            ax.plot(epochs, vals, label=label, color=colour)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Objective / barrier value")
    ax.set_title("Stage-D self-supervised training diagnostics")
    ax.legend(); ax.grid(True, axis="y")
    fig.tight_layout()
    paths["objective"] = save_figure(fig, output_dir / "stage_d_training_objectives")

    safety = nested(("validation", "safety_alpha", "mean"))
    raw_bad = nested(("validation", "raw_path_violation_sample_fraction"))
    if np.any(np.isfinite(safety)) or np.any(np.isfinite(raw_bad)):
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        if np.any(np.isfinite(safety)):
            ax.plot(epochs, safety, label="Mean safety scaling factor", color=PALETTE["safe"])
        if np.any(np.isfinite(raw_bad)):
            ax.plot(epochs, raw_bad, label="Samples with raw path violations", color=PALETTE["warning"])
        ax.set_xlabel("Epoch"); ax.set_ylabel("Fraction / scaling factor")
        ax.set_ylim(-0.03, 1.03); ax.set_title("Stage-D validation safety")
        ax.legend(); ax.grid(True, axis="y")
        fig.tight_layout()
        paths["safety"] = save_figure(fig, output_dir / "stage_d_validation_safety")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export vector figures from a saved GDM training history.")
    parser.add_argument("--history", required=True, help="JSON training history file.")
    parser.add_argument("--output-dir", default=None, help="Output directory; defaults to a sibling 'figures' directory.")
    parser.add_argument("--self-supervised", action="store_true", help="Interpret history as Stage-D self-supervised history.")
    args = parser.parse_args()
    history_path = Path(args.history)
    output_dir = Path(args.output_dir) if args.output_dir else history_path.parent / "figures"
    if args.self_supervised:
        outputs = plot_selfsupervised_history(history_path, output_dir)
    else:
        outputs = plot_training_history(history_path, output_dir)
    for key, paths in outputs.items():
        print(f"{key}: " + ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
