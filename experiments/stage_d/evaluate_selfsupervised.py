"""Evaluate a Stage-D.1 self-supervised checkpoint on held-out MMS splits."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "python"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dataset.dataset import GraphDynamicMeshingDataset, collate_graph_batch
from model.model import build_model_from_config
from experiments.stage_d.train_selfsupervised import _validate, _prepare_baseline_cache
from utils.utils import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/stage_d.yaml")
    parser.add_argument("--checkpoint", default="results/stage_d_selfsupervised/v3_pathwise_safety/checkpoints/best_model.pt")
    parser.add_argument("--dataset-dir", default=None, help="Override della directory dei campioni usata nel training.")
    parser.add_argument("--output", default="results/stage_d_selfsupervised/v3_pathwise_safety/evaluation/heldout_metrics.json")
    parser.add_argument("--splits", nargs="+", default=["test_interp", "test_extrap"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg.get("self_supervised", cfg)
    if args.dataset_dir:
        cfg["dataset"]["raw_dir"] = args.dataset_dir
    torch.set_num_threads(max(1, int(cfg.get("training", {}).get("cpu_num_threads", 1))))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalize = bool(cfg.get("features", {}).get("normalize_field_scale", False))
    outputs = {
        "experiment": "stage_d_selfsupervised_heldout_evaluation",
        "checkpoint": str(ckpt_path), "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "device": str(device), "normalize_field_scale": normalize,
        "training_method": checkpoint.get("training_method", "unknown"), "splits": {},
    }
    for split in args.splits:
        if split not in ("validation", "test_interp", "test_extrap"):
            raise ValueError(f"Split non supportato: {split}")
        ds = GraphDynamicMeshingDataset(cfg["dataset"]["raw_dir"], split, normalize_field_scale=normalize)
        loader = torch.utils.data.DataLoader(ds, batch_size=int(cfg["training"].get("batch_size", 1)), shuffle=False, collate_fn=collate_graph_batch, num_workers=0)
        baseline_cache = _prepare_baseline_cache(ds, cfg, device)
        summary, rows = _validate(model, loader, device, cfg, baseline_cache)
        outputs["splits"][split] = {"summary": summary, "samples": rows}
        print(f"[{split}] objective reduction={summary['objective_reduction_fraction']['mean']:.2%}; safety interventions={summary['safety_intervention_fraction']:.2%}; invalid after filter={summary['samples_with_inverted_cells_after_filter']}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"Risultati salvati in: {output_path}")


if __name__ == "__main__":
    main()
