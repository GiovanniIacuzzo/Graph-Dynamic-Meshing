"""Command-line inference entry point for a trained GDM mesh mover."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "python"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from inference import run_single_sample_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/stage_a.yaml")
    parser.add_argument(
        "--checkpoint",
        default="results/stage_a_oracle_v4/checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--sample",
        default="data/raw_oracle_v4/sample_00000_train.pkl",
        help="Pickle sample (.pkl) or array bundle (.npz).",
    )
    args = parser.parse_args()
    run_single_sample_demo(args.config, args.checkpoint, args.sample)


if __name__ == "__main__":
    main()
