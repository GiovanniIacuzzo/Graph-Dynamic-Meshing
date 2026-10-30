"""Command-line interface for reproducible GDM plotting and data audits."""
from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import plot_benchmark_raw_file, plot_benchmark_summary_file
from .dataset import audit_dataset
from .training import plot_selfsupervised_history, plot_training_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("benchmark", help="Export Stage-A benchmark figures from JSON.")
    bench_src = bench.add_mutually_exclusive_group(required=True)
    bench_src.add_argument("--summary", help="Aggregate summary JSON (means and standard deviations).")
    bench_src.add_argument("--raw-results", help="Per-sample benchmark JSON.")
    bench.add_argument("--output-dir", default=None)
    bench.add_argument("--epsilon", type=float, default=1.0e-4)

    train = sub.add_parser("training", help="Export figures from an existing training history.")
    train.add_argument("--history", required=True)
    train.add_argument("--output-dir", default=None)
    train.add_argument("--self-supervised", action="store_true")

    data = sub.add_parser("dataset", help="Audit an existing dataset and export diagnostic figures.")
    data.add_argument("--manifest", default="data/manifest_oracle_v4.json")
    data.add_argument("--raw-dir", default="data/raw_oracle_v4")
    data.add_argument("--output-dir", default="results/stage_a_oracle_v4/figures/dataset_audit")
    data.add_argument("--max-iter", type=int, default=2000)

    args = parser.parse_args()
    if args.command == "benchmark":
        if args.summary:
            paths = plot_benchmark_summary_file(args.summary, args.output_dir, epsilon=args.epsilon)
        else:
            paths = plot_benchmark_raw_file(args.raw_results, args.output_dir, epsilon=args.epsilon)
        for name, files in paths.items():
            print(f"{name}: " + ", ".join(str(path) for path in files))
    elif args.command == "training":
        source = Path(args.history)
        output = Path(args.output_dir) if args.output_dir else source.parent / "figures"
        plots = plot_selfsupervised_history(source, output) if args.self_supervised else plot_training_history(source, output)
        for name, files in plots.items():
            print(f"{name}: " + ", ".join(str(path) for path in files))
    else:
        summary = audit_dataset(args.manifest, args.raw_dir, args.output_dir, max_iter=args.max_iter)
        print(f"Dataset audit status: {summary['audit_status']}")


if __name__ == "__main__":
    main()
