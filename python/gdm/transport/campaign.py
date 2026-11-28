"""Execute the numerical and graph-learning campaigns in a versioned run directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from .cli import (
    case_selection, fit, generate, record_suite, report, write_json,
)
from .design import read_config


STAGES = ("verify", "mechanism", "dataset", "train", "evaluate", "report")


def _require_complete(file):
    if not file.is_file() or json.loads(file.read_text(encoding="utf-8"))["status"] != "complete":
        raise RuntimeError(f"Required campaign output is missing or incomplete: {file}")


def run_campaign(config, root, stage, build_dir, device, smoke=False):
    cfg = read_config(config)
    root.mkdir(parents=True, exist_ok=True)
    run_spec = root / "campaign.json"
    if smoke:
        cfg["learning"].update(epochs=2, seeds=cfg["learning"]["seeds"][:1])
        cfg["campaigns"].update(timing_repeats=1, figure_resolution=8, figure_dt=cfg["time"]["steps"][0])
    # Use one thread budget for the numerical controls and the measured
    # graph inference. Applied before importing the native and Torch modules.
    threads = str(cfg["learning"]["threads"])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, threads)
    identity = {"configuration": cfg, "smoke": smoke, "training_device": device}
    if run_spec.exists():
        if json.loads(run_spec.read_text(encoding="utf-8")) != identity:
            raise ValueError("Run directory belongs to another campaign configuration.")
    else:
        write_json(run_spec, identity)

    paths = {name: root / f"{i:02d}_{name}" for i, name in enumerate(STAGES)}
    steps = [(int(n), float(dt)) for n in cfg["resolutions"] for dt in cfg["time"]["steps"]]
    if smoke:
        steps = [(8, steps[0][1])]
    replicates = int(cfg["campaigns"]["timing_repeats"])
    figure_setting = int(cfg["campaigns"]["figure_resolution"]), float(cfg["campaigns"]["figure_dt"])
    if figure_setting not in steps or replicates < 1:
        raise ValueError("The primary figure setting must be in the design and timing needs repetitions.")
    seeds = cfg["learning"]["seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Configure at least one distinct training seed.")
    development_cases = case_selection(cfg, {"validation"}, smoke)
    test_cases = case_selection(cfg, {"test"}, smoke)
    checkpoints = [paths["train"] / f"{target}-{seed}" / "best.pt"
                   for seed in cfg["learning"]["seeds"] for target in ("instantaneous", "window")]
    for name in STAGES if stage == "all" else (stage,):
        print(f"Campaign {name}: {paths[name]}", flush=True)
        if name == "verify":
            if not (build_dir / "CTestTestfile.cmake").is_file():
                raise FileNotFoundError(f"No native CTest build in {build_dir}; run prepare.sh first.")
            subprocess.run(["ctest", "--test-dir", str(build_dir), "--output-on-failure", "--no-tests=error"],
                           check=True)
            record_suite(paths[name], cfg, [(8, steps[0][1])], case_selection(cfg, {"validation"}, True),
                         [], True, True, 1)
        elif name == "mechanism":
            record_suite(paths[name], cfg, steps, development_cases, [], True, True, replicates)
            report(paths[name], paths[name] / "analysis", figures=True)
        elif name == "dataset":
            generate(paths[name], cfg, steps[0], case_selection(cfg, {"train", "validation"}, smoke))
        elif name == "train":
            _require_complete(paths["dataset"] / "generation.json")
            for checkpoint in checkpoints:
                output = checkpoint.parent
                target, seed = output.name.rsplit("-", 1)
                if output.exists():
                    _require_complete(output / "training.json")
                    continue
                fit(output, paths["dataset"], cfg, target, device, int(seed))
        elif name == "evaluate":
            for checkpoint in checkpoints:
                _require_complete(checkpoint.with_name("training.json"))
            record_suite(paths[name], cfg, steps, test_cases, checkpoints, False, True, replicates)
        elif name == "report":
            report(paths["evaluate"], paths[name], figures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "all"))
    parser.add_argument("--config", type=Path, default=Path("config/transport.yaml"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--device", default="cpu", help="training device; all timed inference is on CPU")
    parser.add_argument("--smoke", action="store_true", help="two held-out cases, one seed and two epochs")
    args = parser.parse_args()
    run_campaign(args.config, args.root, args.stage, args.build_dir, args.device, args.smoke)


if __name__ == "__main__":
    main()
