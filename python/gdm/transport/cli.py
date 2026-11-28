"""Reproducible entry points for design, replay, target generation and evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np

from .design import ACTIONS, REFERENCE_ACTION, action_features, make_cases, protocol_id, read_config


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_selection(cfg, splits, smoke):
    cases = [case for case in make_cases(cfg) if case.split in splits]
    if smoke:
        cases = list({(case.split, case.profile): case for case in reversed(cases)}.values())
        cases.sort(key=lambda case: case.id)
    if not cases:
        raise ValueError(f"No cases for {splits}.")
    return cases


def provenance(cfg):
    from gdm import _gdm_core

    binary = Path(_gdm_core.__file__)
    repository = Path(__file__).resolve().parents[3]
    sources = [repository / "CMakeLists.txt", repository / "pyproject.toml"]
    sources += sorted((repository / "cpp_core").rglob("*.hpp"))
    sources += sorted((repository / "cpp_core").rglob("*.cpp"))
    sources += sorted((repository / "python" / "gdm").rglob("*.py"))
    return {
        "status": "running", "configuration": cfg, "native_binary_sha256": sha256(binary),
        "source_sha256": {str(path.relative_to(repository)): sha256(path) for path in sources if path.is_file()},
        "python": platform.python_version(), "platform": platform.platform(),
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def record_suite(directory, cfg, settings, cases, checkpoints, defect, include_remap, timing_repeats):
    from .solver import TransportExperiment, mean_in_time
    from .projection import mass_matrix

    metadata = provenance(cfg)
    metadata["cases"] = [case.id for case in cases]
    metadata["settings"] = settings
    metadata["one_step_defect"] = defect
    metadata["include_remap"] = include_remap
    metadata["timing_repeats"] = timing_repeats
    metadata["checkpoint_sha256"] = [sha256(path) for path in checkpoints]
    models = {}
    if checkpoints:
        from gdm.learning.train import load_policy
        import torch

        torch.set_num_threads(int(cfg["learning"]["threads"]))
        manifest_hashes = set()
        for path in checkpoints:
            model, identity = load_policy(path, protocol=protocol_id(cfg), actions=action_features())
            key = (f'gnn_{identity["target"]}', int(identity["seed"]))
            if key in models:
                raise ValueError(f"Duplicate checkpoint identity: {key}")
            models[key] = model
            manifest_hashes.add(identity["manifest_sha256"])
        seeds_by_target = [{seed for name, seed in models if name == target}
                           for target in ("gnn_instantaneous", "gnn_window")]
        if len(manifest_hashes) != 1 or seeds_by_target[0] != seeds_by_target[1]:
            raise ValueError("Use paired seeds and one common training dataset.")
        metadata["training_manifest_sha256"] = manifest_hashes.pop()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "run.json", metadata)
    summaries, steps, windows, timings = [], [], [], []
    for resolution, dt in settings:
        for case in cases:
            exp = TransportExperiment(case, cfg, resolution, dt)
            def run_all(diagnostic_pass):
                diagnostics = defect if diagnostic_pass else False
                outcomes = {(mode, ""): result for mode, result in
                            exp.paired_paths(defect=diagnostics, include_remap=include_remap,
                                             measure=diagnostic_pass).items()}
                for mode in ("fixed_fem", "fixed_ale", "field", "transport"):
                    outcomes[mode, ""] = exp.baseline(mode, defect=diagnostics, measure=diagnostic_pass)
                for mode, model in models.items():
                    if not diagnostic_pass:
                        model.action(exp.state(exp.reference, exp.initial))
                    outcomes[mode] = exp.run_policy(
                        lambda mesh, field: model.action(exp.state(mesh, field)),
                        defect=diagnostics, measure=diagnostic_pass
                    )
                return outcomes

            results = run_all(True)
            samples = {method: [] for method in results}
            for repeat in range(timing_repeats):
                for (method, seed), (_, _, seconds, _) in run_all(False).items():
                    samples[method, seed].append(seconds)
                    timings.append({"case_id": case.id, "resolution": resolution, "dt": dt,
                                    "method": method, "training_seed": seed, "repeat": repeat + 1, "seconds": seconds})
            mass = mass_matrix(exp.nodes, exp.cells)
            delta = results["fixed_fem", ""][3] - results["fixed_ale", ""][3]
            fixed_difference = float(np.sqrt(max(0.0, delta @ (mass @ delta))))
            for (mode, seed), (rows, events, _, _) in results.items():
                case_key = {"case_id": case.id, "profile": case.profile, "resolution": resolution,
                            "dt": dt, "method": mode, "training_seed": seed, "bx": case.bx, "by": case.by, "nu": case.nu,
                            "sigma": case.sigma, "turn": case.turn}
                summaries.append({
                    **case_key, "relative_l2_average": mean_in_time(
                        [row["l2"] for row in rows], [row["time"] for row in rows]),
                    "relative_l2_final": rows[-1]["l2"],
                    "minimum_jacobian_ratio": min(row["min_jacobian_ratio"] for row in rows),
                    "minimum_escobar": min(row["min_escobar_quality"] for row in rows),
                    "max_mesh_speed": max(row.get("max_mesh_speed", 0.0) for row in rows),
                    "solver_and_controller_seconds": float(np.median(samples[mode, seed])),
                    "fixed_fem_vs_static_ale_l2_final": fixed_difference,
                })
                steps.extend({**case_key, **row} for row in rows)
                windows.extend({**case_key, **row} for row in events)
            print(f"{case.id} n={resolution} dt={dt}: {len(results)} methods", flush=True)
    write_csv(directory / "summary.csv", summaries)
    write_csv(directory / "steps.csv", steps)
    write_csv(directory / "windows.csv", windows)
    write_csv(directory / "timings.csv", timings)
    metadata["status"] = "complete"
    write_json(directory / "run.json", metadata)


def generate(directory, cfg, setting, cases):
    from .solver import TransportExperiment

    metadata = provenance(cfg)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "generation.json", metadata)
    rows = []
    for case in cases:
        exp = TransportExperiment(case, cfg, *setting)
        mesh, field = exp.reference, exp.initial
        for window in range(exp.windows):
            t = window * exp.horizon
            scores = exp.reference_targets(mesh, field, t)
            filename = f"{case.id}-window-{window:03d}.npz"
            np.savez_compressed(directory / filename, **exp.state(mesh, field), scores=scores)
            rows.append({"case_id": case.id, "split": case.split, "window": window,
                         "file": filename, "sha256": sha256(directory / filename)})
            endpoint, _ = exp.endpoint(mesh, field, ACTIONS[REFERENCE_ACTION])
            mesh, field, _, _, _ = exp.advance(mesh, field, endpoint, t, 0, measure=False)
        print(f"{case.id}: {exp.windows} labelled windows", flush=True)
    write_json(directory / "manifest.json", {"schema": 2, "protocol": protocol_id(cfg), "samples": rows,
                                             "actions": action_features().tolist(),
                                             "resolution": setting[0], "dt": setting[1]})
    metadata["status"] = "complete"
    write_json(directory / "generation.json", metadata)


def fit(directory, dataset, cfg, target, device, seed):
    from gdm.learning.train import train

    actions = action_features()
    # The instantaneous criterion has exact timing ties for a fixed
    # endpoint. Resolve these ties by using the neutral time schedule.
    eligible = (actions[:, 1] == 0) if target == "instantaneous" else np.ones(len(actions), dtype=bool)
    train(directory, dataset, protocol=protocol_id(cfg), actions=actions, eligible=eligible,
          options=cfg["learning"], target=target, device=device, seed=seed)


def report(evaluation, output, figures=False):
    from .statistics import paired_summary, seeded_summary

    metadata = json.loads((evaluation / "run.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete":
        raise ValueError("The evaluation is incomplete.")
    with (evaluation / "summary.csv").open(encoding="utf-8") as stream:
        summaries = list(csv.DictReader(stream))
    seed = int(metadata["configuration"]["seed"])
    contrasts = paired_summary(summaries, seed)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "paired_contrasts.csv", contrasts)
    write_csv(output / "seed_sensitivity.csv", seeded_summary(summaries, seed))
    if figures:
        from .figures import write_figures

        with (evaluation / "windows.csv").open(encoding="utf-8") as stream:
            windows = list(csv.DictReader(stream))
        write_figures(output, summaries, contrasts, windows, metadata["configuration"])



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/transport.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("design", help="print the case manifest, without loading deal.II or PyTorch")
    replay = sub.add_parser("evaluate", help="paired ALE, projection, fixed and causal controls")
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--smoke", action="store_true")
    replay.add_argument("--no-remap", action="store_true")
    replay.add_argument("--timing-repeats", type=int, default=1)
    replay.add_argument("--defect", action="store_true", help="additional reference-started one-step solves")
    replay.add_argument("--checkpoint", type=Path, action="append", default=[])
    data = sub.add_parser("generate", help="produce paired instantaneous and window targets")
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--smoke", action="store_true")
    train = sub.add_parser("train", help="fit a graph controller")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--target", choices=("instantaneous", "window"), required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--device", default="cpu")
    report = sub.add_parser("report", help="compute paired contrasts from a completed evaluation")
    report.add_argument("--evaluation", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    if args.command == "evaluate" and args.timing_repeats < 1:
        parser.error("--timing-repeats must be positive")
    cfg = read_config(args.config)
    settings = [(int(n), float(dt)) for n in cfg["resolutions"] for dt in cfg["time"]["steps"]]
    if args.command == "design":
        print(json.dumps([vars(case) for case in make_cases(cfg)], indent=2))
    elif args.command == "generate":
        generate(args.output, cfg, (8, settings[0][1]) if args.smoke else settings[0],
                 case_selection(cfg, {"train", "validation"}, args.smoke))
    elif args.command == "evaluate":
        record_suite(args.output, cfg, [(8, settings[0][1])] if args.smoke else settings,
                     case_selection(cfg, {"test"}, args.smoke), args.checkpoint, args.defect,
                     not args.no_remap, args.timing_repeats)
    elif args.command == "report":
        report(args.evaluation, args.output, args.figures)
    else:
        fit(args.output, args.dataset, cfg, args.target, args.device, args.seed)


if __name__ == "__main__":
    main()
