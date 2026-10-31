"""Orchestrate E1 campaign after a successful independent smoke test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import atomic_write_json, configure_logging, load_config, prepare_output_tree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="run all E1 analyses (requires successful smoke_test.json)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if not args.all:
        parser.error("specifica --all; lo smoke test si esegue separatamente con python -m experiments.stage_e.smoke_test")
    cfg = load_config(args.config) if args.config else load_config()
    root = prepare_output_tree(cfg)
    smoke_path = root / "smoke_test.json"
    if not smoke_path.exists():
        raise SystemExit("Smoke test assente. Eseguire prima: python -m experiments.stage_e.smoke_test")
    with smoke_path.open("r", encoding="utf-8") as f:
        smoke = json.load(f)
    if smoke.get("status") != "passed":
        raise SystemExit("Lo smoke test non risulta superato; campagna E1 non avviata.")
    logger = configure_logging(cfg)
    from . import e1_temporal, e1_spatial, e1_peclet, e1_supg_comparison, e1_dgcl, e1_work_precision
    tasks = [
        ("E1-A temporal convergence", e1_temporal.run),
        ("E1-A spatial convergence and temporal separation", e1_spatial.run),
        ("E1-B local Péclet statistics", e1_peclet.run),
        ("E1-C Galerkin/SUPG comparison", e1_supg_comparison.run),
        ("E1-D discrete geometric conservation law", e1_dgcl.run),
        ("E1-E fixed-grid work-precision", e1_work_precision.run),
    ]
    completed = []
    for name, function in tasks:
        logger.info("Avvio: %s", name)
        result = function(args.config)
        completed.append({"task": name, "status": "completed", "summary": result if isinstance(result, (dict, list, str, int, float, bool)) else str(result)})
    summary = {
        "experiment": cfg["experiment"]["name"],
        "configuration": cfg["_config_path"],
        "smoke_test_status": smoke["status"],
        "tasks": completed,
        "scientific_interpretation_required": True,
        "note": "Completamento operativo della pipeline non equivale automaticamente a superamento dei criteri scientifici: consultare i CSV e i flag di validazione per eventuali esiti non conclusivi.",
    }
    atomic_write_json(root / "experiment_summary.json", summary, merge=False)
    logger.info("Campagna E1 terminata; summary=%s", root / "experiment_summary.json")


if __name__ == "__main__":
    main()
