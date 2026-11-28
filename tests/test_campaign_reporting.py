import csv
import json
from pathlib import Path

import pytest

from gdm.transport import campaign
from gdm.transport.cli import report, write_csv
from gdm.transport.design import read_config


CONFIG = Path(__file__).resolve().parents[1] / "config" / "transport.yaml"


def test_campaign_smoke_uses_one_common_dataset_and_paired_seeds(tmp_path, monkeypatch):
    calls = []

    def create(directory, manifest, payload):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / manifest).write_text(json.dumps({"status": "complete", **payload}))

    def record(directory, cfg, settings, cases, checkpoints, defect, remap, repeats):
        create(directory, "run.json", {"settings": settings})
        calls.append((directory.name, len(cases), repeats, len(checkpoints)))

    def dataset(directory, cfg, setting, cases):
        assert {case.split for case in cases} == {"train", "validation"}
        create(directory, "generation.json", {"setting": setting})

    def fit(directory, dataset_dir, cfg, target, device, seed):
        assert cfg["learning"]["epochs"] == 2
        create(directory, "training.json", {"target": target, "seed": seed})
        (directory / "best.pt").touch()
        calls.append((target, dataset_dir.name, seed))

    build = tmp_path / "build"
    build.mkdir()
    (build / "CTestTestfile.cmake").touch()
    monkeypatch.setattr(campaign.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(campaign, "record_suite", record)
    monkeypatch.setattr(campaign, "generate", dataset)
    monkeypatch.setattr(campaign, "fit", fit)
    monkeypatch.setattr(campaign, "report", lambda *args, **kwargs: None)
    root = tmp_path / "campaign"
    campaign.run_campaign(CONFIG, root, "all", build, "cpu", smoke=True)
    assert calls == [("00_verify", 2, 1, 0), ("01_mechanism", 2, 1, 0),
                     ("instantaneous", "02_dataset", 4983), ("window", "02_dataset", 4983),
                     ("04_evaluate", 2, 1, 2)]
    with pytest.raises(ValueError, match="another campaign"):
        campaign.run_campaign(CONFIG, root, "dataset", build, "cpu", smoke=False)


def test_report_pdf_and_case_level_intervals(tmp_path):
    pytest.importorskip("matplotlib")
    cfg = read_config(CONFIG)
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "run.json").write_text(json.dumps({"status": "complete", "configuration": cfg}))
    rows, windows = [], []
    for profile in ("transport", "turning"):
        for case in ("a", "b"):
            for method, seeds in (("fixed_fem", [""]), ("transport", [""]), ("uniform", [""]),
                                   ("early", [""]), ("remap", [""]),
                                   ("gnn_instantaneous", [5, 6]), ("gnn_window", [5, 6])):
                for seed in seeds:
                    err = .12 if method in ("uniform", "gnn_instantaneous") else .10
                    rows.append(dict(profile=profile, resolution=24, dt=.005, method=method,
                                     training_seed=seed, case_id=case, relative_l2_average=err,
                                     relative_l2_final=err, solver_and_controller_seconds=.2))
                    if str(seed):
                        windows.append({**rows[-1], "action": 1 if method == "gnn_window" else 0})
    write_csv(evaluation / "summary.csv", rows)
    write_csv(evaluation / "windows.csv", windows)
    output = tmp_path / "analysis"
    report(evaluation, output, figures=True)
    for name in ("01_trajectory_mechanism.pdf", "02_learning_ablation.pdf", "03_work_precision.pdf"):
        assert (output / name).read_bytes().startswith(b"%PDF")
    with (output / "paired_contrasts.csv").open() as stream:
        selected = [row for row in csv.DictReader(stream) if row["candidate"] == "gnn_window"]
    assert selected and all(row["cases"] == "2" for row in selected)
    with (output / "seed_sensitivity.csv").open() as stream:
        assert {row["training_seed"] for row in csv.DictReader(stream)} == {"5", "6"}
