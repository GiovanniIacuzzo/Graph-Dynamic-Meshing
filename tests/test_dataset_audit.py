from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from plotting.dataset import audit_dataset


def test_dataset_audit_reports_convergence_and_vector_figures(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    split_ids = {
        "train": ["s_train"],
        "validation": ["s_val"],
        "test_interp": ["s_interp"],
        "test_extrap": ["s_extrap"],
    }
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    displacement = np.array([[0.0, 0.0], [0.0, 0.0], [0.01, 0.0], [0.0, 0.0]])
    for split, ids in split_ids.items():
        sample = {
            "mesh": {"nodes": nodes},
            "target": {"displacement": displacement},
            "metadata": {
                "sample_id": ids[0],
                "split": split,
                "parameters": {"mu_x": 0.4, "mu_y": 0.6, "sigma": 0.05, "amplitude": 1.0},
                "converged": True,
                "iterations": 12,
                "oracle_metrics": {
                    "status": 0,
                    "iterations_last_pass": 12,
                    "min_jacobian": 3.0e-4,
                    "num_inverted_cells": 0,
                    "num_degenerate_cells": 0,
                    "mean_escobar_quality": 0.84,
                    "safety_alpha_applied": 1.0,
                    "constraint_penalty": 0.0,
                    "success": True,
                    "candidate_is_valid": True,
                    "candidate_min_jacobian": 3.0e-4,
                    "safety_filter_active": False,
                    "projected_gradient_inf_norm": 1.0e-6,
                    "projected_gradient_l2_norm": 2.0e-5,
                    "gradient_tolerance": 1.0e-5,
                    "gradient_tolerance_met": True,
                    "termination_by_function_reduction": False,
                    "termination_by_projected_gradient": True,
                    "objective_reduction_fraction": 0.75,
                    "interpolation_error_reduction_fraction": 0.80,
                },
            },
        }
        with (raw_dir / f"{ids[0]}.pkl").open("wb") as stream:
            pickle.dump(sample, stream)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(split_ids), encoding="utf-8")

    out = tmp_path / "audit"
    report = audit_dataset(manifest_path, raw_dir, out, max_iter=2000)
    assert report["audit_status"] == "PASS"
    assert report["n_read"] == 4
    assert report["splits"]["train"]["objective_reduction_fraction"]["median"] == 0.75
    assert report["splits"]["train"]["target_acceptance_rate"] == 1.0
    for stem in (
        "dataset_oracle_convergence",
        "dataset_target_displacement",
        "dataset_oracle_geometry_iterations",
        "dataset_target_improvement",
    ):
        assert (out / f"{stem}.pdf").is_file()
    saved = json.loads((out / "dataset_quality_audit.json").read_text(encoding="utf-8"))
    assert saved["audit_status"] == "PASS"
    assert saved["splits"]["train"]["first_order_stationarity_rate"] == 1.0
    assert (out / "sample_diagnostics.csv").is_file()
    assert (out / "sample_diagnostics.json").is_file()
    csv_text = (out / "sample_diagnostics.csv").read_text(encoding="utf-8")
    assert "projected_gradient_inf_norm" in csv_text.splitlines()[0]
    assert "sigma" in csv_text.splitlines()[0]


def test_dataset_audit_warns_when_objective_descent_diagnostics_are_missing(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    displacement = np.zeros_like(nodes)
    sample = {
        "mesh": {"nodes": nodes},
        "target": {"displacement": displacement},
        "metadata": {
            "sample_id": "legacy", "split": "train", "converged": True, "iterations": 10,
            "oracle_metrics": {"status": 0, "success": True, "min_jacobian": 0.5},
        },
    }
    with (raw_dir / "legacy.pkl").open("wb") as stream:
        pickle.dump(sample, stream)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": ["legacy"]}), encoding="utf-8")
    audit = audit_dataset(manifest, raw_dir, tmp_path / "audit")
    assert audit["audit_status"] == "WARN"
    assert audit["audit_checks"]["missing_improvement_diagnostics"] is True


def test_dataset_audit_rejects_mixed_or_mismatched_quadrature(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    sample = {
        "mesh": {"nodes": nodes},
        "target": {"displacement": np.zeros_like(nodes)},
        "metadata": {
            "sample_id": "sample_q7_l2",
            "split": "train",
            "parameters": {"mu_x": 0.4, "mu_y": 0.6, "sigma": 0.05, "amplitude": 1.0},
            "converged": True,
            "iterations": 10,
            "oracle_objective": {
                "quadrature_rule": "dunavant_7_composite",
                "quadrature_subdivision_level": 2,
                "quadrature_points_per_cell": 112,
            },
            "oracle_metrics": {
                "status": 0,
                "success": True,
                "min_jacobian": 0.5,
                "num_inverted_cells": 0,
                "num_degenerate_cells": 0,
                "mean_escobar_quality": 0.8,
                "safety_alpha_applied": 1.0,
                "safety_filter_active": False,
                "constraint_penalty": 0.0,
                "candidate_is_valid": True,
                "candidate_min_jacobian": 0.5,
                "objective_reduction_fraction": 0.75,
                "interpolation_error_reduction_fraction": 0.80,
                "projected_gradient_inf_norm": 1.0e-6,
                "projected_gradient_l2_norm": 2.0e-5,
                "gradient_tolerance": 1.0e-5,
                "gradient_tolerance_met": True,
                "quadrature_rule": "dunavant_7_composite",
                "quadrature_subdivision_level": 2,
                "quadrature_points_per_cell": 112,
            },
        },
    }
    with (raw_dir / "sample_q7_l2.pkl").open("wb") as stream:
        pickle.dump(sample, stream)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": ["sample_q7_l2"]}), encoding="utf-8")

    expected = audit_dataset(
        manifest, raw_dir, tmp_path / "audit_pass",
        expected_quadrature_rule="dunavant_7_composite",
        expected_quadrature_subdivision_level=2,
    )
    assert expected["audit_status"] == "PASS"
    assert expected["oracle_quadrature"]["observed_subdivision_levels"] == [2]

    mismatch = audit_dataset(
        manifest, raw_dir, tmp_path / "audit_mismatch",
        expected_quadrature_rule="dunavant_7_composite",
        expected_quadrature_subdivision_level=1,
    )
    assert mismatch["audit_status"] == "WARN"
    assert mismatch["audit_checks"]["quadrature_mismatched_sample_ids"] == ["sample_q7_l2"]
