from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from plotting.dataset import audit_dataset


def test_dataset_audit_accepts_quality_checked_budgeted_target_without_stationarity(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    disp = np.array([[0., 0.], [0., 0.], [0.01, 0.], [0., 0.]])
    splits = {"train": ["s0"], "validation": ["s1"], "test_interp": ["s2"], "test_extrap": ["s3"]}
    for split, ids in splits.items():
        sample = {
            "mesh": {"nodes": nodes},
            "target": {"displacement": disp},
            "metadata": {
                "sample_id": ids[0], "split": split, "converged": False, "target_accepted": True, "iterations": 10,
                "parameters": {"mu_x": 0.4, "mu_y": 0.6, "sigma": 0.05, "amplitude": 1.0},
                "oracle_objective": {"quadrature_rule": "dunavant_7_composite", "quadrature_subdivision_level": 2},
                "oracle_metrics": {
                    "status": 0, "success": True, "message": "CONVERGENCE: RELATIVE REDUCTION OF F",
                    "min_jacobian": 3e-4, "num_inverted_cells": 0, "num_degenerate_cells": 0,
                    "mean_escobar_quality": 0.8, "safety_alpha_applied": 1.0,
                    "safety_filter_active": False, "constraint_penalty": 0.0, "candidate_is_valid": True,
                    "candidate_min_jacobian": 3e-4, "projected_gradient_inf_norm": 1e-1,
                    "projected_gradient_l2_norm": 2e-1, "gradient_tolerance": 1e-5,
                    "gradient_tolerance_met": False, "termination_by_function_reduction": True,
                    "termination_by_projected_gradient": False, "objective_reduction_fraction": 0.8,
                    "interpolation_error_reduction_fraction": 0.8, "target_accepted": True,
                },
            },
        }
        with (raw / f"{ids[0]}.pkl").open("wb") as stream:
            pickle.dump(sample, stream)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(splits), encoding="utf-8")
    report = audit_dataset(
        manifest, raw, tmp_path / "audit", expected_quadrature_rule="dunavant_7_composite",
        expected_quadrature_subdivision_level=2, minimum_first_order_stationarity_rate=0.95,
    )
    assert report["audit_status"] == "PASS"
    assert report["audit_checks"]["first_order_stationarity_is_diagnostic_only"] is True
    assert report["audit_checks"]["insufficient_first_order_stationarity"] is True
    assert all(v["first_order_stationarity_rate"] == 0.0 for v in report["splits"].values())
    assert all(v["target_acceptance_rate"] == 1.0 for v in report["splits"].values())
