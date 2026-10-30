from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plotting.benchmark import plot_benchmark_results, plot_benchmark_summary_file


def _summary_payload() -> dict:
    methods = {
        "B0_fixed": (0.20, 0.10, 9.7e-4, 0.866, 0.0),
        "B_zero": (0.20, 0.10, 9.7e-4, 0.866, 0.0),
        "B3_gnn_raw": (0.13, 0.08, 6.0e-5, 0.863, 10.0),
        "B4_gnn_safety": (0.13, 0.08, 1.0e-4, 0.863, 12.0),
        "B5_oracle": (0.03, 0.02, 3.0e-4, 0.831, 6000.0),
    }
    payload = {}
    for split, multiplier in (("test_interp", 1.0), ("test_extrap", 2.0)):
        payload[split] = {}
        for key, (mean, sd, jac, quality, timing) in methods.items():
            payload[split][key] = {
                "n_samples": 4,
                "mean_interp_error_x1e4": mean * multiplier,
                "std_interp_error_x1e4": sd * multiplier,
                "min_jacobian": jac,
                "mean_quality": quality,
                "mean_time_ms": timing * multiplier,
            }
    return payload


def _assert_vector_outputs(output_dir: Path, expected_stems: list[str]) -> None:
    for stem in expected_stems:
        pdf = output_dir / f"{stem}.pdf"
        assert pdf.is_file() and pdf.stat().st_size > 1000
        assert pdf.read_bytes().startswith(b"%PDF-")
        assert not (output_dir / f"{stem}.svg").exists()
        assert not (output_dir / f"{stem}.png").exists()


def test_summary_plotting_exports_separate_vector_figures(tmp_path: Path) -> None:
    source = tmp_path / "benchmark_summary.json"
    source.write_text(json.dumps(_summary_payload()), encoding="utf-8")
    out_dir = tmp_path / "figures"
    outputs = plot_benchmark_summary_file(source, out_dir, epsilon=1e-4)
    assert set(outputs) == {"error", "safety", "quality", "runtime"}
    _assert_vector_outputs(out_dir, [
        "stage_a_error_comparison",
        "stage_a_mesh_admissibility",
        "stage_a_element_quality",
        "stage_a_runtime_comparison",
    ])


def test_raw_result_plotting_exports_separate_vector_figures(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    raw = {}
    for split, scale in (("test_interp", 1.0), ("test_extrap", 2.0)):
        raw[split] = {}
        for key, base_error, jac, quality, timing in (
            ("B0_fixed", 2e-5, 9.7e-4, 0.866, 0.0),
            ("B_zero", 2e-5, 9.7e-4, 0.866, 0.0),
            ("B3_gnn_raw", 1.3e-5, 6.0e-5, 0.863, 10.0),
            ("B4_gnn_safety", 1.3e-5, 1.0e-4, 0.863, 12.0),
            ("B5_oracle", 3e-6, 3.0e-4, 0.831, 6000.0),
        ):
            raw[split][key] = {
                "interp_error": (base_error * scale * (1 + 0.1 * rng.random(5))).tolist(),
                "min_jacobian": [jac] * 5,
                "mean_quality": [quality + float(x) for x in rng.normal(0, 0.001, 5)],
                "timing_ms": [timing * (1 + 0.1 * float(x)) for x in rng.random(5)],
            }
    out_dir = tmp_path / "raw_figures"
    outputs = plot_benchmark_results(raw, out_dir / "compatibility.pdf")
    assert set(outputs) == {"error", "safety", "quality", "runtime"}
    _assert_vector_outputs(out_dir, [
        "stage_a_error_comparison",
        "stage_a_mesh_admissibility",
        "stage_a_element_quality",
        "stage_a_runtime_comparison",
    ])
