from pathlib import Path

from experiments.stage_f.safety_tracking_audit import (
    _attach_case_metadata,
    _plot_tracking,
)


def test_initial_tracking_row_receives_case_metadata():
    # t=0 row reproduces the row that was missing case_id in the first audit.
    rows = [
        {"time": 0.0, "step": 0, "annulus_h_refinement_ratio": 1.0,
         "source_center_x": 0.3, "source_center_y": 0.35},
        {"time": 0.1, "step": 2, "annulus_h_refinement_ratio": 1.2,
         "source_center_x": 0.38, "source_center_y": 0.41,
         "case_id": "case_01", "split": "validation", "seed": 7},
    ]
    enriched = _attach_case_metadata(
        rows, case_id="case_01", split="validation", seed=7
    )
    assert all(row["case_id"] == "case_01" for row in enriched)
    assert all(row["split"] == "validation" for row in enriched)
    assert all(row["seed"] == 7 for row in enriched)


def test_tracking_plot_accepts_initial_and_evolved_rows(tmp_path: Path):
    rows = _attach_case_metadata(
        [
            {"time": 0.0, "step": 0, "annulus_h_refinement_ratio": 1.0,
             "source_center_x": 0.3, "source_center_y": 0.35},
            {"time": 0.1, "step": 2, "annulus_h_refinement_ratio": 1.1,
             "source_center_x": 0.4, "source_center_y": 0.42},
        ],
        case_id="case_01", split="validation", seed=7,
    )
    out = tmp_path / "tracking.pdf"
    _plot_tracking(rows, out)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_conflicting_case_metadata_is_rejected():
    rows = [{"time": 0.0, "case_id": "other_case"}]
    try:
        _attach_case_metadata(
            rows, case_id="expected_case", split="validation", seed=7
        )
    except ValueError as exc:
        assert "inconsistent case_id" in str(exc)
    else:
        raise AssertionError("Expected inconsistent case metadata to fail")
