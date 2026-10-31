import numpy as np

from gdm.safety_diagnostics import (
    mesh_source_tracking_metrics,
    safety_event_diagnostics,
    signed_triangle_jacobians,
)


def square_mesh():
    nodes = np.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0],
    ])
    cells = np.array([
        [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
        [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7],
    ], dtype=np.int64)
    boundary = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.int64)
    return nodes, cells, boundary


def test_signed_jacobians_positive_for_counterclockwise_triangles():
    nodes, cells, _ = square_mesh()
    assert np.allclose(signed_triangle_jacobians(nodes, cells), 0.25)


def test_safety_event_diagnostic_confirms_global_scalar_filter_identity():
    nodes, cells, boundary = square_mesh()
    raw = np.zeros_like(nodes)
    raw[4] = [0.05, -0.02]
    alpha = 0.4
    safe = nodes + alpha * raw
    result = safety_event_diagnostics(
        nodes=nodes, cells=cells, boundary_nodes=boundary,
        raw_displacement=raw,
        filter_result={
            "alpha_global": alpha, "safe_nodes": safe,
            "is_safe": False, "filter_activated": True,
            "limiting_cell_index": 3,
        },
        epsilon=1.0e-4, event_time=0.1, source_center=(0.4, 0.4),
    )
    assert abs(result["displacement_retention_ratio"] - alpha) < 1.0e-12
    assert result["scalar_filter_identity_relative_residual"] < 1.0e-12
    assert result["min_jacobian_before"] > 0.0
    assert result["limiting_cell_index"] == 3


def test_mesh_tracking_metric_is_finite_on_uniform_mesh():
    nodes, cells, _ = square_mesh()
    row = mesh_source_tracking_metrics(
        nodes=nodes, cells=cells, source_center=(0.5, 0.5), sigma=0.12,
        domain={"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
        time=0.25, step=5,
    )
    assert row["annulus_element_count"] > 0
    assert np.isfinite(row["annulus_h_refinement_ratio"])
    assert row["nearest_node_distance_to_source"] == 0.0
    assert abs(row["annulus_density_enrichment"] - 1.0) < 0.5


def test_mesh_tracking_requires_positive_orientation():
    nodes, cells, _ = square_mesh()
    bad = cells.copy()
    bad[0] = bad[0, [0, 2, 1]]
    try:
        mesh_source_tracking_metrics(
            nodes=nodes, cells=bad, source_center=(0.5, 0.5), sigma=0.12,
            domain={"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
            time=0.25, step=5,
        )
    except ValueError as exc:
        assert "positively oriented" in str(exc)
    else:
        raise AssertionError("Expected invalid orientation to be rejected")
