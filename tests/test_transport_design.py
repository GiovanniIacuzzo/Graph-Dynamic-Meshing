from pathlib import Path

import numpy as np
import pytest

from gdm.transport.design import ACTIONS, make_cases, path_minimum, progress, proposed_end, read_config
from gdm.transport.projection import project
from gdm.transport.statistics import paired_summary


def mesh():
    return (np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
            np.array([[0, 1, 2], [0, 2, 3]]), np.array([[0, 1, 3], [1, 2, 3]]))


def test_clipped_projection_reproduces_affine_fields_and_conserves_mass():
    nodes, diagonal, crossing = mesh()
    field = 2 + nodes[:, 0] - 3 * nodes[:, 1]
    projected, report = project(nodes, diagonal, field, nodes, crossing)
    np.testing.assert_allclose(projected, field, atol=1e-14)
    assert report["mass_after_projection"] == pytest.approx(report["mass_before_projection"])
    assert report["projection_l2_loss"] < 1e-7


def test_clipped_projection_reports_loss_on_crossing_piecewise_affine_fields():
    nodes, diagonal, crossing = mesh()
    field = np.array([1., 0., 1., 0.])
    same, report = project(nodes, diagonal, field, nodes, diagonal)
    np.testing.assert_allclose(same, field, atol=1e-14)
    changed, report = project(nodes, diagonal, field, nodes, crossing)
    assert not np.allclose(changed, field)
    assert report["mass_after_projection"] == pytest.approx(report["mass_before_projection"])
    assert report["projection_l2_loss"] > 0


def test_path_minimum_detects_an_interior_dip():
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]])
    mirrored = -nodes
    minimum = path_minimum(nodes, mirrored, np.array([[0, 1, 2]]))
    assert minimum[0] == pytest.approx(0)


def test_profiles_share_endpoints_and_respect_velocity_cap():
    vertices = np.array([[0., 0.], [0.4, 0.5], [1., 1.]])
    boundary = np.array([0, 2])
    horizon, speed = 0.1, 0.2
    for action in ACTIONS:
        mesh_step = proposed_end(vertices, vertices, boundary, [0.9, 0.6], 0.15, [0.4, 0.0],
                                 action, horizon, speed, 0.6)
        np.testing.assert_array_equal(mesh_step[boundary], 0)
        assert progress(0, action.timing) == 0
        assert progress(1, action.timing) == 1
        velocity = np.linalg.norm(mesh_step, axis=1).max() * (1 + abs(action.timing)) / horizon
        assert velocity <= speed + 1e-14


def test_held_out_turn_directions_and_case_ids():
    cfg = read_config(Path(__file__).resolve().parents[1] / "config" / "transport.yaml")
    cases = make_cases(cfg)
    assert cases == make_cases(cfg)
    assert len({case.id for case in cases}) == len(cases)
    assert all(case.turn < 0 if case.split == "test" else case.turn > 0
               for case in cases if case.profile == "turning")


def test_paired_contrasts_preserve_case_as_unit():
    common = {"profile": "transport", "resolution": 16, "dt": 0.005}
    rows = [{**common, "method": method, "case_id": name,
             "relative_l2_average": error, "relative_l2_final": error}
            for method, name, error in [("uniform", "a", .2), ("uniform", "b", .3),
                                        ("early", "a", .1), ("early", "b", .15)]]
    results = paired_summary(rows, 13)
    assert all(row["cases"] == row["wins"] == 2 for row in results)
    assert results[0]["mean_improvement"] == pytest.approx(.125)
    with pytest.raises(ValueError, match="Incomplete"):
        paired_summary(rows[:-1], 13)


def test_training_count_changes_do_not_redraw_test_trajectories():
    cfg = read_config(Path(__file__).resolve().parents[1] / "config" / "transport.yaml")
    expected = [case for case in make_cases(cfg) if case.split == "test"]
    cfg["cases"]["train"]["transport"] += 5
    cfg["cases"]["validation"]["turning"] += 3
    assert [case for case in make_cases(cfg) if case.split == "test"] == expected
