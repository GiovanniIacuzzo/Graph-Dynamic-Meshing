from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gdm._gdm_core", reason="The deal.II extension needs to be built.", exc_type=ImportError)

from gdm.native_api import create_pde_model
from gdm.transport.design import make_cases, read_config
from gdm.transport.solver import TransportExperiment


def test_free_transport_exact_solution_and_turning_forcing():
    points = np.array([[.35, .45], [.42, .36]])
    pde = create_pde_model(profile="transport", bx=.2, by=-.1, nu=.001, sigma=.08,
                           center_x=.35, center_y=.45, T=.4)
    t = .2
    center = np.array([.35 + .2*t, .45 - .1*t])
    variance = .08**2 + 2*.001*t
    expected = .08**2 / variance * np.exp(-((points - center)**2).sum(1)/(2*variance))
    np.testing.assert_allclose(pde.evaluate_solution(points, t), expected)
    np.testing.assert_array_equal(pde.evaluate_exact(points[:, 0], points[:, 1], t)[-1], 0)


def test_paired_rollout_shares_endpoints_and_agrees_on_zero_motion_limit():
    cfg = read_config(Path(__file__).resolve().parents[1] / "config" / "transport.yaml")
    case = next(c for c in make_cases(cfg) if c.split == "test" and c.profile == "transport")
    experiment = TransportExperiment(case, cfg, 8, .025)
    variants = experiment.paired_paths(defect=True)
    assert all(len(data[0]) == 17 for data in variants.values())
    expected = [window["endpoint_sha256"] for window in variants["uniform"][1]]
    assert all([window["endpoint_sha256"] for window in results[1]] == expected
               for results in variants.values())
    assert max(abs(w["mass_after_projection"] - w["mass_before_projection"])
               for w in variants["remap"][1]) < 1e-11
    assert all(np.isfinite(r["one_step_state_defect_l2"]) for r in variants["early"][0][1:])
    fixed = experiment.baseline("fixed_fem")
    static = experiment.baseline("fixed_ale")
    np.testing.assert_allclose(fixed[3], static[3], atol=5e-10, rtol=1e-9)
    scores = experiment.reference_targets(experiment.reference, experiment.initial, 0.0)
    assert np.isfinite(scores).all()
