"""Pure Python unit tests for Stage-E1 close-out helper mathematics."""
from __future__ import annotations

import math
import numpy as np

from experiments.stage_e.closeout_common import tau_decomposition


def test_tau_frequency_fractions_sum_to_one():
    speed = np.array([0.5, 0.25, 0.0], dtype=float)
    h_beta = np.array([0.1, 0.2, 1.0], dtype=float)
    result = tau_decomposition(speed, h_beta, 0.0025, 1.0e-3, 2.0, 2.0, 4.0)
    total = result["temporal_fraction"] + result["advective_fraction"] + result["diffusive_fraction"]
    assert np.allclose(total, 1.0, rtol=0.0, atol=2e-15)
    assert np.all(np.isfinite(result["tau"]))
    assert np.all(result["tau"] > 0.0)


def test_uniform_scaling_of_tau_coefficients_scales_tau_linearly():
    speed = np.array([0.5, 0.25], dtype=float)
    h_beta = np.array([0.1, 0.2], dtype=float)
    base = tau_decomposition(speed, h_beta, 0.0025, 1.0e-3, 2.0, 2.0, 4.0)
    scaled = tau_decomposition(speed, h_beta, 0.0025, 1.0e-3, 1.0, 1.0, 2.0)
    assert np.allclose(scaled["tau"], 2.0 * base["tau"], rtol=2e-15, atol=0.0)
    assert np.allclose(scaled["temporal_fraction"], base["temporal_fraction"], rtol=2e-15, atol=0.0)


def test_time_dominance_can_be_quantified_without_tuning():
    speed = np.array([math.hypot(0.4, 0.2)])
    h_beta = np.array([2.0 * speed[0] / (1.2 / (1.0 / 32.0))])
    result = tau_decomposition(speed, h_beta, 0.0025, 0.001, 2.0, 2.0, 4.0)
    assert result["temporal_fraction"][0] > 0.99
