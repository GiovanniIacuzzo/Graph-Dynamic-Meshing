"""Study design, candidate paths and causal numerical features."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    profile: str
    bx: float
    by: float
    nu: float
    sigma: float
    center_x: float
    center_y: float
    turn: float

    def pde_parameters(self, duration):
        params = asdict(self)
        del params["id"], params["split"]
        return {**params, "T": duration}


@dataclass(frozen=True)
class Action:
    anticipation: float
    timing: float


# Order is part of the dataset contract. Keep ties deterministic.
ACTIONS = tuple(Action(anticipation, timing)
                for anticipation in (0.0, 0.5, 1.0)
                for timing in (0.0, 0.7, -0.7))
REFERENCE_ACTION = ACTIONS.index(Action(1.0, 0.0))


def action_features():
    return np.array([(action.anticipation, action.timing) for action in ACTIONS])


def protocol_id(cfg):
    import hashlib
    import json

    numerical = {key: value for key, value in cfg.items() if key not in {"learning", "campaigns"}}
    fingerprint = {"configuration": numerical, "actions": action_features().tolist(), "schema": 2}
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    duration, horizon = (float(cfg["time"][key]) for key in ("duration", "horizon"))
    windows = step_count(duration, horizon)
    if windows < 2:
        raise ValueError("At least two adaptation windows are required.")
    for dt in cfg["time"]["steps"]:
        if step_count(horizon, float(dt)) < 2:
            raise ValueError("Each adaptation window needs at least two ALE steps.")
    if not cfg["resolutions"] or any(not isinstance(n, int) or n < 2 for n in cfg["resolutions"]):
        raise ValueError("Resolutions must be integers of at least two.")
    if not cfg["time"]["steps"]:
        raise ValueError("Configure at least one time step.")
    if not 0 < float(cfg["motion"]["jacobian_ratio"]) < 1:
        raise ValueError("jacobian_ratio must lie in (0, 1).")
    if float(cfg["motion"]["max_speed"]) <= 0 or not 0 < float(cfg["motion"]["strength"]) <= 1:
        raise ValueError("Use positive max_speed and strength in (0, 1].")
    return cfg


def step_count(duration, step):
    if not np.isfinite(duration) or not np.isfinite(step) or duration <= 0 or step <= 0:
        raise ValueError("Times must be finite and strictly positive.")
    count = round(duration / step)
    if count < 1 or not np.isclose(count * step, duration, rtol=1e-12, atol=1e-14):
        raise ValueError(f"{step} does not divide {duration}.")
    return count


def make_cases(cfg):
    cases = []
    for split_id, split in enumerate(("train", "validation", "test")):
        counts = cfg["cases"][split]
        if set(counts) - {"transport", "turning"}:
            raise ValueError(f"Unknown profile in split {split}.")
        for profile_id, profile in enumerate(("transport", "turning")):
            count = counts.get(profile, 0)
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"Invalid case count for {split}/{profile}: {count}")
            rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], split_id, profile_id]))
            for i in range(count):
                speed = rng.uniform(0.2, 0.65)
                theta = rng.uniform(-np.pi, np.pi)
                center = rng.uniform(0.35, 0.65, size=2)
                # Held-out turning cases have opposite curvature.
                turn = rng.uniform(0.2, 0.45) * (-1 if split == "test" else 1)
                cases.append(Case(
                    f"{split}-{profile}-{i:03d}", split, profile,
                    float(speed * np.cos(theta)), float(speed * np.sin(theta)),
                    float(10 ** rng.uniform(-3.7, -2.1)), float(rng.uniform(0.055, 0.12)),
                    float(center[0]), float(center[1]), float(turn if profile == "turning" else 0),
                ))
    return cases


def progress(fraction, timing):
    if not 0 <= fraction <= 1 or not -1 < timing < 1:
        raise ValueError("Invalid path fraction or timing.")
    return fraction + timing * fraction * (1 - fraction)


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def determinants(nodes, cells):
    tri = nodes[cells]
    return cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


def path_minimum(start, end, cells):
    x = start[cells]
    d = (end - start)[cells]
    u, v = x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]
    du, dv = d[:, 1] - d[:, 0], d[:, 2] - d[:, 0]
    a, b, c = cross(du, dv), cross(du, v) + cross(u, dv), cross(u, v)
    vertex = np.divide(-b, 2 * a, out=np.zeros_like(a), where=a > 0)
    interior = (vertex > 0) & (vertex < 1)
    minimum = np.minimum(c, determinants(end, cells))
    minimum[interior] = np.minimum(minimum[interior],
                                   c[interior] + vertex[interior] * (b[interior] + a[interior] * vertex[interior]))
    return minimum


def field_moments(nodes, cells, field):
    areas = 0.5 * determinants(nodes, cells)
    nodal_mass = np.bincount(cells.ravel(), np.repeat(areas / 3, 3), minlength=len(nodes))
    weights = nodal_mass * np.square(field)
    total = weights.sum()
    if total <= np.finfo(float).tiny:
        return np.array([0.5, 0.5]), 0.25
    center = np.average(nodes, axis=0, weights=weights)
    variance = np.average(np.sum((nodes - center) ** 2, axis=1), weights=weights)
    return center, float(np.sqrt(variance))


def proposed_end(reference, current, boundary, center, width, velocity, action,
                 horizon, max_speed, strength):
    monitor_center = center + action.anticipation * horizon * np.asarray(velocity)
    # A smooth candidate family with fixed connectivity. It is intentionally
    # small enough to exhaustively evaluate with the actual solver offline.
    offset = monitor_center - reference
    squared_distance = np.sum(offset ** 2, axis=1)
    weights = np.exp(-squared_distance / (8 * max(width, 1e-3) ** 2))
    displacement = reference + strength * weights[:, None] * offset - current
    displacement[boundary] = 0.0
    peak = np.linalg.norm(displacement, axis=1).max()
    # Common bound across every time parametrization of a given endpoint.
    timing_bound = 1 + max(abs(a.timing) for a in ACTIONS)
    if peak > 0:
        displacement *= min(1.0, max_speed * horizon / (timing_bound * peak))
    return displacement
