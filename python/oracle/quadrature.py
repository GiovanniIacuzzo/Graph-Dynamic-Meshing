"""Triangle quadrature rules used by the analytical Gaussian oracle.

The degree-five seven-point Dunavant rule is composed over a uniformly
subdivided reference triangle.  Subdivision level L yields 7 * 4**L points
per original element.  Coordinates are barycentric in the original triangle;
weights sum to 1/2, matching the reference-triangle area convention used by
``OracleObjective`` (the physical signed Jacobian determinant supplies the
remaining area scaling).
"""
from __future__ import annotations

from itertools import permutations
import numpy as np


def dunavant_7_rule() -> tuple[np.ndarray, np.ndarray]:
    """Return the degree-five symmetric 7-point triangle rule."""
    points: list[tuple[float, float, float]] = [(1.0 / 3.0,) * 3]
    weights: list[float] = [0.1125]
    for values, weight in (
        ((0.059715871789770, 0.470142064105115, 0.470142064105115),
         0.066197076394253),
        ((0.797426985353087, 0.101286507323456, 0.101286507323456),
         0.0629695902724135),
    ):
        points.extend(list(dict.fromkeys(permutations(values, 3))))
        weights.extend([weight] * 3)

    bary = np.asarray(points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if bary.shape != (7, 3) or w.shape != (7,):
        raise RuntimeError("Internal error: invalid Dunavant-7 rule shape")
    if not np.allclose(bary.sum(axis=1), 1.0, atol=1e-14):
        raise RuntimeError("Internal error: barycentric coordinates do not sum to one")
    if not np.isclose(w.sum(), 0.5, atol=1e-12):
        raise RuntimeError("Internal error: Dunavant-7 weights must sum to 1/2")
    return np.ascontiguousarray(bary), np.ascontiguousarray(w)


def composite_triangle_rule(level: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return a composite Dunavant-7 rule on the reference triangle.

    Every refinement splits each triangle into four equal-area children.
    Level 0 is the unsplit seven-point rule; level 1 uses 28 points; level 2
    uses 112 points.  The returned rule is fixed for a given level, so its
    integration functional remains differentiable with respect to mesh nodes.
    """
    if isinstance(level, bool) or int(level) != level or level < 0 or level > 4:
        raise ValueError("quadrature subdivision level must be an integer in [0, 4]")
    level = int(level)
    base_bary, base_weights = dunavant_7_rule()
    children = [np.eye(3, dtype=np.float64)]
    for _ in range(level):
        refined = []
        for verts in children:
            a, b, c = verts
            ab = 0.5 * (a + b)
            bc = 0.5 * (b + c)
            ca = 0.5 * (c + a)
            refined.extend((
                np.stack((a, ab, ca)),
                np.stack((ab, b, bc)),
                np.stack((ca, bc, c)),
                np.stack((ab, bc, ca)),
            ))
        children = refined

    scale = 4.0 ** level
    points = [base_bary @ child for child in children]
    weights = [base_weights / scale for _ in children]
    bary = np.ascontiguousarray(np.concatenate(points, axis=0))
    w = np.ascontiguousarray(np.concatenate(weights, axis=0))
    if bary.shape != (7 * (4 ** level), 3):
        raise RuntimeError("Internal error: unexpected composite rule shape")
    if not np.isclose(w.sum(), 0.5, atol=1e-11):
        raise RuntimeError("Internal error: composite weights must sum to 1/2")
    return bary, w
