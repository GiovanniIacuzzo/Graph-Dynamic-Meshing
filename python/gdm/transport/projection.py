"""Conservative L² control on coincident planar domains using triangle intersections."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .design import cross


QUADRATURE = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
LOCAL_MASS = (np.ones((3, 3)) + np.eye(3)) / 12


def clip_triangle(subject, clipper):
    polygon = list(subject)
    for a, b in zip(clipper, np.roll(clipper, -1, axis=0)):
        if not polygon:
            break
        previous = polygon[-1]
        previous_side = cross(b - a, previous - a)
        output = []
        for point in polygon:
            side = cross(b - a, point - a)
            if (side >= 0) != (previous_side >= 0):
                output.append(previous + (point - previous) * (previous_side / (previous_side - side)))
            if side >= 0:
                output.append(point)
            previous, previous_side = point, side
        polygon = output
    return polygon


def barycentric(points, tri):
    edge1, edge2 = tri[1:] - tri[0]
    det = cross(edge1, edge2)
    shifted = points - tri[0]
    l1 = cross(shifted, edge2) / det
    l2 = cross(edge1, shifted) / det
    return np.column_stack((1 - l1 - l2, l1, l2))


def mass_matrix(nodes, cells):
    tri = nodes[cells]
    areas = 0.5 * cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if np.any(areas <= 0):
        raise ValueError("Projection requires positively oriented triangles.")
    entries = areas[:, None, None] * LOCAL_MASS
    rows = np.repeat(cells, 3, axis=1)
    cols = np.tile(cells, (1, 3))
    return coo_matrix((entries.ravel(), (rows.ravel(), cols.ravel())),
                      shape=(len(nodes), len(nodes))).tocsr()


def project(old_nodes, old_cells, field, new_nodes, new_cells):
    old_tri = old_nodes[old_cells]
    old_lo, old_hi = old_tri.min(axis=1), old_tri.max(axis=1)
    old_mass = mass_matrix(old_nodes, old_cells)
    new_mass = mass_matrix(new_nodes, new_cells)
    rhs = np.zeros(len(new_nodes))
    for ids in new_cells:
        tri = new_nodes[ids]
        lo, hi = tri.min(axis=0), tri.max(axis=0)
        overlaps = np.flatnonzero(np.all((old_lo <= hi) & (old_hi >= lo), axis=1))
        covered = 0.0
        for i in overlaps:
            polygon = clip_triangle(tri, old_tri[i])
            for j in range(1, len(polygon) - 1):
                piece = np.array([polygon[0], polygon[j], polygon[j + 1]])
                area = 0.5 * abs(cross(piece[1] - piece[0], piece[2] - piece[0]))
                if area == 0:
                    continue
                points = QUADRATURE @ piece
                values = barycentric(points, old_tri[i]) @ field[old_cells[i]]
                rhs[ids] += (area / 3) * (values @ barycentric(points, tri))
                covered += area
        area = 0.5 * cross(tri[1] - tri[0], tri[2] - tri[0])
        if not np.isclose(covered, area, rtol=1e-8, atol=1e-14):
            raise ValueError("Triangle intersections do not cover the destination mesh.")
    projected = spsolve(new_mass, rhs)
    if not np.all(np.isfinite(projected)):
        raise FloatingPointError("Projection returned nonfinite coefficients.")
    before = float((old_mass @ field).sum())
    after = float((new_mass @ projected).sum())
    square_loss = float(field @ (old_mass @ field) - projected @ rhs)
    return projected, {"mass_before_projection": before, "mass_after_projection": after,
                       "projection_l2_loss": float(np.sqrt(max(0.0, square_loss)))}
