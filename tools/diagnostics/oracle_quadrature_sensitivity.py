#!/usr/bin/env python3
"""Compare triangle-quadrature accuracy for saved GDM Gaussian-oracle samples.

Uses only Python's standard library and NumPy. It does not modify the samples or
project source. Copy into tools/diagnostics/oracle_quadrature_sensitivity.py and run it from
the repository root, passing one or more existing sample pickle paths.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _permutations3(values: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    # Stable unique permutations: some rules contain repeated barycentric values.
    return list(dict.fromkeys(itertools.permutations(values, 3)))


def triangle_rules() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return degree-2 and degree-5 symmetric triangle rules.

    Weights integrate over the reference triangle with area 1/2. The Q7
    rule is also used in a composite form obtained by recursively subdividing
    each triangle into 4^level subtriangles; this is more informative for a
    narrow Gaussian than assuming a single higher-degree rule is a reference.
    """
    r3_bary = np.array([
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ], dtype=np.float64)
    r3_w = np.full(3, 1.0 / 6.0, dtype=np.float64)

    r7_points: list[tuple[float, float, float]] = [(1.0 / 3.0,) * 3]
    r7_weights: list[float] = [0.1125]
    for values, weight in (
        ((0.059715871789770, 0.470142064105115, 0.470142064105115), 0.066197076394253),
        ((0.797426985353087, 0.101286507323456, 0.101286507323456), 0.0629695902724135),
    ):
        perms = _permutations3(values)
        r7_points.extend(perms)
        r7_weights.extend([weight] * len(perms))

    rules = {
        "q3": (r3_bary, r3_w),
        "q7": (np.asarray(r7_points, dtype=np.float64), np.asarray(r7_weights, dtype=np.float64)),
    }
    for name, (bary, weights) in rules.items():
        if not np.allclose(bary.sum(axis=1), 1.0, atol=1e-14):
            raise RuntimeError(f"{name}: invalid barycentric coordinates")
        if not np.isclose(weights.sum(), 0.5, atol=1e-12):
            raise RuntimeError(f"{name}: weights sum to {weights.sum()}, expected 0.5")
    if rules["q7"][0].shape[0] != 7:
        raise RuntimeError("Unexpected Q7 rule point count")
    return rules


def composite_rule(
    base_bary: np.ndarray, base_weights: np.ndarray, level: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a composite quadrature rule on a reference triangle.

    Every subdivision splits one triangle into four equal-area children, so
    each child quadrature weight is scaled by 1/4 per subdivision level.
    The returned points remain barycentric coordinates of the original cell.
    """
    if level < 0:
        raise ValueError("level must be non-negative")
    children = [np.eye(3, dtype=np.float64)]
    for _ in range(level):
        new_children = []
        for verts in children:
            a, b, c = verts
            ab = 0.5 * (a + b)
            bc = 0.5 * (b + c)
            ca = 0.5 * (c + a)
            new_children.extend([
                np.stack([a, ab, ca]),
                np.stack([ab, b, bc]),
                np.stack([ca, bc, c]),
                np.stack([ab, bc, ca]),
            ])
        children = new_children
    points = []
    weights = []
    scale = 4.0 ** level
    for verts in children:
        points.append(base_bary @ verts)
        weights.append(base_weights / scale)
    bary_all = np.concatenate(points, axis=0)
    weights_all = np.concatenate(weights, axis=0)
    if not np.isclose(weights_all.sum(), 0.5, atol=1e-11):
        raise RuntimeError(f"Composite weights sum to {weights_all.sum()} at level {level}")
    return bary_all, weights_all


def gaussian_value(x: np.ndarray, y: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    sigma = float(p["sigma"])
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"Invalid sigma={sigma}")
    dx = np.asarray(x, dtype=np.float64) - float(p["mu_x"])
    dy = np.asarray(y, dtype=np.float64) - float(p["mu_y"])
    return float(p["amplitude"]) * np.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))


def interpolation_error_integral(
    nodes: np.ndarray,
    cells: np.ndarray,
    params: dict[str, Any],
    bary: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, int]:
    nodes = np.asarray(nodes, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"nodes must have shape [N,2], got {nodes.shape}")
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError(f"cells must have shape [M,3], got {cells.shape}")
    if not np.isfinite(nodes).all():
        raise ValueError("non-finite mesh coordinates")
    tri = nodes[cells]  # [M,3,2]
    p0, p1, p2 = tri[:, 0], tri[:, 1], tri[:, 2]
    signed_det = (
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )
    invalid_count = int(np.count_nonzero(signed_det <= 0.0))
    det_measure = np.abs(signed_det)
    nodal_u = gaussian_value(nodes[:, 0], nodes[:, 1], params)
    tri_u = nodal_u[cells]
    total = 0.0
    for bary_q, weight in zip(bary, weights):
        xq = np.einsum("i,mij->mj", bary_q, tri)[:, 0]
        yq = np.einsum("i,mij->mj", bary_q, tri)[:, 1]
        u_exact = gaussian_value(xq, yq, params)
        u_interp = np.einsum("i,mi->m", bary_q, tri_u)
        total += float(np.sum((u_exact - u_interp) ** 2 * det_measure * weight))
    return total, float(np.min(signed_det)), invalid_count


def load_meshes(sample_path: Path) -> list[tuple[str, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]]:
    with sample_path.open("rb") as f:
        sample = pickle.load(f)
    metadata = sample.get("metadata", {})
    params = metadata.get("parameters", {})
    required = ("mu_x", "mu_y", "sigma", "amplitude")
    if not all(k in params for k in required):
        raise ValueError(f"{sample_path}: missing Gaussian parameters in metadata")
    initial_nodes = np.asarray(sample["mesh"]["nodes"], dtype=np.float64)
    cells = np.asarray(sample["mesh"]["cells"], dtype=np.int64)
    target = sample.get("target", {})
    if "optimal_nodes" in target:
        target_nodes = np.asarray(target["optimal_nodes"], dtype=np.float64)
    elif "displacement" in target:
        target_nodes = initial_nodes + np.asarray(target["displacement"], dtype=np.float64)
    else:
        raise ValueError(f"{sample_path}: target lacks optimal_nodes/displacement")
    prefix = str(metadata.get("sample_id", sample_path.stem))
    split = str(metadata.get("split", "unknown"))
    return [
        ("initial", initial_nodes, cells, params, {"sample_id": prefix, "split": split, "sample_path": str(sample_path)}),
        ("target", target_nodes, cells, params, {"sample_id": prefix, "split": split, "sample_path": str(sample_path)}),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="+", type=Path, help="Existing sample .pkl files; no files are modified")
    parser.add_argument("--csv", type=Path, default=Path("results/oracle_quadrature_sensitivity.csv"), help="Output CSV path")
    parser.add_argument("--json", type=Path, default=Path("results/oracle_quadrature_sensitivity.json"), help="Output JSON path")
    args = parser.parse_args()

    base_rules = triangle_rules()
    q3_bary, q3_weights = base_rules["q3"]
    q7_bary, q7_weights = base_rules["q7"]
    rules = {"q3_base": (q3_bary, q3_weights), "q7_base": (q7_bary, q7_weights)}
    for level in range(1, 5):
        rules[f"q7_composite_l{level}"] = composite_rule(q7_bary, q7_weights, level)
    rows: list[dict[str, Any]] = []
    for sample_path in args.samples:
        if not sample_path.is_file():
            raise FileNotFoundError(sample_path)
        for mesh_name, nodes, cells, params, metadata in load_meshes(sample_path):
            q_results: dict[str, float] = {}
            signed_det_min = float("nan")
            inverted_count = 0
            for qname, (bary, weights) in rules.items():
                val, min_det, n_inverted = interpolation_error_integral(nodes, cells, params, bary, weights)
                q_results[qname] = val
                signed_det_min = min_det
                inverted_count = n_inverted
            ref = q_results["q7_composite_l4"]
            denom = max(abs(ref), np.finfo(np.float64).tiny)
            row = {
                **metadata,
                "mesh": mesh_name,
                "sigma": float(params["sigma"]),
                "h_over_sigma_nominal": (1.0 / 32.0) / float(params["sigma"]),
                "n_nodes": int(nodes.shape[0]),
                "n_cells": int(cells.shape[0]),
                "min_signed_jacobian": signed_det_min,
                "nonpositive_jacobian_cells": inverted_count,
                **{f"interpolation_error_{k}": v for k, v in q_results.items()},
                "q3_relative_difference_vs_composite_l4_pct": 100.0 * abs(q_results["q3_base"] - ref) / denom,
                "q7_relative_difference_vs_composite_l4_pct": 100.0 * abs(q_results["q7_base"] - ref) / denom,
                "composite_l3_relative_difference_vs_l4_pct": 100.0 * abs(q_results["q7_composite_l3"] - ref) / denom,
            }
            rows.append(row)
            print(
                f"{row['sample_id']} | {mesh_name:7s} | sigma={row['sigma']:.6g} | "
                f"h/sigma={row['h_over_sigma_nominal']:.3g} | "
                f"E(Q3)={q_results['q3_base']:.8e} | "
                f"E(Q7)={q_results['q7_base']:.8e} | "
                f"E(comp.L2)={q_results['q7_composite_l2']:.8e} | "
                f"E(comp.L3)={q_results['q7_composite_l3']:.8e} | "
                f"E(comp.L4)={ref:.8e} | "
                f"rel(L3,L4)={row['composite_l3_relative_difference_vs_l4_pct']:.4g}%"
            )

    for output in (args.csv, args.json):
        output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.json.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved CSV: {args.csv.resolve()}")
    print(f"Saved JSON: {args.json.resolve()}")
    print("Interpret composite refinement differences as a sensitivity diagnostic, not a proof of exact quadrature.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
