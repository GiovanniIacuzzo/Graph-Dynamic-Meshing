"""Case-paired contrasts without treating time steps or repetitions as cases."""
from collections import defaultdict

import numpy as np


CONTRASTS = (
    ("uniform", "early"),
    ("remap", "uniform"),
    ("fixed_fem", "transport"),
    ("fixed_ale", "transport"),
    ("field", "transport"),
    ("field", "gnn_window"),
    ("transport", "gnn_window"),
    ("gnn_instantaneous", "gnn_window"),
)


def average_seeds(rows):
    """Average errors over trained seeds within each case/method, not across cases."""
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[name] for name in ("profile", "resolution", "dt", "method", "case_id"))
        training_seed = str(row.get("training_seed", ""))
        if training_seed in groups[key]:
            raise ValueError(f"Duplicate method/case/seed row: {key}, {training_seed}")
        groups[key][training_seed] = row
    expected_seeds = {}
    result = []
    for (_, _, _, method, _), replicas in groups.items():
        seeds = frozenset(replicas)
        if seeds != expected_seeds.setdefault(method, seeds):
            raise ValueError(f"Incomplete seed coverage for {method}.")
        values = list(replicas.values())
        representative = dict(values[0], seed_count=len(seeds))
        for metric in ("relative_l2_average", "relative_l2_final"):
            representative[metric] = float(np.mean([float(row[metric]) for row in values]))
        if "solver_and_controller_seconds" in representative:
            representative["solver_and_controller_seconds"] = float(np.median(
                [float(row["solver_and_controller_seconds"]) for row in values]))
        result.append(representative)
    return result


def seeded_summary(rows, seed):
    deterministic = [row for row in rows if str(row.get("training_seed", "")) == ""]
    seeds = sorted({str(row.get("training_seed", "")) for row in rows} - {""})
    return [{"training_seed": training_seed, **result}
            for training_seed in seeds
            for result in paired_summary(
                deterministic + [row for row in rows if str(row.get("training_seed", "")) == training_seed], seed)
            if result["candidate"].startswith("gnn_")]


def paired_summary(rows, seed):
    groups = defaultdict(dict)
    for row in average_seeds(rows):
        stratum = (row["profile"], int(row["resolution"]), float(row["dt"]))
        key = (row["method"], row["case_id"])
        if key in groups[stratum]:
            raise ValueError(f"Duplicate method/case row: {stratum}, {key}")
        groups[stratum][key] = row
    output = []
    rng = np.random.default_rng(seed)
    for (profile, resolution, dt), observations in sorted(groups.items()):
        methods = {method for method, _ in observations}
        cases = sorted({case for _, case in observations})
        if set(observations) != {(method, case) for method in methods for case in cases}:
            raise ValueError(f"Incomplete method/case pairs for {profile}, {resolution}, {dt}.")
        for baseline, candidate in CONTRASTS:
            if not {baseline, candidate} <= methods:
                continue
            indices = rng.integers(0, len(cases), size=(5000, len(cases)))
            for metric in ("relative_l2_average", "relative_l2_final"):
                difference = np.array([float(observations[baseline, case][metric]) -
                                       float(observations[candidate, case][metric]) for case in cases])
                if not np.isfinite(difference).all():
                    raise ValueError("Nonfinite metric.")
                low, high = (np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
                             if len(cases) >= 2 else (None, None))
                output.append({
                    "profile": profile, "resolution": resolution, "dt": dt, "metric": metric,
                    "baseline": baseline, "candidate": candidate, "cases": len(cases),
                    "mean_improvement": float(difference.mean()),
                    "bootstrap_low": low, "bootstrap_high": high,
                    "wins": int((difference > 0).sum()), "ties": int((difference == 0).sum()),
                    "losses": int((difference < 0).sum()),
                })
    return output
