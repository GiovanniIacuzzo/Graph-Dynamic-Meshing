"""Stage-C campaign whose numerical kernels run exclusively in C++/deal.II.

Python remains responsible for orchestration, plotting, the trained GNN, and the
low-budget offline oracle used only to propose C4 target meshes. FEM, ALE,
continuous error integration, mesh metrics, safety filtering and both remappers
are executed by ``gdm._gdm_core``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.utils import ensure_dir, load_config
from gdm import NativeDynamicRuntime, NativeMeshAdapter
from inference import MeshMoverInferenceEngine


NON_SERIALIZABLE = {"final_mesh", "final_u", "snapshots", "recorded_targets"}


def _plain_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep finite, JSON-compatible scalar/series metrics; omit heavy state arrays."""
    clean: dict[str, Any] = {}
    for key, value in result.items():
        if key in NON_SERIALIZABLE:
            continue
        if isinstance(value, np.ndarray):
            clean[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer, np.bool_)):
            clean[key] = value.item()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            try:
                clean[key] = np.asarray(value).tolist()
            except Exception:
                continue
    return clean


def _timing_summary(samples: list[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "total_wall_time_s", "fem_time_s", "adapt_time_s", "safety_time_s",
        "remap_time_s", "sum_components_s", "other_overhead_s",
    )
    out: dict[str, float] = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in samples], dtype=np.float64)
        out[field] = float(np.median(values))
        out[f"{field}_q25"] = float(np.quantile(values, 0.25))
        out[f"{field}_q75"] = float(np.quantile(values, 0.75))
    walls = np.asarray([float(row["total_wall_time_s"]) for row in samples], dtype=np.float64)
    out["wall_time_iqr_s"] = float(np.quantile(walls, 0.75) - np.quantile(walls, 0.25))
    out["n_timing_repeats"] = int(walls.size)
    out["raw_wall_times_s"] = walls.tolist()
    return out


class StageCCampaign:
    METHODS = [
        ("C0_fixed", "fixed", "Fixed FEM"),
        ("C1_gdm_p1", "gdm_p1", r"GDM + $P_1$ remap"),
        ("C2_gdm_l2", "gdm_l2", r"GDM + $L^2$ projection"),
        ("C3_gdm_ale", "gdm_ale", "GDM + ALE"),
        ("C4_oracle_ale", "oracle_ale", "Low-budget oracle + ALE"),
    ]

    def __init__(
        self,
        config_path: str | Path = "config/config.yaml",
        checkpoint_path: str | Path = "results/stage_a/checkpoints/best_model.pt",
        output_dir: str | Path = "results/stage_c",
        n_timing_repeats: int = 5,
        nx: int = 32,
        ny: int = 32,
        T: float = 0.5,
        dt: float = 0.005,
        adapt_interval: int = 10,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.config = load_config(str(self.config_path))
        self.output_dir = ensure_dir(output_dir)
        self.figures_dir = ensure_dir(self.output_dir / "figures")
        self.tables_dir = ensure_dir(self.output_dir / "tables")
        self.raw_dir = ensure_dir(self.output_dir / "raw_metrics")
        self.repeats = max(1, int(n_timing_repeats))
        self.nx, self.ny = int(nx), int(ny)
        self.T, self.dt = float(T), float(dt)
        self.adapt_interval = int(adapt_interval)
        self.domain = dict(self.config["geometry"]["domain"])
        self.epsilon = float(self.config["geometry"]["safety"]["epsilon"])
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.engine = None
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint GNN non trovato: {self.checkpoint_path}")
        self.engine = MeshMoverInferenceEngine(
            checkpoint_path=self.checkpoint_path, config=self.config, device="cpu"
        )
        oracle_cfg = {
            "objective": self.config.get("oracle", {}).get("objective", {}),
            "epsilon": self.epsilon,
            # This is intentionally a low-budget diagnostic; not a converged oracle bound.
            "max_iter": 3,
        }
        self.runtime = NativeDynamicRuntime(
            domain_cfg=self.domain,
            nx=self.nx,
            ny=self.ny,
            bx=0.4,
            by=0.2,
            nu=1.0e-3,
            sigma=0.06,
            beta=0.5,
            T=self.T,
            dt=self.dt,
            adapt_interval=self.adapt_interval,
            inference_engine=self.engine,
            oracle_cfg=oracle_cfg,
        )

    def _run_with_timing_repeats(
        self,
        method: str,
        replay_targets: list[np.ndarray] | None = None,
        record_targets: bool = False,
    ) -> dict[str, Any]:
        detailed = self.runtime.run_episode(
            method=method,
            replay_targets=replay_targets,
            compute_diagnostics=True,
            record_targets=record_targets,
        )
        # Timing runs omit diagnostics so the wall time excludes error integration/plot snapshots.
        samples: list[dict[str, Any]] = []
        for _ in range(self.repeats):
            samples.append(self.runtime.run_episode(
                method=method,
                replay_targets=replay_targets,
                compute_diagnostics=False,
                record_targets=False,
            ))
        detailed.update(_timing_summary(samples))
        detailed["backend"] = "C++20/deal.II via pybind11; Python GNN inference/oracle proposal only"
        return detailed

    def run_benchmark(self) -> dict[str, Any]:
        print("Stage C: native C++/deal.II FEM/ALE/remapping campaign")
        fixed_mesh = NativeMeshAdapter.from_domain(self.domain, nx=self.nx, ny=self.ny)
        results: dict[str, dict[str, Any]] = {}
        for key, method, label in self.METHODS:
            print(f"  -> {key}: {label}")
            results[key] = self._run_with_timing_repeats(
                method, record_targets=(key == "C3_gdm_ale")
            )
            print(
                f"     final L2={results[key]['final_l2'] * 100:.3f}% | "
                f"time-int L2={results[key]['time_int_l2'] * 100:.3f}% | "
                f"wall={results[key]['total_wall_time_s']:.3f}s"
            )

        target_nodes = results["C3_gdm_ale"]["recorded_targets"]
        target_replay: dict[str, dict[str, Any]] = {}
        for key, method in (
            ("R1_gdm_p1_replay", "gdm_p1"),
            ("R2_gdm_l2_replay", "gdm_l2"),
            ("R3_gdm_ale_replay", "gdm_ale"),
        ):
            print(f"  -> Controlled target replay: {key}")
            target_replay[key] = self._run_with_timing_repeats(method, replay_targets=target_nodes)

        target_archive = self.raw_dir / "stage_c_replay_targets.npz"
        np.savez_compressed(
            target_archive,
            target_nodes=np.asarray(target_nodes, dtype=np.float64),
            adaptation_steps=np.asarray(results["C3_gdm_ale"]["adaptation_steps"], dtype=np.int64),
            adaptation_times=np.asarray(results["C3_gdm_ale"]["adaptation_times"], dtype=np.float64),
            cells=fixed_mesh.cells,
            initial_nodes=fixed_mesh.nodes,
            nx=self.nx, ny=self.ny, dt=self.dt, T=self.T, adapt_interval=self.adapt_interval,
            checkpoint_sha256=self._sha256(self.checkpoint_path),
        )

        payload = {
            "metadata": {
                "backend": "C++20/deal.II native numerical kernels via pybind11",
                "python_components": ["GNN inference", "low-budget oracle target proposal for C4"],
                "T": self.T, "dt": self.dt, "nx": self.nx, "ny": self.ny,
                "adapt_interval": self.adapt_interval, "timing_repeats": self.repeats,
                "checkpoint": str(self.checkpoint_path),
                "checkpoint_sha256": self._sha256(self.checkpoint_path),
                "oracle_caveat": "C4 target optimizer is limited to 3 L-BFGS iterations; not a converged oracle upper bound.",
                "replay_targets": str(target_archive),
            },
            "closed_loop": {k: _plain_result(v) for k, v in results.items()},
            "target_replay": {k: _plain_result(v) for k, v in target_replay.items()},
        }
        (self.raw_dir / "stage_c_benchmark_results.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        self._write_latex_table(results)
        self._write_figures(results, fixed_mesh.cells)
        print(f"Results: {self.raw_dir / 'stage_c_benchmark_results.json'}")
        return payload

    @staticmethod
    def _sha256(path: Path) -> str:
        import hashlib
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _write_latex_table(self, results: dict[str, dict[str, Any]]) -> None:
        rows = []
        for key, _method, label in self.METHODS:
            r = results[key]
            adapt_s = float(r["adapt_time_s"]) + float(r["safety_time_s"])
            rows.append(
                f"{label} & {100*r['final_l2']:.3f}\\% & {100*r['time_int_l2']:.3f}\\% & "
                f"{r['min_jacobian']:.3e} & {r['fem_time_s']:.3f} & {r['remap_time_s']:.3f} & "
                f"{adapt_s:.3f} & {r['total_wall_time_s']:.3f} $\\pm$ {r['wall_time_iqr_s']:.3f} \\\\"
            )
        table = (
            "\\begin{table}[htbp]\n\\centering\n\\footnotesize\n"
            "\\caption{Stage-C benchmark with C++/deal.II numerical kernels. Wall-clock medians and IQRs are measured on runs without diagnostic integration.}\n"
            "\\label{tab:stage_c_native_summary}\n"
            "\\begin{tabular}{lccccccc}\n\\toprule\n"
            "Method & $E_{L^2}(T)$ & $\\overline E_{L^2}$ & $\\min J_e$ & "
            "$T_{\\mathrm{FEM/ALE}}$ [s] & $T_{\\mathrm{remap}}$ [s] & $T_{\\mathrm{adapt}}$ [s] & Wall [s]\\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )
        (self.tables_dir / "table_stage_c_summary.tex").write_text(table, encoding="utf-8")

    def _write_figures(self, results: dict[str, dict[str, Any]], cells: np.ndarray) -> None:
        t = results["C0_fixed"]["time"]
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        for key, _method, label in self.METHODS:
            ax.plot(t, 100*np.asarray(results[key]["rel_l2_error"]), label=label)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Relative $L^2$ error [%]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig_c3_dynamic_error.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for key in ("C0_fixed", "C2_gdm_l2", "C3_gdm_ale", "C4_oracle_ale"):
            ax.plot(t, results[key]["min_j_series"], label=key)
        ax.axhline(self.epsilon, linestyle=":", label=rf"Safety floor $\\epsilon={self.epsilon:g}$")
        ax.set_yscale("log")
        ax.set_xlabel("Time $t$")
        ax.set_ylabel(r"Minimum oriented Jacobian $\min_e J_e$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig_c4_mesh_validity.png", dpi=300)
        plt.close(fig)

        keys = ["C0_fixed", "C1_gdm_p1", "C2_gdm_l2", "C3_gdm_ale"]
        labels = ["Fixed", "$P_1$", "$L^2$", "ALE"]
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        x = np.arange(len(keys))
        bottom = np.zeros(len(keys))
        for field, label in (
            ("fem_time_s", "FEM/ALE"), ("remap_time_s", "Remapping"),
            ("adapt_time_s", "GNN/oracle"), ("safety_time_s", "Safety filter"),
            ("other_overhead_s", "Other overhead"),
        ):
            v = np.asarray([results[k][field] for k in keys], dtype=np.float64)
            ax.bar(x, v, bottom=bottom, label=label)
            bottom += v
        ax.set_xticks(x, labels)
        ax.set_ylabel("Component time [s]")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig_c5_runtime_decomposition.png", dpi=300)
        plt.close(fig)

        snaps = results["C3_gdm_ale"]["snapshots"]
        fig, axes = plt.subplots(2, len(snaps), figsize=(3.0*len(snaps), 5.5), squeeze=False)
        for col, (snap_t, snap_nodes, snap_u) in enumerate(snaps):
            tri = mtri.Triangulation(snap_nodes[:, 0], snap_nodes[:, 1], cells)
            axes[0, col].triplot(tri, color="tab:blue", linewidth=0.35)
            axes[0, col].set_title(f"Mesh t={snap_t:.3f}")
            axes[1, col].tricontourf(tri, snap_u, levels=24)
            axes[1, col].set_title(f"$u_h$, t={snap_t:.3f}")
            for row in range(2):
                axes[row, col].set_aspect("equal")
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
        fig.tight_layout()
        fig.savefig(self.figures_dir / "fig_c6_qualitative_tracking.png", dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default="results/stage_a/checkpoints/best_model.pt")
    parser.add_argument("--output_dir", default="results/stage_c")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--adapt-interval", type=int, default=10)
    args = parser.parse_args()
    campaign = StageCCampaign(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        n_timing_repeats=args.repeats,
        nx=args.nx,
        ny=args.ny,
        T=args.T,
        dt=args.dt,
        adapt_interval=args.adapt_interval,
    )
    campaign.run_benchmark()


if __name__ == "__main__":
    main()
