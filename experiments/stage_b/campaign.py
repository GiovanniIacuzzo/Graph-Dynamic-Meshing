"""Stage-B dynamic comparison running all numerical kernels through C++/deal.II."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.utils import ensure_dir, load_config
from gdm import NativeDynamicRuntime
from inference import MeshMoverInferenceEngine
from plotting.dynamics import plot_time_series


def _clean(row: dict) -> dict:
    ignore = {"final_mesh", "final_u", "snapshots", "recorded_targets"}
    output = {}
    for k, v in row.items():
        if k in ignore:
            continue
        if isinstance(v, np.ndarray):
            output[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer, np.bool_)):
            output[k] = v.item()
        elif isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            output[k] = v
    return output


class StageBCampaign:
    def __init__(
        self,
        config_path: str = "config/stage_b.yaml",
        checkpoint_path: str = "results/stage_a_oracle_v4/checkpoints/best_model.pt",
        output_dir: str | Path = "results/stage_b_v4",
    ):
        self.cfg = load_config(str(Path(config_path).resolve()))
        self.out = ensure_dir(output_dir)
        self.figures = ensure_dir(self.out / "figures")
        self.tables = ensure_dir(self.out / "tables")
        self.raw = ensure_dir(self.out / "raw_metrics")
        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint GNN non trovato: {checkpoint}")
        engine = MeshMoverInferenceEngine(checkpoint, self.cfg, device="cpu")
        domain = self.cfg["geometry"]["domain"]
        self.runtime = NativeDynamicRuntime(
            domain_cfg=domain, nx=32, ny=32, bx=0.4, by=0.2, nu=1e-3,
            sigma=0.06, beta=0.5, T=0.5, dt=0.005, adapt_interval=10,
            inference_engine=engine,
            oracle_cfg={"objective": self.cfg["oracle"]["objective"],
                        "epsilon": self.cfg["geometry"]["safety"]["epsilon"],
                        "max_iter": 3},
        )

    def run_benchmark(self) -> dict:
        methods = [
            ("B0_fixed", "fixed", "Fixed FEM"),
            ("B1_gdm_p1", "gdm_p1", "GDM + native P1 remap"),
            ("B2_gdm_l2", "gdm_l2", "GDM + native L2 remap"),
            ("B3_oracle_ale_low_budget", "oracle_ale", "Low-budget oracle target + native ALE"),
        ]
        rows = {}
        for key, method, label in methods:
            print(f"[Stage B] {label}")
            result = self.runtime.run_episode(method, compute_diagnostics=True)
            rows[key] = result
            (self.raw / f"{key}_metrics.json").write_text(json.dumps(_clean(result), indent=2), encoding="utf-8")
            print(f"  final L2={result['final_l2']*100:.3f}%  time-integrated L2={result['time_int_l2']*100:.3f}%")
        payload = {
            "backend": "C++20/deal.II for mesh, safety filter, FEM, ALE, remapping, metrics and errors; Python for GNN and oracle target proposal",
            "oracle_note": "B3 target optimizer has a low iteration budget (3 iterations); it is not a converged oracle bound.",
            "results": {k: _clean(v) for k, v in rows.items()},
        }
        (self.raw / "stage_b_native_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        latex = ["\\begin{table}[t]", "\\centering", "\\small", "\\caption{Stage-B native dynamic benchmark ($T=0.5$, $\\Delta t=0.005$, mesh $32\\times32$).}", "\\label{tab:stage_b_native}", "\\begin{tabular}{lrrrr}", "\\toprule", "Method & Final $L^2$ [\\%] & Time-integrated $L^2$ [\\%] & Min $J_e$ & Wall [s] \\\\", "\\midrule"]
        for key, _method, label in methods:
            r = rows[key]
            latex.append(f"{label} & {100*r['final_l2']:.3f} & {100*r['time_int_l2']:.3f} & {r['min_jacobian']:.3e} & {r['total_wall_time_s']:.3f} " + r"\\")
        latex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        (self.tables / "table_stage_b_summary.tex").write_text("\n".join(latex)+"\n", encoding="utf-8")
        series = {key: 100 * np.asarray(rows[key]["rel_l2_error"]) for key, _method, _label in methods}
        labels = {key: label for key, _method, label in methods}
        plot_time_series(
            rows["B0_fixed"]["time"], series, labels,
            self.figures / "stage_b_native_error_curves.pdf",
            ylabel=r"Relative $L^2$ error [\%]", title="Stage-B transient PDE error",
        )
        return payload


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/stage_b.yaml")
    parser.add_argument("--checkpoint", default="results/stage_a_oracle_v4/checkpoints/best_model.pt")
    parser.add_argument("--output_dir", default="results/stage_b_v4")
    args = parser.parse_args()
    StageBCampaign(args.config, args.checkpoint, args.output_dir).run_benchmark()


if __name__ == "__main__":
    main()
