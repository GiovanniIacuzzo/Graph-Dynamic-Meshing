"""Native diagnostics for remapping accumulation and controlled transient replay."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.utils import ensure_dir, load_config
from gdm import (
    NativeMeshAdapter, NativeDynamicRuntime, create_pde_model,
    compute_native_continuous_errors, compute_native_field_gradients,
    apply_native_safety_filter, remap_p1_native, remap_l2_native,
)
from inference import MeshMoverInferenceEngine


class StageBDiagnostics:
    def __init__(self, config_path: str = "config/config.yaml", checkpoint_path: str = "results/stage_a/checkpoints/best_model.pt"):
        self.cfg = load_config(str(Path(config_path).resolve()))
        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint GNN non trovato: {checkpoint}")
        self.engine = MeshMoverInferenceEngine(checkpoint, self.cfg, device="cpu")
        self.output_dir = ensure_dir("results/stage_b/native_diagnostics")
        self.domain = self.cfg["geometry"]["domain"]
        self.mesh = NativeMeshAdapter.from_domain(self.domain, nx=32, ny=32)
        self.pde = create_pde_model(bx=0.4, by=0.2, nu=1e-3, sigma=0.06, beta=0.5, T=0.5)
        self.runtime = NativeDynamicRuntime(
            domain_cfg=self.domain, nx=32, ny=32, bx=0.4, by=0.2,
            nu=1e-3, sigma=0.06, beta=0.5, T=0.5, dt=0.005,
            adapt_interval=10, inference_engine=self.engine,
        )

    def run(self) -> dict:
        u0 = np.asarray(self.pde.evaluate_solution(self.mesh.nodes, 0.0), dtype=np.float64)
        e0, h0 = compute_native_continuous_errors(self.mesh, u0, self.pde, 0.0)
        grad_u0 = compute_native_field_gradients(self.mesh, u0, self.pde, 0.0)
        raw = self.engine.predict_raw(self.mesh, u0, grad_u0)
        filtered = apply_native_safety_filter(
            self.mesh, raw["raw_displacement"], epsilon=float(self.cfg["geometry"]["safety"]["epsilon"]),
            safety_margin=float(self.cfg["geometry"]["safety"].get("safety_margin", 1e-6)), lock_boundary_nodes=True,
        )
        adapted = self.mesh.update_coordinates(filtered["safe_nodes"], enforce_fixed_boundary=True)
        p1_once = remap_p1_native(self.mesh, u0, adapted)["remapped_field"]
        l2_once = remap_l2_native(self.mesh, u0, adapted)["remapped_field"]
        e_p1, h_p1 = compute_native_continuous_errors(adapted, p1_once, self.pde, 0.0)
        e_l2, h_l2 = compute_native_continuous_errors(adapted, l2_once, self.pde, 0.0)

        p1_loop, l2_loop = u0.copy(), u0.copy()
        for _ in range(5):
            p1_loop = remap_p1_native(self.mesh, p1_loop, adapted)["remapped_field"]
            p1_loop = remap_p1_native(adapted, p1_loop, self.mesh)["remapped_field"]
            l2_loop = remap_l2_native(self.mesh, l2_loop, adapted)["remapped_field"]
            l2_loop = remap_l2_native(adapted, l2_loop, self.mesh)["remapped_field"]
        e_p1_loop, _ = compute_native_continuous_errors(self.mesh, p1_loop, self.pde, 0.0)
        e_l2_loop, _ = compute_native_continuous_errors(self.mesh, l2_loop, self.pde, 0.0)

        # The learned target path is generated once and then replayed unchanged for each method.
        target_result = self.runtime.run_episode("gdm_ale", record_targets=True, compute_diagnostics=True)
        targets = target_result["recorded_targets"]
        transient = {"fixed": self.runtime.run_episode("fixed", compute_diagnostics=True)}
        for method in ("gdm_p1", "gdm_l2", "gdm_ale"):
            transient[method] = self.runtime.run_episode(method, replay_targets=targets, compute_diagnostics=True)
        dynamic = {
            key: {
                "final_l2": float(value["final_l2"]), "final_h1": float(value["final_h1"]),
                "time_int_l2": float(value["time_int_l2"]), "min_jacobian": float(value["min_jacobian"]),
                "total_wall_time_s": float(value["total_wall_time_s"]),
            } for key, value in transient.items()
        }
        payload = {
            "backend": "C++20/deal.II for numerical kernels; Python only for GNN forward pass",
            "static_remap": {
                "initial_relative_l2": e0, "initial_relative_h1": h0,
                "p1_after_one_transfer_l2": float(e_p1), "p1_after_one_transfer_h1": float(h_p1),
                "l2_after_one_transfer_l2": float(e_l2), "l2_after_one_transfer_h1": float(h_l2),
                "p1_after_10_transfers_l2": float(e_p1_loop), "l2_after_10_transfers_l2": float(e_l2_loop),
                "safety_alpha": float(filtered["alpha_global"]),
            },
            "dynamic_replay": dynamic,
            "interpretation_note": "Replay isolates transfer method for a fixed sequence of native-safe target meshes; it is not closed-loop performance.",
        }
        (self.output_dir / "stage_b_native_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload["static_remap"], indent=2))
        print(json.dumps(dynamic, indent=2))
        return payload


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default="results/stage_a/checkpoints/best_model.pt")
    args = parser.parse_args()
    StageBDiagnostics(args.config, args.checkpoint).run()


if __name__ == "__main__":
    main()
