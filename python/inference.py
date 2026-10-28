"""
inference.py
Motore di inferenza per il framework Graph-Dynamic Meshing (GDM).

Carica un checkpoint addestrato della MeshMoverGNN, costruisce il grafo a partire
dalla mesh corrente e dallo stato fisico (soluzione u e gradiente grad_u),
predice lo spostamento nodale \\Delta X_{GNN} e applica il Deterministic Safety Filter
O(M) per certificare matematicamente che \\min_e J_e(\\alpha_{global}) >= \\epsilon.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import torch

from utils.utils import load_config, get_device, load_checkpoint
from dataset.dataset import build_graph_sample_from_arrays
from model.model import build_model_from_config
from gdm.native_api import (
    NativeMeshAdapter, apply_native_safety_filter, compute_native_mesh_metrics,
)


class MeshMoverInferenceEngine:
    """
    Incapsula il modello GNN addestrato e il filtro di sicurezza geometrico
    per l'uso sia nei test statici (Stage A) sia nel loop dinamico PDE (Stage B).
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")

        if device is None:
            device_str = config.get("experiment", {}).get("device", "auto") if config else "auto"
            self.device = get_device(device_str)
        elif isinstance(device, str):
            self.device = get_device(device)
        else:
            self.device = device

        # Carica il payload del checkpoint per recuperare eventuale config salvata
        try:
            ckpt_payload = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )
        except TypeError:  # Compatibilità con versioni di PyTorch meno recenti.
            ckpt_payload = torch.load(self.checkpoint_path, map_location=self.device)
        self.config = config if config is not None else ckpt_payload.get("config", {})
        self.normalize_field_scale = bool(
            self.config.get("features", {}).get("normalize_field_scale", False)
        )
        if not self.config:
            raise ValueError("Configuration dictionary is required if not embedded in checkpoint.")

        # Inizializza l'architettura GNN e carica i pesi
        self.model = build_model_from_config(self.config).to(self.device)
        load_checkpoint(self.checkpoint_path, self.model, optimizer=None, device=self.device)
        self.model.eval()

        # Parametri del Safety Filter
        geometry_cfg = self.config.get("geometry", {})
        safety_cfg = self.config.get("safety", geometry_cfg.get("safety", {}))
        self.epsilon: float = float(safety_cfg.get("epsilon", 1e-4))
        self.safety_margin: float = float(safety_cfg.get("safety_margin", 1e-6))
        self.tol_deg: float = float(safety_cfg.get("tol_deg", 1e-12))

    def prepare_graph(
        self,
        mesh: NativeMeshAdapter,
        u: np.ndarray,
        grad_u: np.ndarray,
    ):
        """Costruisce l'oggetto PyG Data normalizzato e lo sposta sul device."""
        graph_data = build_graph_sample_from_arrays(
            nodes=mesh.nodes,
            cells=mesh.cells,
            u=u,
            grad_u=grad_u,
            target_disp=None,
            normalize_field_scale=self.normalize_field_scale,
            native_mesh=mesh,
        )
        return graph_data.to(self.device)

    @torch.no_grad()
    def predict_raw(
        self,
        mesh: NativeMeshAdapter,
        u: np.ndarray,
        grad_u: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Esegue esclusivamente l'inferenza neurale (GNN forward pass) e restituisce
        lo spostamento nodale grezzo [N, 2] con condizioni al contorno (spostamento
        nullo sui nodi di bordo) imposte dal modello, SENZA applicare alcun safety filter.
        """
        u_arr = np.asarray(u, dtype=np.float64)
        grad_arr = np.asarray(grad_u, dtype=np.float64)
        if u_arr.shape != (mesh.num_nodes,):
            raise ValueError(
                f"Expected u of shape ({mesh.num_nodes},), got {u_arr.shape}."
            )
        if grad_arr.shape != (mesh.num_nodes, 2):
            raise ValueError(
                f"Expected grad_u of shape ({mesh.num_nodes}, 2), got {grad_arr.shape}."
            )
        if not np.all(np.isfinite(u_arr)) or not np.all(np.isfinite(grad_arr)):
            raise ValueError("Input fields u or grad_u contain non-finite (NaN/Inf) values.")

        t0_infer = time.perf_counter()
        graph_data = self.prepare_graph(mesh, u_arr, grad_arr)
        pred_disp_tensor = self.model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        disp_raw = pred_disp_tensor.detach().cpu().numpy().astype(np.float64)

        if disp_raw.shape != (mesh.num_nodes, 2):
            raise RuntimeError(
                f"Malformed GNN output shape: expected ({mesh.num_nodes}, 2), got {disp_raw.shape}."
            )
        if not np.all(np.isfinite(disp_raw)):
            raise RuntimeError("GNN predicted non-finite (NaN or Inf) nodal displacements.")

        # Garanzia esplicita del vincolo di Dirichlet geometrico sui nodi di bordo
        if mesh.boundary_nodes.size > 0:
            disp_raw[mesh.boundary_nodes] = 0.0

        t_infer = time.perf_counter() - t0_infer
        return {
            "raw_displacement": disp_raw,
            "inference_time": t_infer,
        }

    @torch.no_grad()
    def predict(
        self,
        mesh: NativeMeshAdapter,
        u: np.ndarray,
        grad_u: np.ndarray,
    ) -> Dict[str, Any]:
        """Run GNN inference followed by the native C++ deterministic filter.

        The PDE state, mesh, geometric diagnostics and safety decision are all
        handled through ``gdm._gdm_core``. Python/PyTorch is used only to build
        the graph features and evaluate the neural network.
        """
        native_mesh = mesh if isinstance(mesh, NativeMeshAdapter) else NativeMeshAdapter.from_arrays(
            mesh.nodes, mesh.cells, getattr(mesh, "boundary_nodes", None)
        )
        raw_res = self.predict_raw(native_mesh, u, grad_u)
        disp_raw = np.asarray(raw_res["raw_displacement"], dtype=np.float64)
        t_filter = time.perf_counter()
        filt = apply_native_safety_filter(
            native_mesh,
            disp_raw,
            epsilon=self.epsilon,
            safety_margin=self.safety_margin,
            tol_deg=self.tol_deg,
            lock_boundary_nodes=True,
        )
        t_filter = time.perf_counter() - t_filter

        safe_nodes = np.asarray(filt["safe_nodes"], dtype=np.float64)
        safe_disp = safe_nodes - native_mesh.nodes
        adapted_mesh = native_mesh.update_coordinates(safe_nodes, enforce_fixed_boundary=True)
        metrics_before = compute_native_mesh_metrics(native_mesh)
        min_j_before = float(metrics_before["min_jacobian"])
        min_j_after = float(filt["min_jacobian_after"])
        alpha = float(filt["alpha_global"])
        t_infer = float(raw_res["inference_time"])
        is_safe = bool(alpha >= 1.0 - 1e-12 and filt.get("is_safe", False))
        return {
            # Canonical names.
            "displacement": safe_disp,
            "raw_displacement": disp_raw,
            "adapted_mesh": adapted_mesh,
            "alpha": alpha,
            "safety_alpha": alpha,
            "is_safe": is_safe,
            "min_j_before": min_j_before,
            "min_j_after": min_j_after,
            "inference_time": t_infer,
            "filter_time": t_filter,
            "total_time": t_infer + t_filter,
            "timing": {"gnn_time": t_infer, "safety_time": t_filter},
            # Legacy aliases used by earlier Stage-A experiment scripts.
            "predicted_displacement": disp_raw,
            "new_nodes": safe_nodes,
            "filter_activated": bool(filt.get("filter_activated", alpha < 1.0 - 1e-12)),
        }


def run_single_sample_demo(config_path: str, checkpoint_path: str, sample_npz: str) -> None:
    """Esegue un test rapido di inferenza su un singolo file .npz."""
    config = load_config(config_path)
    engine = MeshMoverInferenceEngine(checkpoint_path=checkpoint_path, config=config)

    with np.load(sample_npz) as data:
        nodes = data["nodes"]
        cells = data["cells"]
        u = data["u"]
        grad_u = data["grad_u"]
        oracle_disp = data["oracle_disp"] if "oracle_disp" in data else None

    mesh = NativeMeshAdapter.from_arrays(nodes=nodes, cells=cells)
    result = engine.predict(mesh, u, grad_u)

    print("=" * 65)
    print(f"GDM INFERENCE DEMO SU SAMPLE: {Path(sample_npz).name}")
    print("=" * 65)
    print(f"Numero nodi / celle     : {mesh.num_nodes} / {mesh.num_cells}")
    print(f"Alpha Safety Filter     : {result['alpha']:.6f} (is_safe={result['is_safe']})")
    print(f"Min Jacobian (Init->Out): {result['min_j_before']:.3e} -> {result['min_j_after']:.3e}")
    print(f"Tempo Inferenza GNN     : {result['inference_time']*1000:.3f} ms")
    print(f"Tempo Safety Filter     : {result['filter_time']*1000:.3f} ms")
    if oracle_disp is not None:
        rmse = np.sqrt(np.mean(np.sum((result["displacement"] - oracle_disp) ** 2, axis=1)))
        print(f"RMSE vs Oracle Disp     : {rmse:.6e}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferenza GDM MeshMoverGNN + Safety Filter.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path config YAML.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/stage_a/checkpoints/best_model.pt",
        help="Path al checkpoint del modello.",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default="data/raw/test_id/sample_00000.npz",
        help="Path a un sample .npz su cui testare l'inferenza.",
    )
    args = parser.parse_args()
    run_single_sample_demo(args.config, args.checkpoint, args.sample)