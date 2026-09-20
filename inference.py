"""
inference.py
Entry point per l'inferenza della GNN Mesh Mover in Graph-Dynamic Meshing (GDM).

Pipeline di esecuzione:
    1. Caricamento configurazione e pesi del modello addestrato (best_model.pt)
    2. Ricezione mesh corrente M_t e stato fisico u_t
    3. Costruzione delle feature di grafo (nodi ed archi)
    4. Forward pass della GNN -> predizione spostamento grezzo Delta X_pred
    5. Applicazione del Safety Filter deterministico O(M) -> alpha_global e X_safe
    6. Valutazione delle metriche geometriche post-adattamento
    7. Profiling temporale disaccoppiato (T_GNN vs T_safety)

Non esegue solutori FEM/ALE: circoscritto alla pura predizione geometrica dello Stage A.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch

# Inclusione root nel path di sistema per esecuzione diretta
sys.path.append(str(Path(__file__).resolve().parent))

from model.model import MeshMoverGNN
from geometry.mesh import TriangularMesh2D
from geometry.geometry import compute_mesh_metrics, compute_mesh_quality_summary
from geometry.safety import apply_safety_filter
from oracle.field import GaussianBumpField
from utils.utils import load_config, ensure_dir, setup_logger, Timer


class MeshMoverInferenceEngine:
    """
    Motore di inferenza per la predizione dello spostamento nodale e validazione geometrica.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        self.checkpoint_path = Path(checkpoint_path)
        assert self.checkpoint_path.exists(), f"Checkpoint non trovato: {self.checkpoint_path.resolve()}"

        # 1. Caricamento del checkpoint e della configurazione
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.config = config if config is not None else checkpoint.get("config", {})
        assert self.config, "Configurazione non presente nel checkpoint né fornita esplicitamente."

        # 2. Selezione dispositivo di calcolo
        if device is None:
            dev_cfg = self.config.get("experiment", {}).get("device", "cpu")
            if dev_cfg == "mps" and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif dev_cfg == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # 3. Istanziazione e caricamento pesi del modello GNN
        m_cfg = self.config["model"]
        self.model = MeshMoverGNN(
            node_in_dim=m_cfg["node_in_dim"],
            edge_in_dim=m_cfg["edge_in_dim"],
            out_dim=m_cfg["out_dim"],
            latent_dim=m_cfg["latent_dim"],
            num_message_passing_steps=m_cfg["num_message_passing_steps"],
            mlp_num_layers=m_cfg["mlp_num_layers"]
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 4. Parametri geometrici e soglie di sicurezza
        geom_cfg = self.config.get("geometry", {}).get("safety", {})
        self.epsilon = float(geom_cfg.get("epsilon", 1e-4))
        self.tol_deg = float(geom_cfg.get("tol_deg", 1e-12))
        self.safety_margin = float(geom_cfg.get("safety_margin", 1e-6))

    def _prepare_graph_tensors(
        self,
        mesh: TriangularMesh2D,
        u: np.ndarray,
        grad_u: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Costruisce i tensori di input per la GNN a partire dalla mesh e dal campo fisico.
        """
        nodes = mesh.nodes
        edges = mesh.edges
        num_nodes = mesh.num_nodes

        # Maschera booleana/numerica nodi di contorno
        is_boundary = np.zeros((num_nodes, 1), dtype=np.float64)
        is_boundary[mesh.boundary_nodes] = 1.0

        # Node features: [x, y, u, grad_u_x, grad_u_y, is_boundary] -> [N, 6]
        u_col = u[:, np.newaxis]
        node_feats_np = np.hstack([nodes, u_col, grad_u, is_boundary])

        # Edge features: [dx, dy, ||dx||] -> [E, 3]
        src_pts = nodes[edges[:, 0]]
        dst_pts = nodes[edges[:, 1]]
        diff = dst_pts - src_pts
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        edge_feats_np = np.hstack([diff, dist])

        node_features = torch.tensor(node_feats_np, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(edges.T, dtype=torch.long, device=self.device)
        edge_features = torch.tensor(edge_feats_np, dtype=torch.float32, device=self.device)

        return node_features, edge_index, edge_features

    def predict(
        self,
        mesh: TriangularMesh2D,
        u: np.ndarray,
        grad_u: np.ndarray
    ) -> Dict[str, Any]:
        """
        Esegue la predizione end-to-end dello spostamento nodale con filtro di sicurezza.

        Input:
            mesh  : istanza TriangularMesh2D corrente
            u     : vettore dello stato scalare ai nodi [N], float64
            grad_u: gradiente cartesiano ai nodi [N, 2], float64

        Output:
            Dizionario contenente:
                - 'predicted_displacement': spostamento continuo grezzo della GNN Delta X_pred, [N, 2]
                - 'alpha_global'          : fattore di scala deterministico in [0, 1]
                - 'is_safe'               : True se alpha_global == 1.0
                - 'new_nodes'             : nuove coordinate nodali X_safe, [N, 2]
                - 'adapted_mesh'          : nuova istanza TriangularMesh2D
                - 'geometric_metrics'     : riepilogo scalare della qualità geometrica
                - 'timing'                : breakdown cronometrico (gnn_time, safety_time, total_time)
        """
        assert len(u) == mesh.num_nodes, "Incompatibilità dimensionale tra stato u e nodi della mesh"
        assert grad_u.shape == (mesh.num_nodes, 2), "Incompatibilità dimensionale per grad_u"

        # 1. Costruzione dei tensori di grafo
        node_feats, edge_index, edge_feats = self._prepare_graph_tensors(mesh, u, grad_u)

        # 2. Forward pass GNN con profilazione cronometrica
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

        t_gnn_start = time.perf_counter()

        with torch.no_grad():
            disp_pred_tensor = self.model(node_feats, edge_index, edge_feats)

        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

        gnn_time = time.perf_counter() - t_gnn_start

        predicted_displacement = disp_pred_tensor.cpu().numpy().astype(np.float64)

        # I nodi di bordo devono rimanere rigorosamente fissi
        predicted_displacement[mesh.boundary_nodes] = 0.0

        # 3. Applicazione del Safety Filter deterministico O(M)
        t_safety_start = time.perf_counter()

        alpha_global, safe_nodes, is_safe, min_j_after = apply_safety_filter(
            nodes=mesh.nodes,
            displacement=predicted_displacement,
            cells=mesh.cells,
            epsilon=self.epsilon,
            tol_deg=self.tol_deg,
            safety_margin=self.safety_margin
        )

        safety_time = time.perf_counter() - t_safety_start

        # 4. Creazione della mesh adattata e calcolo metriche geometriche
        adapted_mesh = mesh.update_coordinates(safe_nodes)
        raw_metrics = compute_mesh_metrics(adapted_mesh.nodes, adapted_mesh.cells)
        geom_summary = compute_mesh_quality_summary(raw_metrics)

        total_time = gnn_time + safety_time

        return {
            "predicted_displacement": predicted_displacement,
            "alpha_global": float(alpha_global),
            "is_safe": bool(is_safe),
            "new_nodes": safe_nodes,
            "adapted_mesh": adapted_mesh,
            "geometric_metrics": geom_summary,
            "timing": {
                "gnn_time": float(gnn_time),
                "safety_time": float(safety_time),
                "total_time": float(total_time)
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Inference standalone per Graph-Dynamic Meshing (Stage A)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Percorso file YAML")
    parser.add_argument("--checkpoint", type=str, default="results/stage_a/checkpoints/best_model.pt", help="Pesi del modello")
    parser.add_argument("--mu_x", type=float, default=0.6, help="Centro X della Gaussian bump di test")
    parser.add_argument("--mu_y", type=float, default=0.6, help="Centro Y della Gaussian bump di test")
    parser.add_argument("--sigma", type=float, default=0.04, help="Larghezza sigma della Gaussian bump")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("Inference")
    logger.info("=== Avvio Inferenza Mesh Mover GNN ===")

    # 1. Inizializzazione motore di inferenza
    engine = MeshMoverInferenceEngine(checkpoint_path=args.checkpoint, config=config)

    # 2. Generazione di una mesh uniforme nominale per il test
    dom = config["geometry"]["domain"]
    grid_cfg = config["geometry"]["initial_grid"]
    from generate_dataset import create_initial_mesh_2d
    initial_mesh = create_initial_mesh_2d(dom, nx=grid_cfg["nx"], ny=grid_cfg["ny"])

    # 3. Istanziazione del campo fisico di prova
    field = GaussianBumpField(mu_x=args.mu_x, mu_y=args.mu_y, sigma=args.sigma, amplitude=1.0)
    u_init, grad_u_init, _, _ = field.evaluate(initial_mesh.nodes[:, 0], initial_mesh.nodes[:, 1])

    # 4. Esecuzione predizione con validazione
    result = engine.predict(mesh=initial_mesh, u=u_init, grad_u=grad_u_init)

    # 5. Output dei risultati e telemetria
    m = result["geometric_metrics"]
    t = result["timing"]
    logger.info("--- Risultati dell'Inferenza ---")
    logger.info(f"Alpha globale di sicurezza: {result['alpha_global']:.6f} (Pienamente ammissibile: {result['is_safe']})")
    logger.info(f"Minimo Jacobiano risultante: {m['min_jacobian']:.4e} (Soglia epsilon: {engine.epsilon:.4e})")
    logger.info(f"Qualità media Escobar-Knupp : {m['mean_escobar_quality']:.4f}")
    logger.info(f"Aspect ratio massimo       : {m['max_aspect_ratio']:.2f}")
    logger.info(f"Elementi invertiti         : {m['num_inverted_cells']}")
    logger.info(f"Tempi: GNN={t['gnn_time']*1000:.2f} ms | Safety={t['safety_time']*1000:.2f} ms | Totale={t['total_time']*1000:.2f} ms")


if __name__ == "__main__":
    main()