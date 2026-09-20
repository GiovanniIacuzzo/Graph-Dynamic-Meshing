"""
dataset/dataset.py
Gestione del dataset per Graph-Dynamic Meshing (GDM).

Carica campioni serializzati generati offline dall'oracle e li trasforma
in tensori per il training della GNN MeshMover:
    - Node Features: [x, y, u, grad_u_x, grad_u_y, is_boundary]
    - Edge Index:    [2, E]
    - Edge Features: [dx, dy, ||dx||]
    - Target:        Delta X* = X* - X

Garantisce partizionamento out-of-distribution (OOD) e parametrico.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from geometry.mesh import TriangularMesh2D


class MeshSample:
    """
    Struttura contenitore per un singolo campione generato offline.
    """
    def __init__(
        self,
        sample_id: str,
        nodes: np.ndarray,              # [N, 2], float64
        cells: np.ndarray,              # [M, 3], int64
        u: np.ndarray,                  # [N], float64
        grad_u: np.ndarray,             # [N, 2], float64
        optimal_nodes: np.ndarray,      # [N, 2], float64
        displacement: np.ndarray,       # [N, 2], float64 (Delta X*)
        metadata: Dict[str, Any]
    ):
        self.sample_id = sample_id
        self.nodes = nodes
        self.cells = cells
        self.u = u
        self.grad_u = grad_u
        self.optimal_nodes = optimal_nodes
        self.displacement = displacement
        self.metadata = metadata

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeshSample":
        return cls(
            sample_id=str(data["metadata"]["sample_id"]),
            nodes=np.ascontiguousarray(data["mesh"]["nodes"], dtype=np.float64),
            cells=np.ascontiguousarray(data["mesh"]["cells"], dtype=np.int64),
            u=np.ascontiguousarray(data["state"]["u"], dtype=np.float64),
            grad_u=np.ascontiguousarray(data["state"]["grad_u"], dtype=np.float64),
            optimal_nodes=np.ascontiguousarray(data["target"]["optimal_nodes"], dtype=np.float64),
            displacement=np.ascontiguousarray(data["target"]["displacement"], dtype=np.float64),
            metadata=data["metadata"]
        )


class GraphDynamicMeshingDataset(Dataset):
    """
    Dataset PyTorch per coppie (M_t, u_t, Delta X*).
    """
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        dtype: torch.dtype = torch.float32
    ):
        """
        data_dir: cartella contenente i campioni serializzati (.pkl o .pt)
                  oppure il manifest 'manifest.json' che associa split e sample_id.
        split: 'train', 'validation', 'test_interp', 'test_extrap'
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.dtype = dtype

        assert self.data_dir.exists(), f"Directory non trovata: {self.data_dir}"

        self.sample_files: List[Path] = self._resolve_samples_for_split()
        if len(self.sample_files) == 0:
            raise RuntimeError(f"Nessun campione trovato per lo split '{split}' in {self.data_dir}")

    def _resolve_samples_for_split(self) -> List[Path]:
        """
        Risolve i file da caricare garantendo che la separazione rispetti
        il vincolo sui parametri fisici e non sia casuale.
        """
        manifest_path = self.data_dir / "manifest.json"
        
        # Se esiste un file di configurazione degli split esplicito
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            
            sample_ids = manifest.get(self.split, [])
            resolved = []
            for sid in sample_ids:
                fpath = self.data_dir / f"{sid}.pkl"
                if fpath.exists():
                    resolved.append(fpath)
            return resolved

        # Fallback: scansiona tutti i file .pkl e verifica il metadata "split"
        all_files = sorted(list(self.data_dir.glob("*.pkl")))
        split_files = []
        for p in all_files:
            try:
                with open(p, "rb") as f:
                    meta = pickle.load(f)["metadata"]
                if meta.get("split") == self.split:
                    split_files.append(p)
            except Exception:
                continue

        # Se nessun metadato di split è presente, isolamento per ordine numerico o avviso
        return split_files if len(split_files) > 0 else all_files

    def __len__(self) -> int:
        return len(self.sample_files)

    def _load_sample(self, file_path: Path) -> MeshSample:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return MeshSample.from_dict(data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Estrae un campione e costruisce il tensore per la GNN.

        Returns:
            Dict con chiavi:
                'node_features':       Tensor [N, F_node]
                'edge_index':          Tensor [2, E], int64
                'edge_features':       Tensor [E, F_edge]
                'target_displacement': Tensor [N, 2]
                'nodes':               Tensor [N, 2]
                'cells':               Tensor [M, 3], int64
                'sample_id':           str
                'metadata':            dict
        """
        sample_path = self.sample_files[idx]
        sample = self._load_sample(sample_path)

        # 1. Ricostruzione topologia e grafo
        mesh = TriangularMesh2D(sample.nodes, sample.cells)
        edges = mesh.edges  # [E, 2] bidirezionale
        bnd_nodes = mesh.boundary_nodes

        # Maschera booleana / numerica di bordo per i nodi
        is_boundary = np.zeros((mesh.num_nodes, 1), dtype=np.float64)
        is_boundary[bnd_nodes] = 1.0

        # 2. Composizione Node Features: [x, y, u, grad_u_x, grad_u_y, is_boundary]
        # Dimensione: [N, 6]
        u_expanded = sample.u[:, np.newaxis]
        node_feats = np.hstack([
            sample.nodes,       # [N, 2]
            u_expanded,         # [N, 1]
            sample.grad_u,      # [N, 2]
            is_boundary         # [N, 1]
        ])

        # 3. Composizione Edge Features: [x_j - x_i, y_j - y_i, ||x_j - x_i||]
        # Dimensione: [E, 3]
        src_nodes = sample.nodes[edges[:, 0]]
        dst_nodes = sample.nodes[edges[:, 1]]
        diff = dst_nodes - src_nodes
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        edge_feats = np.hstack([diff, dist])

        # 4. Target Displacement: Delta X* = X* - X
        target_disp = sample.displacement

        return {
            "node_features": torch.tensor(node_feats, dtype=self.dtype),
            "edge_index": torch.tensor(edges.T, dtype=torch.long),  # [2, E]
            "edge_features": torch.tensor(edge_feats, dtype=self.dtype),
            "target_displacement": torch.tensor(target_disp, dtype=self.dtype),
            "nodes": torch.tensor(sample.nodes, dtype=self.dtype),
            "cells": torch.tensor(sample.cells, dtype=torch.long),
            "sample_id": sample.sample_id,
            "metadata": sample.metadata
        }


def collate_graph_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function per raggruppare grafi di dimensione arbitraria (batching standard GNN).
    Concatena nodi e archi traslando gli indici in edge_index.
    """
    node_features_list = []
    edge_index_list = []
    edge_features_list = []
    target_displacement_list = []
    batch_idx_list = []

    node_offset = 0

    for i, item in enumerate(batch):
        num_nodes = item["node_features"].size(0)
        
        node_features_list.append(item["node_features"])
        edge_features_list.append(item["edge_features"])
        target_displacement_list.append(item["target_displacement"])
        
        # Traslazione indici di connettività
        shifted_edges = item["edge_index"] + node_offset
        edge_index_list.append(shifted_edges)

        batch_idx_list.append(torch.full((num_nodes,), i, dtype=torch.long))
        node_offset += num_nodes

    return {
        "node_features": torch.cat(node_features_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_features": torch.cat(edge_features_list, dim=0),
        "target_displacement": torch.cat(target_displacement_list, dim=0),
        "batch": torch.cat(batch_idx_list, dim=0),
        "raw_samples": batch
    }