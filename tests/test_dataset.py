"""
tests/test_dataset.py
Test di integrità del dataset e verifica partizionamento out-of-distribution.
"""

import json
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("gdm._gdm_core", reason="Il preprocessing topologico richiede il core C++/deal.II", exc_type=ImportError)
import torch

from dataset.dataset import GraphDynamicMeshingDataset, collate_graph_batch
from utils.utils import save_pickle, ensure_dir


@pytest.fixture
def mock_dataset_directory(tmp_path):
    """
    Costruisce una directory temporanea con mock samples e relativo manifest.json
    per testare il caricamento conforme, il batching e la separazione OOD.
    """
    data_dir = tmp_path / "data" / "raw"
    ensure_dir(data_dir)

    # Creazione mesh di prova (4 nodi, 2 celle)
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    u = np.array([0.1, 0.2, 0.9, 0.3], dtype=np.float64)
    grad_u = np.array([[0.0, 0.0], [0.1, 0.1], [-0.5, -0.5], [0.1, 0.2]], dtype=np.float64)

    # Due campioni per 'train' (parametri in range) e due per 'test_extrap' (parametri estremi)
    sample_registry = {
        "train": ["sample_00001_train", "sample_00002_train"],
        "test_extrap": ["sample_00003_test_extrap", "sample_00004_test_extrap"]
    }

    manifest = {"train": [], "test_extrap": []}

    for split, sids in sample_registry.items():
        for sid in sids:
            # Parametro sigma differenziato per verificare OOD
            sigma_val = 0.05 if split == "train" else 0.015
            optimal_nodes = nodes + 0.01
            displacement = optimal_nodes - nodes

            sample_data = {
                "mesh": {"nodes": nodes, "cells": cells},
                "state": {"u": u, "grad_u": grad_u},
                "target": {"optimal_nodes": optimal_nodes, "displacement": displacement},
                "metadata": {
                    "sample_id": sid,
                    "split": split,
                    "parameters": {"mu_x": 0.5, "mu_y": 0.5, "sigma": sigma_val, "amplitude": 1.0},
                    "converged": True,
                    "iterations": 15,
                    "oracle_metrics": {"min_jacobian": 0.8}
                }
            }
            save_pickle(sample_data, data_dir / f"{sid}.pkl")
            manifest[split].append(sid)

    # Salvataggio manifest.json
    with open(data_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return data_dir


def test_dataset_loading_and_features_dimensions(mock_dataset_directory):
    """Verifica caricamento e forma corretta dei tensori estratti dal dataset."""
    train_dataset = GraphDynamicMeshingDataset(data_dir=mock_dataset_directory, split="train")

    assert len(train_dataset) == 2
    item = train_dataset[0]

    # Node features attese: [N, 6] -> (x, y, u, grad_u_x, grad_u_y, is_boundary)
    assert item["node_features"].shape == (4, 6)
    # Connettività bidirezionale: 5 lati unici -> 10 archi
    assert item["edge_index"].shape == (2, 10)
    # Edge features attese: [E, 3] -> (dx, dy, norm)
    assert item["edge_features"].shape == (10, 3)
    # Target displacement: [N, 2]
    assert item["target_displacement"].shape == (4, 2)
    assert item["target_displacement"].dtype == torch.float32


def test_out_of_distribution_partition_isolation(mock_dataset_directory):
    """
    Verifica che gli split train e test_extrap siano mutuamente esclusivi
    e mantengano la separazione rigorosa sui parametri fisici (sigma).
    """
    train_ds = GraphDynamicMeshingDataset(data_dir=mock_dataset_directory, split="train")
    extrap_ds = GraphDynamicMeshingDataset(data_dir=mock_dataset_directory, split="test_extrap")

    train_ids = [train_ds[i]["sample_id"] for i in range(len(train_ds))]
    extrap_ids = [extrap_ds[i]["sample_id"] for i in range(len(extrap_ds))]

    # Nessuna intersezione di sample ID
    assert len(set(train_ids).intersection(set(extrap_ids))) == 0

    # Verifica isolamento parametrico
    train_sigmas = [train_ds[i]["metadata"]["parameters"]["sigma"] for i in range(len(train_ds))]
    extrap_sigmas = [extrap_ds[i]["metadata"]["parameters"]["sigma"] for i in range(len(extrap_ds))]

    assert all(s == 0.05 for s in train_sigmas)
    assert all(s == 0.015 for s in extrap_sigmas)


def test_collate_graph_batch_concatenation(mock_dataset_directory):
    """
    Verifica che collate_graph_batch impacchetti correttamente i campioni
    traslando gli indici degli archi ed evitando sovrapposizioni spurie.
    """
    train_ds = GraphDynamicMeshingDataset(data_dir=mock_dataset_directory, split="train")
    batch_raw = [train_ds[0], train_ds[1]]

    batched = collate_graph_batch(batch_raw)

    # 4 nodi per campione -> 8 nodi totali
    assert batched["node_features"].shape == (8, 6)
    # 10 archi per campione -> 20 archi totali
    assert batched["edge_index"].shape == (2, 20)
    assert batched["edge_features"].shape == (20, 3)
    assert batched["target_displacement"].shape == (8, 2)
    # Batch tensor di assegnazione nodale
    assert batched["batch"].shape == (8,)
    np.testing.assert_array_equal(batched["batch"].numpy(), [0, 0, 0, 0, 1, 1, 1, 1])

    # Verifica corretta traslazione di connettività per il secondo grafo (offset = 4)
    second_graph_edges = batched["edge_index"][:, 10:].numpy()
    assert np.all(second_graph_edges >= 4) and np.all(second_graph_edges < 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])