"""
geometry/mesh.py
Rappresentazione topologica e geometrica di mesh triangolari 2D.

Fornisce la struttura dati principale `TriangularMesh2D` per la gestione
di coordinate nodali, connettività simpliciale (triangoli), estrazione
automatica degli archi orientati (per PyTorch Geometric) e identificazione
dei nodi di bordo (vincolo di Dirichlet / boundary lock).
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np


class TriangularMesh2D:
    """
    Struttura dati per una mesh triangolare conforme in 2D.

    Attributes:
        nodes (np.ndarray): Coordinate nodali di shape [N, 2], dtype float64.
        cells (np.ndarray): Connettività dei triangoli di shape [M, 3], dtype int64
                            (ordinamento antiorario standard).
    """

    def __init__(
        self,
        nodes: np.ndarray,
        cells: np.ndarray,
        boundary_nodes: Optional[np.ndarray] = None,
    ) -> None:
        self.nodes: np.ndarray = np.asarray(nodes, dtype=np.float64)
        self.cells: np.ndarray = np.asarray(cells, dtype=np.int64)

        self._validate_inputs()

        self._provided_boundary_nodes = (
            np.asarray(boundary_nodes, dtype=np.int64)
            if boundary_nodes is not None
            else None
        )

        # Cache per strutture topologiche derivate
        self._undirected_edges: Optional[np.ndarray] = None
        self._edges: Optional[np.ndarray] = None
        self._boundary_nodes: Optional[np.ndarray] = None
        self._boundary_mask: Optional[np.ndarray] = None

        self._build_topology()

    def _validate_inputs(self) -> None:
        """Verifica la consistenza dimensionale, la finitezza e i limiti degli indici."""
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError(
                f"Expected nodes of shape [N, 2], got {self.nodes.shape}."
            )
        if self.nodes.shape[0] < 3:
            raise ValueError(
                f"A 2D mesh requires at least 3 nodes, got {self.nodes.shape[0]}."
            )
        if not np.all(np.isfinite(self.nodes)):
            raise ValueError("Mesh nodes contain non-finite (NaN or Inf) coordinates.")

        if self.cells.ndim != 2 or self.cells.shape[1] != 3:
            raise ValueError(
                f"Expected cells of shape [M, 3], got {self.cells.shape}."
            )
        if self.cells.shape[0] < 1:
            raise ValueError("A 2D mesh requires at least 1 triangular cell.")

        if np.min(self.cells) < 0 or np.max(self.cells) >= self.nodes.shape[0]:
            raise ValueError(
                "Cell indices out of bounds for the provided nodes array."
            )
        if np.any(
            (self.cells[:, 0] == self.cells[:, 1])
            | (self.cells[:, 1] == self.cells[:, 2])
            | (self.cells[:, 0] == self.cells[:, 2])
        ):
            raise ValueError(
                "Cells cannot contain duplicate vertex indices within the same triangle."
            )

    def _build_topology(self) -> None:
        """
        Estrae gli archi orientati (bidirezionali per grafi GNN) e rileva
        automaticamente i nodi di bordo (appartenenti a lati condivisi da 1 sola cella)
        qualora non siano stati forniti esplicitamente.
        """
        # Estrazione dei 3 lati per ogni triangolo: (0,1), (1,2), (2,0)
        e01 = self.cells[:, [0, 1]]
        e12 = self.cells[:, [1, 2]]
        e20 = self.cells[:, [2, 0]]
        all_half_edges = np.vstack([e01, e12, e20])

        # Ordina i vertici di ciascun lato (min, max) per trovare i lati unici non orientati
        sorted_edges = np.sort(all_half_edges, axis=1)
        unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("Non-manifold mesh: un lato è condiviso da più di due triangoli.")

        self._undirected_edges = unique_edges.astype(np.int64)

        # Un lato di bordo in una mesh conforme 2D appartiene esattamente a 1 triangolo
        unique_boundary_edges = unique_edges[counts == 1]
        boundary_from_topology = np.unique(unique_boundary_edges)

        if self._provided_boundary_nodes is not None:
            b_nodes = np.asarray(self._provided_boundary_nodes, dtype=np.int64)
            if b_nodes.ndim != 1:
                raise ValueError("boundary_nodes must be a 1D array of vertex indices.")
            if b_nodes.size > 0 and (
                np.min(b_nodes) < 0 or np.max(b_nodes) >= self.nodes.shape[0]
            ):
                raise ValueError("boundary_nodes contains out-of-bounds vertex indices.")
            unique_b = np.unique(b_nodes)
            if not np.array_equal(unique_b, boundary_from_topology):
                raise ValueError(
                    "Provided boundary_nodes must exactly match the topological boundary "
                    "(neither omissions nor interior vertices are allowed)."
                )
            self._boundary_nodes = unique_b
        else:
            self._boundary_nodes = boundary_from_topology

        self._boundary_mask = np.zeros(self.nodes.shape[0], dtype=bool)
        self._boundary_mask[self._boundary_nodes] = True

        # Costruzione degli archi orientati (i -> j) e (j -> i) per il message passing
        rev_edges = unique_edges[:, [1, 0]]
        directed_edges = np.vstack([unique_edges, rev_edges])

        # Ordinamento lessicografico per riproducibilità deterministica
        order = np.lexsort((directed_edges[:, 1], directed_edges[:, 0]))
        self._edges = directed_edges[order].astype(np.int64)

    @property
    def num_nodes(self) -> int:
        """Numero totale di nodi N."""
        return int(self.nodes.shape[0])

    @property
    def num_cells(self) -> int:
        """Numero totale di elementi triangolari M."""
        return int(self.cells.shape[0])

    @property
    def num_edges(self) -> int:
        """Numero totale di archi orientati 2E."""
        return int(self._edges.shape[0])

    @property
    def edges(self) -> np.ndarray:
        """Restituisce gli archi orientati bidirezionali di shape [2E, 2] (dtype=int64) per PyG."""
        return self._edges

    @property
    def directed_edges(self) -> np.ndarray:
        """Alias esplicito per gli archi orientati bidirezionali di shape [2E, 2] (dtype=int64)."""
        return self._edges

    @property
    def undirected_edges(self) -> np.ndarray:
        """Restituisce gli archi non orientati canonici (i < j) di shape [E, 2] (dtype=int64)."""
        return self._undirected_edges

    @property
    def boundary_nodes(self) -> np.ndarray:
        """Restituisce gli indici ordinati dei nodi di bordo [B]."""
        return self._boundary_nodes

    @property
    def boundary_mask(self) -> np.ndarray:
        """Restituisce una maschera booleana [N] (True per i nodi di bordo)."""
        return self._boundary_mask

    @property
    def interior_mask(self) -> np.ndarray:
        """Restituisce una maschera booleana [N] (True per i nodi interni liberi)."""
        return ~self._boundary_mask

    def clone(self) -> "TriangularMesh2D":
        """Restituisce una copia profonda della mesh."""
        return TriangularMesh2D(
            nodes=self.nodes.copy(),
            cells=self.cells.copy(),
            boundary_nodes=self._boundary_nodes.copy(),
        )

    def update_coordinates(self, new_nodes: np.ndarray) -> "TriangularMesh2D":
        """
        Restituisce una nuova istanza di TriangularMesh2D con coordinate aggiornate
        ma identica topologia (r-adaptation: connettività invariante).
        """
        new_nodes_arr = np.asarray(new_nodes, dtype=np.float64)
        if new_nodes_arr.shape != self.nodes.shape:
            raise ValueError(
                f"Shape mismatch in update_coordinates: expected {self.nodes.shape}, "
                f"got {new_nodes_arr.shape}."
            )
        if not np.all(np.isfinite(new_nodes_arr)):
            raise ValueError("New coordinates contain non-finite (NaN or Inf) values.")
        tri = new_nodes_arr[self.cells]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        det_j = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
        if np.any(~np.isfinite(det_j)) or np.any(det_j <= 1.0e-15):
            bad = np.flatnonzero((~np.isfinite(det_j)) | (det_j <= 1.0e-15))
            first = int(bad[0])
            raise ValueError(
                "New coordinates would create an inverted or degenerate triangle "
                f"(cell_index={first}, det_J={det_j[first]:.6e})."
            )

        # Creazione istanza senza ricalcolare la topologia (ottimizzazione per loop dinamici)
        updated_mesh = TriangularMesh2D.__new__(TriangularMesh2D)
        updated_mesh.nodes = new_nodes_arr.copy()
        updated_mesh.cells = self.cells
        updated_mesh._provided_boundary_nodes = self._provided_boundary_nodes
        updated_mesh._undirected_edges = self._undirected_edges
        updated_mesh._edges = self._edges
        updated_mesh._boundary_nodes = self._boundary_nodes
        updated_mesh._boundary_mask = self._boundary_mask
        return updated_mesh

    def compute_edge_vectors_and_lengths(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcola i vettori relativi e_{ij} = x_j - x_i e le lunghezze Euclidee ||e_{ij}||
        per tutti gli archi orientati.

        Returns:
            edge_vectors: Array [2E, 2] con (dx, dy).
            edge_lengths: Array [2E, 1] con le lunghezze.
        """
        src, dst = self._edges[:, 0], self._edges[:, 1]
        edge_vectors = self.nodes[dst] - self.nodes[src]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1, keepdims=True)
        return edge_vectors, edge_lengths

    def to_dict(self) -> Dict[str, Any]:
        """Serializza la mesh in un dizionario di array NumPy."""
        return {
            "nodes": self.nodes,
            "cells": self.cells,
            "edges": self._edges,
            "boundary_nodes": self._boundary_nodes,
        }