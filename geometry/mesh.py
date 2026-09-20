"""
geometry/mesh.py
Rappresentazione e manipolazione topologica di mesh triangolari 2D.

Gestisce la struttura dati per la discretizzazione conforme:
    M = (X, T)
dove:
    X in R^{N x 2} (coordinate nodali)
    T in N^{M x 3} (connettivita' triangolare)

Fornisce primitive topologiche disaccoppiate da qualsiasi logica
di metriche numeriche o reti neurali.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class TriangularMesh2D:
    """
    Rappresenta una mesh triangolare bidimensionale conforme a connettivita' fissa.
    """

    def __init__(self, nodes: np.ndarray, cells: np.ndarray):
        """
        Inizializza la mesh ed esegue controlli formali di consistenza dimensionale.

        Input:
            nodes: coordinate nodali, shape [N, 2], dtype float64
            cells: connettività dei triangoli, shape [M, 3], dtype int64
        """
        assert nodes.ndim == 2 and nodes.shape[1] == 2, (
            f"Forma attesa per nodes: [N, 2], ottenuta: {nodes.shape}"
        )
        assert cells.ndim == 2 and cells.shape[1] == 3, (
            f"Forma attesa per cells: [M, 3], ottenuta: {cells.shape}"
        )

        self._nodes = np.ascontiguousarray(nodes, dtype=np.float64)
        self._cells = np.ascontiguousarray(cells, dtype=np.int64)

        self._num_nodes: int = self._nodes.shape[0]
        self._num_cells: int = self._cells.shape[0]

        # Validazione range indici
        assert np.all(self._cells >= 0) and np.all(self._cells < self._num_nodes), (
            "La matrice delle celle contiene indici nodali non validi o fuori intervallo."
        )

        # Cache lazy per primitive topologiche
        self._boundary_nodes: Optional[np.ndarray] = None
        self._boundary_edges: Optional[np.ndarray] = None
        self._edges: Optional[np.ndarray] = None
        self._node_to_cells: Optional[List[np.ndarray]] = None

    @property
    def nodes(self) -> np.ndarray:
        """Restituisce le coordinate nodali correnti [N, 2]."""
        return self._nodes

    @property
    def cells(self) -> np.ndarray:
        """Restituisce la matrice di connettività delle celle [M, 3]."""
        return self._cells

    @property
    def num_nodes(self) -> int:
        """Numero totale di nodi N."""
        return self._num_nodes

    @property
    def num_cells(self) -> int:
        """Numero totale di elementi triangolari M."""
        return self._num_cells

    def _extract_edges_and_boundary(self) -> None:
        """
        Estrae i lati della triangolazione e classifica i bordi
        in base alla frequenza dei segmenti condivisi.
        """
        # Estrazione ordinata di ogni lato per cella (3 lati per triangolo)
        e1 = np.sort(self._cells[:, [0, 1]], axis=1)
        e2 = np.sort(self._cells[:, [1, 2]], axis=1)
        e3 = np.sort(self._cells[:, [2, 0]], axis=1)
        all_edges = np.vstack([e1, e2, e3])

        # Identificazione lati unici e conteggio molteplicità
        edges_unique, counts = np.unique(all_edges, axis=0, return_counts=True)

        # In una mesh 2D conforme:
        # - lati interni compaiono esattamente 2 volte (condivisi da due triangoli adiacenti)
        # - lati sul bordo d_Omega compaiono esattamente 1 volta
        is_boundary = counts == 1
        self._boundary_edges = edges_unique[is_boundary]
        self._boundary_nodes = np.unique(self._boundary_edges)

        # Costruzione dell'elenco archi bidirezionali non orientati per GNN message-passing
        # Evita duplicazioni e self-loops
        edges_bidirectional = np.vstack([
            edges_unique,
            edges_unique[:, [1, 0]]
        ])
        self._edges = np.ascontiguousarray(edges_bidirectional, dtype=np.int64)

    @property
    def boundary_nodes(self) -> np.ndarray:
        """
        Identifica gli indici dei nodi appartenenti al bordo esterno d_Omega.
        Returns:
            array 1D ordinato con gli indici dei nodi di bordo, shape [N_bnd], dtype int64.
        """
        if self._boundary_nodes is None:
            self._extract_edges_and_boundary()
        return self._boundary_nodes

    @property
    def boundary_edges(self) -> np.ndarray:
        """
        Identifica i segmenti che compongono il bordo d_Omega.
        Returns:
            array di lati sul bordo [E_bnd, 2], dtype int64.
        """
        if self._boundary_edges is None:
            self._extract_edges_and_boundary()
        return self._boundary_edges

    @property
    def edges(self) -> np.ndarray:
        """
        Estrae la lista degli archi per la costruzione del grafo GNN.
        Rappresentazione bidirezionale (u -> v e v -> u) senza nodi isolati né self-loops.
        Returns:
            array [2 * E_unici, 2], dtype int64.
        """
        if self._edges is None:
            self._extract_edges_and_boundary()
        return self._edges

    @property
    def node_to_cells(self) -> List[np.ndarray]:
        """
        Costruisce la lista di adiacenza vertice-elementi (patch locale / 1-ring).
        Returns:
            Lista di lunghezza N in cui l'i-esimo elemento contiene gli indici
            dei triangoli che condividono il nodo i.
        """
        if self._node_to_cells is None:
            patch_dict: Dict[int, List[int]] = {i: [] for i in range(self._num_nodes)}
            for cell_idx, cell in enumerate(self._cells):
                for node_idx in cell:
                    patch_dict[int(node_idx)].append(cell_idx)

            self._node_to_cells = [
                np.array(patch_dict[i], dtype=np.int64) for i in range(self._num_nodes)
            ]
        return self._node_to_cells

    def get_local_patch(self, node_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restituisce la connettività locale (1-ring) attorno a un nodo.

        Input:
            node_index: indice del vertice di riferimento

        Output:
            cell_indices: indici delle celle che compongono il ring [K]
            neighbor_nodes: indici dei vertici adiacenti connessi da un lato [P]
        """
        assert 0 <= node_index < self._num_nodes, f"Indice nodo non valido: {node_index}"
        cells_patch = self.node_to_cells[node_index]
        nodes_in_cells = np.unique(self._cells[cells_patch])
        neighbor_nodes = nodes_in_cells[nodes_in_cells != node_index]
        return cells_patch, neighbor_nodes

    def update_coordinates(self, new_nodes: np.ndarray) -> "TriangularMesh2D":
        """
        Genera una nuova istanza della mesh aggiornando unicamente le coordinate spaziali X,
        mantenendo rigorosamente inalterata la connettività topologica T (r-refinement puro).

        Input:
            new_nodes: nuovo tensore nodale [N, 2], dtype float64

        Output:
            nuova istanza di TriangularMesh2D
        """
        assert new_nodes.shape == self._nodes.shape, (
            f"Dimensione incompatibile per new_nodes. Attesa: {self._nodes.shape}, ricevuta: {new_nodes.shape}"
        )
        return TriangularMesh2D(nodes=new_nodes, cells=self._cells)