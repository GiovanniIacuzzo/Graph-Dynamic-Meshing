"""
numerics/remap.py
Trasferimento di stato e riproiezione di campo tra mesh deformate successive.

Gestisce la transizione:
    (M_old, state_old, M_new) -> (state_new, conservation_error)

Supporta due modalità operative distinte per ablation study:
    1. 'standard': interpolazione continua pointwise P1 standard (non conservativa).
    2. 'conservative': riproiezione con correzione integrale per preservare la massa globale:
           int_Omega u_new dOmega = int_Omega u_old dOmega
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np

from geometry.mesh import TriangularMesh2D


class MeshStateRemapper:
    """
    Esegue il remapping di campi scalari nodali tra due configurazioni di mesh
    aventi la medesima connettività topologica (r-refinement puro).
    """

    def __init__(self, tol_bary: float = 1e-10):
        """
        Input:
            tol_bary: tolleranza per consentire coordinate baricentriche leggermente
                      esterne dovute ad approssimazione floating-point.
        """
        self.tol_bary = tol_bary

    @staticmethod
    def compute_total_mass(mesh: TriangularMesh2D, state: np.ndarray) -> float:
        """
        Calcola l'integrale di campo sul dominio:
            Massa = int_Omega u dOmega = sum_e (Area_e / 3) * (u_1 + u_2 + u_3)
        per elementi finiti lineari P1.
        """
        nodes = mesh.nodes
        cells = mesh.cells

        # Vertici di ogni triangolo
        x1, y1 = nodes[cells[:, 0], 0], nodes[cells[:, 0], 1]
        x2, y2 = nodes[cells[:, 1], 0], nodes[cells[:, 1], 1]
        x3, y3 = nodes[cells[:, 2], 0], nodes[cells[:, 2], 1]

        # Aree elementari: 0.5 * det(J_e)
        det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        areas = 0.5 * det_J

        # Media dei valori nodali per ciascun triangolo
        u_elem_mean = np.mean(state[cells], axis=1)

        total_mass = float(np.sum(areas * u_elem_mean))
        return total_mass

    def _interpolate_pointwise(
        self,
        mesh_old: TriangularMesh2D,
        mesh_new: TriangularMesh2D,
        state_old: np.ndarray
    ) -> np.ndarray:
        """
        Interpola lo stato dai nodi di mesh_old ai nuovi punti nodali di mesh_new.
        Sfrutta la connettività per circoscrivere la ricerca locale nei triangoli del ring.
        """
        nodes_old = mesh_old.nodes
        cells_old = mesh_old.cells
        nodes_new = mesh_new.nodes
        num_nodes = mesh_new.num_nodes

        state_new = np.empty(num_nodes, dtype=np.float64)

        # Matrici dei triangoli vecchi per il calcolo baricentrico rapido
        p1 = nodes_old[cells_old[:, 0]]
        p2 = nodes_old[cells_old[:, 1]]
        p3 = nodes_old[cells_old[:, 2]]

        v0 = p2 - p1
        v1 = p3 - p1

        # Determinante 2D (prodotto vettoriale z): v0_x * v1_y - v0_y * v1_x
        det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        inv_det = 1.0 / np.where(np.abs(det) < 1e-15, 1e-15, det)

        # Ricerca della cella contenitore per ciascun nodo target
        # Poiché lo spostamento da r-refinement è continuo e moderato,
        # testiamo prioritariamente le celle adiacenti (1-ring) del nodo originale
        node_to_cells = mesh_old.node_to_cells

        for i in range(num_nodes):
            pt = nodes_new[i]
            candidate_cells = node_to_cells[i]

            found = False
            # 1. Test nei triangoli candidati della patch locale
            for c_idx in candidate_cells:
                w = pt - p1[c_idx]
                lambda2 = (w[0] * v1[c_idx, 1] - w[1] * v1[c_idx, 0]) * inv_det[c_idx]
                lambda3 = (v0[c_idx, 0] * w[1] - v0[c_idx, 1] * w[0]) * inv_det[c_idx]
                lambda1 = 1.0 - lambda2 - lambda3

                if (
                    lambda1 >= -self.tol_bary
                    and lambda2 >= -self.tol_bary
                    and lambda3 >= -self.tol_bary
                ):
                    # Punto localizzato: interpolazione lineare P1 esatta
                    cell_nodes = cells_old[c_idx]
                    vals = state_old[cell_nodes]
                    state_new[i] = lambda1 * vals[0] + lambda2 * vals[1] + lambda3 * vals[2]
                    found = True
                    break

            if not found:
                # 2. Ricerca esaustiva globale vettorizzata se fuori dalla patch immediata
                w_all = pt - p1  # [M, 2]
                l2_all = (w_all[:, 0] * v1[:, 1] - w_all[:, 1] * v1[:, 0]) * inv_det
                l3_all = (v0[:, 0] * w_all[:, 1] - v0[:, 1] * w_all[:, 0]) * inv_det
                l1_all = 1.0 - l2_all - l3_all

                inside = (
                    (l1_all >= -self.tol_bary)
                    & (l2_all >= -self.tol_bary)
                    & (l3_all >= -self.tol_bary)
                )
                matching_indices = np.where(inside)[0]

                if len(matching_indices) > 0:
                    c_idx = matching_indices[0]
                    cell_nodes = cells_old[c_idx]
                    vals = state_old[cell_nodes]
                    state_new[i] = (
                        l1_all[c_idx] * vals[0]
                        + l2_all[c_idx] * vals[1]
                        + l3_all[c_idx] * vals[2]
                    )
                else:
                    # Fallback conservativo per nodi al bordo esterno: nearest-neighbor
                    dists = np.linalg.norm(nodes_old - pt, axis=1)
                    nearest_idx = np.argmin(dists)
                    state_new[i] = state_old[nearest_idx]

        return state_new

    def remap(
        self,
        mesh_old: TriangularMesh2D,
        mesh_new: TriangularMesh2D,
        state_old: np.ndarray,
        mode: str = "conservative"
    ) -> Tuple[np.ndarray, float, Dict[str, float]]:
        """
        Esegue il trasferimento dello stato fisico dalla vecchia alla nuova griglia.

        Input:
            mesh_old : istanza TriangularMesh2D di partenza
            mesh_new : istanza TriangularMesh2D deformata
            state_old: array del campo nodale su mesh_old, shape [N]
            mode     : 'standard' (pura interpolazione pointwise)
                       'conservative' (correzione integrale globale della massa)

        Output:
            state_new          : campo nodale proiettato su mesh_new, shape [N]
            conservation_error : errore assoluto |Massa_new - Massa_old|
            diagnostics        : dizionario con masse e fattore di scala correttivo
        """
        assert mode in ["standard", "conservative"], (
            f"Modalità sconosciuta: '{mode}'. Valori ammessi: 'standard', 'conservative'"
        )
        assert len(state_old) == mesh_old.num_nodes, "Dimensione dello stato incompatibile con mesh_old"

        # 1. Massa iniziale sul dominio
        mass_old = self.compute_total_mass(mesh_old, state_old)

        # 2. Interpolazione continua standard sui nuovi nodi
        state_interp = self._interpolate_pointwise(mesh_old, mesh_new, state_old)

        # 3. Applicazione del vincolo di conservazione se richiesto
        if mode == "standard":
            state_new = state_interp
            mass_new = self.compute_total_mass(mesh_new, state_new)
            correction_factor = 1.0
        else:
            # Calcolo della massa non corretta
            mass_interp = self.compute_total_mass(mesh_new, state_interp)

            # Correzione integrale globale:
            # Scala uniformemente il campo per forzare identità delle masse integrali
            if np.abs(mass_interp) > 1e-14:
                correction_factor = mass_old / mass_interp
                state_new = state_interp * correction_factor
            else:
                # Se la massa totale è circa nulla, correzione additiva uniforme
                # Delta_u * Area_totale = mass_old - mass_interp
                nodes_n = mesh_new.nodes
                cells_n = mesh_new.cells
                det_J = (nodes_n[cells_n[:, 1], 0] - nodes_n[cells_n[:, 0], 0]) * (
                    nodes_n[cells_n[:, 2], 1] - nodes_n[cells_n[:, 0], 1]
                ) - (nodes_n[cells_n[:, 2], 0] - nodes_n[cells_n[:, 0], 0]) * (
                    nodes_n[cells_n[:, 1], 1] - nodes_n[cells_n[:, 0], 1]
                )
                total_area = float(np.sum(0.5 * det_J))
                shift = (mass_old - mass_interp) / max(total_area, 1e-12)
                state_new = state_interp + shift
                correction_factor = 1.0

            mass_new = self.compute_total_mass(mesh_new, state_new)

        # 4. Errore di conservazione scalare assoluto
        conservation_error = float(np.abs(mass_new - mass_old))

        diagnostics = {
            "mass_before": mass_old,
            "mass_after": mass_new,
            "correction_factor": float(correction_factor),
            "mode": mode
        }

        return state_new, conservation_error, diagnostics