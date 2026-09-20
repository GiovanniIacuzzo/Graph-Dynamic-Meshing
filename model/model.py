"""
model/model.py
Modello GNN per Graph-Dynamic Meshing (GDM).

Implementa la mappatura funzionale:
    F_theta: (G_t, u_t) -> Delta X_theta

Architettura:
    - Node Encoder: R^{F_node} -> R^{hidden_dim}
    - Edge Encoder: R^{F_edge} -> R^{hidden_dim}
    - Processore: Sequenza di blocchi Message-Passing con connessioni residue
    - Node Decoder: R^{hidden_dim} -> R^2 (spostamento nodale Delta x, Delta y)

Nessuna dipendenza da solver numerici o filtri geometrici.
"""

from typing import Optional
import torch
import torch.nn as nn


def build_mlp(
    in_features: int,
    out_features: int,
    hidden_dim: int,
    num_layers: int,
    activate_final: bool = False
) -> nn.Sequential:
    """
    Costruisce un blocco Multi-Layer Perceptron (MLP) con LayerNorm e attivazione SiLU.
    """
    layers = []
    current_dim = in_features
    for i in range(num_layers - 1):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        current_dim = hidden_dim
    
    layers.append(nn.Linear(current_dim, out_features))
    if activate_final:
        layers.append(nn.LayerNorm(out_features))
        layers.append(nn.SiLU())
        
    return nn.Sequential(*layers)


class ResidualMessagePassingBlock(nn.Module):
    """
    Singolo blocco di Message Passing con aggiornamento congiunto di Edge e Node embeddings
    e connessioni skip/residue.
    """
    def __init__(self, latent_dim: int, mlp_num_layers: int = 2):
        super().__init__()
        # Input all'edge network: [v_i, v_j, e_ij] -> dimensione 3 * latent_dim
        self.edge_mlp = build_mlp(
            in_features=3 * latent_dim,
            out_features=latent_dim,
            hidden_dim=latent_dim,
            num_layers=mlp_num_layers,
            activate_final=False
        )
        # Input al node network: [v_i, sum_j(e_ij)] -> dimensione 2 * latent_dim
        self.node_mlp = build_mlp(
            in_features=2 * latent_dim,
            out_features=latent_dim,
            hidden_dim=latent_dim,
            num_layers=mlp_num_layers,
            activate_final=False
        )
        
        self.edge_norm = nn.LayerNorm(latent_dim)
        self.node_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        node_latents: torch.Tensor,
        edge_index: torch.Tensor,
        edge_latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        node_latents: [N, latent_dim]
        edge_index:   [2, E] (indici [src, dst])
        edge_latents: [E, latent_dim]
        """
        src, dst = edge_index[0], edge_index[1]
        
        # 1. Edge Update: concatena nodo sorgente, nodo target ed edge features correnti
        edge_inputs = torch.cat([node_latents[src], node_latents[dst], edge_latents], dim=-1)
        new_edge_latents = self.edge_mlp(edge_inputs)
        # Connessione residua e normalizzazione
        updated_edge_latents = self.edge_norm(edge_latents + new_edge_latents)

        # 2. Aggregazione dei messaggi nodali (somma sui nodi target)
        num_nodes = node_latents.size(0)
        aggregated_edges = torch.zeros(
            (num_nodes, node_latents.size(1)),
            device=node_latents.device,
            dtype=node_latents.dtype
        )
        aggregated_edges.index_add_(0, dst, updated_edge_latents)

        # 3. Node Update: concatena il nodo corrente con i messaggi aggregati
        node_inputs = torch.cat([node_latents, aggregated_edges], dim=-1)
        new_node_latents = self.node_mlp(node_inputs)
        # Connessione residua e normalizzazione
        updated_node_latents = self.node_norm(node_latents + new_node_latents)

        return updated_node_latents, updated_edge_latents


class MeshMoverGNN(nn.Module):
    """
    Rete neurale principale per la predizione dello spostamento dei nodi:
    F_theta(G_t, u_t) -> Delta X
    """
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        out_dim: int = 2,
        latent_dim: int = 128,
        num_message_passing_steps: int = 6,
        mlp_num_layers: int = 2
    ):
        """
        Parametri:
            node_in_dim: dimensionalità feature nodo (es. x, y, u, grad_u_x, grad_u_y, is_bnd)
            edge_in_dim: dimensionalità feature arco (es. dx, dy, norm)
            out_dim: coordinate dello spostamento predette (default 2: Delta x, Delta y)
            latent_dim: canale interno dello spazio latente
            num_message_passing_steps: numero di blocchi di propagazione residua
            mlp_num_layers: numero di strati interni per ciascun MLP
        """
        super().__init__()
        
        self.node_encoder = build_mlp(
            in_features=node_in_dim,
            out_features=latent_dim,
            hidden_dim=latent_dim,
            num_layers=mlp_num_layers,
            activate_final=False
        )
        
        self.edge_encoder = build_mlp(
            in_features=edge_in_dim,
            out_features=latent_dim,
            hidden_dim=latent_dim,
            num_layers=mlp_num_layers,
            activate_final=False
        )
        
        self.processor = nn.ModuleList([
            ResidualMessagePassingBlock(latent_dim=latent_dim, mlp_num_layers=mlp_num_layers)
            for _ in range(num_message_passing_steps)
        ])
        
        self.node_decoder = build_mlp(
            in_features=latent_dim,
            out_features=out_dim,
            hidden_dim=latent_dim,
            num_layers=mlp_num_layers,
            activate_final=False
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass del modello.

        Input:
            node_features: [N, node_in_dim]
            edge_index:    [2, E] (int64)
            edge_features: [E, edge_in_dim]

        Output:
            displacement:  [N, out_dim] (float32 o float64 in base ai tensori di input)
        """
        # 1. Encoding
        v = self.node_encoder(node_features)
        e = self.edge_encoder(edge_features)

        # 2. Elaborazione (Message Passing sequenziale con residui interni)
        for block in self.processor:
            v, e = block(v, edge_index, e)

        # 3. Decoding: estrazione del campo di spostamento nodale
        displacement = self.node_decoder(v)
        
        return displacement