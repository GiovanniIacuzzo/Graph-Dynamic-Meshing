"""Reusable residual edge and node message passing."""

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
