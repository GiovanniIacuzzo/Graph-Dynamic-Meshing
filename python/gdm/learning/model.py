"""Candidate-conditioned graph scores with mesh-area and feature-weighted pooling."""
import numpy as np
import torch
from torch import nn

from .graph import collate_graphs, encode_state
from .message_passing import ResidualMessagePassingBlock, build_mlp


class TrajectoryRanker(nn.Module):
    def __init__(self, latent_dim, message_passing_steps, action_dimension):
        super().__init__()
        self.node_encoder = build_mlp(6, latent_dim, latent_dim, 2)
        self.edge_encoder = build_mlp(4, latent_dim, latent_dim, 2)
        self.processor = nn.ModuleList(ResidualMessagePassingBlock(latent_dim)
                                       for _ in range(message_passing_steps))
        self.graph_encoder = build_mlp(2 * latent_dim + 4, latent_dim, latent_dim, 2)
        self.action_encoder = build_mlp(action_dimension, latent_dim, latent_dim, 2)
        self.score = build_mlp(3 * latent_dim, 1, latent_dim, 2)

    def forward(self, batch, actions):
        nodes = self.node_encoder(batch.nodes)
        edges = self.edge_encoder(batch.attributes)
        for block in self.processor:
            nodes, edges = block(nodes, batch.edges, edges)
        pooling = torch.cat((nodes * batch.mass[:, None], nodes * batch.attention[:, None]), dim=1)
        n_graphs = batch.context.size(0)
        pooled = nodes.new_zeros((n_graphs, pooling.size(1))).index_add(0, batch.graph, pooling)
        state = self.graph_encoder(torch.cat((pooled, batch.context), dim=1))[:, None, :]
        candidates = self.action_encoder(actions)[None, :, :]
        interactions = state * candidates
        features = torch.cat((state.expand_as(interactions), candidates.expand_as(interactions),
                              interactions), dim=-1)
        return self.score(features).squeeze(-1)


class TrajectoryPolicy:
    """Rank any compatible action descriptors from a graph observation."""
    def __init__(self, model, actions, eligible=None):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        self.eligible = torch.as_tensor(
            np.ones(len(actions), dtype=bool) if eligible is None else eligible, device=self.device)
        if self.eligible.shape != (len(actions),) or not self.eligible.any():
            raise ValueError("No eligible candidates.")

    @torch.inference_mode()
    def scores(self, state):
        batch = collate_graphs([encode_state(state)]).to(self.device)
        return self.model(batch, self.actions)[0].cpu().numpy()

    @torch.inference_mode()
    def action(self, state):
        scores = self.scores(state)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite candidate scores.")
        return int(np.argmin(np.where(self.eligible.cpu().numpy(), scores, np.inf)))
