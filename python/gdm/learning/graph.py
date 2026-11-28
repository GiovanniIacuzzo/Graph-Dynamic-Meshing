"""Graph encoding for scalar PDE states on 2D triangular meshes."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GraphBatch:
    nodes: torch.Tensor
    edges: torch.Tensor
    attributes: torch.Tensor
    graph: torch.Tensor
    mass: torch.Tensor
    attention: torch.Tensor
    context: torch.Tensor

    def to(self, device):
        return GraphBatch(*(value.to(device) for value in vars(self).values()))


def encode_state(state):
    coords = np.asarray(state["nodes"], dtype=np.float64)
    edges = np.asarray(state["edges"], dtype=np.int64)
    field = np.asarray(state["field"], dtype=np.float64)
    mass = np.asarray(state["node_mass"], dtype=np.float64)
    boundary = np.asarray(state["boundary"])
    params = np.asarray(state["parameters"], dtype=np.float64)
    n = len(coords)
    if (coords.shape != (n, 2) or n == 0 or edges.ndim != 2 or edges.shape[1] != 2
            or not len(edges) or edges.min() < 0 or edges.max() >= n
            or field.shape != (n,) or mass.shape != (n,) or boundary.shape != (n,)
            or params.shape != (4,) or not all(np.isfinite(x).all() for x in
                                               (coords, field, mass, boundary, params))
            or np.any(mass <= 0) or params[2] <= 0 or params[3] <= 0):
        raise ValueError("Invalid scalar mesh observation.")
    mass = mass / mass.sum()
    scaled = field / max(float(np.max(np.abs(field))), 1e-12)
    # Normalize coordinates with a domain scale and edge lengths with the
    # nominal mesh scale. Context retains both transport and diffusion scales.
    origin = coords.min(axis=0)
    domain = float(np.linalg.norm(coords.max(axis=0) - origin))
    if domain <= 0:
        raise ValueError("The graph domain has zero extent.")
    h = domain / np.sqrt(n)
    src, dst = edges.T
    vectors = coords[dst] - coords[src]
    jump = scaled[dst] - scaled[src]
    local_variation = np.bincount(dst, np.abs(jump), minlength=n)
    local_variation /= np.maximum(1, np.bincount(dst, minlength=n))
    bx, by, nu, horizon = params
    nodes = np.column_stack(((coords - origin) / domain, scaled, boundary,
                             np.log1p(n * mass), local_variation))
    edge_features = np.column_stack((vectors / h, np.linalg.norm(vectors, axis=1) / h, jump))
    context = np.array([bx * horizon / domain, by * horizon / domain,
                        np.log1p(nu * horizon / h**2), h / domain])
    weight = mass * (np.abs(scaled) + local_variation)
    attention = weight / weight.sum() if weight.sum() > 0 else mass
    return nodes, edges, edge_features, mass, attention, context


def collate_graphs(encoded):
    nodes, edges, attributes, mass, attention, context = zip(*encoded)
    offsets = np.cumsum([0, *[len(item) for item in nodes[:-1]]])
    graph_ids = np.repeat(np.arange(len(nodes)), [len(item) for item in nodes])
    tensors = [torch.as_tensor(np.concatenate(seq), dtype=torch.float32)
               for seq in (nodes, attributes, mass, attention)]
    edge_index = np.concatenate([edge + offset for edge, offset in zip(edges, offsets)]).T
    return GraphBatch(tensors[0], torch.as_tensor(edge_index, dtype=torch.long),
                      tensors[1], torch.as_tensor(graph_ids, dtype=torch.long),
                      tensors[2], tensors[3], torch.as_tensor(np.stack(context), dtype=torch.float32))
