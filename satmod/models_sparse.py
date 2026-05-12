"""Sparse / deep GCN models for large graphs (Flickr, Reddit2, ogbn-arxiv)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DeepGCNModel(nn.Module):
    """
    Deeper GCN with per-layer residual + LayerNorm and temperature-scaled softmax.

    Designed for large graphs where 2-layer GCN over-smooths and collapses to
    few clusters. Temperature < 1 sharpens assignments; > 1 softens them.
    """

    def __init__(self, in_dim, hidden_dim, n_clusters,
                 dropout=0.3, n_layers=4, temperature=0.5):
        super().__init__()
        self.temperature = temperature

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, n_clusters)
        self.residual = nn.Linear(in_dim, n_clusters, bias=False)

    def forward(self, x, edge_index):
        res = self.residual(x)
        x = F.relu(self.input_proj(x))

        for conv, norm in zip(self.convs, self.norms):
            identity = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + identity

        x = self.output(x) + 0.1 * res
        return F.softmax(x / self.temperature, dim=1)


class SparseGCNModel(nn.Module):
    """
    Sparse-matmul GCN with a pre-built normalized adjacency (D^{-1/2}(A+I)D^{-1/2}).

    Adjacency is built on CPU then transferred to GPU, avoiding OOM on graphs
    like Reddit2 where intermediate edge tensors would not fit on device.
    """

    def __init__(self, in_dim, hidden_dim, n_clusters,
                 dropout=0.3, n_layers=4, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.n_layers = n_layers

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, n_clusters)
        self.residual = nn.Linear(in_dim, n_clusters, bias=False)

        # Filled lazily by set_adjacency() or on first forward().
        self._A_hat = None

        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def set_adjacency(self, edge_index, num_nodes, device):
        """Build D^{-1/2}(A+I)D^{-1/2} on CPU, then move to ``device``."""
        cpu = torch.device('cpu')

        row, col = edge_index.cpu()
        self_loops = torch.arange(num_nodes, device=cpu)
        row = torch.cat([row, self_loops])
        col = torch.cat([col, self_loops])

        deg = torch.zeros(num_nodes).float()
        deg.scatter_add_(0, row, torch.ones(len(row)))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        A_hat_cpu = torch.sparse_coo_tensor(
            torch.stack([row, col]),
            edge_weight,
            size=(num_nodes, num_nodes),
        ).coalesce()

        self._A_hat = A_hat_cpu.to(device)
        del A_hat_cpu
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    def forward(self, x, edge_index):
        device = x.device
        num_nodes = x.shape[0]
        res = self.residual(x)

        if self._A_hat is None or self._A_hat.device != device:
            self.set_adjacency(edge_index, num_nodes, device)

        x = F.relu(self.input_proj(x))

        for W, norm in zip(self.weights, self.norms):
            identity = x
            x = torch.sparse.mm(self._A_hat, x)
            x = x @ W
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + identity

        x = self.output(x) + 0.1 * res
        return F.softmax(x / self.temperature, dim=1)


def get_model(model_type, in_dim, hidden_dim, n_clusters,
              dropout=0.3, n_layers=4, temperature=0.5):
    if model_type == 'deep_gcn':
        return DeepGCNModel(in_dim, hidden_dim, n_clusters,
                            dropout=dropout, n_layers=n_layers,
                            temperature=temperature)
    if model_type == 'sparse_gcn':
        return SparseGCNModel(in_dim, hidden_dim, n_clusters,
                              dropout=dropout, n_layers=n_layers,
                              temperature=temperature)
    raise ValueError(f"Unknown sparse model type: {model_type}")
