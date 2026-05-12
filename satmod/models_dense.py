"""Dense GCN / GraphSAGE models for small-to-mid graphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GCNModel(nn.Module):
    """Two-layer GCN with leaky-ReLU and input-to-output residual."""

    def __init__(self, in_dim, hidden_dim, n_clusters,
                 dropout=0.5, leaky_relu_negative_slope=0.2):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, n_clusters)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_negative_slope)
        self.residual = nn.Linear(in_dim, n_clusters, bias=False)

    def forward(self, x, edge_index):
        res = self.residual(x)
        x = self.gcn1(x, edge_index)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        x = x + 0.1 * res
        return F.softmax(x, dim=1)


class GraphSAGEModel(nn.Module):
    """Two-layer GraphSAGE with leaky-ReLU and residual."""

    def __init__(self, in_dim, hidden_dim, n_clusters,
                 dropout=0.5, leaky_relu_negative_slope=0.2):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, n_clusters)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_negative_slope)
        self.residual = nn.Linear(in_dim, n_clusters, bias=False)

    def forward(self, x, edge_index):
        res = self.residual(x)
        x = self.sage1(x, edge_index)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.sage2(x, edge_index)
        x = x + 0.1 * res
        return F.softmax(x, dim=1)


def get_model(model_type, in_dim, hidden_dim, n_clusters,
              dropout=0.5, leaky_relu_negative_slope=0.2):
    if model_type == 'gcn':
        return GCNModel(in_dim, hidden_dim, n_clusters,
                        dropout, leaky_relu_negative_slope)
    if model_type == 'graphsage':
        return GraphSAGEModel(in_dim, hidden_dim, n_clusters,
                              dropout, leaky_relu_negative_slope)
    raise ValueError(f"Unknown model type: {model_type}")
