"""Dataset loaders for real benchmarks (citation, co-purchase, attributed, airports, OGB, Flickr, Reddit2)."""

import numpy as np
import torch
from scipy.sparse import coo_array, coo_matrix
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    CitationFull,
    Airports,
    AttributedGraphDataset,
    Flickr,
    Reddit2,
)
from torch_geometric.utils import coalesce
from ogb.nodeproppred import PygNodePropPredDataset

# PyTorch 2.6+ / OGB compatibility patch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
torch.serialization.add_safe_globals([
    DataEdgeAttr, DataTensorAttr, GlobalStorage,
])

# Graphs where building a dense normalized adjacency is infeasible.
LARGE_GRAPHS = ['ogbn-arxiv', 'Flickr', 'Reddit2']


def normalize_adj(adj):
    """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = coo_matrix(np.diag(d_inv_sqrt))
    return D_inv_sqrt @ adj @ D_inv_sqrt


def collect_data(dataset_name, root='./data'):
    """
    Unified loader for real benchmark datasets.

    Returns
    -------
    norm_adj : scipy sparse matrix or None
        D^{-1/2} A D^{-1/2}. None for LARGE_GRAPHS (built lazily by the model).
    data : torch_geometric.data.Data
    nclasses : int
    """

    # Citation networks
    if dataset_name in ['Cora_ML', 'DBLP']:
        dataset = CitationFull(root=root, name=dataset_name)
        data = dataset[0]
        nclasses = dataset.num_classes

    elif dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=root, name=dataset_name)
        data = dataset[0]
        nclasses = dataset.num_classes

    # Co-purchase networks
    elif dataset_name in ['Computers', 'Photo', 'Amazon_PC', 'Amazon_Computers']:
        name = 'Computers' if dataset_name in ['Amazon_PC', 'Amazon_Computers'] else dataset_name
        dataset = Amazon(root=root, name=name)
        data = dataset[0]
        nclasses = dataset.num_classes

    # Attributed networks
    elif dataset_name in ['WIKI', 'BLOGCATALOG', 'FACEBOOK']:
        dataset = AttributedGraphDataset(root=root, name=dataset_name)
        data = dataset[0]
        nclasses = dataset.num_classes

    # Airport networks
    elif dataset_name in ['USA', 'Brazil', 'Europe']:
        dataset = Airports(root=root, name=dataset_name)
        data = dataset[0]
        nclasses = dataset.num_classes

    # Flickr
    elif dataset_name == 'Flickr':
        dataset = Flickr(root=f'{root}/Flickr')
        data = dataset[0]
        nclasses = dataset.num_classes

    # Reddit2
    elif dataset_name == 'Reddit2':
        dataset = Reddit2(root=f'{root}/Reddit2')
        data = dataset[0]
        nclasses = dataset.num_classes

    # ogbn-arxiv
    elif dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(
            name='ogbn-arxiv',
            root=f'{root}/ogbn_arxiv',
        )
        data = dataset[0]
        data.y = data.y.squeeze(1)

        # ogbn-arxiv is directed: symmetrize.
        row, col = data.edge_index
        row_sym = torch.cat([row, col])
        col_sym = torch.cat([col, row])
        data.edge_index = torch.stack([row_sym, col_sym], dim=0)
        data.edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)

        nclasses = dataset.num_classes

    else:
        raise ValueError(
            f"Dataset '{dataset_name}' not recognized. Supported: "
            "Cora, CiteSeer, PubMed, Cora_ML, DBLP, Computers, Photo, "
            "WIKI, BLOGCATALOG, FACEBOOK, USA, Brazil, Europe, "
            "Flickr, Reddit2, ogbn-arxiv."
        )

    print(f"Dataset '{dataset_name}' loaded.")
    print(f"  Nodes    : {data.x.shape[0]:,}")
    print(f"  Features : {data.x.shape[1]}")
    print(f"  Edges    : {data.edge_index.shape[1]:,}")
    print(f"  Classes  : {nclasses}")

    nnodes = data.x.shape[0]

    if dataset_name in LARGE_GRAPHS:
        # Adjacency built lazily inside SparseGCN to avoid dense materialization.
        norm_adj = None
    else:
        row_np = data.edge_index[0].numpy()
        col_np = data.edge_index[1].numpy()
        weights = np.ones(len(row_np), dtype=np.float32)
        adj = coo_array(
            (weights, (row_np, col_np)),
            shape=(nnodes, nnodes),
        ).tocsr()
        norm_adj = normalize_adj(adj)

    return norm_adj, data, nclasses
