"""
Sparse Q_abs modularity loss for large graphs.

Avoids materializing dense modularity matrices. Structural modularity is
computed by indexing into edge endpoints; attributed modularity uses a
sparse top-k k-NN cosine graph instead of the full similarity matrix.

In addition to the dense version's collapse and entropy terms, this variant
adds:
  - balance_penalty : negative entropy of cluster-size distribution; pushes
                      column masses toward uniform.
  - occupancy_gap   : hinge penalty for any cluster falling below 1/k mass.
  - qabs_clip       : soft floor on Q_abs to prevent runaway negative values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


class ModularityLossSparse(nn.Module):
    def __init__(self, n_clusters, X, edge_index, num_nodes,
                 initial_alpha=0.5, initial_beta=1.0,
                 initial_gamma=0.001, initial_delta=0.5,
                 initial_eta=1.0,
                 k_neighbors=10, chunk_size=256,
                 edge_chunk_size=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.chunk_size = chunk_size
        self.edge_chunk_size = edge_chunk_size

        # Learnable scalars.
        self.alpha_raw = nn.Parameter(torch.tensor(
            torch.logit(torch.tensor(initial_alpha)).item(),
            dtype=torch.float32,
        ))
        self.beta = nn.Parameter(torch.tensor(initial_beta, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(initial_gamma, dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(initial_delta, dtype=torch.float32))
        self.eta = nn.Parameter(torch.tensor(initial_eta, dtype=torch.float32))

        # Structural buffers.
        deg = degree(edge_index[0], num_nodes=num_nodes).float()
        m = deg.sum() / 2.0
        self.register_buffer('edge_index_buf', edge_index)
        self.register_buffer('deg', deg)
        self.register_buffer('m', m)
        self.num_nodes = num_nodes

        # Attributed: sparse symmetric top-k k-NN graph.
        W_sparse, s, w = self._build_sparse_W(X, k_neighbors, chunk_size)
        self.register_buffer('W_sparse', W_sparse)
        self.register_buffer('s_attr', s)
        self.register_buffer('w_attr', w)

    @torch.no_grad()
    def _build_sparse_W(self, X, k, chunk_size):
        """Chunked top-k cosine-similarity graph, symmetrized."""
        n = X.shape[0]
        device = X.device
        x_norm = F.normalize(X, p=2, dim=1)

        rows_list, cols_list, vals_list = [], [], []

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            sim_chunk = torch.mm(x_norm[start:end], x_norm.t()).clamp(min=0)

            # Zero self-similarity.
            for i in range(end - start):
                sim_chunk[i, start + i] = 0.0

            k_eff = min(k, n - 1)
            topk_vals, topk_idx = torch.topk(sim_chunk, k=k_eff, dim=1)

            mask = topk_vals > 0
            r = torch.arange(start, end, device=device)\
                     .unsqueeze(1).expand(-1, k_eff)[mask]
            c = topk_idx[mask]
            v = topk_vals[mask]

            rows_list.append(r.cpu())
            cols_list.append(c.cpu())
            vals_list.append(v.cpu())

            del sim_chunk
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        rows = torch.cat(rows_list)
        cols = torch.cat(cols_list)
        vals = torch.cat(vals_list)

        rows_sym = torch.cat([rows, cols])
        cols_sym = torch.cat([cols, rows])
        vals_sym = torch.cat([vals, vals]) / 2.0

        indices = torch.stack([rows_sym, cols_sym])
        W_sparse = torch.sparse_coo_tensor(
            indices, vals_sym, size=(n, n),
        ).coalesce()

        s = torch.sparse.sum(W_sparse, dim=1).to_dense()
        w = s.sum() / 2.0

        return W_sparse, s, w

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def _structural_loss(self, C, row, col, deg, m):
        n = C.shape[0]
        device = C.device
        num_edges = row.shape[0]
        chunk_size = self.edge_chunk_size or num_edges

        trace_AC = torch.tensor(0.0, device=device)
        for start in range(0, num_edges, chunk_size):
            end = min(start + chunk_size, num_edges)
            trace_AC = trace_AC + (C[row[start:end]] * C[col[start:end]]).sum()

        trace_AC = trace_AC / n
        degC = deg.to(device) @ C
        null_mod = (degC * degC).sum() / (2.0 * m * n + 1e-8)

        return -(trace_AC - null_mod)

    def _attributed_loss(self, C, W_sp, s, w):
        n = C.shape[0]
        device = C.device
        WC = torch.sparse.mm(W_sp, C)
        trace_WC = (C * WC).sum() / n
        sC = s.to(device) @ C
        null_attr = (sC * sC).sum() / (2.0 * w * n + 1e-8)
        return -(trace_WC - null_attr)

    def _regularisation(self, C):
        device = C.device
        n, k = C.shape

        ones = torch.ones((n, 1), device=device)
        collapse_term = torch.norm(C.t() @ ones - 1, p=1)
        collapse_reg = (
            torch.sqrt(torch.tensor(k * k, dtype=torch.float32, device=device)) / n
        ) * collapse_term

        # Node-level entropy.
        entropy = -torch.sum(C * torch.log(C + 1e-10)) / n

        # Cluster-size entropy: maximized when columns are equal-mass.
        cluster_sizes = C.sum(dim=0) / n
        size_entropy = -(cluster_sizes * torch.log(cluster_sizes + 1e-10)).sum()
        balance_penalty = -size_entropy

        # Hinge for any cluster below 1/k mass.
        min_expected = 1.0 / k
        occupancy_gap = F.relu(min_expected - cluster_sizes).sum()

        return collapse_reg, entropy, balance_penalty, occupancy_gap

    def _combine(self, mod_loss, attr_loss,
                 collapse_reg, entropy,
                 balance_penalty, occupancy_gap):
        alpha = self.alpha
        Qabs = alpha * mod_loss + (1 - alpha) * attr_loss

        # Soft floor: discourage Q_abs from going below -0.5.
        qabs_clip = F.relu(-Qabs - 0.5)

        loss = (Qabs
                + self.beta * collapse_reg
                - self.gamma * entropy
                + self.delta * balance_penalty
                + self.eta * occupancy_gap
                + 1.0 * qabs_clip)

        return Qabs, loss

    def forward(self, C):
        device = C.device
        row = self.edge_index_buf[0].to(device)
        col = self.edge_index_buf[1].to(device)
        W_sp = self.W_sparse.to(device)

        mod_loss = self._structural_loss(C, row, col, self.deg, self.m)
        attr_loss = self._attributed_loss(C, W_sp, self.s_attr, self.w_attr)

        (collapse_reg, entropy,
         balance_penalty, occupancy_gap) = self._regularisation(C)

        Qabs, loss = self._combine(
            mod_loss, attr_loss,
            collapse_reg, entropy,
            balance_penalty, occupancy_gap,
        )

        return (Qabs, mod_loss, attr_loss, loss,
                collapse_reg, entropy,
                self.alpha, self.beta, self.gamma)
