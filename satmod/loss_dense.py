"""
Dense Q_abs modularity loss.

For graph G with adjacency A (degrees d, total m) and attribute similarity
graph G' with weights W (strengths s, total w), define modularity matrices
    B      = A - d d^T / (2m)
    B_attr = W - s s^T / (2w)

For a soft assignment C in R^{n x k}, structural and attributed modularity are
    Q_struct = tr(C^T B C) / (2m)
    Q_attr   = tr(C^T B_attr C) / (2w)

Q_abs = alpha * Q_struct + (1 - alpha) * Q_attr, with alpha learnable via a
logit reparameterization (sigmoid(alpha_raw) keeps alpha in (0, 1)).

The loss minimizes the negation, plus a collapse regularizer that penalizes
unbalanced cluster column sums and an entropy term that encourages confident
node-level assignments.
"""

import torch
import torch.nn as nn


class ModularityLoss(nn.Module):
    def __init__(self, n_clusters, X, adj,
                 initial_alpha=0.5, initial_beta=1.0, initial_gamma=0.001):
        super().__init__()
        self.n_clusters = n_clusters

        # Learnable mixing coefficient via logit reparameterization.
        self.alpha_raw = nn.Parameter(torch.tensor(
            torch.logit(torch.tensor(initial_alpha)).item(),
            dtype=torch.float32,
        ))
        self.beta = nn.Parameter(torch.tensor(initial_beta, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(initial_gamma, dtype=torch.float32))

        # Precompute structural modularity matrix B = A - dd^T / (2m).
        if not torch.is_tensor(adj):
            adj_tensor = torch.tensor(adj.toarray(), dtype=torch.float32)
        elif adj.is_sparse:
            adj_tensor = adj.to_dense()
        else:
            adj_tensor = adj.float()

        deg = torch.sum(adj_tensor, dim=1)
        m = torch.sum(deg) / 2
        B = adj_tensor - torch.outer(deg, deg) / (2 * m)
        self.register_buffer('B', B)
        self.register_buffer('m', m)

        # Precompute attributed modularity matrix B_attr.
        # W = cos-sim(X, X) clipped to non-negative entries.
        x_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
        W = torch.mm(x_norm, x_norm.t()).clamp(min=0)
        s = torch.sum(W, dim=1)
        w = torch.sum(s) / 2
        B_attr = W - torch.outer(s, s) / (2 * w + 1e-8)
        self.register_buffer('B_attr', B_attr)
        self.register_buffer('w', w)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def forward(self, C):
        device = C.device
        n, k = C.shape

        # Modularity terms (negated for minimization).
        mod_loss = -(1 / (2 * self.m)) * torch.trace(C.t() @ self.B @ C)
        attr_loss = -(1 / (2 * self.w + 1e-8)) * torch.trace(C.t() @ self.B_attr @ C)

        # Collapse regularization: penalize column-sum deviation from uniform.
        ones = torch.ones((n, 1), device=device)
        collapse_term = torch.norm(C.t() @ ones - 1, p=1)
        collapse_reg = (
            torch.sqrt(torch.tensor(k * k, dtype=torch.float32, device=device)) / n
        ) * collapse_term

        # Node-level entropy (encourage confident assignments; subtracted below).
        eps = 1e-10
        entropy = -torch.sum(C * torch.log(C + eps)) / n

        alpha = self.alpha
        Qabs = alpha * mod_loss + (1 - alpha) * attr_loss

        loss = Qabs + self.beta * collapse_reg - self.gamma * entropy

        return (Qabs, mod_loss, attr_loss, loss,
                collapse_reg, entropy,
                alpha, self.beta, self.gamma)
