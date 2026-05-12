"""t-SNE visualization of soft assignments / embeddings."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(embeddings, labels, dataset_name,
              out_dir='figures', tag='final', dpi=300, show=False):
    """
    Project embeddings to 2D with t-SNE and color by ``labels``.

    Parameters
    ----------
    embeddings : np.ndarray or torch.Tensor of shape (n, d)
    labels     : list, np.ndarray, or torch.Tensor of length n
    dataset_name : str
    out_dir : str
        Directory where the PNG is written.
    tag : str
        Filename suffix (e.g. 'final', 'epoch100').
    """
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=labels, cmap='tab10', s=10,
    )
    plt.colorbar(scatter, label='Labels')
    plt.title(f't-SNE of {dataset_name} ({tag})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'{dataset_name}_{tag}.png')
    plt.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    return out_path
