# SAT-MoD: End-to-End Deep Graph Clustering via Structural and Attributed Modularity

Reference implementation of **SAT-MoD**, a GNN-based deep graph clustering method that jointly optimizes a convex combination of **structural** and **attributed** modularity, denoted $Q_{abs}$. The weight coefficient $\alpha$ is learned end-to-end via a logit reparameterization, removing the need for dataset-specific tuning of the structure–attribute trade-off.

Two training entry points are provided:

- `train_dense.py` &nbsp;— small-to-mid graphs (Cora, CiteSeer, PubMed, Cora_ML, DBLP, Photo, Computers, WIKI, BLOGCATALOG).
- `train_sparse.py` — large graphs (ogbn-arxiv, Flickr, Reddit2).

The two scripts share the same dataset, model, and metric utilities; they differ in how they construct the modularity matrices (dense $B$, $B_{\text{attr}}$ vs. sparse edge-indexed and top-$k$ k-NN representations) and in the regularizers used.

---

## Method at a glance

For graph $G$ with adjacency $A$ (degrees $d$, total mass $m$) and attribute similarity graph $W$ (strengths $s$, total mass $w$), let

$$B = A - \frac{d d^\top}{2m}, \qquad B_{\text{attr}} = W - \frac{s s^\top}{2w}.$$

For a soft assignment $C \in \mathbb{R}^{n \times k}$,

$$Q_{abs}(C) = \alpha \cdot \frac{Tr(C^\top B C)}{2m} + (1 - \alpha) \cdot \frac{Tr(C^\top B_{\text{attr}} C)}{2w},$$

with $\alpha = \sigma(\alpha_{\text{raw}}) \in (0, 1)$ learnable. The training objective minimizes $-Q_{abs}(C)$ plus collapse, entropy, and (in the sparse variant) balance / occupancy regularizers.

---

## Usage

### Dense loss (small / mid graphs)

```bash
python train_dense.py --dataset Cora     --epochs 200 --runs 10 --hidden 512
python train_dense.py --dataset Photo    --epochs 200 --runs 10 --hidden 512
python train_dense.py --dataset Cora_ML  --epochs 200 --runs 10 --hidden 512 --plot
```

### Sparse loss (large graphs)

```bash
python train_sparse.py --dataset ogbn-arxiv --epochs 200 --runs 5 --hidden 256
python train_sparse.py --dataset Flickr     --epochs 200  --runs 5 --hidden 256
python train_sparse.py --dataset Reddit2    --epochs 200  --runs 5 --hidden 128
```

`train_sparse.py` auto-selects `sparse_gcn` for Reddit2 (CPU adjacency build → GPU sparse matmul) and `deep_gcn` for the others; override with `--model`.

### Common flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset` | required | One of the supported names below. |
| `--data-root` | `./data` | Root directory where PyG / OGB caches go. |
| `--epochs` | 500 / 1000 | Total epochs per run. |
| `--runs` | 5 / 3 | Independent runs; final results are mean ± std. |
| `--hidden` | 64 / 256 | GNN hidden dimension. |
| `--alpha` | 0.5 | Initial $\alpha$ in $(0, 1)$; learned. |
| `--beta` | 1.0 | Initial collapse-regularization weight. |
| `--gamma` | 0.001 | Initial entropy weight. |
| `--seed` | None | Set for reproducible runs. |
| `--plot` | off | Save a t-SNE plot of the best run to `figures/`. |
| `--device` | `auto` | `auto` / `cuda` / `cpu`. |

Sparse-only flags include `--delta`, `--eta` (balance / occupancy regularizers), `--temperature`, `--n-layers`, `--k-neighbors`, `--chunk-size`, `--edge-chunk-size`, `--eval-every`, `--patience`, and `--auto-scale-beta`. Run `python train_sparse.py --help` for the full list.

---

## Supported datasets

| Script | Dataset names |
| --- | --- |
| `train_dense.py` | `Cora`, `CiteSeer`, `PubMed`, `Cora_ML`, `DBLP`, `Computers`, `Photo`, `WIKI`, `BLOGCATALOG` |
| `train_sparse.py` | All of the above plus `ogbn-arxiv`, `Flickr`, `Reddit2` |

`ogbn-arxiv`, `Flickr`, and `Reddit2` are too large for the dense modularity matrices and will be rejected by `train_dense.py`.

---

## Outputs

Each run writes:

- `saved_embeddings/<dataset>_best_run_embeddings.npz` — soft-assignment matrix, predicted labels, best ACC, and run index for the best run across `--runs`.
- `logs/<dataset>_best_run_log.txt` — per-epoch loss / metric log of the best run.
- `figures/<dataset>_best_run.png` — t-SNE of the best run's soft assignments (only with `--plot`).

The final stdout block reports mean ± std of ACC, NMI, ARI, F1, runtime, and peak memory across all valid runs.

---

## Repository layout

```
.
├── satmod/
│   ├── __init__.py
│   ├── datasets.py        # collect_data() for all benchmarks
│   ├── metrics.py         # ACC / NMI / ARI / F1 with Hungarian alignment
│   ├── models_dense.py    # GCN, GraphSAGE for the dense variant
│   ├── models_sparse.py   # DeepGCN, SparseGCN for the sparse variant
│   ├── loss_dense.py      # ModularityLoss        (dense B, B_attr)
│   ├── loss_sparse.py     # ModularityLossSparse  (edge-indexed + top-k k-NN)
│   └── plotting.py        # t-SNE utility
├── train_dense.py
├── train_sparse.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
