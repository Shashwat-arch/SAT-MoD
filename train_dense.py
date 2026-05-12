"""
SAT-MoD training with dense Q_abs loss (for small-to-mid graphs).

Example
-------
    python train_dense.py --dataset Cora --epochs 500 --runs 5 --hidden 64
"""

import argparse
import os
import time
import traceback

import numpy as np
import psutil
import torch

from satmod.datasets import collect_data, LARGE_GRAPHS
from satmod.loss_dense import ModularityLoss
from satmod.metrics import clustering_metrics
from satmod.models_dense import get_model
from satmod.plotting import plot_tsne


def parse_args():
    p = argparse.ArgumentParser(
        description='SAT-MoD with dense modularity loss.',
    )
    p.add_argument('--dataset', type=str, required=True,
                   help='Dataset name (Cora, CiteSeer, PubMed, Cora_ML, DBLP, '
                        'Computers, Photo, WIKI, BLOGCATALOG, FACEBOOK, '
                        'USA, Brazil, Europe).')
    p.add_argument('--data-root', type=str, default='./data',
                   help='Root directory for PyG datasets.')
    p.add_argument('--model', type=str, default='gcn',
                   choices=['gcn', 'graphsage'])
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--alpha', type=float, default=0.5,
                   help='Initial Q_abs mixing coefficient (in (0, 1)).')
    p.add_argument('--beta', type=float, default=1.0,
                   help='Initial collapse-regularization weight.')
    p.add_argument('--gamma', type=float, default=0.001,
                   help='Initial entropy-regularization weight.')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--log-dir', type=str, default='logs')
    p.add_argument('--save-dir', type=str, default='saved_embeddings')
    p.add_argument('--plot', action='store_true',
                   help='Save a t-SNE plot of the best run.')
    p.add_argument('--device', type=str, default='auto',
                   choices=['auto', 'cuda', 'cpu'])
    return p.parse_args()


def resolve_device(flag):
    if flag == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(flag)


def run(args):
    if args.dataset in LARGE_GRAPHS:
        raise ValueError(
            f"Dataset '{args.dataset}' is too large for the dense loss. "
            "Use `train_sparse.py` instead."
        )

    device = resolve_device(args.device)
    print(f'Using device: {device}')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    acc_list, nmi_list, ari_list, f1_list = [], [], [], []
    alpha_list, time_list, mem_list = [], [], []

    best_acc = -1.0
    best_run_id = None
    best_embeddings = None
    best_pred_labels = None
    best_labels = None
    best_run_log = []

    process = psutil.Process()

    for run_id in range(args.runs):
        print(f'\n========== Run {run_id + 1}/{args.runs} ==========')

        try:
            start_time = time.time()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            norm_adj, data, nclasses = collect_data(
                args.dataset, root=args.data_root,
            )
            features = data.x.to(device)
            edge_index = data.edge_index.to(device)
            labels = data.y.tolist()
            in_dim = features.shape[1]

            adj_cpu = (
                torch.tensor(norm_adj.toarray(), dtype=torch.float32)
                if not torch.is_tensor(norm_adj) else norm_adj.float().cpu()
            )

            model = get_model(
                args.model, in_dim, args.hidden, nclasses,
                dropout=args.dropout,
            ).to(device)

            loss_fn = ModularityLoss(
                n_clusters=nclasses,
                X=data.x,
                adj=adj_cpu,
                initial_alpha=args.alpha,
                initial_beta=args.beta,
                initial_gamma=args.gamma,
            ).to(device)

            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(loss_fn.parameters()),
                lr=args.lr,
            )

            current_run_log = [
                f'========== Run {run_id + 1}/{args.runs} ==========\n',
            ]

            for epoch in range(1, args.epochs + 1):
                model.train()
                optimizer.zero_grad()
                out = model(features, edge_index)

                (Qabs, mod_loss, attr_loss, loss,
                 collapse_reg, entropy,
                 alpha, beta, gamma) = loss_fn(out)

                loss.backward()
                optimizer.step()

                current_run_log.append(
                    f' {epoch} {Qabs.item():.4f} {alpha.item():.4f}\n'
                )

            # Final evaluation.
            model.eval()
            with torch.no_grad():
                out = model(features, edge_index)
                pred_labels = out.argmax(dim=1).cpu().numpy()
                embeddings = out.detach().cpu().numpy()

            acc, nmi, ari, f1_macro = clustering_metrics(labels, pred_labels)

            acc_list.append(acc)
            nmi_list.append(nmi)
            ari_list.append(ari)
            f1_list.append(f1_macro)
            alpha_list.append(alpha.item())

            elapsed = time.time() - start_time
            time_list.append(elapsed)

            mem_used = (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
                if device.type == 'cuda'
                else process.memory_info().rss / (1024 ** 2)
            )
            mem_list.append(mem_used)

            summary = (
                f'Run {run_id + 1} | Alpha={alpha.item():.4f} '
                f'ACC={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}, F1={f1_macro:.4f} '
                f'| Time={elapsed:.2f}s, Mem={mem_used:.2f}MB\n'
            )
            print(summary, end='')
            current_run_log.append(summary)

            if acc > best_acc:
                best_acc = acc
                best_run_id = run_id + 1
                best_embeddings = embeddings
                best_pred_labels = pred_labels
                best_labels = labels
                best_run_log = current_run_log

        except Exception:
            print(f'[WARNING] Run {run_id + 1} failed.')
            traceback.print_exc()
            continue

    # Persist artifacts from the best run.
    if best_embeddings is not None:
        emb_path = os.path.join(
            args.save_dir, f'{args.dataset}_best_run_embeddings.npz',
        )
        np.savez(
            emb_path,
            embeddings=best_embeddings,
            pred_labels=best_pred_labels,
            best_acc=best_acc,
            best_run=best_run_id,
        )
        print(f'\n[INFO] Best run {best_run_id} | ACC={best_acc:.4f}')
        print(f'[INFO] Saved embeddings -> {emb_path}')

    if best_run_log:
        log_path = os.path.join(args.log_dir, f'{args.dataset}_best_run_log.txt')
        with open(log_path, 'w') as f:
            f.writelines(best_run_log)
        print(f'[INFO] Saved best run log -> {log_path}')

    if args.plot and best_embeddings is not None:
        path = plot_tsne(
            best_embeddings, best_labels, args.dataset,
            out_dir='figures', tag='best_run',
        )
        print(f'[INFO] Saved t-SNE plot   -> {path}')

    def stats(arr):
        if not arr:
            return None, None, None, None
        return np.min(arr), np.max(arr), np.mean(arr), np.std(arr)

    print('\n========== FINAL RESULTS ==========')
    print(f'Valid runs: {len(acc_list)}/{args.runs}\n')

    if not acc_list:
        print('[WARNING] No valid runs.')
        return

    acc_s = stats(acc_list)
    nmi_s = stats(nmi_list)
    ari_s = stats(ari_list)
    f1_s = stats(f1_list)
    time_s = stats(time_list)
    mem_s = stats(mem_list)

    print(f'ACC  : mean={acc_s[2]:.4f} ± {acc_s[3]:.4f}')
    print(f'NMI  : mean={nmi_s[2]:.4f} ± {nmi_s[3]:.4f}')
    print(f'ARI  : mean={ari_s[2]:.4f} ± {ari_s[3]:.4f}')
    print(f'F1   : mean={f1_s[2]:.4f} ± {f1_s[3]:.4f}')
    print(f'Time : mean={time_s[2]:.2f}s ± {time_s[3]:.2f}s')
    print(f'Mem  : mean={mem_s[2]:.2f}MB ± {mem_s[3]:.2f}MB')


if __name__ == '__main__':
    run(parse_args())
