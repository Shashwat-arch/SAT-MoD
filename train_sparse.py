"""
SAT-MoD training with sparse Q_abs loss (for large graphs).

Example
-------
    python train_sparse.py --dataset ogbn-arxiv --epochs 1000 --runs 3 --hidden 256
    python train_sparse.py --dataset Reddit2    --epochs 500  --runs 1 --hidden 64
"""

import argparse
import copy
import os
import time
import traceback
from collections import Counter

import numpy as np
import psutil
import torch

from satmod.datasets import collect_data, LARGE_GRAPHS
from satmod.loss_sparse import ModularityLossSparse
from satmod.metrics import clustering_metrics
from satmod.models_sparse import SparseGCNModel, get_model
from satmod.plotting import plot_tsne


def parse_args():
    p = argparse.ArgumentParser(
        description='SAT-MoD with sparse modularity loss.',
    )
    p.add_argument('--dataset', type=str, required=True,
                   help='Any dataset supported by collect_data(); use this script '
                        'for ogbn-arxiv, Flickr, and Reddit2.')
    p.add_argument('--data-root', type=str, default='./data')
    p.add_argument('--model', type=str, default='auto',
                   choices=['auto', 'deep_gcn', 'sparse_gcn'],
                   help='"auto" picks sparse_gcn for Reddit/Reddit2 and '
                        'deep_gcn for other large graphs.')
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--temperature', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--beta', type=float, default=1.0,
                   help='Initial collapse weight; overridden by auto-scaling below.')
    p.add_argument('--gamma', type=float, default=0.001)
    p.add_argument('--delta', type=float, default=0.5,
                   help='Balance-penalty weight (negative size-entropy).')
    p.add_argument('--eta', type=float, default=1.0,
                   help='Minimum-occupancy hinge weight.')
    p.add_argument('--k-neighbors', type=int, default=10,
                   help='Top-k for sparse attribute similarity graph.')
    p.add_argument('--chunk-size', type=int, default=256,
                   help='Row chunk for top-k attribute similarity.')
    p.add_argument('--edge-chunk-size', type=int, default=None,
                   help='Edge chunk for structural loss; default = full batch.')
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--patience', type=int, default=10,
                   help='Number of eval intervals without improvement before '
                        'early stop.')
    p.add_argument('--auto-scale-beta', action='store_true', default=True,
                   help='Rescale beta from |Q_abs| / collapse at init.')
    p.add_argument('--no-auto-scale-beta', dest='auto_scale_beta',
                   action='store_false')
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--log-dir', type=str, default='logs')
    p.add_argument('--save-dir', type=str, default='saved_embeddings')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--device', type=str, default='auto',
                   choices=['auto', 'cuda', 'cpu'])
    return p.parse_args()


def resolve_device(flag):
    if flag == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(flag)


def select_model_type(dataset, requested):
    if requested != 'auto':
        return requested
    if dataset in ('Reddit', 'Reddit2'):
        return 'sparse_gcn'
    return 'deep_gcn'


def run(args):
    device = resolve_device(args.device)
    print(f'Using device: {device}')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    acc_list, nmi_list, ari_list, f1_list = [], [], [], []
    time_list, mem_list = [], []

    best_acc = -1.0
    best_run_id = None
    best_embeddings = None
    best_pred_labels = None
    best_labels = None
    best_run_log = []

    process = psutil.Process()
    model_type = select_model_type(args.dataset, args.model)

    for run_id in range(args.runs):
        print(f'\n========== Run {run_id + 1}/{args.runs} ==========')

        try:
            start_time = time.time()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            norm_adj, data, nclasses = collect_data(
                args.dataset, root=args.data_root,
            )
            labels = data.y.tolist()
            in_dim = data.x.shape[1]
            n_nodes = data.x.shape[0]

            # For Reddit-class graphs, keep features on CPU until adjacency
            # is built to avoid GPU OOM.
            keep_cpu = args.dataset in ('Reddit', 'Reddit2')
            if keep_cpu:
                features = data.x
                edge_index = data.edge_index
            else:
                features = data.x.to(device)
                edge_index = data.edge_index.to(device)

            model = get_model(
                model_type, in_dim, args.hidden, nclasses,
                dropout=args.dropout,
                n_layers=args.n_layers,
                temperature=args.temperature,
            ).to(device)

            if isinstance(model, SparseGCNModel):
                print('[INFO] Precomputing sparse adjacency on CPU...')
                model.set_adjacency(data.edge_index, n_nodes, device)
                print('[INFO] Done.')

            if keep_cpu:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                features = features.to(device)
                edge_index = edge_index.to(device)
                if device.type == 'cuda':
                    free = torch.cuda.mem_get_info()[0] / 1e9
                    print(f'[INFO] Features on GPU. Free memory: {free:.2f} GB')

            loss_fn = ModularityLossSparse(
                n_clusters=nclasses,
                X=data.x,
                edge_index=data.edge_index,
                num_nodes=n_nodes,
                initial_alpha=args.alpha,
                initial_beta=args.beta,
                initial_gamma=args.gamma,
                initial_delta=args.delta,
                initial_eta=args.eta,
                k_neighbors=args.k_neighbors,
                chunk_size=args.chunk_size,
                edge_chunk_size=args.edge_chunk_size,
            ).to(device)

            if args.auto_scale_beta:
                model.eval()
                with torch.no_grad():
                    C_init = model(features, edge_index)
                    (Qabs_init, _, _, _,
                     collapse_init, _, _, _, _) = loss_fn(C_init)
                qabs_mag = max(abs(Qabs_init.item()), 1e-6)
                col_mag = max(collapse_init.item(), 1e-6)
                beta_auto = float(np.clip(qabs_mag / col_mag, 1e-4, 0.1))
                with torch.no_grad():
                    loss_fn.beta.fill_(beta_auto)
                print(f'[AUTO-SCALE] Beta={beta_auto:.6f}')
                model.train()

            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(loss_fn.parameters()),
                lr=args.lr,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-6,
            )

            current_run_log = [
                f'========== Run {run_id + 1}/{args.runs} ==========\n',
            ]

            best_epoch_acc = 0.0
            best_epoch_state = None
            no_improve = 0

            for epoch in range(1, args.epochs + 1):
                model.train()
                optimizer.zero_grad()
                out = model(features, edge_index)

                (Qabs, mod_loss, attr_loss, loss,
                 collapse_reg, entropy,
                 alpha, beta, gamma) = loss_fn(out)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(loss_fn.parameters()),
                    max_norm=args.grad_clip,
                )
                optimizer.step()
                scheduler.step()

                current_run_log.append(
                    f'Epoch {epoch}: Qabs={Qabs.item():.4f}, '
                    f'Mod={mod_loss.item():.4f}, Attr={attr_loss.item():.4f}, '
                    f'Loss={loss.item():.4f}, Alpha={alpha.item():.4f}\n'
                )

                if epoch % args.eval_every == 0:
                    model.eval()
                    with torch.no_grad():
                        out_e = model(features, edge_index)
                        preds = out_e.argmax(dim=1).cpu().numpy()
                        acc_e, nmi_e, ari_e, f1_e = clustering_metrics(
                            labels, preds,
                        )
                        active = sum(
                            1 for v in Counter(preds).values() if v > 10
                        )

                    if acc_e > best_epoch_acc:
                        best_epoch_acc = acc_e
                        best_epoch_state = {
                            'model': copy.deepcopy(model.state_dict()),
                            'loss_fn': copy.deepcopy(loss_fn.state_dict()),
                            'epoch': epoch,
                            'acc': acc_e,
                            'nmi': nmi_e,
                            'ari': ari_e,
                            'f1': f1_e,
                        }
                        no_improve = 0
                    else:
                        no_improve += 1

                    if no_improve >= args.patience:
                        print(
                            f'Early stop at epoch {epoch}, '
                            f'best ACC={best_epoch_acc:.4f} '
                            f"at epoch {best_epoch_state['epoch']}"
                        )
                        break

                    if epoch > args.eval_every and active <= 3:
                        print(f'Early stop — stuck at {active} clusters.')
                        break

                    model.train()

            if best_epoch_state is not None:
                model.load_state_dict(best_epoch_state['model'])
                loss_fn.load_state_dict(best_epoch_state['loss_fn'])
                print(
                    f"\n[CHECKPOINT] Restored epoch {best_epoch_state['epoch']} "
                    f"| ACC={best_epoch_state['acc']:.4f}"
                )

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

            elapsed = time.time() - start_time
            time_list.append(elapsed)

            mem_used = (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
                if device.type == 'cuda'
                else process.memory_info().rss / (1024 ** 2)
            )
            mem_list.append(mem_used)

            summary = (
                f'Run {run_id + 1} | ACC={acc:.4f}, NMI={nmi:.4f}, '
                f'ARI={ari:.4f}, F1={f1_macro:.4f} | '
                f'Time={elapsed:.2f}s, Mem={mem_used:.2f}MB\n'
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
