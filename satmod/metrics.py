"""Clustering evaluation metrics with Hungarian label alignment."""

import numpy as np
from sklearn import metrics
from munkres import Munkres


def clustering_metrics(true_labels, pred_labels):
    """
    Compute ACC, NMI, ARI, and macro-F1 with Hungarian-aligned predictions.

    ACC and F1 use aligned labels; NMI and ARI are alignment-invariant.
    """
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    l1 = list(set(true_labels))
    l2 = list(set(pred_labels))
    D = max(len(l1), len(l2))

    cost = np.zeros((D, D), dtype=int)
    for i, c1 in enumerate(l1):
        idx = np.where(true_labels == c1)[0]
        for j, c2 in enumerate(l2):
            cost[i, j] = np.sum(pred_labels[idx] == c2)

    indexes = Munkres().compute((-cost).tolist())

    new_pred = np.zeros(len(pred_labels))
    for i, j in indexes:
        if i < len(l1) and j < len(l2):
            new_pred[pred_labels == l2[j]] = l1[i]

    acc = metrics.accuracy_score(true_labels, new_pred)
    f1_macro = metrics.f1_score(true_labels, new_pred, average='macro')
    nmi = metrics.normalized_mutual_info_score(true_labels, pred_labels)
    ari = metrics.adjusted_rand_score(true_labels, pred_labels)

    return acc, nmi, ari, f1_macro
