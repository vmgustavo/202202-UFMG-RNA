from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_samples, silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from .graph import GabrielGraph


def sil_neg_samples_score(X, labels):
    res = silhouette_samples(X, labels)

    counts = Counter(res > 0)
    return counts[False] / (counts[False] + counts[True])


class GGMetrics:
    def __init__(self, X, labels):
        gg = GabrielGraph(X)
        gg.adjacency()

        self.adj_mat = gg.adj_mat_

        scores = list()
        for i in range(X.shape[0]):
            neighs = self.adj_mat.getrow(i)
            indexes = neighs.tolil().rows[0]

            diff_class = pd.Series(labels[indexes] != labels[i])

            perc = diff_class.astype("int").mean()
            scores.append(perc)

        self.scores = np.array(scores)

    def gg_neigh_index(self, *args, **kwargs):  # noqa
        return np.mean(self.scores)

    def gg_neigh_count(self, *args, **kwargs):  # noqa
        return np.sum(self.scores > 0.5) / self.scores.shape[0]


def cluster_evaluate(X, labels):
    gg_metrics = GGMetrics(X, labels)

    metrics = [
        gg_metrics.gg_neigh_index,
        gg_metrics.gg_neigh_count,
        silhouette_score,
        sil_neg_samples_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    ]

    return {
        metric.__name__: metric(X=X, labels=labels)
        for metric in metrics
    }
