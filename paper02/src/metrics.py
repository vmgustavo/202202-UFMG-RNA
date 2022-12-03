from collections import Counter

import numpy as np
from sklearn.metrics import (
    silhouette_samples, silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from .graph import GabrielGraph, Topology


def sil_neg_samples_score(X, labels):
    res = silhouette_samples(X, labels)

    counts = Counter(res > 0)
    return counts[False] / (counts[False] + counts[True])


class GGMetrics:
    def __init__(self, X, labels):
        gg = GabrielGraph(X)
        adj_mat = gg.adjacency()
        tpl = Topology(X, labels, adj_mat)
        self.quality = tpl.class_quality()
        self.scores = np.array(self.quality["link_prop"].values)

    def gg_neigh_index(self, *args, **kwargs):  # noqa
        return np.mean(self.scores)

    def gg_class_quality(self):
        return (
            self.quality
            .groupby(["target", "quality"])
            .agg({"quality": "count"})
        )


def cluster_evaluate(X, labels):
    gg_metrics = GGMetrics(X, labels)

    metrics = [
        gg_metrics.gg_neigh_index,
        silhouette_score,
        sil_neg_samples_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    ]

    return {
        metric.__name__: metric(X=X, labels=labels)
        for metric in metrics
    }
