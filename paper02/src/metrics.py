from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_samples, silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from graph import GabrielGraph


def sil_neg_samples_score(X, labels):
    res = silhouette_samples(X, labels)

    counts = Counter(res > 0)
    return counts[False] / (counts[False] + counts[True])


class GGMetrics:
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels

        gg = GabrielGraph(X)
        gg.adjacency()

        self.adj_mat = gg.adj_mat_

        scores = list()
        weights = list()
        for i in range(self.X.shape[0]):
            neighs = self.adj_mat.getrow(i)
            indexes = neighs.tolil().rows[0]

            diff_class = pd.Series(self.labels.iloc[indexes] != self.labels.iloc[i])

            perc = diff_class.astype("int").mean()
            scores.append(perc)

            obs_weights = np.sqrt(np.power(X.iloc[indexes] - X.iloc[i], 2).sum(axis=1))
            weight_i = np.exp(obs_weights.sum())
            weights.append(weight_i)

        self.weights = np.array(weights)
        self.scores = np.array(scores)

    def gg_neigh_index(self, *args, **kwargs):
        return np.mean(self.scores)

    def gg_neigh_count(self, *args, **kwargs):
        return np.sum(self.scores > 0) / self.X.shape[0]

    def gg_weighted_index(self, *args, **kwargs):
        return np.mean(self.scores * self.weights)


def cluster_evaluate(X, labels):
    gg_metrics = GGMetrics(X, labels)

    metrics = [
        gg_metrics.gg_neigh_index,
        gg_metrics.gg_neigh_count,
        gg_metrics.gg_weighted_index,
        silhouette_score,
        sil_neg_samples_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    ]

    return {
        metric.__name__: metric(X=X, labels=labels)
        for metric in metrics
    }


if __name__ == '__main__':
    from datasets import get_linear

    data, target = get_linear(n_obs=100)
    print(cluster_evaluate(data, target))
