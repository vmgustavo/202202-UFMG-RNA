from collections import Counter

from sklearn.metrics import (
    silhouette_samples, silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def sil_neg_samples_score(X, labels):
    res = silhouette_samples(X, labels)

    counts = Counter(res > 0)
    return counts[False] / (counts[False] + counts[True])


def cluster_evaluate(X, labels):
    metrics = [
        silhouette_score,
        sil_neg_samples_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    ]

    return {
        metric.__name__: metric(X=X, labels=labels)
        for metric in metrics
    }
