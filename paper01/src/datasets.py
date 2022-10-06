import numpy as np
from sklearn import datasets


def get_linear(n_obs: int, n_feats: int):
    data, target = datasets.make_blobs(
        n_samples=n_obs, n_features=n_feats,
        centers=[[-2, -2], [2, 2]],
        cluster_std=1.5,
    )
    target = (target == 1) * 2 - 1
    return data, target


def get_blobs(n_obs: int, n_feats: int):
    data, target = datasets.make_blobs(
        n_samples=n_obs, n_features=n_feats,
        centers=[
            [-2, -2], [2, 2],
            [-2, 2], [2, -2],
        ],
        cluster_std=1.2,
    )
    target = np.isin(target, [0, 1]) * 2 - 1
    return data, target


def get_moons(n_obs: int):
    data, target = datasets.make_moons(
        n_samples=n_obs, noise=0.2,
    )

    target = (target == 1) * 2 - 1
    data = data - [data[:, 0].mean(), data[:, 1].mean()]
    return data, target
