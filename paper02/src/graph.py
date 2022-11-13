from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.sparse import coo_matrix


class GabrielGraph:
    adj_mat_: coo_matrix
    distmat_: np.array

    def __init__(self, X: pd.DataFrame) -> None:
        self.X = X

    def adjacency(self) -> coo_matrix:
        n_obs = self.X.shape[0]
        self.distmat_ = np.power(distance.squareform(distance.pdist(self.X)), 2)
        self.distmat_[np.diag_indices(n_obs)] = np.inf

        with Pool() as pool:
            params = product(range(n_obs), range(n_obs))
            res = pool.starmap(self.calc, params)
        res = np.array(list(filter(lambda x: x is not None, res)))

        n_adjs = len(res)
        adj_mat = coo_matrix((np.ones(n_adjs), (res[:, 0], res[:, 1])), shape=(n_obs, n_obs))

        self.adj_mat_ = adj_mat
        return adj_mat

    def calc(self, i: int, j: int):
        minimum = np.min(self.distmat_[i, :] + self.distmat_[j, :])
        if self.distmat_[i, j] <= minimum:
            return [i, j]

    def plot(self):
        graph = nx.from_scipy_sparse_array(self.adj_mat_)
        pos = {i: tuple(elem) for i, elem in self.X.iterrows()}

        options = {
            "font_size": 1,
            "node_size": 50,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 1,
            "width": 1,
        }
        nx.draw_networkx(graph, pos, **options)

        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.show()
