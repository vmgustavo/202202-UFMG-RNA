import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.sparse import coo_matrix


class GabrielGraph:
    adj_mat_: coo_matrix

    def __init__(self, X: pd.DataFrame) -> None:
        self.X = X

    def adjacency(self) -> coo_matrix:
        n_obs = self.X.shape[0]
        distmat = np.power(distance.squareform(distance.pdist(self.X)), 2)
        distmat[np.diag_indices(n_obs)] = np.inf

        adjs_i = list()
        adjs_j = list()
        for i in range(n_obs):
            for j in range(n_obs):
                minimum = np.min(distmat[i, :] + distmat[j, :])
                if distmat[i, j] <= minimum:
                    adjs_i.extend([i, j])
                    adjs_j.extend([j, i])

        n_adjs = len(adjs_i)
        adj_mat = coo_matrix((np.ones(n_adjs), (adjs_i, adjs_j)), shape=(n_obs, n_obs))

        self.adj_mat_ = adj_mat
        return adj_mat

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


if __name__ == '__main__':
    from datasets import get_linear

    data, target = get_linear(n_obs=100)
    gg = GabrielGraph(X=data)
    gg.adjacency()

    gg.plot()
