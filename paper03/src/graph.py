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


class Topology:
    def __init__(self, data: pd.DataFrame, target: pd.Series, adjacency: coo_matrix):
        self.data = data
        self.target = target
        self.adjacency = adjacency

    @staticmethod
    def get_quality(curr_obs: int, adjacency: coo_matrix, target_: pd.Series):
        class_links = target_[np.where((adjacency.getcol(curr_obs) > 0).toarray())[0]]
        return pd.DataFrame(class_links == target_[curr_obs]).astype("int").mean().iloc[0]

    def executor(self, x):
        return self.get_quality(x, self.adjacency, self.target)

    def class_quality(self):
        with Pool() as pool:
            results = pool.map(self.executor, range(self.data.shape[0]))

        class_quality = pd.DataFrame([self.target, pd.Series(results)]).transpose()
        class_quality.columns = ["target", "link_prop"]

        def quality(x: float):
            if x == 0:
                return "isolated"
            elif x == 1:
                return "normal"
            else:
                return "border"

        class_quality["quality"] = class_quality["link_prop"].apply(quality)
        return class_quality
