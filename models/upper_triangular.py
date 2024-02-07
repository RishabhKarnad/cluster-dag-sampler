import numpy as np
from itertools import combinations
import jax.random as random
import scipy.stats as stats

from rng import random_state


class UpperTriangular:
    def __init__(self, n):
        self.n = n
        self.mask = np.zeros((self.n, self.n))
        for i in range(n):
            for j in range(i+1, n):
                self.mask[i, j] = 1

    def sample(self):
        seed = random_state.get_random_number()
        return self.mask * stats.randint.rvs(low=0, high=2, size=(self.n, self.n), random_state=seed)

    def sample_n(self, n_samples=1):
        return np.array([self.sample() for _ in range(n_samples)])

    def get_graph(self, g_ord):
        """
        Enumerates all upper triangular matrices from numbers 0 to 2**k - 1
        where k = (n**2 - n) / 2
        """
        # TODO: Optimize / rethink
        n = self.n
        k = int((n**2 - n) / 2)

        g_ord_bin = bin(g_ord).split('b')[1]
        g_ord_bin_padded = g_ord_bin.zfill(k)
        adj_flattened = [int(c) for c in list(g_ord_bin_padded)]
        g = np.zeros((self.n, self.n))

        k = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                g[i, j] = adj_flattened[k]
                k += 1

        return g

    def all_graphs(self):
        n = self.n
        k = (n**2 - n) / 2
        n_graphs = int(2 ** k)
        return [self.get_graph(i) for i in range(n_graphs)]

    def _check_k_parents(self, k, graph):
        n_parents = graph.sum(axis=0)
        return (n_parents <= k).all()

    def all_k_parent_graphs(self, k):
        all_graphs = self.all_graphs()
        k_parent_graphs = list(
            filter(lambda g: self._check_k_parents(k, g), all_graphs))
        return k_parent_graphs
