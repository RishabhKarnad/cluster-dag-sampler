import numpy as np


def h(G):
    m, m = G.shape
    return np.trace(np.linalg.matrix_power((np.eye(m) + G*G), m)) - m


class SparseDAGDistribution:
    def __init__(self, n_vars, gibbs_temp=10., sparsity_factor=0.1):
        self.n_vars = n_vars
        self.gibbs_temp = gibbs_temp
        self.sparsity_factor = sparsity_factor

    def logpmf(self, G):
        dagness = h(G)
        return -self.gibbs_temp*dagness - self.sparsity_factor*G.sum()

    def sample(self, n_samples=1):
        raise NotImplementedError
