import numpy as np
import scipy.stats as stats

from models.truncated_poisson import TruncatedPoisson
from models.clustering_distribution import ClusteringDistribution
from models.graph_distribution import SparseDAGDistribution
from models.cluster_linear_gaussian_network import ClusterLinearGaussianNetwork


def to_matrix(C, k):
    n = np.sum([len(C_i) for C_i in C])
    m = np.zeros((n, k))
    for i, C_i in enumerate(C):
        for X_j in C_i:
            m[X_j, i] = 1
    return m


class CDAGJointDistribution:
    def __init__(self,
                 n_vars,
                 theta,
                 Cov,
                 min_clusters,
                 mean_clusters,
                 max_clusters,
                 cluster_prior=ClusteringDistribution,
                 graph_prior=SparseDAGDistribution,
                 likelihood=ClusterLinearGaussianNetwork):
        self.n_vars = n_vars
        self.theta = theta
        self.Cov = Cov
        self.min_clusters = min_clusters
        self.mean_clusters = mean_clusters
        self.max_clusters = max_clusters
        self.cluster_prior = cluster_prior
        self.graph_prior = graph_prior
        self.likelihood = likelihood

    def pmf(self, C, G, X):
        return np.exp(self.logpmf(C, G, X))

    def logpmf(self, C, G, X):
        k = len(C)
        C = to_matrix(C, len(C))
        # return (TruncatedPoisson(self.min_clusters, self.mean_clusters, self.max_clusters).logpmf(k)
        #         + self.cluster_prior(self.n_vars, k).logpmf(C)
        #         + self.graph_prior(self.n_vars).logpmf(G)
        #         + self.likelihood(self.n_vars).logpmf(X, self.theta, self.Cov, C, G))
        return (stats.randint(self.min_clusters, self.max_clusters+1).logpmf(k)
                + self.cluster_prior(self.n_vars, k).logpmf(C)
                + self.graph_prior(self.n_vars).logpmf(G)
                + self.likelihood(self.n_vars).logpmf(X, self.theta, self.Cov, C, G))

    def sample(self):
        raise NotImplementedError
