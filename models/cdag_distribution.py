import numpy as np
import scipy.stats as stats

from models.truncated_poisson import TruncatedPoisson
from models.clustering_distribution import ClusteringDistribution
from models.graph_distribution import SparseDAGDistribution
from models.cluster_linear_gaussian_network import ClusterLinearGaussianNetwork
from utils.c_dag import clustering_to_matrix


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
        C = clustering_to_matrix(C, len(C))
        p_k = stats.randint(self.min_clusters, self.max_clusters+1).logpmf(k)
        p_c = self.cluster_prior(self.n_vars, k).logpmf(C)
        p_g = self.graph_prior(self.n_vars).logpmf(G)
        p_d = self.likelihood(self.n_vars).logpmf(
            X, self.theta, self.Cov, C, G)

        return p_k + p_c + p_g + p_d

    def sample(self):
        raise NotImplementedError
