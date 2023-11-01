import numpy as np
import scipy.stats as stats


class ClusteringDistribution:
    def __init__(self, n_vars, n_clusters, alpha=100, lambda_s=0.1):
        self.n_vars = n_vars
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.lambda_s = lambda_s

    def pmf(self, C):
        return np.exp(self.logpmf(C))

    def logpmf(self, C):
        return (stats.poisson(np.log(self.n_vars)).logpmf(self.n_clusters)
                + self.lambda_s * np.prod(np.tanh(self.alpha * np.sum(C, axis=0))))

    def sample(self, n_samples=1):
        raise NotImplementedError
