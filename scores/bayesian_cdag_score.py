from models.cdag_distribution import CDAGJointDistribution


class BayesianCDAGScore:
    def __init__(self, data, min_clusters, mean_clusters, max_clusters):
        m, n_vars = data.shape
        self.data = data
        self.dist = CDAGJointDistribution(n_vars,
                                          min_clusters=min_clusters,
                                          mean_clusters=mean_clusters,
                                          max_clusters=max_clusters)

    def __call__(self, G_C, theta):
        return self.dist.logpmf(G_C[0], G_C[1], self.data, theta)
