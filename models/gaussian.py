import numpy as np
import scipy.stats as stats


class GaussianDistribution:
    def __init__(self, n_vars=1, *, mean=None, cov=None):
        self.n_vars = n_vars
        self.mean = mean if mean is not None else np.zeros(n_vars)
        self.cov = cov if cov is not None else np.eye(n_vars)
        self.dist = stats.multivariate_normal(mean, cov)

    def fit(self, data):
        m, n = data.shape

        if n != self.n_vars:
            raise RuntimeError(
                f'Dimension of data {n} does not match number of variables {self.n_vars} of distribution')

        self.mean = np.mean(data, axis=0)
        self.cov = np.cov(data.T)

        self.dist = stats.multivariate_normal(self.mean, self.cov)

    def prob(self, sample):
        return self.dist.pdf(sample)

    def sample(self, n_samples):
        return self.dist.rvs(n_samples)

    def marginal(self, *, dims):
        mean = self.mean[dims]
        cov = self.cov[np.ix_(dims, dims)]
        return stats.multivariate_normal(mean, cov)

    def prob_marginal(self, sample, *, dims):
        return self.marginal(dims=dims).pdf(sample)

    def sample_marginal(self, n_samples, *, dims):
        return self.marginal(dims=dims).rvs(n_samples)

    def prob_condtional(self, sample, *, dims, dims_conditioned):
        raise NotImplementedError()

    def sample_condtional(self, *, dims, dims_conditioned):
        raise NotImplementedError()

    def total_correlation(self, dims):
        dist_marginal = self.marginal(dims=dims)
        cov_indep = np.diag(np.diag(dist_marginal.cov))
        dist_indep = GaussianDistribution(
            n_vars=len(dims), mean=dist_marginal.mean, cov=cov_indep)
        return GaussianDistribution.kl_div(dist_marginal, dist_indep)

    def group_mutual_information(self, K_i, K_j):
        dims_combined = np.hstack([K_i, K_j])

        dist_joint = self.marginal(dims=dims_combined)

        dist_k_i = self.marginal(dims=K_i)
        dist_k_j = self.marginal(dims=K_j)

        mean_indep = np.hstack(
            [dist_k_i.mean, dist_k_j.mean])
        cov_indep = np.diag(
            np.hstack([np.diag(dist_k_i.cov), np.diag(dist_k_j.cov)]))
        dist_indep = GaussianDistribution(
            n_vars=dims_combined.size, mean=mean_indep, cov=cov_indep)

        return GaussianDistribution.kl_div(dist_joint, dist_indep)

    def mutual_information(self, i, j):
        dist_joint = self.marginal(dims=np.array([i, j]))

        dist_i = self.marginal(dims=[i])
        dist_j = self.marginal(dims=[j])

        mean_indep = np.hstack(
            [dist_i.mean, dist_j.mean])
        cov_indep = np.diag(
            np.hstack([np.diag(dist_i.cov), np.diag(dist_j.cov)]))
        dist_indep = GaussianDistribution(
            n_vars=mean_indep.size, mean=mean_indep, cov=cov_indep)

        return GaussianDistribution.kl_div(dist_joint, dist_indep)

    @staticmethod
    def kl_div(p, q):
        if p.mean.size != q.mean.size:
            raise RuntimeError(
                f'Dimension mismatch: dimension of p ({p.n_vars}) does not match that of q ({q.n_vars})')

        q_cov_inv = np.linalg.inv(q.cov)

        return ((np.log2(np.linalg.det(q.cov))
                - np.log2(np.linalg.det(p.cov))
                - p.mean.size
                + ((p.mean - q.mean).reshape(1, -1) @ q_cov_inv
                    @ (p.mean - q.mean).reshape(-1, 1))[0, 0]
                + np.trace(q_cov_inv @ p.cov)
                 ) / 2)


def test():
    g1 = GaussianDistribution(n_vars=1, mean=np.array([0]), cov=np.eye(1))
    g2 = GaussianDistribution(n_vars=1, mean=np.array([0]), cov=np.eye(1))
    print(GaussianDistribution.kl_div(g1, g2))

    g1 = GaussianDistribution(n_vars=2, mean=np.array([0, 0]), cov=np.eye(2))
    g2 = GaussianDistribution(n_vars=2, mean=np.array([1, 1]), cov=np.eye(2))
    print(GaussianDistribution.kl_div(g1, g2))

    g1 = GaussianDistribution(n_vars=1, mean=np.array([0]), cov=np.eye(1))
    g2 = GaussianDistribution(n_vars=2, mean=np.array([1, 1]), cov=np.eye(2))
    print(GaussianDistribution.kl_div(g1, g2))


if __name__ == '__main__':
    test()
