import jax.numpy as jnp
import jax.random as random
import jax.scipy.stats as stats


class GaussianDistributionInternal:
    def __init__(self, mean, cov, *, random_seed):
        self.mean = mean
        self.cov = cov
        self.key = random.PRNGKey(random_seed)

    def pdf(self, x):
        return stats.multivariate_normal.pdf(x, self.mean, self.cov)

    def logpdf(self, x):
        return stats.multivariate_normal.logpdf(x, self.mean, self.cov)

    def rvs(self, shape=None):
        self.key, subk = random.split(self.key)
        return random.multivariate_normal(subk, self.mean, self.cov, shape)


class GaussianDistribution:
    def __init__(self, n_vars=1, *, mean=None, cov=None, random_seed=42):
        self.random_seed = random_seed
        self.n_vars = n_vars
        self.mean = mean if mean is not None else jnp.zeros(n_vars)
        self.cov = cov if cov is not None else jnp.eye(n_vars)
        self.dist = GaussianDistributionInternal(
            mean, cov, random_seed=random_seed)

    def fit(self, data):
        m, n = data.shape

        if n != self.n_vars:
            raise RuntimeError(
                f'Dimension of data {n} does not match number of variables {self.n_vars} of distribution')

        self.mean = jnp.mean(data, axis=0)
        self.cov = jnp.cov(data.T)

        self.dist = GaussianDistributionInternal(
            self.mean, self.cov, random_seed=self.random_seed)

    def prob(self, sample):
        return self.dist.pdf(sample)

    def sample(self, n_samples):
        return self.dist.rvs(n_samples)

    def marginal(self, *, dims):
        dims = jnp.array(dims)
        mean = self.mean[dims]
        cov = self.cov[jnp.ix_(dims, dims)]
        return GaussianDistributionInternal(mean, cov, random_seed=self.random_seed)

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
        cov_indep = jnp.diag(jnp.diag(dist_marginal.cov))
        dist_indep = GaussianDistribution(
            n_vars=len(dims), mean=dist_marginal.mean, cov=cov_indep)
        return GaussianDistribution.kl_div(dist_marginal, dist_indep)

    def group_mutual_information(self, K_i, K_j):
        dims_combined = jnp.hstack([K_i, K_j])

        dist_joint = self.marginal(dims=dims_combined)

        dist_k_i = self.marginal(dims=K_i)
        dist_k_j = self.marginal(dims=K_j)

        mean_indep = jnp.hstack(
            [dist_k_i.mean, dist_k_j.mean])
        cov_indep = jnp.diag(
            jnp.hstack([jnp.diag(dist_k_i.cov), jnp.diag(dist_k_j.cov)]))
        dist_indep = GaussianDistribution(
            n_vars=dims_combined.size, mean=mean_indep, cov=cov_indep)

        return GaussianDistribution.kl_div(dist_joint, dist_indep)

    def mutual_information(self, i, j):
        dist_joint = self.marginal(dims=jnp.array([i, j]))

        dist_i = self.marginal(dims=[i])
        dist_j = self.marginal(dims=[j])

        mean_indep = jnp.hstack(
            [dist_i.mean, dist_j.mean])
        cov_indep = jnp.diag(
            jnp.hstack([jnp.diag(dist_i.cov), jnp.diag(dist_j.cov)]))
        dist_indep = GaussianDistribution(
            n_vars=mean_indep.size, mean=mean_indep, cov=cov_indep)

        return GaussianDistribution.kl_div(dist_joint, dist_indep)

    @staticmethod
    def kl_div(p, q):
        if p.mean.size != q.mean.size:
            raise RuntimeError(
                f'Dimension mismatch: dimension of p ({p.n_vars}) does not match that of q ({q.n_vars})')

        q_cov_inv = jnp.linalg.inv(q.cov)

        return ((jnp.log2(jnp.linalg.det(q.cov))
                - jnp.log2(jnp.linalg.det(p.cov))
                - p.mean.size
                + ((p.mean - q.mean).reshape(1, -1) @ q_cov_inv
                    @ (p.mean - q.mean).reshape(-1, 1))[0, 0]
                + jnp.trace(q_cov_inv @ p.cov)
                 ) / 2)


def test():
    g1 = GaussianDistribution(n_vars=1, mean=jnp.array([0]), cov=jnp.eye(1))
    g2 = GaussianDistribution(n_vars=1, mean=jnp.array([0]), cov=jnp.eye(1))
    print(GaussianDistribution.kl_div(g1, g2))

    g1 = GaussianDistribution(n_vars=2, mean=jnp.array([0, 0]), cov=jnp.eye(2))
    g2 = GaussianDistribution(n_vars=2, mean=jnp.array([1, 1]), cov=jnp.eye(2))
    print(GaussianDistribution.kl_div(g1, g2))

    g1 = GaussianDistribution(n_vars=1, mean=jnp.array([0]), cov=jnp.eye(1))
    g2 = GaussianDistribution(n_vars=2, mean=jnp.array([1, 1]), cov=jnp.eye(2))
    print(GaussianDistribution.kl_div(g1, g2))


if __name__ == '__main__':
    test()
