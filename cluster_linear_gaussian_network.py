import jax.numpy as jnp
from jax.scipy import stats

from c_dag_mcmc import CDAGSampler
from models.gaussian import GaussianDistribution


class ClusterLinearGaussianNetwork:
    def __init__(self, *, n_posterior_samples, n_warmup_samples):
        self.posterior_sampler_class = CDAGSampler
        self.n_posterior_samples = n_posterior_samples
        self.n_warmup_samples = n_warmup_samples

    def fit(self, data):
        m, n = data.shape

        self.posterior_sampler = self.posterior_sampler_class(
            data=data, dist=GaussianDistribution)

        self.posterior_sampler.sample(
            n_samples=self.n_posterior_samples, n_warmup=self.n_warmup_samples)

        self.latent_samples = self.posterior_sampler.samples[0:self.n_posterior_samples]

        self.theta = jnp.zeros((n, n))

    def log_data_likelihood(self, X, theta_, Cov_, soft_graph, C):
        G_expand = C@soft_graph@C.T
        mean_expected = X@(G_expand*theta_)
        G_cov = C@C.T
        Cov_mask = G_cov*Cov_
        return jnp.sum([stats.multivariate_normal.logpdf(
            X[i, :], mean_expected[i, :], Cov_mask) for i in range(X.shape[0])])
