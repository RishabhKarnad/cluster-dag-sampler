import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy import stats
from jax import random
import optax
from tqdm import tqdm

# from c_dag_mcmc import CDAGSampler
# from models.gaussian import GaussianDistribution


def to_matrix(C, k):
    n = np.sum([len(C_i) for C_i in C])
    m = np.zeros((n, k))
    for i, C_i in enumerate(C):
        for X_j in C_i:
            m[X_j, i] = 1
    return m


def zero_pad(A, k):
    m, n = A.shape
    p_rows, p_cols = k - m, k-n
    if p_cols > 0:
        A = jnp.hstack([A, jnp.zeros((m, p_cols))])
    if p_rows > 0:
        A = jnp.vstack([A, jnp.zeros((p_rows, k))])
    return A


class ClusterLinearGaussianNetwork:
    def __init__(self, *, cdag_sampler):
        self.cdag_sampler = cdag_sampler
        self.key = random.PRNGKey(42)
        self.loss_trace = []

    def fit(self, data, *, Cov, max_iter=100, run_sampler=False, n_posterior_samples=1000, n_warmup_samples=500):
        self.loss_trace = []

        m, n = data.shape

        if run_sampler:
            self.cdag_sampler.sample(
                n_samples=n_posterior_samples, n_warmup=n_warmup_samples)

        ks = jnp.array(list(
            map(lambda G_C: len(G_C[0]), self.cdag_sampler.get_samples())))

        k_max = jnp.max(ks).item()

        Cs = jnp.array(list(
            map(lambda G_C: (to_matrix(G_C[0], k=k_max)), self.cdag_sampler.get_samples())))

        Gs = jnp.array(list(
            map(lambda G_C: (zero_pad(G_C[1], k=k_max)), self.cdag_sampler.get_samples())))

        key_, subk = random.split(self.key)
        params = {'theta': random.normal(subk, (n, n))}
        optimizer = optax.adam(0.01)
        opt_state = optimizer.init(params)

        def loss_fn(params): return self.loss(
            data, params['theta'], Cov, Cs, Gs)
        for _ in tqdm(range(max_iter), 'Estimating theta'):
            l = self.loss(data, params['theta'], Cov, Cs, Gs)
            self.loss_trace.append(l)
            grads = jax.grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        self.theta = params['theta']

    def loss(self, X, theta, Cov, Cs, Gs):
        return -jnp.sum(jax.vmap(self.log_data_likelihood, (None, None, None, 0, 0), 0)(X, theta, Cov, Cs, Gs))

    def log_data_likelihood(self, X, theta, Cov, C, G):
        G_expand = C@G@C.T
        mean_expected = X@(G_expand*theta)
        G_cov = C@C.T
        Cov_mask = G_cov*Cov
        return jnp.sum(stats.multivariate_normal.logpdf(X, mean_expected, Cov_mask))
