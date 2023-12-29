import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy import stats
import optax
from tqdm import tqdm


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
    def __init__(self, n_vars):
        self.n_vars = n_vars

    def fit(self, data, *,
            theta,
            Cov,
            max_mle_iters=100,
            mle_step_size=0.01,
            samples,
            scores,
            cb):
        m, n = data.shape

        params = {'theta': theta}

        ks = jnp.array(list(map(lambda G_C: len(G_C[0]), samples)))

        k_max = jnp.max(ks).item()

        Cs = jnp.array(
            list(map(lambda G_C: (to_matrix(G_C[0], k=k_max)), samples)))

        Gs = jnp.array(
            list(map(lambda G_C: (zero_pad(G_C[1], k=k_max)), samples)))

        optimizer = optax.adam(mle_step_size)
        opt_state = optimizer.init(params)

        def loss_fn(params):
            return self.loss(data, params['theta'], Cov, Cs, Gs)

        for _ in tqdm(range(max_mle_iters), 'Estimating theta'):
            l = self.loss(data, params['theta'], Cov, Cs, Gs)
            cb(l.item())
            grads = jax.grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        self.theta = params['theta']
        self.cdag_samples = samples
        self.cdag_scores = scores

    def loss(self, X, theta, Cov, Cs, Gs):
        return -jnp.mean(jax.vmap(self.logpmf, (None, None, None, 0, 0), 0)(X, theta, Cov, Cs, Gs))

    def logpmf(self, X, theta, Cov, C, G):
        G_expand = C@G@C.T
        mean_expected = X@(G_expand*theta)
        G_cov = C@C.T
        Cov_mask = G_cov*Cov
        return jnp.mean(stats.multivariate_normal.logpdf(X, mean_expected, Cov_mask))

    def pmf(self, X, theta, Cov, C, G):
        return jnp.exp(self.logpmf(X, theta, Cov, C, G))
