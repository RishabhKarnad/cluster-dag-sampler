import jax
import jax.numpy as jnp
from jax.scipy import stats
import optax
from tqdm import tqdm

from utils.c_dag import clustering_to_matrix, get_covariance_for_clustering


def zero_pad(A, k):
    m, n = A.shape
    p_rows, p_cols = k - m, k-n
    if p_cols > 0:
        A = jnp.hstack([A, jnp.zeros((m, p_cols))])
    if p_rows > 0:
        A = jnp.vstack([A, jnp.zeros((p_rows, k))])
    return A


class ClusterLinearGaussianNetwork:
    def __init__(self, n_vars, *, mask_covariance=False):
        self.n_vars = n_vars
        self.mask_covariance = mask_covariance

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
            list(map(lambda G_C: (clustering_to_matrix(G_C[0], k=k_max)), samples)))

        Gs = jnp.array(
            list(map(lambda G_C: (zero_pad(G_C[1], k=k_max)), samples)))

        optimizer = optax.adam(mle_step_size)
        opt_state = optimizer.init(params)

        Covs = jax.vmap(get_covariance_for_clustering, (0), 0)(Cs)

        def loss_fn(params):
            return self.loss(data, params['theta'], Covs, Cs, Gs)

        for _ in tqdm(range(max_mle_iters), 'Estimating theta'):
            l = self.loss(data, params['theta'], Covs, Cs, Gs)
            cb(l.item())
            grads = jax.grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        self.theta = params['theta']
        self.cdag_samples = samples
        self.cdag_scores = scores

    def loss(self, X, theta, Covs, Cs, Gs):
        return -jnp.mean(jax.vmap(self.logpmf, (None, None, 0, 0, 0), 0)(X, theta, Covs, Cs, Gs))

    def logpmf(self, X, theta, Cov_, C, G):
        G_expand = C@G@C.T
        mean_expected = X@(G_expand*theta)
        Cov = get_covariance_for_clustering(C)
        if self.mask_covariance:
            G_cov = C@C.T
            Cov_mask = G_cov*Cov
            return jnp.mean(stats.multivariate_normal.logpdf(X, mean_expected, Cov_mask))
        else:
            return jnp.mean(stats.multivariate_normal.logpdf(X, mean_expected, Cov))

    def pmf(self, X, theta, Cov, C, G):
        return jnp.exp(self.logpmf(X, theta, Cov, C, G))
