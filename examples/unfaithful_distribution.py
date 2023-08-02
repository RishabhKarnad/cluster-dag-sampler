import numpy as np
import scipy.stats as stats

import jax
import jax.random as random
import jax.numpy as jnp

from dibs.models.linearGaussian import BGe
from dibs.models.linearGaussian import LinearGaussian
from dibs.models.graph import ErdosReniDAGDistribution
from dibs.target import make_linear_gaussian_equivalent_model

from dibs.inference.svgd import MarginalDiBS

jax.devices()

key = random.PRNGKey(123)
print(f"JAX backend: {jax.default_backend()}")

key, subk = random.split(key)


def generate_data_continuous(n_samples=1000, padding_dims=17):
    a = 10
    b = 2
    c = -20

    samples = []

    for i in range(n_samples):
        n1 = stats.norm(0, 1).rvs()
        n2 = stats.norm(0, 1).rvs()
        n3 = stats.norm(0, 1).rvs()

        x1 = n1
        x2 = a*x1 + n2
        x3 = b*x2 + c*x1 + n3
        xs = stats.norm(0, 1).rvs(padding_dims)

        samples.append(np.hstack([[x1, x2, x3], xs]))

    return jnp.array(samples)


graph_dist = ErdosReniDAGDistribution(n_vars=20, n_edges_per_node=1)
bge = BGe(graph_dist=graph_dist)

data = generate_data_continuous(n_samples=100)
# data = jnp.zeros((100, 3))
# data, model = make_linear_gaussian_equivalent_model(
#     key=subk, n_vars=20, graph_prior_str="er")

print(data)


dibs = MarginalDiBS(x=data, interv_mask=None, inference_model=bge)

key, subk = random.split(key)
gs = dibs.sample(key=subk, n_particles=20, steps=2000,
                 callback_every=200, callback=dibs.visualize_callback(ipython=False, save_path='./results/unfaithful'))
