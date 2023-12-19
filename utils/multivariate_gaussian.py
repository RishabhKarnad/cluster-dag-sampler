import jax.numpy as jnp
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD
import networkx as nx


def get_joint_gaussian_covariance(g_true, theta_true):
    lgbn = nx.from_numpy_array(
        g_true, create_using=LinearGaussianBayesianNetwork)
    G = nx.from_numpy_array(g_true, create_using=nx.MultiDiGraph())
    obs_noise = 0.1
    factors = []
    for node in lgbn.nodes:
        parents = list(G.predecessors(node))
        if parents:
            theta_ = [0.0]
            theta_ += theta_true[jnp.array(parents), node].tolist()
            factor = LinearGaussianCPD(node, theta_, obs_noise, parents)
        else:
            factor = LinearGaussianCPD(node, [0.0], obs_noise, parents)
        factors.append(factor)

    lgbn.add_cpds(*factors)
    joint_dist = lgbn.to_joint_gaussian()
    Cov_true = joint_dist.covariance

    return Cov_true
