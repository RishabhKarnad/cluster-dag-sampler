import numpy as np
import scipy.stats as stats
import networkx as nx
import igraph as ig
# import jax
import jax.numpy as jnp
# from jax import random
import matplotlib.pyplot as plt
from pgmpy.models import LinearGaussianBayesianNetwork


# jax.default_device = jax.devices('cpu')[0]


def fit_bernoulli(x):
    return stats.bernoulli(np.sum(x) / x.size)


def fit_joint_bernoulli(x):
    freqs = {}
    for v in list(x):
        str_v = ''.join([str(v_i) for v_i in v])
        if str_v not in freqs:
            freqs[str_v] = 0
        freqs[str_v] += 1


def parity(x):
    return np.sum(x, axis=1).reshape(-1, 1) % 2


def majority(x):
    return np.sum(x, axis=1).reshape(-1, 1) > x.size


def logical_and(x, y):
    return x + y == 2


def logical_and_elemwise(x, y):
    return np.vectorize(logical_and)(x, y)


def generate_data_discrete(n_samples=1000, f1=parity, f2=parity, f3=parity):
    X = stats.bernoulli(0.5).rvs((n_samples, 6))
    X_7 = logical_and_elemwise(f1(X[:, 0:3]), f2(X[:, 3:6]))
    X_8 = logical_and_elemwise(X_7, f3(X[:, 3:6]))
    D = np.hstack([X, X_7, X_8])
    return D


def generate_data_discrete_v2(n_samples=1000):
    samples = []
    for i in range(n_samples):
        X1 = stats.bernoulli(0.5).rvs()
        X2 = stats.bernoulli(0.5).rvs()
        eps3 = stats.bernoulli(0.3).rvs()
        X3 = (X1 ^ X2) ^ eps3
        eps4 = stats.bernoulli(0.75).rvs()
        X4 = X3 ^ eps4
        samples.append([X1, X2, X3, X4])
    return np.array(samples)


def generate_data_continuous(*, n_samples=1000, n_dims=3):
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
        xs = stats.norm(0, 1).rvs(n_dims - 3)

        samples.append(np.hstack([[x1, x2, x3], xs]))

    return np.array(samples)


def generate_data_continuous_v2(*, n_samples=1000, n_dims=3):
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
        xs = stats.norm(0, 1).rvs(n_dims - 3)

        samples.append(np.hstack([[x1, x3], xs, [x2]]))

    return np.array(samples)


def generate_data_continuous_v3(n_samples=100, return_scm=False):
    adjacency_matrix = np.zeros((5, 5))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[0, 2] = 1
    adjacency_matrix[1, 2] = 1
    adjacency_matrix[1, 3] = 1
    adjacency_matrix[3, 4] = 1

    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.MultiDiGraph())
    g = ig.Graph.from_networkx(G)
    g.vs['label'] = g.vs['_nx_name']

    obs_noise = 0.1
    N = n_samples  # no. of samples
    nvars = len(g.vs)
    nclus = 3
    # key = random.PRNGKey(135)  # (123)
    # key_, subkey = random.split(key)

    theta = np.random.normal(size=(nvars, nvars)) * \
        100  # jnp.zeros((nvars,nvars)) #
    theta[0, 1] = 1
    theta[0, 2] = 1
    theta[0, 3] = 0
    theta[2, 3] = 0
    # theta=theta[1,2].set(-5)
    theta[1, 2] = -1
    theta[1, 3] = 2
    theta[3, 4] = 3

    # key_, subkey = random.split(key_)
    noise = np.sqrt(obs_noise)*np.random.normal(size=(N, nvars))

    toporder = g.topological_sorting()
    print(toporder)

    X = np.zeros((N, nvars))

    for j in toporder:
        parent_edges = g.incident(j, mode='in')
        parents = list(g.es[e].source for e in parent_edges)

        if parents:
            mean = X[:, np.array(parents)] @ theta[np.array(parents), j]
            X[:, j] = mean + noise[:, j]

        else:
            X[:, j] = noise[:, j]

    if return_scm:
        return X, (adjacency_matrix, theta)

    return X
