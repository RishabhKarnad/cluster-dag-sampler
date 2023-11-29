import numpy as np
import scipy.stats as stats


def parity(x):
    return np.sum(x, axis=1).reshape(-1, 1) % 2


def majority(x):
    return np.sum(x, axis=1).reshape(-1, 1) > x.size


def logical_and(x, y):
    return x + y == 2


def logical_and_elemwise(x, y):
    return np.vectorize(logical_and)(x, y)


def generate_data_discrete_4(n_samples=1000):
    g_true = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])

    samples = []
    for i in range(n_samples):
        X1 = stats.bernoulli(0.5).rvs()
        X2 = stats.bernoulli(0.5).rvs()
        eps3 = stats.bernoulli(0.3).rvs()
        X3 = (X1 ^ X2) ^ eps3
        eps4 = stats.bernoulli(0.75).rvs()
        X4 = X3 ^ eps4
        samples.append([X1, X2, X3, X4])
    return np.array(samples), (g_true,)


def generate_data_discrete_8(n_samples=1000, f1=parity, f2=parity, f3=parity):
    g_true = np.array([[0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0]])

    X = stats.bernoulli(0.5).rvs((n_samples, 6))
    X_7 = logical_and_elemwise(f1(X[:, 0:3]), f2(X[:, 3:6]))
    X_8 = logical_and_elemwise(X_7, f3(X[:, 3:6]))
    D = np.hstack([X, X_7, X_8])
    return D, (g_true,)
