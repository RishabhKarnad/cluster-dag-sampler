import numpy as np
import scipy.stats as stats


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
