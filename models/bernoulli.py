import numpy as np
import scipy.stats as stats
from scipy.special import rel_entr
from utils.permutation import make_permutation_matrix

EPS = 1e-10


class MultivariateBernoulliDistribution:
    def __init__(self, n_vars=1, params=None):
        self.n_vars = n_vars

        if params is not None:
            self.params = params
        else:
            self.params = np.empty(tuple(2*np.ones(n_vars, dtype='int')))
            support = self.params.size
            self.params.fill(1/support)

    def make_dist(self, params):
        params = params.flatten()
        support = params.size
        return stats.rv_discrete(
            name='categorical', a=0, b=(support), inc=1, values=(np.arange(support), params))

    def dist(self):
        return self.make_dist(self.params)

    def fit(self, data):
        m, n = data.shape

        self.params = np.zeros(tuple(2*np.ones(self.n_vars, dtype='int')))

        if n != self.n_vars:
            raise RuntimeError(
                f'Dimension of data {n} does not match number of variables {self.n_vars} of distribution')

        for i in range(m):
            self.params[tuple(data[i, :])] += 1

        self.params += EPS

        self.params /= self.params.sum()

    def prob(self, sample):
        return self.dist().pdf(idx)

    def sample(self, n_samples):
        samples_int = self.dist().rvs(size=n_samples)
        support = self.n_vars
        return np.array([list(map(int, list(format(sample, f'0{support}b'))))
                         for sample in samples_int])

    def marginal(self, *, dims):
        dims_rest = list(set(range(self.n_vars)).difference(set(dims)))
        params = self.params.sum(axis=tuple(dims_rest))
        return self.make_dist(params)

    def prob_marginal(self, sample, *, dims):
        return self.marginal(dims=dims).pdf(sample)

    def sample_marginal(self, n_samples, *, dims):
        samples_int = self.marginal(dims=dims).rvs(size=n_samples)
        support = len(dims)
        return np.array([list(map(int, list(format(sample, f'0{support}b'))))
                         for sample in samples_int])

    def prob_condtional(self, sample, *, dims, dims_conditioned):
        raise NotImplementedError()

    def sample_condtional(self, *, dims, dims_conditioned):
        raise NotImplementedError()

    def total_correlation(self, dims):
        dims = sorted(dims)
        dist_marginal = self.marginal(dims=dims)
        marginal_pmfs = [self.marginal(dims=[i]).pk for i in dims]
        params = marginal_pmfs[0]
        for i in range(1, len(marginal_pmfs)):
            params = np.multiply.outer(params, marginal_pmfs[i])
        dist_indep = self.make_dist(params.squeeze())
        return MultivariateBernoulliDistribution.kl_div(dist_marginal, dist_indep)

    def group_mutual_information(self, K_i, K_j):
        if isinstance(K_i, int):
            K_i = [K_i]
        elif isinstance(K_i, np.ndarray):
            K_i = list(K_i)

        if isinstance(K_j, int):
            K_j = [K_j]
        elif isinstance(K_j, np.ndarray):
            K_j = list(K_j)

        K_i = sorted(K_i)
        K_j = sorted(K_j)
        if K_i[0] > K_j[0]:
            K_i, K_j = K_j, K_i
        dims_combined = K_i+K_j
        dist_marginal = self.marginal(dims=(dims_combined))
        marginal_i, marginal_j = (
            self.marginal(dims=K_i).pk, self.marginal(dims=K_j).pk)
        P = make_permutation_matrix(dims_combined)
        permute_order = np.arange(len(dims_combined), dtype=int) @ P.T
        params = (np.multiply.outer(marginal_i, marginal_j)
                  .reshape(np.ones(len(dims_combined), dtype=int)*2)
                  .transpose(tuple(permute_order)))
        dist_indep = self.make_dist(params.squeeze())
        return MultivariateBernoulliDistribution.kl_div(dist_marginal, dist_indep)

    def mutual_information(self, i, j):
        if i > j:
            i, j = j, i
        dist_marginal = self.marginal(dims=[i, j])
        marginal_i, marginal_j = (
            self.marginal(dims=[i]).pk, self.marginal(dims=[j]).pk)
        params = np.multiply.outer(marginal_i, marginal_j)
        dist_indep = self.make_dist(params.squeeze())
        return MultivariateBernoulliDistribution.kl_div(dist_marginal, dist_indep)

    def kl_div(p, q):
        if hasattr(p, 'dist'):
            p_pdf = p.dist().pk
        else:
            p_pdf = p.pk
        if hasattr(q, 'dist'):
            q_pdf = q.dist().pk
        else:
            q_pdf = q.pk

        if p_pdf.size != q_pdf.size:
            raise RuntimeError(
                f'Size of support of distributions do not match: {p_pdf.size} and {q_pdf.size}')

        if (q_pdf == 0).any():
            raise RuntimeError('Divide by zero')

        return np.sum(p_pdf * np.log2(p_pdf / q_pdf))


def test():
    dist1 = MultivariateBernoulliDistribution(n_vars=2, params=np.array([[0.25, 0.25],
                                                                         [0.25, 0.25]]))
    tc = dist1.total_correlation(dims=[0, 1])
    assert (tc == 0.0)

    dist2 = MultivariateBernoulliDistribution(n_vars=2, params=np.array([[0.125, 0.375],
                                                                         [0.125, 0.375]]))
    tc = dist2.total_correlation(dims=[0, 1])
    assert (tc == 0.0)

    dist3 = MultivariateBernoulliDistribution(n_vars=2, params=np.array([[(3/4 * 1/8), (3/4 * 7/8)],
                                                                         [(1/4 * 7/8), (1/4 * 1/8)]]))
    tc1 = dist3.total_correlation(dims=[0, 1])
    tc2 = dist3.total_correlation(dims=[1, 0])
    assert (tc1 == tc2)

    mi1 = dist3.mutual_information(0, 1)
    mi2 = dist3.mutual_information(1, 0)
    assert (mi1 == mi2)

    delta = 1/3
    gamma = 1/7
    p1 = 1/2 * 1/2 * (1-delta) * (1-gamma)
    p2 = 1/2 * 1/2 * (1-delta) * (gamma)
    p3 = 1/2 * 1/2 * (delta) * (gamma)
    p4 = 1/2 * 1/2 * (delta) * (1-gamma)
    dist4 = MultivariateBernoulliDistribution(n_vars=4, params=np.array(
        [p1, p2, p3, p4, p4, p3, p2, p1, p4, p3, p2, p1, p1, p2, p3, p4]).reshape(2, 2, 2, 2))
    assert (dist4.mutual_information(0, 1) == 0)
    assert (dist4.mutual_information(0, 2) == 0)
    assert (dist4.mutual_information(1, 2) == 0)
    assert (dist4.mutual_information(0, 3) == 0)
    assert (dist4.mutual_information(1, 3) == 0)
    assert (
        dist4.total_correlation(dims=[2, 3]) == dist4.mutual_information(2, 3))

    print('All tests pass')

    print(dist4.sample(10))


if __name__ == '__main__':
    test()
