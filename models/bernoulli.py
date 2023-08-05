import numpy as np
import scipy.stats as stats
from scipy.special import rel_entr


class MultivariateBernoulliDistribution:
    def __init__(self, n_vars=1):
        self.n_vars = n_vars
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

        if n != self.n_vars:
            raise RuntimeError(
                f'Dimension of data {n} does not match number of variables {self.n_vars} of distribution')

        for i in range(m):
            self.params[data[i, :]] += 1

        self.params /= self.params.sum()

    def prob(self, sample):
        return self.dist().pdf(idx)

    def sample(self, n_samples):
        return self.dist().rvs(n_samples)

    def marginal(self, *, dims):
        dims_rest = list(set(range(self.n_vars)).difference(set(dims)))
        params = self.params.sum(axis=tuple(dims_rest))
        return self.make_dist(params)

    def prob_marginal(self, sample, *, dims):
        return self.marginal(dims=dims).pdf(sample)

    def sample_marginal(self, n_samples, *, dims):
        return self.marginal(dims=dims).rvs(n_samples)

    def prob_condtional(self, sample, *, dims, dims_conditioned):
        raise NotImplementedError()

    def sample_condtional(self, *, dims, dims_conditioned):
        raise NotImplementedError()

    def total_correlation(self, dims):
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

        dist_marginal = self.marginal(dims=(K_i+K_j))
        marginal_i, marginal_j = (
            self.marginal(dims=K_i).pk, self.marginal(dims=K_j).pk)
        params = np.multiply.outer(marginal_i, marginal_j)
        dist_indep = self.make_dist(params.squeeze())
        return MultivariateBernoulliDistribution.kl_div(dist_marginal, dist_indep)

    def mutual_information(self, i, j):
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

        return stats.entropy(p_pdf, q_pdf)


def test():
    g1 = MultivariateBernoulliDistribution(
        n_vars=1)
    g2 = MultivariateBernoulliDistribution(
        n_vars=1)
    print(MultivariateBernoulliDistribution.kl_div(g1, g2))

    g1 = MultivariateBernoulliDistribution(
        n_vars=2)
    g2 = MultivariateBernoulliDistribution(
        n_vars=2)
    print(MultivariateBernoulliDistribution.kl_div(g1, g2))

    g1 = MultivariateBernoulliDistribution(
        n_vars=1)
    g2 = MultivariateBernoulliDistribution(
        n_vars=2)
    print(MultivariateBernoulliDistribution.kl_div(g1, g2))


if __name__ == '__main__':
    test()
