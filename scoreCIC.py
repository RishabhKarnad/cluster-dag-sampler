import numpy as np
from sympy.functions.combinatorial.numbers import stirling


class ScoreCIC:
    def __init__(self, data, dist):
        m, n = data.shape
        self._dist = dist
        self.dist = dist(n_vars=n)
        self.dist.fit(data)
        self.N = m

    def total_correlation_all(self, G_C):
        C, _ = G_C
        return np.sum([self.dist.total_correlation(list(K_i)) for K_i in C])

    def g2(self, G_C):
        C, E_C = G_C
        g2_score = 0
        for i, K_i in enumerate(C):
            parent_indices = E_C[:, i].nonzero()[0]
            if parent_indices.size > 0:
                parents = np.hstack([list(C[j]) for j in parent_indices])
                g2_score += self.dist.group_mutual_information(
                    list(K_i), parents)

        return g2_score

    def pairwise_MI(self, G_C, K_i, parents):
        mi = 0

        for i in K_i:
            for j in parents:
                mi += self.dist.mutual_information(i, j)

        return mi

    def pairwise_MI_all(self, G_C):
        C, E_C = G_C

        pairwise_MI = 0

        for i, K_i in enumerate(C):
            parent_indices = E_C[:, i].nonzero()[0]
            if parent_indices.size > 0:
                parents = np.hstack([list(C[j]) for j in parent_indices])
                pairwise_MI += self.pairwise_MI(G_C, list(K_i), parents)

        return pairwise_MI

    def penalty(self, G_C):
        C, _ = G_C

        m = len(C)
        log_m = np.log2(m)

        return (np.log2(int(stirling(self.dist.n_vars, len(C))))
                + np.sum([log_m*len(K_i) for K_i in C])
                # + (np.sum([[X_i for X_i in list(K_i)] for K_i in C]))
                )

    def __call__(self, G_C, *, penalize_complexity=True):
        penalty = self.penalty(G_C) if penalize_complexity else 0
        N = self.N
        return (2 * N * (self.total_correlation_all(G_C) + self.g2(G_C) - self.pairwise_MI_all(G_C))
                - (np.log2(N) * penalty / 2))


def test():
    K = [[0, 1, 2], [3], [4]]
    G = np.array([[0, 1, 1],
                  [0, 0, 1],
                  [0, 0, 0]])
    G_C = (K, G)

    data = generate_data_continuous(n_samples=100, n_dims=5)
    score_CIC = ScoreCIC(data=data, dist=GaussianDistribution)

    # data = generate_data_discrete(n_samples=100)
    # score_CIC = ScoreCIC(data=data, dist=MultivariateBernoulliDistribution)

    score = score_CIC(G_C)
    print(f'CIC            : {score}')
    print(f'  TC           : {score_CIC.total_correlation_all(G_C)}')
    print(f'  G2 score     : {score_CIC.g2(G_C)}')
    print(f'  Pairwise MI  : {score_CIC.pairwise_MI_all(G_C)}')
    print(f'  Penalty      : {score_CIC.penalty(G_C)}')

    samples_4_vars = generate_data_discrete_v2()
    # dist = MultivariateBernoulliDistribution(n_vars=4)
    # dist.fit(samples_4_vars)
    score = ScoreCIC(samples_4_vars, dist=MultivariateBernoulliDistribution)
    G_C = ([{0, 1}, {2, 3}], np.array([[0, 1], [0, 0]]))
    print(score(G_C))


if __name__ == '__main__':
    from data import generate_data_continuous, generate_data_discrete, generate_data_discrete_v2
    from models.gaussian import GaussianDistribution
    from models.bernoulli import MultivariateBernoulliDistribution

    test()
