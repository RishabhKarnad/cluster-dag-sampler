import numpy as np
from sympy.functions.combinatorial.numbers import stirling


class ScoreCIC:
    def __init__(self, data, dist, parameters=None):
        m, n = data.shape
        self._dist = dist

        if parameters is not None:
            self.dist = dist(n_vars=n, **parameters)
        else:
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

    def __call__(self, G_C):
        penalty = self.penalty(G_C)
        N = self.N
        return (2 * N * (self.total_correlation_all(G_C) + self.g2(G_C) - self.pairwise_MI_all(G_C))
                - (np.log2(N) * penalty / 2))
