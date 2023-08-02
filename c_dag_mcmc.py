import numpy as np
import scipy.stats as stats
from scipy.special import comb
from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm

# from ER import ErdosReniDAGDistribution
# from BGe import BGe
from scoreCIC import ScoreCIC
from models.upper_triangular import UpperTriangular


class CDAGSampler:
    def __init__(self, *, data, dist, penalize_complexity=True):
        m, n = data.shape

        self.n_nodes = n
        self.samples = [self.make_random_partitioning()]
        self.U = stats.uniform(0, 1)

        self.data = data
        self.score = ScoreCIC(data, dist=dist)
        self.penalize_complexity = penalize_complexity
        self.scores = []

    def _reset(self):
        self.samples = [self.make_random_partitioning()]
        self.scores = []

    def make_random_partitioning(self):
        variables = range(self.n_nodes)
        variables = np.random.permutation(variables)
        partitioning = np.array_split(variables, self.n_nodes // 2)
        partitioning = [set(K_i) for K_i in partitioning]
        return partitioning

    def sample(self, n_samples=5000, n_warmup=1000):
        for i in tqdm(range(n_warmup)):
            K_t = self.step()
            self.samples.append(K_t)

        for i in tqdm(range(n_samples)):
            K_t = self.step()
            self.samples.append(K_t)
            self.scores.append(self.cluster_score(K_t))

    def step(self):
        alpha = self.U.rvs()

        if alpha < 0.01:
            return self.samples[-1]
        else:
            K_star = self.sample_K()
            u = self.U.rvs()
            a = self.prob_accept(K_star)
            if u < a:
                return K_star
            else:
                return self.samples[-1]

    def sample_K(self):
        K_prev = self.samples[-1]
        K = deepcopy(K_prev)

        n_neighbours = self.count_neighbours(K_prev)
        j = stats.randint(0, n_neighbours).rvs()

        if j < len(K_prev) - 1:
            K[j].update(K.pop(j+1))
        else:
            i_star = self.find_i_star(K, j)
            c_star = self.find_c_star(K, j, i_star)
            if c_star != len(K[i_star]):
                c_new = set(np.random.choice(
                    sorted(K[i_star]), c_star, replace=False))
                K[i_star] -= c_new
                K.insert(i_star + 1, c_new)

        return K

    def find_i_star(self, K, j):
        # TODO: optimize
        i_star = 0
        for i_lim in range(len(K)):
            neighbours_lim = self.count_neighbours(K, upto=i_lim)
            if neighbours_lim > j:
                return i_star
            else:
                i_star = i_lim

    def find_c_star(self, K, j, i_star):
        c_star = 0
        k_i_star = len(K[i_star])

        n_neighbours_i_star = self.count_neighbours(K, upto=i_star)

        for c_lim in range(len(K[i_star])):
            n_splits_c_lim = np.sum([comb(k_i_star, c)
                                    for c in range(1, c_lim)])
            if n_neighbours_i_star + n_splits_c_lim > j:
                return c_star + 1  # index + 1 -> count
            else:
                c_star = c_lim

    def count_neighbours(self, K, upto=None):
        m = len(K)

        n_merges = m - 1

        n_splits = 0

        lim = m or upto
        for i in range(lim):
            k_i = len(K[i])
            for c in range(1, k_i):
                n_splits += comb(k_i, c)

        return n_merges + n_splits

    def prob_accept(self, K_star):
        K_prev = self.samples[-1]

        nbd_K_star = self.count_neighbours(K_star)
        nbd_K_prev = self.count_neighbours(K_prev)

        prob_K_star = self.cluster_score(K_star)
        prob_K_prev = self.cluster_score(K_prev)

        rho = (nbd_K_prev * prob_K_star) / (nbd_K_star * prob_K_prev)

        return min(1.0, rho)

    def cluster_score(self, K):
        score = 0
        for E in self.sample_graphs(K):
            V = [sorted(K_i) for K_i in K]
            G_C = (V, E)
            score += self.graph_score(G_C)
        return score

    def sample_graphs(self, K, *, n_samples=None):
        if n_samples is None:
            return UpperTriangular(len(K)).all_graphs()
        return UpperTriangular(len(K)).sample_n(n_samples)

    def graph_score(self, G_C):
        return self.score(G_C, penalize_complexity=self.penalize_complexity)


def test():
    data = generate_data_continuous(n_samples=100, n_dims=5)
    dist = GaussianDistribution

    # data = generate_data_discrete(n_samples=100)
    # dist = MultivariateBernoulliDistribution

    sampler = CDAGSampler(data=data, dist=dist)
    sampler.sample()
    print(sampler.samples[-1])


if __name__ == '__main__':
    from data import generate_data_continuous, generate_data_discrete
    from models.gaussian import GaussianDistribution
    # from models.bernoulli import MultivariateBernoulliDistribution

    test()
