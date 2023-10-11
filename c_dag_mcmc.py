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


MAX_PARENTS = 2

N_WARMUP = 100
N_SAMPLES = 500


class CDAGSampler:
    def __init__(self, *, data, dist, penalize_complexity=True):
        m, n = data.shape

        self.n_nodes = n
        K_init = self.make_random_partitioning()
        G_init = UpperTriangular(len(K_init)).sample()
        self.samples = [(K_init, G_init)]
        self.U = stats.uniform(0, 1)

        self.data = data
        self.score = ScoreCIC(data, dist=dist)
        self.penalize_complexity = penalize_complexity
        self.scores = []

        self.ctx = {}

    def _reset(self):
        K_init = self.make_random_partitioning()
        G_init = UpperTriangular(len(K_init)).sample()
        self.samples = [(K_init, G_init)]
        self.scores = []

    def make_random_partitioning(self):
        variables = range(self.n_nodes)
        variables = np.random.permutation(variables)
        partitioning = np.array_split(variables, self.n_nodes // 2)
        partitioning = [set(K_i) for K_i in partitioning]
        return partitioning

    def sample(self, n_samples=5000, n_warmup=1000):
        for i in tqdm(range(n_warmup)):
            K_t, G_t = self.step()
            self.samples.append((K_t, G_t))

        for i in tqdm(range(n_samples)):
            K_t, G_t = self.step()
            self.samples.append((K_t, G_t))
            self.scores.append(
                (self.cluster_score(K_t), self.score((K_t, G_t))))

    def make_graph_dist(self, scores):
        support = len(scores)
        params = np.array(scores)
        params -= min(params.min(), 0)
        params /= params.sum()
        return stats.rv_discrete(
            name='categorical', a=0, b=(support), inc=1, values=(np.arange(support), params))

    def step(self):
        alpha = self.U.rvs()

        if alpha < 0.01:
            return self.samples[-1]
        else:
            K_star = self.sample_K()
            u = self.U.rvs()
            a = self.prob_accept(K_star)
            if u < a:
                graph_index = self.make_graph_dist(
                    self.ctx['graph_scores']).rvs()
                graph = self.ctx['graphs'][graph_index]
                return K_star, graph
            else:
                graph_index = self.make_graph_dist(
                    self.ctx['prev_graph_scores']).rvs()
                graph = self.ctx['prev_graphs'][graph_index]
                return self.samples[-1][0], graph

    def sample_K(self):
        K_prev, _ = self.samples[-1]
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
        for i_lim in range(len(K)):
            neighbours_lim = self.count_neighbours(K, upto=i_lim)
            if j <= neighbours_lim:
                return i_lim

    def find_c_star(self, K, j, i_star):
        k_i_star = len(K[i_star])

        n_neighbours_i_star = self.count_neighbours(K, upto=i_star-1)

        for c_lim in range(1, len(K[i_star])+1):
            n_splits_c_lim = np.sum([comb(k_i_star, c)
                                    for c in range(1, c_lim)])
            if j <= n_neighbours_i_star + n_splits_c_lim:
                return c_lim

    def count_neighbours(self, K, upto=None):
        m = len(K)

        n_merges = m - 1

        n_splits = 0

        lim = m or (upto + 1)
        for i in range(lim):
            k_i = len(K[i])
            for c in range(1, k_i):
                n_splits += comb(k_i, c)

        return n_merges + n_splits

    def prob_accept(self, K_star):
        K_prev, _ = self.samples[-1]

        nbd_K_star = self.count_neighbours(K_star)
        nbd_K_prev = self.count_neighbours(K_prev)

        prob_K_star, graphs, graph_scores = self.cluster_score(K_star)
        prob_K_prev, prev_graphs, prev_graph_scores = self.cluster_score(
            K_prev)

        self.ctx['graphs'] = graphs
        self.ctx['prev_graphs'] = prev_graphs
        self.ctx['graph_scores'] = graph_scores
        self.ctx['prev_graph_scores'] = prev_graph_scores

        rho = (nbd_K_prev * prob_K_star) / (nbd_K_star * prob_K_prev)

        return min(1.0, rho)

    def cluster_score(self, K):
        score = 0
        graphs = []
        graph_scores = []
        for E in self.sample_graphs(K):
            V = [sorted(K_i) for K_i in K]
            G_C = (V, E)
            graph_score = self.graph_score(G_C)
            score += graph_score
            graphs.append(E)
            graph_scores.append(graph_score)
        return score, graphs, graph_scores

    def sample_graphs(self, K, *, n_samples=None):
        if n_samples is None:
            # return UpperTriangular(len(K)).all_graphs()
            return UpperTriangular(len(K)).all_k_parent_graphs(k=MAX_PARENTS)
        return UpperTriangular(len(K)).sample_n(n_samples)

    def graph_score(self, G_C):
        return self.score(G_C, penalize_complexity=self.penalize_complexity)


def test():
    # data = generate_data_continuous(n_samples=100, n_dims=5)
    # dist = GaussianDistribution

    # data = generate_data_discrete(n_samples=100)
    data = generate_data_discrete_v2(n_samples=100)
    # data = generate_data_discrete_v2()
    dist = MultivariateBernoulliDistribution

    sampler = CDAGSampler(data=data, dist=dist)
    sampler.count_neighbours([{1}, {2, 3}, {4, 5}])
    sampler.sample(n_warmup=N_WARMUP, n_samples=N_SAMPLES)
    for i, sample in enumerate(sampler.samples[N_WARMUP:-1]):
        print(f'[Sample {i}]')
        C, G = sample
        score_C, score_CIC = sampler.scores[i]
        print(f'    C: {C}')
        print(f'    G: {G}')
        print(f'    cluster score: {score_C}')
        print(f'    graph_score: {score_CIC}')
    print('=========================')

    g_true = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    # g_true = np.array([[0, 0, 0, 0, 0, 0, 1, 0],
    #                    [0, 0, 0, 0, 0, 0, 1, 0],
    #                    [0, 0, 0, 0, 0, 0, 1, 0],
    #                    [0, 0, 0, 0, 0, 0, 1, 1],
    #                    [0, 0, 0, 0, 0, 0, 1, 1],
    #                    [0, 0, 0, 0, 0, 0, 1, 1],
    #                    [0, 0, 0, 0, 0, 0, 0, 1],
    #                    [0, 0, 0, 0, 0, 0, 0, 0]])
    ecshd = expected_cluster_shd(g_true, sampler.samples[-20:-1])
    print(f'E-CSHD: {ecshd}')


if __name__ == '__main__':
    from data import generate_data_continuous, generate_data_discrete, generate_data_discrete_v2
    from models.gaussian import GaussianDistribution
    from models.bernoulli import MultivariateBernoulliDistribution
    from metrics import expected_cluster_shd

    test()
