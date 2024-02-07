import numpy as np
import scipy.stats as stats
from scipy.special import comb
from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm
import jax.random as random
import jax.numpy as jnp
from functools import reduce

from models.upper_triangular import UpperTriangular

from utils.sys import debugger_is_active
from utils.c_dag import matrix_to_clustering, count_toposorts

from rng import random_state

MAX_PARENTS = 2

N_WARMUP = 100
N_SAMPLES = 500

eps = 1 / 10**256
log_eps = np.log(eps)


class ProposalDistribution:
    def __init__(self, C, min_clusters, max_clusters):
        self.C = C
        self.k = len(C)

        self.neighbours = []
        self.neighbour_counts = []
        self.total_neighbours = 0

        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

        self.populate_neighbours()

    def populate_neighbours(self):
        # For convenience, popped later
        self.neighbours.append(None)
        self.neighbour_counts.append(0)

        # Merges
        if self.max_clusters > self.k:
            for i in range(self.k-1):
                self.neighbours.append(f'merge-{i}')
                last_count = self.neighbour_counts[-1]
                self.neighbour_counts.append(last_count+1)

        # Splits
        if self.min_clusters < self.k:
            for i in range(self.k):
                for c in range(1, len(self.C[i])):
                    self.neighbours.append(f'split-{i}-{c}')
                    last_count = self.neighbour_counts[-1]
                    self.neighbour_counts.append(
                        last_count + comb(len(self.C[i]), c))

        # Reversals
        for i in range(self.k-1):
            self.neighbours.append(f'reverse-{i}')
            last_count = self.neighbour_counts[-1]
            self.neighbour_counts.append(last_count+1)

        # Exchanges
        for i in range(self.k-1):
            for c1 in range(1, len(self.C[i])+1):
                for c2 in range(1, len(self.C[i+1])+1):
                    self.neighbours.append(f'exchange-{i}-{c1}-{c2}')
                    last_count = self.neighbour_counts[-1]
                    self.neighbour_counts.append(
                        last_count + (comb(len(self.C[i]), c1) * comb(len(self.C[i+1]), c2)))

        self.total_neighbours = self.neighbour_counts[-1]

        # Remove the placeholders added in first line of this method
        self.neighbours.pop(0)
        self.neighbour_counts.pop(0)

    def gen_neighbour(self, spec):
        neighbour = deepcopy(self.C)

        match spec.split('-'):
            case ['merge', i]:
                i = int(i)
                neighbour[i].update(neighbour.pop(i+1))
            case ['split', i, c]:
                i, c = int(i), int(c)
                subk = random_state.get_key()
                c_new = set(random.choice(
                    subk, jnp.array(sorted(neighbour[i])), (c,), replace=False).tolist())
                neighbour[i] -= c_new
                neighbour.insert(i + 1, c_new)
            case ['reverse', i]:
                i = int(i)
                neighbour[i], neighbour[i+1] = neighbour[i+1], neighbour[i]
            case ['exchange', i, c1, c2]:
                i, c1, c2 = int(i), int(c1), int(c2)
                subk = random_state.get_key()
                c1_subset = set(random.choice(
                    subk, jnp.array(sorted(neighbour[i])), (c1,), replace=False).tolist())
                subk = random_state.get_key()
                c2_subset = set(random.choice(
                    subk, jnp.array(sorted(neighbour[i+1])), (c2,), replace=False).tolist())
                neighbour[i] -= c1_subset
                neighbour[i+1].update(c1_subset)
                neighbour[i+1] -= c2_subset
                neighbour[i].update(c2_subset)

        return neighbour

    def sample(self):
        seed = random_state.get_random_number()
        j = stats.randint(0, self.total_neighbours).rvs(
            random_state=seed)
        for idx in reversed(range(len(self.neighbour_counts))):
            if j < self.neighbour_counts[idx]:
                return self.gen_neighbour(self.neighbours[idx])
        raise RuntimeError('No neighbours were generated')

    def pdf(self, C_star):
        # Uniform probability over neighbours
        return 1 / self.total_neighbours

    def logpdf(self, C_star):
        # Uniform probability over neighbours
        return -np.log(self.total_neighbours)


class CDAGSampler:
    def __init__(self, *, data, score, min_clusters=None, max_clusters=None, initial_sample=None):
        m, n = data.shape

        self.n_nodes = n

        self.min_clusters = min_clusters or 2
        self.max_clusters = max_clusters or n

        if initial_sample is None:
            K_init = matrix_to_clustering(
                self.make_random_clustering(self.max_clusters))
            G_init = UpperTriangular(len(K_init)).sample()
            initial_sample = (K_init, G_init)

        self.samples = [initial_sample]
        self.U = stats.uniform(0, 1)

        self.data = data
        self.score = score
        self.scores = []

        self.n_samples = N_SAMPLES
        self.n_warmup = N_WARMUP

        self.ctx = {}

        self.theta = None
        self.Cov = None

        self.debug = debugger_is_active()

    def _reset(self, K_init, G_init):
        self.samples = [(K_init, G_init)]
        self.scores = []

    def set_parameters(self, theta, Cov):
        self.theta = theta
        self.Cov = Cov

    def make_random_clustering(self, n_clusters):
        done = False
        while not done:
            C = np.zeros((self.n_nodes, n_clusters))
            for i in range(self.n_nodes):
                subk = random_state.get_key()
                j = random.choice(subk, n_clusters)
                C[i, j] = 1
            if (np.sum(C, axis=0) > 0).all():
                done = True
        return C

    def sample(self, n_samples=N_SAMPLES, n_warmup=N_WARMUP):
        self.n_samples = n_samples
        self.n_warmup = n_warmup

        it = tqdm(range(n_warmup), 'MCMC warmup')
        for i in it:
            K_t, G_t = self.step(
                cb=lambda K: it.set_postfix_str(f'{len(K)} clusters'))
            self.samples.append((K_t, G_t))
            self.scores.append(
                (self.cluster_score(K_t), self.score((K_t, G_t), self.theta, self.Cov)))

        it = tqdm(range(n_samples), 'Sampling with MCMC')
        for i in it:
            K_t, G_t = self.step(
                cb=lambda K: it.set_postfix_str(f'{len(K)} clusters'))
            self.samples.append((K_t, G_t))
            self.scores.append(
                (self.cluster_score(K_t), self.score((K_t, G_t), self.theta, self.Cov)))

    def get_samples(self):
        return self.samples[-(self.n_samples+1):-1]

    def get_scores(self):
        return self.scores[-(self.n_samples+1):-1]

    def make_graph_dist(self, scores):
        support = len(scores)
        params = np.array(scores)
        params -= min(params.min(), 0)
        params /= params.sum()
        return stats.rv_discrete(
            name='categorical', a=0, b=(support), inc=1, values=(np.arange(support), params))

    def step(self, cb=None):
        seed = random_state.get_random_number()
        alpha = self.U.rvs(random_state=seed)

        if alpha < 0.01:
            # Small probability of staying in same state to make Markov Chain ergodic
            return self.samples[-1]
        else:
            K_prev, _ = self.samples[-1]
            K_star = ProposalDistribution(
                K_prev, self.min_clusters, self.max_clusters).sample()

            if cb is not None:
                cb(K_star)

            u = self.U.rvs()
            a = self.log_prob_accept(K_star)
            if np.log(u) < a:
                seed = random_state.get_random_number()
                graph_index = self.make_graph_dist(
                    self.ctx['graph_scores']).rvs(random_state=seed)
                graph = self.ctx['graphs'][graph_index]
                return K_star, graph
            else:
                seed = random_state.get_random_number()
                graph_index = self.make_graph_dist(
                    self.ctx['prev_graph_scores']).rvs(random_state=seed)
                graph = self.ctx['prev_graphs'][graph_index]
                return self.samples[-1][0], graph

    def log_prob_accept(self, K_star):
        K_prev, _ = self.samples[-1]

        log_q_K_prev = ProposalDistribution(
            K_star, self.min_clusters, self.max_clusters).logpdf(K_prev)
        log_q_K_star = ProposalDistribution(
            K_prev, self.min_clusters, self.max_clusters).logpdf(K_star)

        log_prob_K_star, graphs, graph_scores = self.cluster_score(K_star)
        log_prob_K_prev, prev_graphs, prev_graph_scores = self.cluster_score(
            K_prev)

        self.ctx['graphs'] = graphs
        self.ctx['prev_graphs'] = prev_graphs
        self.ctx['graph_scores'] = graph_scores
        self.ctx['prev_graph_scores'] = prev_graph_scores

        rho = ((log_q_K_prev + log_prob_K_star)
               - (log_q_K_star + log_prob_K_prev))

        return min(0, rho)

    def cluster_score(self, K):
        graphs = []
        graph_scores = []
        for E in self.sample_graphs(K):
            V = [sorted(K_i) for K_i in K]
            G_C = (V, E)
            graph_score = self.graph_score(G_C)
            graphs.append(E)
            graph_scores.append(graph_score)
        score = reduce(np.logaddexp, graph_scores)
        return score, graphs, graph_scores

    def sample_graphs(self, K, *, n_samples=None):
        if n_samples is None:
            # return UpperTriangular(len(K)).all_graphs()
            return UpperTriangular(len(K)).all_k_parent_graphs(k=MAX_PARENTS)
        return UpperTriangular(len(K)).sample_n(n_samples)

    def graph_score(self, G_C):
        return self.score(G_C, self.theta, self.Cov) - np.log(count_toposorts(G_C[1]))


def test():
    data, _ = generate_data_continuous_5(n_samples=100)
    dist = GaussianDistribution

    sampler = CDAGSampler(data=data, dist=dist)
    print(sampler.count_neighbours([{1}, {2, 3}, {4, 5}]))
    sampler.sample(n_warmup=N_WARMUP, n_samples=N_SAMPLES)
    for i, sample in enumerate(sampler.samples[N_WARMUP:-1]):
        print(f'[Sample {i}]')
        C, G = sample
        score_C, score_CIC = sampler.scores[i]
        print(f'    C: {C}')
        print(f'    G: {G}')
        print(f'    cluster score: {score_C[0]}')
        print(f'    graph_score: {score_CIC}')
    print('=========================')

    g_true = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])
    ecshd, ecshd_stddev = expected_cluster_shd(g_true, sampler.get_samples())
    print(f'E-CSHD: {ecshd}+-{ecshd_stddev}')


if __name__ == '__main__':
    from data.continuous import generate_data_continuous_5
    from models.gaussian import GaussianDistribution
    from utils.metrics import expected_cluster_shd

    test()
