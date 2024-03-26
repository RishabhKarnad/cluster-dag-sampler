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
from utils.c_dag import matrix_to_clustering, clustering_to_matrix

from rng import random_state

MAX_PARENTS = 2

N_WARMUP = 100
N_SAMPLES = 500


class ClusteringProposalDistribution:
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
                self.neighbour_counts.append(1)

        # Splits
        if self.min_clusters < self.k:
            for i in range(self.k):
                for c in range(1, len(self.C[i])):
                    self.neighbours.append(f'split-{i}-{c}')
                    self.neighbour_counts.append(
                        comb(len(self.C[i]), c))

        # Reversals
        for i in range(self.k-1):
            self.neighbours.append(f'reverse-{i}')
            self.neighbour_counts.append(1)

        # Exchanges
        for i in range(self.k-1):
            for c1 in range(1, len(self.C[i])+1):
                for c2 in range(1, len(self.C[i+1])+1):
                    self.neighbours.append(f'exchange-{i}-{c1}-{c2}')
                    self.neighbour_counts.append(
                        (comb(len(self.C[i]), c1) * comb(len(self.C[i+1]), c2)))

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
        subk = random_state.get_key()
        j = random.choice(
            subk, np.arange(len(self.neighbours)), p=np.array(self.neighbour_counts)/len(self.neighbour_counts))
        return self.gen_neighbour(self.neighbours[j])

    def pdf(self, C_star):
        # Uniform probability over neighbours
        return 1 / self.total_neighbours

    def logpdf(self, C_star):
        # Uniform probability over neighbours
        return -np.log(self.total_neighbours)


class GraphProposalDistribution:
    def __init__(self, G):
        self.G = G
        m, m = G.shape
        self.n_nodes = m

        self.total_neighbours = m*(m-1) / 2

    def sample(self):
        G = deepcopy(self.G)

        subk = random_state.get_key()
        [i, j] = random.choice(subk, np.arange(
            self.n_nodes), (2,), replace=False)

        if i > j:
            i, j = j, i

        G[i, j] = 1 - G[i, j]

        return G

    def pdf(self, G_star):
        # Uniform probability over neighbours
        return 1 / self.total_neighbours

    def logpdf(self, G_star):
        # Uniform probability over neighbours
        return -np.log(self.total_neighbours)


class CDAGProposalDistribution:
    def __init__(self, G_C, min_clusters, max_clusters):
        C, E_C = G_C
        self.q_c = ClusteringProposalDistribution(
            C, min_clusters, max_clusters)
        self.q_e_c = GraphProposalDistribution(E_C)

    def sample(self):
        C_star = self.q_c.sample()
        E_C_star = self.q_e_c.sample()
        return (C_star, E_C_star)

    def pdf(self, G_C):
        C, E_C = G_C
        return self.q_c.pdf(C) * self.q_e_c.pdf(E_C)

    def logpdf(self, G_C):
        C, E_C = G_C
        return self.q_c.logpdf(C) + self.q_e_c.logpdf(E_C)


class CDAGSampler:
    def __init__(self, *, data, score, min_clusters=None, max_clusters=None, initial_sample=None):
        m, n = data.shape

        self.n_nodes = n

        self.min_clusters = min_clusters or 2
        self.max_clusters = max_clusters or n

        if initial_sample is None:
            C_init = matrix_to_clustering(
                self.make_random_clustering(self.max_clusters))
            G_init = UpperTriangular(len(C_init)).sample()
            initial_sample = (C_init, G_init)

        self.samples = [initial_sample]
        self.G_C_proposed = []
        self.U = stats.uniform(0, 1)

        self.data = data
        self.score = score
        self.scores = []

        self.n_samples = N_SAMPLES
        self.n_warmup = N_WARMUP

        self.theta = None

        self.debug = debugger_is_active()

    def _reset(self, C_init, G_init):
        self.samples = [(C_init, G_init)]
        self.scores = []
        self.G_C_proposed = []

    def set_parameters(self, theta, Cov):
        self.theta = theta

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
            C_t, G_t = self.step(
                cb=lambda K: it.set_postfix_str(f'{len(K)} clusters'))
            self.samples.append((C_t, G_t))
            self.scores.append(self.cdag_score((C_t, G_t)))

        it = tqdm(range(n_samples), 'Sampling with MCMC')
        for i in it:
            C_t, G_t = self.step(
                cb=lambda K: it.set_postfix_str(f'{len(K)} clusters'))
            self.samples.append((C_t, G_t))
            self.scores.append(self.cdag_score((C_t, G_t)))

    def get_samples(self):
        return self.samples[-(self.n_samples+1):-1]

    def get_scores(self):
        return self.scores[-(self.n_samples+1):-1]

    def step(self, cb=None):
        seed = random_state.get_random_number()
        alpha = self.U.rvs(random_state=seed)

        if alpha < 0.01:
            # Small probability of staying in same state to make Markov Chain ergodic
            return self.samples[-1]
        else:
            C_prev, E_C_prev = self.samples[-1]

            C_new, E_C_new = C_prev, E_C_prev

            C_star, E_C_star = CDAGProposalDistribution(
                (C_prev, E_C_prev), self.min_clusters, self.max_clusters).sample()
            self.G_C_proposed.append((
                clustering_to_matrix(C_star, k=len(C_star)),
                E_C_star
            ))

            if cb is not None:
                cb((C_star, E_C_star))

            seed = random_state.get_random_number()
            u = self.U.rvs(random_state=seed)
            a = self.log_prob_accept((C_star, E_C_prev))
            if np.log(u) < a:
                C_new = C_star
                E_C_new = E_C_star
            else:
                C_new = C_prev
                E_C_new = E_C_prev

            return C_new, E_C_new

    def log_prob_accept(self, G_C_star):
        C_star, E_C_star = G_C_star

        C_prev, E_C_prev = self.samples[-1]

        log_q_C_prev = ClusteringProposalDistribution(
            C_star, self.min_clusters, self.max_clusters).logpdf(C_prev)
        log_q_C_star = ClusteringProposalDistribution(
            C_prev, self.min_clusters, self.max_clusters).logpdf(C_star)

        log_q_E_C_prev = GraphProposalDistribution(E_C_star).logpdf(E_C_prev)
        log_q_E_C_star = GraphProposalDistribution(E_C_prev).logpdf(E_C_star)

        log_prob_G_C_star = self.cdag_score((C_star, E_C_star))
        log_prob_G_C_prev = self.cdag_score((C_prev, E_C_prev))

        rho = ((log_q_C_prev + log_q_E_C_prev + log_prob_G_C_star)
               - (log_q_C_star + log_q_E_C_star + log_prob_G_C_prev))

        return min(0, rho)

    def cdag_score(self, G_C):
        return self.score(G_C, self.theta)
