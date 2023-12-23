from copy import deepcopy

import numpy as np
import scipy.stats as stats
import scipy.linalg as linalg
from tqdm import tqdm


def sample_clustering(n_dims):
    k = stats.randint(1, n_dims+1).rvs()
    C = [set() for _ in range(k)]
    for i in range(n_dims):
        cluster = stats.randint(0, k).rvs()
        C[cluster].add(i)
    C = list(filter(lambda s: len(s) > 0, C))
    return C


def sample_cdag(C, rho=0.5):
    k = len(C)
    E_C = np.zeros((k, k))
    B = stats.bernoulli(rho)
    for i in range(k):
        for j in range(i+1, k):
            E_C[i, j] = B.rvs()
    G_C = (C, E_C)
    return G_C


def h(G):
    n, _ = G.shape
    return linalg.expm(G*G).trace() - n


def get_neighbours(G_C):
    C, E_C = G_C
    k = len(C)
    G = E_C

    neighbours = [G_C]

    # Add neighbours with one edge removed
    for i in range(k):
        for j in range(k):
            if i != j and G[i, j] == 1:
                G_n = G.copy()
                G_n[i, j] = 0
                neighbours.append((deepcopy(C), G_n))

    # Add neighbours with one edge added
    for i in range(k):
        for j in range(k):
            if i != j and G[i, j] == 0:
                G_n = G.copy()
                G_n[i, j] = 1
                neighbours.append((deepcopy(C), G_n))

    # Add neighbours with one edge reversed
    for i in range(k):
        for j in range(k):
            if i != j and G[i, j] == 1:
                G_n = G.copy()
                G_n[i, j] = 0
                G_n[j, i] = 1
                neighbours.append((deepcopy(C), G_n))

    neighbours = list(filter(lambda n: h(n[1]) == 0, neighbours))

    return neighbours


def greedy_search(G_C, dataset, score):
    neighbours = get_neighbours(G_C)
    scores = [score(G_n) for G_n in neighbours]
    if len(scores) == 0:
        return G_C
    return neighbours[np.argmax(scores)]


def find_cdag(data, *, score, steps=100, rho=0.5):
    m, n = data.shape
    C_star = sample_clustering(n)
    G_star = sample_cdag(C_star, rho)
    score_G = score(G_star)
    scores = []

    for i in tqdm(range(steps)):
        C = sample_clustering(n)
        G_C = sample_cdag(C, rho)
        G_prime = greedy_search(G_C, data, score)
        score_new = score(G_prime)
        if score_new > score_G:
            G_star = G_prime
            score_G = score_new
        scores.append(score_G)
        print(f'[Iter {i}]')
        print(f'    C: {G_prime[0]}')
        print(f'    E: {G_prime[1]}')
        print(f'    Score: {score_new}')
        print(f'    Best: {score_G}')
        accept = "Yes" if score_new > score_G else "No"
        print(f'    Accept: {accept}')
        print()

    return G_prime, scores


def main():
    from scores.cic_score import ScoreCIC
    # from models.bernoulli import MultivariateBernoulliDistribution
    from models.gaussian import GaussianDistribution
    from data.continuous import generate_data_continuous_5
    import matplotlib.pyplot as plt

    # data = generate_data_discrete_v2(n_samples=1000)
    # score_CIC = ScoreCIC(
    #     data=data, dist=MultivariateBernoulliDistribution)

    data, (g_true, theta_true) = generate_data_continuous_5(n_samples=1000)
    score_CIC = ScoreCIC(
        data=data, dist=GaussianDistribution)

    G_C, scores = find_cdag(data, score=score_CIC, steps=1000)
    print(G_C)
    plt.plot(scores)
    plt.savefig('./scores.png')


if __name__ == '__main__':
    main()
