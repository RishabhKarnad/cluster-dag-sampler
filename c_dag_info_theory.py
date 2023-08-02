from copy import deepcopy

import numpy as np
import scipy.stats as stats


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


def get_neighbours(G_C):
    C, E_C = G_C
    k = len(C)
    G = E_C
    # G_clos = G.T

    neighbours = []

    # Add neighbours with one edge removed
    for i in range(k):
        for j in range(i+1, k):
            if G[i, j] == 1:
                G_n = G.copy()
                G_n[i, j] = 0
                neighbours.append((deepcopy(C), G_n))

    # Add neighbours with one edge added
    for i in range(k):
        for j in range(i+1, k):
            if G[i, j] == 0:
                G_n = G.copy()
                G_n[i, j] = 1
                neighbours.append((deepcopy(C), G_n))


def greedy_search(G_C, dataset, score):
    # G_C_prime = G_C
    # score = score(G_C_prime, dataset)
    # neighbours = neighbourhood(G_C_prime)
    # for G_C_neighbour in neighbours:
    #     score_neighbour = score(G_C_neighbour, dataset)
    #     if score_neighbour > score:
    #         G_C_prime = G_C_neighbour
    #         score = score_neighbour
    # return G_C_prime
    neighbours = get_neighbours(G_C)
    scores = [score(G_n) for G_n in neighbours]
    return neighbours[np.argmax(scores)]


def find_cdag(data, *, score, steps=100, rho=0.5):
    m, n = data.shape
    C_star = sample_clustering(n)
    G_star = sample_cdag(C_star, rho)
    score_G = score(G_star)

    for _ in range(steps):
        C = sample_clustering(n)
        G_C = sample_cdag(C, rho)
        G_prime = greedy_search(G_C)
        score_new = score(G_prime)
        if score_new > score_G:
            G_star = G_prime
            score_G = score_new

    return G_prime


def main():
    from scoreCIC import ScoreCIC
    from models.bernoulli import MultivariateBernoulliDistribution
    from data import generate_data_discrete

    data = generate_data_discrete(n_samples=100)
    score_CIC = ScoreCIC(
        data=data, dist=MultivariateBernoulliDistribution)

    G_C = find_cdag(data, score=score_CIC)
    print(G_C)


if __name__ == '__main__':
    main()
