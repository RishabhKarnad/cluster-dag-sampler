from copy import deepcopy
import matplotlib.pyplot as plt
from itertools import combinations

import numpy as np
import scipy.stats as stats
import scipy.linalg as linalg
from tqdm import tqdm
from causallearn.utils.cit import CIT

from utils.c_dag import clustering_to_matrix


def sample_clustering(n_dims):
    # k = stats.randint(1, n_dims+1).rvs()
    k = 3
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

    neighbours = []

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
    if len(neighbours) == 0:
        return G_C
    scores = [score(G_n) for G_n in neighbours]
    return neighbours[np.argmax(scores)]


def has_chain_or_fork(G_C, i, j, k):
    return ((G_C[i, k] == 1 and G_C[k, j] == 1) or
            (G_C[k, i] == 1 and G_C[k, j] == 1) or
            (G_C[k, i] == 1 and G_C[j, k] == 1))


def reorient_v_structures(cdag, data):
    C, G_C = cdag

    is_indep = CIT(np.array(data), 'kci')

    for (i, j) in combinations(range(len(C)), 2):
        for k in range(len(C)):
            C_i, C_j, C_k = list(C[i]), list(C[j]), list(C[k])
            if i != k and j != k:
                if (has_chain_or_fork(G_C, i, j, k)) and is_indep(C_i, C_j, C_k) < 0.9:
                    G_C[k, i] = 0
                    G_C[k, j] = 0
                    G_C[i, k] = 1
                    G_C[j, k] = 1

    return (C, G_C)


def find_cdag(data, *, score, steps=100, rho=0.5, logger, desc):
    m, n = data.shape
    C_star = sample_clustering(n)
    G_star = sample_cdag(C_star, rho)
    score_G = score(G_star)
    scores = []

    for i in tqdm(range(steps), desc=desc):
        C = sample_clustering(n)
        G_C = sample_cdag(C, rho)
        G_prime = greedy_search(G_C, data, score)
        score_new = score(G_prime)
        accept = False
        if score_new > score_G:
            G_star = G_prime
            score_G = score_new
            accept = True
        scores.append(score_G)
        logger.info(f'[Iter {i}]')
        logger.info(f'    C: {G_prime[0]}')
        logger.info(f'    E: {G_prime[1]}')
        logger.info(f'    Score: {score_new}')
        logger.info(f'    Best: {score_G}')
        logger.info(f'    Accept: {accept}\n')

    G_star = reorient_v_structures(G_star, data)

    return G_prime, score_G, scores


def visualize_cdag(C, G_C, filepath):
    plt.imshow(clustering_to_matrix(C, len(C)))
    plt.savefig(f'{filepath}/clustering.png')
    plt.clf()
    plt.imshow(G_C)
    plt.savefig(f'{filepath}/graph.png')
    plt.clf()


def plot_scores(scores, filepath):
    plt.plot(scores)
    plt.savefig(f'{filepath}/scores.png')
    plt.clf()


def main():
    import os
    from datetime import datetime
    import logging
    import pandas as pd

    from scores.cic_score import ScoreCIC
    # from models.bernoulli import MultivariateBernoulliDistribution
    from models.gaussian import GaussianDistribution
    from data.loader import DataGen
    import jax.random as random

    from utils.metrics import shd_expanded_graph, faithfulness_score, metrics_expanded_graph

    # data = generate_data_discrete_v2(n_samples=1000)
    # score_CIC = ScoreCIC(
    #     data=data, dist=MultivariateBernoulliDistribution)

    filepath = f'./results/greedy/{datetime.now().isoformat()}'
    os.makedirs(filepath, exist_ok=True)
    logging.basicConfig(
        filename=f'{filepath}/log.txt', encoding='utf-8', filemode='w', level=logging.INFO)

    key = random.PRNGKey(123)

    data, (g_true, theta_true, Cov_true, C_true, G_C_true) = DataGen(
        key=key, obs_noise=0.1).generate_data_continuous_5(n_samples=1000)
    score_CIC = ScoreCIC(
        data=data, dist=GaussianDistribution)

    stats = pd.DataFrame(
        columns=['run', 'shd', 'precision', 'recall', 'is_faithful'])

    best_score = None
    for i in range(10):
        logging.info(f'RUN {i}')

        (C, G_C), graph_score, scores = find_cdag(
            data, score=score_CIC, steps=1000, logger=logging, desc=f'Run {i}')

        shd = shd_expanded_graph((C, G_C), None, g_true)
        is_faithful = faithfulness_score([(C, G_C)], g_true, key=key) == 1
        shd_vcn, prc, rec = metrics_expanded_graph((C, G_C), None, g_true)

        logging.info(f'SHD: {shd}')
        logging.info(f'Precision: {prc}')
        logging.info(f'Recall: {rec}')
        logging.info(f'Is faithful?: {is_faithful}')
        logging.info(f'Score: {graph_score}')

        stats = pd.concat([stats, pd.DataFrame([{
            'run': i,
            'shd': shd,
            'precision': prc,
            'recall': rec,
            'is_faithful': is_faithful,
            'score': graph_score}])], ignore_index=True)

        graphpath = f'{filepath}/run-{i}'
        os.makedirs(graphpath, exist_ok=True)
        visualize_cdag(C, G_C, filepath=graphpath)
        plot_scores(scores, filepath=graphpath)
        logging.info(
            '================================================================')

        if best_score is None or best_score < graph_score:
            best_score = graph_score
            best_graph = (C, G_C)
            best_run = i

    stats.to_csv(f'{filepath}/metrics.csv')

    logging.info(f'Best run {best_run}')
    logging.info(f'Best clustering {best_graph[0]}')
    logging.info(f'Best graph {best_graph[1]}')
    logging.info(f'Score {best_score}')
    logging.info(
        '================================================================')


if __name__ == '__main__':
    main()
