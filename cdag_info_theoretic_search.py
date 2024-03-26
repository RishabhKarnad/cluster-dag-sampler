from rng import random_state
from utils.c_dag import clustering_to_matrix
from causallearn.utils.cit import CIT
from tqdm import tqdm
import scipy.linalg as linalg
import scipy.stats as stats
import numpy as np
import jax.random as random
from utils.metrics import rand_index, mutual_information_score, shd_expanded_graph, shd_expanded_mixed_graph
from data.loader import DataGen
from models.gaussian import GaussianDistribution
from scores.cic_score import ScoreCIC
from copy import deepcopy
import matplotlib.pyplot as plt
from itertools import combinations
from argparse import ArgumentParser
import os
from datetime import datetime
import logging
import pandas as pd


def sample_clustering(n_dims, n_clus):
    k = n_clus
    done = False
    while not done:
        C = [set() for _ in range(k)]
        for i in range(n_dims):
            seed = random_state.get_random_number()
            cluster = stats.randint(0, k).rvs(random_state=seed)
            C[cluster].add(i)
        C = list(filter(lambda s: len(s) > 0, C))
        if len(C) == n_clus:
            done = True
    return C


def sample_cdag(C, rho=0.5):
    k = len(C)
    E_C = np.zeros((k, k))
    B = stats.bernoulli(rho)
    for i in range(k):
        for j in range(i+1, k):
            seed = random_state.get_random_number()
            E_C[i, j] = B.rvs(random_state=seed)
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


def find_cdag(data, *, n_clus, score, steps=100, rho=0.5, logger, desc):
    m, n = data.shape
    C_star = sample_clustering(n, n_clus)
    G_star = sample_cdag(C_star, rho)
    score_G = score(G_star)
    scores = []

    for i in tqdm(range(steps), desc=desc):
        C = sample_clustering(n, n_clus)
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

    return G_star, score_G, scores


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


def gen_data(dataset, n_data_samples):
    datagen = DataGen(0.1)
    if dataset == '7var':
        return datagen.generate_group_scm_data(
            n_samples=n_data_samples, confounded=True)
    elif dataset == '3var':
        return datagen.generate_group_scm_data_small_dag(
            n_samples=n_data_samples)
    elif dataset == '4var':
        return datagen.generate_group_scm_data_small_dag_4vars(
            n_samples=n_data_samples)
    elif dataset == 'sachs':
        return datagen.get_sachs_data()
    elif dataset == 'housing':
        return datagen.load_housing_data()
    else:
        raise RuntimeError('Invalid dataset')


def main():
    parser = ArgumentParser(prog='Information theoretic greedy C-DAG search',
                            description='Greedy search algorithm based on CIC score')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--n_clusters', type=int, default=2)
    parser.add_argument('--n_data_samples', type=int, default=10000)
    parser.add_argument('--dataset', type=str,
                        choices=['3var', '4var', '7var', 'sachs', 'housing'])
    parser.add_argument('--n_runs', type=int, default=20)
    parser.add_argument('--n_iters', type=int, default=1000)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    filepath = f'./{args.output_path or "results/greedy"}/{datetime.now().isoformat()}'
    os.makedirs(filepath, exist_ok=True)
    logging.basicConfig(
        filename=f'{filepath}/log.txt', encoding='utf-8', filemode='w', level=logging.INFO)

    random_state.set_key(seed=args.random_seed)

    data, scm = gen_data(args.dataset, args.n_data_samples)

    (g_true, theta_true, Cov_true, grouping, group_dag) = scm
    C_true = clustering_to_matrix(grouping, len(grouping))
    G_true = group_dag

    score_CIC = ScoreCIC(
        data=data, dist=GaussianDistribution)

    stats = pd.DataFrame(
        columns=['run', 'rand_index', 'mi', 'shd', 'shd_cpdag', 'score'])

    optimal_score = score_CIC((grouping, group_dag))
    logging.info(f'Optimal graph score {optimal_score}')

    best_score = None
    for i in range(args.n_runs):
        logging.info(f'RUN {i}')

        (C, G_C), graph_score, scores = find_cdag(
            data, n_clus=args.n_clusters, score=score_CIC, steps=args.n_iters, logger=logging, desc=f'Run {i}')
        C_mat = clustering_to_matrix(C, len(C))

        C_dummy = list(map(lambda x: {x}, range(len(C))))

        # shd = shd_expanded_graph((C_dummy, G_C), None, g_true)
        # shd_vcn, prc, rec = metrics_expanded_graph(
        #     (C_dummy, G_C), None, g_true)

        # logging.info(f'SHD: {shd}')
        # logging.info(f'Precision: {prc}')
        # logging.info(f'Recall: {rec}')
        ri = rand_index(C_true, C_mat)
        mi = mutual_information_score(C_true, C_mat)
        shd = shd_expanded_graph((C_true, G_true), (C, G_C))
        shd_cpdag = shd_expanded_mixed_graph((C_true, G_true), (C, G_C))
        logging.info(f'Score: {graph_score}')

        stats = pd.concat([stats, pd.DataFrame([{
            'run': i,
            'rand_index': ri,
            'mi': mi,
            'shd': shd,
            'shd_cpdag': shd_cpdag,
            # 'shd': shd,
            # 'precision': prc,
            # 'recall': rec,
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
