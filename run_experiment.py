import numpy as np
import jax.random as random
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import igraph as ig
from itertools import chain

import os
import logging

from cdag_mcmc_sampler import CDAGSampler
from data.loader import DataGen
from models.gaussian import GaussianDistribution
from scores.cic_score import ScoreCIC
from scores.bayesian_cdag_score import BayesianCDAGScore
from models.cluster_linear_gaussian_network import ClusterLinearGaussianNetwork
from utils.metrics import compute_nlls, compute_mse_theta, compute_mse_theta_all
from utils.c_dag import stringify_cdag, unstringify_cdag, clustering_to_matrix
from utils.sys import initialize_logger

from rng import random_state


def parse_args():
    parser = ArgumentParser(prog='C-DAG MCMC runner',
                            description='Run MCMC sampler for C-DAGs with arguments')

    parser.add_argument('--random_seed', type=int, default=42)

    parser.add_argument('--score', type=str, choices=['CIC', 'Bayesian'])

    parser.add_argument('--dataset', type=str, choices=[
        'full7var',
        '3var',
        '4var',
        '7var',
        'sachs',
        'housing',
    ])

    parser.add_argument('--n_data_samples', type=int)
    parser.add_argument('--n_mcmc_samples', type=int, default=5000)
    parser.add_argument('--n_mcmc_warmup', type=int, default=1000)

    parser.add_argument('--num_chains', type=int, default=1)

    parser.add_argument('--max_em_iters', type=int, default=10)
    parser.add_argument('--max_mle_iters', type=int, default=100)
    parser.add_argument('--min_clusters', type=int, default=None)
    parser.add_argument('--max_clusters', type=int, default=None)

    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()

    return args


def get_graphs_by_count(samples):
    graph_counts = {}
    for graph in samples:
        graph_string = stringify_cdag(graph)
        if graph_string not in graph_counts:
            graph_counts[graph_string] = 0
        graph_counts[graph_string] += 1
    graphs = sorted(graph_counts, key=graph_counts.get, reverse=True)
    graph_counts = [graph_counts[g] for g in graphs]
    graphs = [unstringify_cdag(g) for g in graphs]
    return graphs, graph_counts


def evaluate_samples(*, samples, scores, theta, theta_true, data, filepath):
    nll_mean, nll_stddev = compute_nlls(data, samples, theta)
    logging.info(f'NLL: {nll_mean}+-{nll_stddev}')

    mse_theta_mean, mse_theta_stddev = compute_mse_theta_all(
        samples, theta, theta_true)
    logging.info(f'MSE (Theta): {mse_theta_mean}+-{mse_theta_stddev}')

    # Compute metrics for the CDAG with best score
    logging.info('Best DAG metrics:')
    best_cdag = samples[np.argmax(scores)]

    C_best, G_best = best_cdag
    m, n = data.shape
    nll_best = -ClusterLinearGaussianNetwork(n).logpmf(
        data, theta, clustering_to_matrix(C_best, len(C_best)), G_best)
    logging.info(f'\tNLL: {nll_best}')

    mse_theta_best = compute_mse_theta(best_cdag, theta, theta_true)
    logging.info(f'\tMSE (Theta): {mse_theta_best}')

    # Compute metrics for most frequently sampled CDAG
    logging.info('Most frequent DAG metrics:')

    graphs, graph_counts = get_graphs_by_count(samples)
    np.save(f'{filepath}/clusterings_by_count.npy',
            list(map(lambda x: clustering_to_matrix(x[0], len(x[0])), graphs)))
    np.save(f'{filepath}/graphs_by_count.npy',
            list(map(lambda x: x[1], graphs)))
    np.save(f'{filepath}/cdag_counts.npy', graph_counts)

    mode_dag = graphs[0]

    C_mode, G_mode = mode_dag
    m, n = data.shape
    nll_mode = -ClusterLinearGaussianNetwork(n).logpmf(
        data, theta, clustering_to_matrix(C_mode, len(C_mode)), G_mode)
    logging.info(f'\tNLL: {nll_mode}')

    mse_theta_mode = compute_mse_theta(mode_dag, theta, theta_true)
    logging.info(f'\tMSE (Theta): {mse_theta_mode}')


def plot_graph(g, target):
    c = g[0]
    graphstring = stringify_cdag(g)
    g = ig.Graph.Adjacency(g[1].tolist())
    ig.plot(g,
            target,
            vertex_size=75,
            vertex_color='grey',
            vertex_label=c,
            layout='circle',
            bbox=(0, 0, 300, 300),
            margin=50)


def visualize_graphs(graphs, filename):
    graphs, graph_counts = get_graphs_by_count(graphs)

    selected_graphs = graphs[:5]

    ncols = 2
    nrows = int(np.ceil(len(selected_graphs) / 2))

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*350*px, nrows*350*px))
    ax = ax.reshape(nrows, ncols)
    i, j = 0, 0
    for ni in range(nrows):
        for nj in range(ncols):
            ax[ni, nj].axis('off')
    for graph in selected_graphs:
        plot_graph(graph, ax[i, j])
        if j+1 == ncols:
            j = 0
            i += 1
        else:
            j += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'{filename}.png')
    plt.clf()


def plot_graph_scores(scores, opt_score, filepath):
    lengths = [len(s) for s in scores]
    scores = list(chain(*scores))
    scores = list(map(lambda x: x.item(), scores))
    plt.plot(scores)
    for i, l in enumerate(lengths):
        plt.axvline(l*(i+1), color='green')
    if opt_score is not None:
        plt.axhline(opt_score, color='red', linestyle=':')
    plt.savefig(f'{filepath}/scores.png')
    plt.clf()


def train(data, init_params, score_type, max_em_iters, n_mcmc_samples, n_mcmc_warmup, min_clusters, max_clusters):
    loss_trace = []
    samples = []
    scores = []
    proposal_C = []
    proposal_G = []

    m, n = data.shape
    theta, Cov = init_params['theta'], init_params['Cov']

    initial_cdag_sample = None

    for i in range(max_em_iters):
        print(f'EM iteration {i+1}/{max_em_iters}')

        if score_type == 'CIC':
            score = ScoreCIC(
                data=data,
                dist=GaussianDistribution,
                parameters={
                    'mean': np.array((data@theta).mean(axis=0)),
                    'cov': Cov,
                })
        elif score_type == 'Bayesian':
            score = BayesianCDAGScore(data=data,
                                      min_clusters=min_clusters,
                                      mean_clusters=max_clusters,
                                      max_clusters=max_clusters)

        cdag_sampler = CDAGSampler(data=data,
                                   score=score,
                                   min_clusters=min_clusters,
                                   max_clusters=max_clusters,
                                   initial_sample=initial_cdag_sample)
        cdag_sampler.set_parameters(theta, None)

        cdag_sampler.sample(n_samples=n_mcmc_samples, n_warmup=n_mcmc_warmup)

        clgn = ClusterLinearGaussianNetwork(n_vars=n)

        samples_i, scores_i = cdag_sampler.get_samples(), cdag_sampler.get_scores()
        proposal_C.append(cdag_sampler.C_proposed)
        proposal_G.append(cdag_sampler.G_proposed)

        initial_cdag_sample = samples_i[-1]

        clgn.fit(data,
                 theta=theta,
                 Cov=Cov,
                 max_mle_iters=args.max_mle_iters,
                 samples=samples_i,
                 scores=scores_i,
                 cb=lambda loss_val: loss_trace.append(loss_val))

        theta = clgn.theta

        samples.append(samples_i)
        scores.append(scores_i)

    return samples, scores, theta, loss_trace, proposal_C, proposal_G


def gen_data(args):
    datagen = DataGen(0.1)
    if args.dataset == '7var':
        return datagen.generate_group_scm_data(
            n_samples=args.n_data_samples, confounded=True)
    elif args.dataset == 'full7var':
        return datagen.generate_scm_data(n_samples=args.n_data_samples)
    elif args.dataset == '3var':
        return datagen.generate_group_scm_data_small_dag(
            n_samples=args.n_data_samples)
    elif args.dataset == '4var':
        return datagen.generate_group_scm_data_small_dag_4vars(
            n_samples=args.n_data_samples)
    elif args.dataset == 'sachs':
        return datagen.get_sachs_data()
    elif args.dataset == 'housing':
        return datagen.load_housing_data()
    else:
        raise RuntimeError('Invalid dataset')


def run(args):
    random_state.set_key(seed=args.random_seed)

    data, scm = gen_data(args)

    (g_true, theta_true, Cov_true, grouping, group_dag) = scm

    m, n = data.shape

    logging.info('GROUND TRUTH')
    logging.info(
        '============================================================')
    logging.info('True graph')
    logging.info(g_true)
    logging.info('Grouping')
    logging.info(grouping)
    logging.info('Group DAG')
    logging.info(group_dag)

    if theta_true is not None:
        logging.info('True theta')
        logging.info(theta_true)
        logging.info('True covariance')
        logging.info(Cov_true)
        logging.info('Ground truth NLL')
        logging.info(-ClusterLinearGaussianNetwork(n).logpmf(data,
                                                             theta_true,
                                                             clustering_to_matrix(
                                                                 grouping, len(grouping)),
                                                             group_dag))
    logging.info(
        '============================================================')

    base_path = args.output_path

    key = random_state.get_key()
    seed = random.randint(key, (20,), 500, 999)

    for i in range(args.num_chains):
        random_state.set_key(seed=seed[i])

        chain_id = i + 1

        args.output_path = f'{base_path}/chain-{chain_id}'
        os.makedirs(f'{args.output_path}', exist_ok=True)

        log_filename = f'{args.output_path}/log.txt'
        logging.basicConfig(filename=log_filename, force=True)
        logging.info(f'CHAIN {chain_id}\n\n')

        subk = random_state.get_key()
        theta = random.normal(subk, (n, n))

        init_params = {
            'theta': theta,
            'Cov': Cov_true,
        }

        cdag_samples, cdag_scores, theta, loss_trace, C_proposals, G_proposals = train(data,
                                                                                       init_params,
                                                                                       args.score,
                                                                                       args.max_em_iters,
                                                                                       args.n_mcmc_samples,
                                                                                       args.n_mcmc_warmup,
                                                                                       args.min_clusters,
                                                                                       args.max_clusters)

        if theta_true is not None:
            evaluate_samples(samples=cdag_samples[-1],
                             scores=cdag_scores[-1],
                             g_true=g_true,
                             theta=theta,
                             theta_true=theta_true,
                             Cov=Cov_true,
                             data=data,
                             filepath=args.output_path)

        for i in range(len(cdag_samples)):
            np.save(f'{args.output_path}/em_iter_{i}_clusterings.npy',
                    list(map(lambda G_C: clustering_to_matrix(G_C[0], len(G_C[0])), cdag_samples[i])))
            np.save(f'{args.output_path}/em_iter_{i}_graphs.npy',
                    list(map(lambda G_C: G_C[1], cdag_samples[i])))
            np.save(f'{args.output_path}/em_iter_{i}_scores.npy',
                    cdag_scores[i])
            np.save(f'{args.output_path}/em_iter_{i}_C_proposals.npy',
                    C_proposals[i])
            np.save(f'{args.output_path}/em_iter_{i}_G_proposals.npy',
                    G_proposals[i])

        for i, graphs in enumerate(cdag_samples):
            visualize_graphs(graphs, f'{args.output_path}/em_iter_{i}.png')

        if theta_true is not None:
            if args.score == 'CIC':
                score = ScoreCIC(
                    data=data,
                    dist=GaussianDistribution,
                    parameters={
                        'mean': np.array((data@theta_true).mean(axis=0)),
                        'cov': Cov_true,
                    })
            elif args.score == 'Bayesian':
                score = BayesianCDAGScore(data=data,
                                          min_clusters=args.min_clusters,
                                          mean_clusters=args.max_clusters,
                                          max_clusters=args.max_clusters)

            opt_cdag = (grouping, group_dag)
            opt_score = score(opt_cdag, theta=theta_true*g_true)
        else:
            opt_score = None

        plot_graph_scores(cdag_scores, opt_score, args.output_path)

        np.save(f'{args.output_path}/theta_estimated.npy', theta)

        np.save(f'{args.output_path}/loss_trace.npy', loss_trace)

        plt.plot(loss_trace)
        plt.savefig(f'{args.output_path}/loss_trace.png')


if __name__ == '__main__':
    args = parse_args()

    initialize_logger(output_path=args.output_path)

    run(args)
