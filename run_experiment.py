import numpy as np
import jax.random as random
import argparse
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import igraph as ig
from itertools import chain

import os
import logging
import csv

from cdag_mcmc_sampler import CDAGSampler
from data.loader import DataGen
from models.gaussian import GaussianDistribution
from scores.cic_score import ScoreCIC
from scores.bayesian_cdag_score import BayesianCDAGScore
from models.cluster_linear_gaussian_network import ClusterLinearGaussianNetwork
from utils.metrics import expected_cluster_shd, expected_shd, faithfulness_score
from utils.c_dag import stringify_cdag, unstringify_cdag


MCMC_N_WARMUP = 100
MCMC_N_SAMPLES = 500


def make_arg_parser():
    parser = ArgumentParser(prog='C-DAG MCMC runner',
                            description='Run MCMC sampler for C-DAGs with arguments')

    parser.add_argument('--random_seed', type=int, default=42)

    # parser.add_argument(
    #     '--data', choices=['discrete_4', 'discrete_8', 'continuous_5'])
    parser.add_argument('--score', type=str, choices=['CIC', 'Bayesian'])

    parser.add_argument('--group_faithful',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--faithful', action=argparse.BooleanOptionalAction)
    parser.add_argument('--vstruct', action=argparse.BooleanOptionalAction)
    parser.add_argument('--n_data_samples', type=int)
    parser.add_argument('--n_mcmc_samples', type=int, default=MCMC_N_SAMPLES)
    parser.add_argument('--n_mcmc_warmup', type=int, default=MCMC_N_WARMUP)

    parser.add_argument('--max_em_iters', type=int, default=10)
    parser.add_argument('--max_mle_iters', type=int, default=100)
    parser.add_argument('--min_clusters', type=int, default=None)
    parser.add_argument('--max_clusters', type=int, default=None)

    parser.add_argument('--output_path', type=str)

    return parser


def initialize_logger(output_path=None):
    if output_path is None:
        output_path = './results'

    log_filename = f'{output_path}/log.txt'

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(
        filename=log_filename, encoding='utf-8', filemode='w', level=logging.INFO)


def evaluate_samples(samples, scores, g_true, theta):
    for i, sample in enumerate(samples):
        logging.info(f'[Sample {i}]')
        C, G = sample
        score_C, score_CIC = scores[i]
        logging.info(f'    C: {C}')
        logging.info(f'    G: {G}')
        logging.info(f'    cluster score: {score_C}')
        logging.info(f'    graph_score: {score_CIC}')
    logging.info('=========================')

    ecshd = expected_cluster_shd(g_true, samples)
    logging.info(f'E-CSHD: {ecshd}')

    eshd = expected_shd(samples, theta, g_true)
    logging.info(f'E-SHD: {eshd}')

    faithfulness = faithfulness_score(samples, g_true)
    logging.info(f'Faithfulness score: {faithfulness}')


def display_sample_statistics(samples, filepath):
    graph_counts = {}
    for graph in samples:
        graph_string = stringify_cdag(graph)
        if graph_string not in graph_counts:
            graph_counts[graph_string] = 0
        graph_counts[graph_string] += 1
    graphs = sorted(graph_counts, key=graph_counts.get, reverse=True)
    graph_counts = [graph_counts[g] for g in graphs]
    graph_info = list(zip(graphs, graph_counts))
    logging.info(graph_info[:5])


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
    graph_counts = {}
    for graph in graphs:
        graph_string = stringify_cdag(graph)
        if graph_string not in graph_counts:
            graph_counts[graph_string] = 0
        graph_counts[graph_string] += 1
    graphs = sorted(graph_counts, key=graph_counts.get, reverse=True)
    graph_counts = [graph_counts[g] for g in graphs]
    graph_info = list(zip(graphs, graph_counts))
    logging.info(graph_info[:5])
    graphs = [unstringify_cdag(g) for g in graphs]

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
    scores = list(map(lambda x: x[1].item(), scores))
    plt.plot(scores)
    for l in lengths:
        plt.axvline(l, color='green')
    plt.axhline(opt_score, color='red')
    plt.savefig(f'{filepath}/scores.png')
    plt.clf()


def train(data, init_params, score_type, max_em_iters, n_mcmc_samples, n_mcmc_warmup, min_clusters, max_clusters):
    loss_trace = []
    samples = []
    scores = []

    def record_loss_trace(loss_val):
        loss_trace.append(loss_val)

    m, n = data.shape
    theta, Cov = init_params['theta'], init_params['Cov']

    initial_cdag_sample = None

    for i in range(max_em_iters):
        print(f'EM iteration {i+1}/{max_em_iters}')

        parameters = {
            'mean': np.array((data@theta).mean(axis=0)),
            'cov': Cov,
        }

        if score_type == 'CIC':
            score = ScoreCIC(
                data=data, dist=GaussianDistribution, parameters=parameters)
        elif score_type == 'Bayesian':
            score = BayesianCDAGScore(data=data,
                                      theta=theta,
                                      Cov=Cov,
                                      min_clusters=min_clusters,
                                      mean_clusters=max_clusters,
                                      max_clusters=max_clusters)

        cdag_sampler = CDAGSampler(data=data,
                                   score=score,
                                   min_clusters=min_clusters,
                                   max_clusters=max_clusters,
                                   initial_sample=initial_cdag_sample)

        cdag_sampler.sample(n_samples=n_mcmc_samples, n_warmup=n_mcmc_warmup)

        clgn = ClusterLinearGaussianNetwork(n_vars=n)

        samples_i, scores_i = cdag_sampler.get_samples(), cdag_sampler.get_scores()

        initial_cdag_sample = samples_i[-1]

        clgn.fit(data,
                 theta=theta,
                 Cov=Cov,
                 max_mle_iters=args.max_mle_iters,
                 samples=samples_i,
                 scores=scores_i,
                 cb=record_loss_trace)

        theta = clgn.theta

        samples.append(samples_i)
        scores.append(scores_i)

    return samples, scores, theta, loss_trace


def main(args):
    key = random.PRNGKey(args.random_seed)

    n_samples = args.n_data_samples

    if args.group_faithful:
        data, (g_true, theta_true, Cov_true, grouping, group_dag) = DataGen(
            key, 0.1).generate_data_group_faithful(n_samples=n_samples, N=10, k=3, p=0.2)
    else:
        data, (g_true, theta_true, Cov_true, grouping, group_dag) = DataGen(key, 0.1).generate_data_continuous_5(
            n_samples=n_samples, vstruct=args.vstruct, faithful=args.faithful)

    logging.info('GROUND TRUTH')
    logging.info(
        '============================================================')
    logging.info('True graph')
    logging.info(g_true)
    logging.info('Grouping')
    logging.info(grouping)
    logging.info('Group DAG')
    logging.info(group_dag)
    logging.info('True theta')
    logging.info(theta_true)
    logging.info('True covariance')
    logging.info(Cov_true)
    logging.info(
        '============================================================')

    key_, subk = random.split(key)

    m, n = data.shape
    theta = random.normal(subk, (n, n))

    init_params = {
        'theta': theta,
        'Cov': Cov_true,
    }

    cdag_samples, cdag_scores, theta, loss_trace = train(data,
                                                         init_params,
                                                         args.score,
                                                         args.max_em_iters,
                                                         args.n_mcmc_samples,
                                                         args.n_mcmc_warmup,
                                                         args.min_clusters,
                                                         args.max_clusters)

    evaluate_samples(samples=cdag_samples[-1],
                     scores=cdag_scores[-1], g_true=g_true, theta=theta)
    display_sample_statistics(cdag_samples[-1], filepath=args.output_path)

    for i, graphs in enumerate(cdag_samples):
        visualize_graphs(graphs, f'{args.output_path}/iter-{i}.png')

    if args.score == 'CIC':
        parameters = {
            'mean': np.array((data@theta_true).mean(axis=0)),
            'cov': Cov_true,
        }
        score = ScoreCIC(
            data=data, dist=GaussianDistribution, parameters=parameters)
    elif args.score == 'Bayesian':
        score = BayesianCDAGScore(data=data,
                                  theta=theta_true*g_true,
                                  Cov=Cov_true,
                                  min_clusters=args.min_clusters,
                                  mean_clusters=args.max_clusters,
                                  max_clusters=args.max_clusters)

    opt_cdag = (grouping, group_dag)
    opt_score = score(opt_cdag)

    plot_graph_scores(cdag_scores, opt_score, args.output_path)

    logging.info('Estimated theta')
    logging.info(theta)

    with open(f'{args.output_path}/loss_trace.csv', 'w', newline='') as loss_trace_file:
        wr = csv.writer(loss_trace_file)
        loss_trace = list(map(lambda x: [x], loss_trace))
        wr.writerows(loss_trace)

    plt.plot(loss_trace)
    plt.savefig(f'{args.output_path}/loss_trace.png')


if __name__ == '__main__':
    parser = make_arg_parser()
    args = parser.parse_args()

    initialize_logger(output_path=args.output_path)

    main(args)
