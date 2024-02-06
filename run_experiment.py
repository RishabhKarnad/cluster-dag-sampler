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
from utils.metrics import expected_cluster_shd, expected_shd, expected_metrics, faithfulness_score, compute_nlls, cluster_shd, shd_expanded_graph, metrics_expanded_graph
from utils.c_dag import stringify_cdag, unstringify_cdag, clustering_to_matrix
from utils.sys import initialize_logger


datasets = [
    'faithful_vstruct',
    'faithful_novstruct',
    'unfaithful_vstruct',
    'unfaithful_novstruct',
    'group_faithful',
    'group_scm',
    'group_scm_confounder',
    'group_scm_random',
    '3var',
    '4var',
]


def parse_args():
    parser = ArgumentParser(prog='C-DAG MCMC runner',
                            description='Run MCMC sampler for C-DAGs with arguments')

    parser.add_argument('--random_seed', type=int, default=42)

    # parser.add_argument(
    #     '--data', choices=['discrete_4', 'discrete_8', 'continuous_5'])
    parser.add_argument('--score', type=str, choices=['CIC', 'Bayesian'])

    parser.add_argument('--dataset', type=str, choices=datasets)

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


def evaluate_samples(samples, scores, g_true, theta, Cov, data, ground_truth_is_cdag=False):
    for i, sample in enumerate(samples):
        logging.info(f'[Sample {i}]')
        C, G = sample
        score_C, score_CIC = scores[i]
        logging.info(f'    C: {C}')
        logging.info(f'    G: {G}')
        logging.info(f'    cluster score: {score_C}')
        logging.info(f'    graph_score: {score_CIC}')
    logging.info('=========================')

    if ground_truth_is_cdag:
        samples_metric = list(
            map(lambda G_C: (list(range(len(G_C[0]))), G_C[1]), samples))
    else:
        samples_metric = samples

    if not ground_truth_is_cdag:
        ecshd, ecshd_stddev = expected_cluster_shd(g_true, samples_metric)
        logging.info(f'E-CSHD: {ecshd}+-{ecshd_stddev}')

    if ground_truth_is_cdag:
        pass
    else:
        eshd, eshd_stddev = expected_shd(samples_metric, theta, g_true)
        logging.info(f'E-SHD: {eshd}+-{eshd_stddev}')

    eshd_vcn, eshd_stddev_vcn, eprc, erec = expected_metrics(
        samples_metric, theta, g_true)
    logging.info(
        f'E-SHD (VCN): {eshd_vcn}+-{eshd_stddev_vcn}, E-PRC: {eprc}, E-REC: {erec}')

    nll_mean, nll_stddev = compute_nlls(data, samples, theta, Cov)
    logging.info(f'NLL: {nll_mean}+-{nll_stddev}')

    if not ground_truth_is_cdag:
        faithfulness = faithfulness_score(
            samples, g_true, key=random.PRNGKey(123))
        logging.info(f'Faithfulness score: {faithfulness}')

    top_dag = samples[np.argmax(list(map(lambda x: x[0][0], scores)))]

    if ground_truth_is_cdag:
        top_dag_metric = (list(range(len(top_dag[0]))), top_dag[1])
    else:
        top_dag_metric = top_dag

    logging.info('Best DAG metrics:')

    if not ground_truth_is_cdag:
        cshd = cluster_shd(g_true, top_dag_metric)
        logging.info(f'CSHD: {cshd}')

        shd = shd_expanded_graph(top_dag_metric, theta, g_true)
        logging.info(f'SHD: {eshd}')

    shd, prc, rec = metrics_expanded_graph(top_dag_metric, theta, g_true)
    logging.info(f'SHD (VCN): {shd}, PRC: {prc}, REC: {rec}')

    C_top, G_top = top_dag
    m, n = data.shape
    nll_top = -ClusterLinearGaussianNetwork(n).logpmf(
        data, theta, Cov, clustering_to_matrix(C_top, len(C_top)), G_top)
    logging.info(f'NLL: {nll_top}')

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
    graphs = [unstringify_cdag(g) for g in graphs]

    logging.info('Most frequent DAG metrics:')

    mode_dag = graphs[0]

    if ground_truth_is_cdag:
        mode_dag_metric = (list(range(len(mode_dag[0]))), mode_dag[1])
    else:
        mode_dag_metric = mode_dag

    logging.info('Most frequent DAG metrics:')

    if not ground_truth_is_cdag:
        cshd = cluster_shd(g_true, mode_dag_metric)
        logging.info(f'CSHD: {cshd}')

        shd = shd_expanded_graph(mode_dag_metric, theta, g_true)
        logging.info(f'SHD: {eshd}')

    shd, prc, rec = metrics_expanded_graph(mode_dag_metric, theta, g_true)
    logging.info(f'SHD (VCN): {shd}, PRC: {prc}, REC: {rec}')

    C_top, G_top = mode_dag
    m, n = data.shape
    nll_top = -ClusterLinearGaussianNetwork(n).logpmf(
        data, theta, Cov, clustering_to_matrix(C_top, len(C_top)), G_top)
    logging.info(f'NLL: {nll_top}')


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
    for i, l in enumerate(lengths):
        plt.axvline(l*(i+1), color='green')
    plt.axhline(opt_score, color='red', linestyle=':')
    plt.savefig(f'{filepath}/scores.png')
    plt.clf()


def train(data, init_params, score_type, max_em_iters, n_mcmc_samples, n_mcmc_warmup, min_clusters, max_clusters):
    loss_trace = []
    samples = []
    scores = []

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

    return samples, scores, theta, loss_trace


def gen_data(key, args):
    datagen = DataGen(key, 0.1)
    if args.dataset == 'faithful_vstruct':
        dataobj = (datagen.generate_data_continuous_5(
            n_samples=args.n_data_samples, vstruct=True, faithful=True))
        return dataobj, datagen.key
    elif args.dataset == 'faithful_novstruct':
        dataobj = (datagen.generate_data_continuous_5(
            n_samples=args.n_data_samples, vstruct=False, faithful=True))
        return dataobj, datagen.key
    elif args.dataset == 'unfaithful_vstruct':
        dataobj = (datagen.generate_data_continuous_5(
            n_samples=args.n_data_samples, vstruct=True, faithful=False))
        return dataobj, datagen.key
    elif args.dataset == 'unfaithful_novstruct':
        dataobj = (datagen.generate_data_continuous_5(
            n_samples=args.n_data_samples, vstruct=False, faithful=False))
        return dataobj, datagen.key
    elif args.dataset == 'group_faithful':
        dataobj = (datagen.generate_data_group_faithful(
            n_samples=args.n_data_samples, N=10, k=3, p=0.2))
        return dataobj, datagen.key
    elif args.dataset == 'group_scm':
        dataobj = (datagen.generate_group_scm_data(
            n_samples=args.n_data_samples))
        return dataobj, datagen.key
    elif args.dataset == 'group_scm_confounder':
        dataobj = (datagen.generate_group_scm_data(
            n_samples=args.n_data_samples))
        return dataobj, datagen.key
    elif args.dataset == 'group_scm_random':
        dataobj = (datagen.generate_random_group_scm_data(
            n_samples=args.n_data_samples))
        return dataobj, datagen.key
    elif args.dataset == '3var':
        dataobj = (datagen.generate_group_scm_data_small_dag(
            n_samples=args.n_data_samples))
        return dataobj, datagen.key
    elif args.dataset == '4var':
        dataobj = (datagen.generate_group_scm_data_small_dag_4vars(
            n_samples=args.n_data_samples))
        return dataobj, datagen.key
    else:
        raise RuntimeError('Invalid dataset')


def run(args):
    key = random.PRNGKey(args.random_seed)

    (data, scm), key_ = gen_data(key, args)

    if args.dataset in [
        'group_scm',
        'group_scm_confounder',
        'group_scm_random',
        '3var',
        '4var',
    ]:
        (g_true, theta_true, Cov_true, grouping, group_dag) = scm
    else:
        (g_true,) = scm

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
    logging.info('True theta')
    logging.info(theta_true)
    logging.info('True covariance')
    logging.info(Cov_true)
    logging.info('Ground truth NLL')
    logging.info(-ClusterLinearGaussianNetwork(n).logpmf(data,
                 theta_true, Cov_true, clustering_to_matrix(grouping, len(grouping)), group_dag))
    logging.info(
        '============================================================')

    base_path = args.output_path

    for i in range(args.num_chains):
        chain_id = i + 1

        args.output_path = f'{base_path}/chain-{chain_id}'
        os.makedirs(f'{args.output_path}', exist_ok=True)

        log_filename = f'{args.output_path}/log.txt'
        logging.basicConfig(filename=log_filename, force=True)
        logging.info(f'CHAIN {chain_id}\n\n')

        key_, subk = random.split(key_)
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

        ground_truth_is_cdag = args.dataset in [
            # 'group_scm',
            # 'group_scm_confounder',
            'group_scm_random',
        ]

        evaluate_samples(samples=cdag_samples[-1],
                         scores=cdag_scores[-1],
                         g_true=g_true,
                         theta=theta,
                         Cov=Cov_true,
                         data=data,
                         ground_truth_is_cdag=ground_truth_is_cdag)
        display_sample_statistics(cdag_samples[-1], filepath=args.output_path)

        for i, graphs in enumerate(cdag_samples):
            visualize_graphs(graphs, f'{args.output_path}/iter-{i}.png')

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
        opt_score = score(opt_cdag,
                          theta=theta_true*g_true,
                          Cov=Cov_true)

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
    args = parse_args()

    initialize_logger(output_path=args.output_path)

    run(args)
