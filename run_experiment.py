import numpy as np
import jax.random as random
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import os
import logging

from cdag_mcmc_sampler import CDAGSampler
from data.loader import DataGen
from models.gaussian import GaussianDistribution
from scores.cic_score import ScoreCIC
from scores.bayesian_cdag_score import BayesianCDAGScore
from models.cluster_linear_gaussian_network import ClusterLinearGaussianNetwork
from utils.metrics import nll, expected_nll, mse_theta, expected_mse_theta, rand_index, expected_rand_index, mutual_information_score, expected_mutual_information_score, shd_expanded_graph, expected_shd, shd_expanded_mixed_graph, expected_cpdag_shd
from utils.c_dag import clustering_to_matrix, get_graphs_by_count
from utils.visualization import visualize_graphs, plot_graph_scores
from utils.sys import initialize_logger

from rng import random_state


def parse_args():
    parser = ArgumentParser(prog='C-DAG MCMC runner',
                            description='Run MCMC sampler for C-DAGs with arguments')

    parser.add_argument('--random_seed', type=int, default=42)

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


def evaluate_samples(*, C_true, G_true, samples, scores, theta, theta_true, data, filepath):
    C_samples = list(
        map(lambda x: clustering_to_matrix(x[0]), samples))

    C_true = clustering_to_matrix(C_true)

    # G_expanded = C_true@G_true@C_true.T

    rand_index_mean, rand_index_stddev = expected_rand_index(C_true, C_samples)
    logging.info(f'Rand-index: {rand_index_mean}+-{rand_index_stddev}')

    mi_mean, mi_stddev = expected_mutual_information_score(C_true, C_samples)
    logging.info(f'Cluster MI: {mi_mean}+-{mi_stddev}')

    shd_mean, shd_stddev = expected_shd((C_true, G_true), samples)
    logging.info(f'E-SHD: {shd_mean}+-{shd_stddev}')

    shd_cpdag_mean, shd_cpdag_stddev = expected_cpdag_shd(
        (C_true, G_true), samples)
    logging.info(f'CPDAG E-SHD: {shd_cpdag_mean}+-{shd_cpdag_stddev}')

    nll_mean, nll_stddev = expected_nll(data, samples, theta)
    logging.info(f'NLL: {nll_mean}+-{nll_stddev}')

    mse_theta_mean, mse_theta_stddev = expected_mse_theta(
        samples, theta, theta_true)
    logging.info(f'MSE (Theta): {mse_theta_mean}+-{mse_theta_stddev}')

    # Compute metrics for the CDAG with best score
    logging.info('Best DAG metrics:')
    best_cdag = samples[np.argmax(scores)]

    C_best, G_best = best_cdag
    m, n = data.shape

    C_best_mat = clustering_to_matrix(C_best)

    rand_index_best = rand_index(C_true, C_best_mat)
    logging.info(f'\tRand-index: {rand_index_best}')

    mi_best = mutual_information_score(C_true, C_best_mat)
    logging.info(f'\tCluster MI: {mi_best}')

    shd_best = shd_expanded_graph((C_true, G_true), best_cdag)
    logging.info(f'\tSHD: {shd_best}')

    shd_cpdag_best = shd_expanded_mixed_graph((C_true, G_true), best_cdag)
    logging.info(f'\tCPDAG SHD: {shd_cpdag_best}')

    nll_best = nll(data, C_best, G_best, theta)
    logging.info(f'\tNLL: {nll_best}')

    mse_theta_best = mse_theta(best_cdag, theta, theta_true)
    logging.info(f'\tMSE (Theta): {mse_theta_best}')

    # Compute metrics for most frequently sampled CDAG
    logging.info('Most frequent DAG metrics:')

    graphs, graph_counts = get_graphs_by_count(samples)
    np.save(f'{filepath}/clusterings_by_count.npy',
            list(map(lambda x: clustering_to_matrix(x[0]), graphs)))
    np.save(f'{filepath}/graphs_by_count.npy',
            list(map(lambda x: x[1], graphs)))
    np.save(f'{filepath}/cdag_counts.npy', graph_counts)

    mode_dag = graphs[0]

    C_mode, G_mode = mode_dag
    m, n = data.shape

    C_mode_mat = clustering_to_matrix(C_mode)

    rand_index_mode = rand_index(C_true, C_mode_mat)
    logging.info(f'\tRand-index: {rand_index_mode}')

    mi_mode = mutual_information_score(C_true, C_mode_mat)
    logging.info(f'\tCluster MI: {mi_mode}')

    shd_mode = shd_expanded_graph((C_true, G_true), mode_dag)
    logging.info(f'\tSHD: {shd_mode}')

    shd_cpdag_mode = shd_expanded_mixed_graph((C_true, G_true), mode_dag)
    logging.info(f'\tCPDAG SHD: {shd_cpdag_mode}')

    nll_mode = nll(data, C_mode, G_mode, theta)
    logging.info(f'\tNLL: {nll_mode}')

    mse_theta_mode = mse_theta(mode_dag, theta, theta_true)
    logging.info(f'\tMSE (Theta): {mse_theta_mode}')


def train(data, init_params, max_em_iters, n_mcmc_samples, n_mcmc_warmup, min_clusters, max_clusters):
    loss_trace = []
    samples = []
    scores = []
    proposal_G_C = []

    m, n = data.shape
    theta, Cov = init_params['theta'], init_params['Cov']

    initial_cdag_sample = None

    for i in range(max_em_iters):
        print(f'EM iteration {i+1}/{max_em_iters}')

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
        proposal_G_C.append(cdag_sampler.G_C_proposed)

        initial_cdag_sample = samples_i[-1]

        clgn.fit(data,
                 theta=theta,
                 max_mle_iters=args.max_mle_iters,
                 samples=samples_i,
                 scores=scores_i,
                 cb=lambda loss_val: loss_trace.append(loss_val))

        theta = clgn.theta

        samples.append(samples_i)
        scores.append(scores_i)

    return samples, scores, theta, loss_trace, proposal_G_C


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
        logging.info(-ClusterLinearGaussianNetwork(n)
                     .logpmf(data, theta_true, clustering_to_matrix(grouping), group_dag))
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

        cdag_samples, cdag_scores, theta, loss_trace, G_C_proposals = train(data,
                                                                            init_params,
                                                                            args.max_em_iters,
                                                                            args.n_mcmc_samples,
                                                                            args.n_mcmc_warmup,
                                                                            args.min_clusters,
                                                                            args.max_clusters)

        if theta_true is not None:
            evaluate_samples(C_true=grouping,
                             G_true=group_dag,
                             samples=cdag_samples[-1],
                             scores=cdag_scores[-1],
                             theta=theta,
                             theta_true=theta_true,
                             data=data,
                             filepath=args.output_path)

        for i in range(len(cdag_samples)):
            np.save(f'{args.output_path}/em_iter_{i}_clusterings.npy',
                    list(map(lambda G_C: clustering_to_matrix(G_C[0]), cdag_samples[i])))
            np.save(f'{args.output_path}/em_iter_{i}_graphs.npy',
                    list(map(lambda G_C: G_C[1], cdag_samples[i])))
            np.save(f'{args.output_path}/em_iter_{i}_scores.npy',
                    cdag_scores[i])
            np.save(f'{args.output_path}/em_iter_{i}_C_proposals.npy',
                    list(map(lambda x: x[0], G_C_proposals[i])))
            np.save(f'{args.output_path}/em_iter_{i}_G_proposals.npy',
                    list(map(lambda x: x[1], G_C_proposals[i])))

        for i, graphs in enumerate(cdag_samples):
            visualize_graphs(graphs, f'{args.output_path}/em_iter_{i}.png')

        if theta_true is not None:
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
