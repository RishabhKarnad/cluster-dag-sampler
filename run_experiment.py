import numpy as np
import jax.numpy as jnp
import jax.random as random
from argparse import ArgumentParser
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD
import networkx as nx
import matplotlib.pyplot as plt

import os
import logging
import csv

from cdag_mcmc_sampler import CDAGSampler
from data import generate_data_continuous_5, generate_data_discrete_8, generate_data_discrete_4
from models.gaussian import GaussianDistribution
# from models.bernoulli import MultivariateBernoulliDistribution
from scoreCIC import ScoreCIC
from models.cluster_linear_gaussian_network import ClusterLinearGaussianNetwork
from metrics import expected_cluster_shd
from utils.c_dag import stringify_cdag

MCMC_N_WARMUP = 100
MCMC_N_SAMPLES = 500


def make_arg_parser():
    parser = ArgumentParser(prog='C-DAG MCMC runner',
                            description='Run MCMC sampler for C-DAGs with arguments')

    parser.add_argument('--random_seed', type=int, default=42)

    # parser.add_argument(
    #     '--data', choices=['discrete_4', 'discrete_8', 'continuous_5'])
    parser.add_argument('--score', type=str, choices=['CIC', 'Bayesian'])

    parser.add_argument('--n_data_samples', type=int)
    parser.add_argument('--n_mcmc_samples', type=int, default=MCMC_N_SAMPLES)
    parser.add_argument('--n_mcmc_warmup', type=int, default=MCMC_N_WARMUP)

    parser.add_argument('--max_em_iters', type=int, default=10)
    parser.add_argument('--max_mle_iters', type=int, default=100)

    parser.add_argument('--output_path', type=str)

    return parser


def initialize_logger(output_path=None):
    if output_path is None:
        output_path = './results'

    log_filename = f'{output_path}/log.txt'

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(
        filename=log_filename, encoding='utf-8', filemode='w', level=logging.INFO)


def evaluate_samples(samples, scores, g_true):
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


def display_sample_statistics(samples):
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


def train(data, init_params, score_type, max_em_iters, n_mcmc_samples, n_mcmc_warmup):
    loss_trace = []

    def record_loss_trace(loss_val):
        loss_trace.append(loss_val)

    theta, Cov = init_params['theta'], init_params['Cov']

    for i in range(max_em_iters):
        print(f'EM iteration {i+1}/{max_em_iters}')

        parameters = {
            'mean': np.array((data@theta).mean(axis=0)),
            'cov': Cov,
        }

        score = score_type(
            data=data, dist=GaussianDistribution, parameters=parameters)

        cdag_sampler = CDAGSampler(data=data, score=score)
        cdag_sampler.sample(n_samples=n_mcmc_samples, n_warmup=n_mcmc_warmup)

        clgn = ClusterLinearGaussianNetwork()

        samples, scores = cdag_sampler.get_samples(), cdag_sampler.get_scores()

        clgn.fit(data,
                 theta=theta,
                 Cov=Cov,
                 max_mle_iters=args.max_mle_iters,
                 samples=samples,
                 scores=scores,
                 cb=record_loss_trace)

        theta = clgn.theta

    return samples, scores, theta, loss_trace


if __name__ == '__main__':
    parser = make_arg_parser()
    args = parser.parse_args()

    key = random.PRNGKey(args.random_seed)

    initialize_logger(output_path=args.output_path)

    # dist = (MultivariateBernoulliDistribution
    #         if args.data in ['discrete_4', 'discrete_8']
    #         else GaussianDistribution)

    n_samples = args.n_data_samples

    # if args.data == 'discrete_4':
    #     data, (g_true,) = generate_data_discrete_4(n_samples=n_samples)
    # elif args.data == 'discrete_8':
    #     data, (g_true,) = generate_data_discrete_8(n_samples=n_samples)
    # elif args.data == 'continuous_5':
    #     data, (g_true, theta_true) = generate_data_continuous_5(
    #         n_samples=n_samples)

    data, (g_true, theta_true) = generate_data_continuous_5(n_samples=n_samples)

    lgbn = nx.from_numpy_array(
        g_true, create_using=LinearGaussianBayesianNetwork)
    G = nx.from_numpy_array(g_true, create_using=nx.MultiDiGraph())
    obs_noise = 0.1
    factors = []
    for node in lgbn.nodes:
        parents = list(G.predecessors(node))
        if parents:
            theta_ = [0.0]
            theta_ += theta_true[jnp.array(parents), node].tolist()
            factor = LinearGaussianCPD(node, theta_, obs_noise, parents)
        else:
            factor = LinearGaussianCPD(node, [0.0], obs_noise, parents)
        factors.append(factor)

    lgbn.add_cpds(*factors)
    joint_dist = lgbn.to_joint_gaussian()
    Cov_true = joint_dist.covariance

    if args.score == 'CIC':
        score_type = ScoreCIC
    # elif args.score == 'Bayesian':
    #     score_type = BayesianCDAGScore

    key_, subk = random.split(key)

    m, n = data.shape
    theta = random.normal(subk, (n, n))

    init_params = {
        'theta': theta,
        'Cov': Cov_true,
    }

    cdag_samples, cdag_scores, theta, loss_trace = train(
        data, init_params, score_type, args.max_em_iters, args.n_mcmc_samples, args.n_mcmc_warmup)

    evaluate_samples(samples=cdag_samples,
                     scores=cdag_scores, g_true=g_true)
    display_sample_statistics(cdag_samples)

    logging.info('True theta')
    logging.info(theta_true)
    logging.info('Estimated theta')
    logging.info(theta)

    with open(f'{args.output_path}/loss_trace.csv', 'w', newline='') as loss_trace_file:
        wr = csv.writer(loss_trace_file, quoting=csv.QUOTE_ALL)
        wr.writerow(loss_trace)

    plt.plot(loss_trace)
    plt.savefig(f'{args.output_path}/loss_trace.png')
