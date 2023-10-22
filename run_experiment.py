import jax.numpy as jnp
from argparse import ArgumentParser, BooleanOptionalAction
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD
import networkx as nx
import matplotlib.pyplot as plt

from c_dag_mcmc import CDAGSampler
from data import generate_data_continuous_5, generate_data_discrete_8, generate_data_discrete_4
from models.gaussian import GaussianDistribution
from models.bernoulli import MultivariateBernoulliDistribution
from cluster_linear_gaussian_network import ClusterLinearGaussianNetwork
from metrics import expected_cluster_shd
from utils.c_dag import stringify_cdag

MCMC_N_WARMUP = 100
MCMC_N_SAMPLES = 500


def make_arg_parser():
    parser = ArgumentParser(prog='C-DAG MCMC runner',
                            description='Run MCMC sampler for C-DAGs with arguments')
    parser.add_argument(
        '--data', choices=['discrete_4', 'discrete_8', 'continuous_5'])

    parser.add_argument('--n_data_samples', type=int)
    parser.add_argument('--n_mcmc_samples', type=int, default=MCMC_N_SAMPLES)
    parser.add_argument('--n_mcmc_warmup', type=int, default=MCMC_N_WARMUP)

    parser.add_argument('--estimate_parameters', action=BooleanOptionalAction)

    return parser


def run_sampler(data, dist, n_samples, n_warmup):
    sampler = CDAGSampler(data=data, dist=dist)
    sampler.sample(n_warmup=n_warmup, n_samples=n_samples)

    return sampler


def evaluate_samples(sampler, g_true):
    for i, sample in enumerate(sampler.get_samples()):
        print(f'[Sample {i}]')
        C, G = sample
        score_C, score_CIC = sampler.scores[i]
        print(f'    C: {C}')
        print(f'    G: {G}')
        print(f'    cluster score: {score_C}')
        print(f'    graph_score: {score_CIC}')
    print('=========================')

    ecshd = expected_cluster_shd(g_true, sampler.get_samples())
    print(f'E-CSHD: {ecshd}')


def display_sample_statistics(sampler):
    graph_counts = {}
    for graph in sampler.get_samples():
        graph_string = stringify_cdag(graph)
        if graph_string not in graph_counts:
            graph_counts[graph_string] = 0
        graph_counts[graph_string] += 1
    graphs = sorted(graph_counts, key=graph_counts.get, reverse=True)
    graph_counts = [graph_counts[g] for g in graphs]
    graph_info = list(zip(graphs, graph_counts))
    print(graph_info[:5])


if __name__ == '__main__':
    parser = make_arg_parser()
    args = parser.parse_args()

    dist = (MultivariateBernoulliDistribution
            if args.data in ['discrete_4', 'discrete_8']
            else GaussianDistribution)

    n_samples = args.n_data_samples

    if args.data == 'discrete_4':
        data, (g_true,) = generate_data_discrete_4(n_samples=n_samples)
    elif args.data == 'discrete_8':
        data, (g_true,) = generate_data_discrete_8(n_samples=n_samples)
    elif args.data == 'continuous_5':
        data, (g_true, theta_true) = generate_data_continuous_5(
            n_samples=n_samples)

    # sampler = run_sampler(
    #     data=data, dist=dist, n_samples=args.n_mcmc_samples, n_warmup=args.n_mcmc_warmup)

    sampler = CDAGSampler(data=data, dist=dist)
    sampler.sample(n_warmup=args.n_mcmc_warmup, n_samples=args.n_mcmc_samples)

    evaluate_samples(sampler=sampler, g_true=g_true)
    display_sample_statistics(sampler)

    if args.estimate_parameters:
        clgn = ClusterLinearGaussianNetwork(cdag_sampler=sampler)

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

        clgn.fit(data, Cov=Cov_true, max_iter=300)
        print('True theta')
        print(theta_true)
        print('Estimated theta')
        print(clgn.theta)

        plt.plot(clgn.loss_trace)
        plt.savefig('./loss_trace.png')
