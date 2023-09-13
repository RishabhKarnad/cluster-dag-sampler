import numpy as np
import scipy.stats as stats
from scipy.special import comb
from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm

# from ER import ErdosReniDAGDistribution
# from BGe import BGe
from scoreCIC import ScoreCIC
from models.upper_triangular import UpperTriangular

from c_dag_mcmc import CDAGSampler
from data import generate_data_continuous_v3
from models.gaussian import GaussianDistribution
from metrics import expected_cluster_shd
from utils.c_dag import stringify_cdag

MAX_PARENTS = 2

N_WARMUP = 100
N_SAMPLES = 500


data = generate_data_continuous_v3(n_samples=100)
print(data)

dist = GaussianDistribution

sampler = CDAGSampler(data=data, dist=dist)
sampler.sample(n_warmup=N_WARMUP, n_samples=N_SAMPLES)

for i, sample in enumerate(sampler.samples[N_WARMUP:-1]):
    print(f'[Sample {i}]')
    C, G = sample
    score_C, score_CIC = sampler.scores[i]
    print(f'    C: {C}')
    print(f'    G: {G}')
    print(f'    cluster score: {score_C}')
    print(f'    graph_score: {score_CIC}')
print('=========================')

g_true = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0]])

ecshd = expected_cluster_shd(g_true, sampler.samples[-20:-1])
print(f'E-CSHD: {ecshd}')

graph_counts = {}
for graph in sampler.samples:
    graph_string = stringify_cdag(graph)
    if graph_string not in graph_counts:
        graph_counts[graph_string] = 0
    graph_counts[graph_string] += 1
graph_counts = sorted(graph_counts, key=graph_counts.get, reverse=True)
print(graph_counts[:5])
