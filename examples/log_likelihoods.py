from itertools import permutations

import numpy as np
from tqdm import tqdm

from data.continuous import generate_data_continuous_5
from models.cluster_linear_gaussian_network import ClusterLinearGaussianNetwork
from utils.c_dag import clustering_to_matrix


def all_partitionings(l):
    partitionings = []
    for i in range(1, len(l)-1):
        for j in range(i+1, len(l)):
            partitionings.append([l[:i], l[i:j], l[j:]])
    return partitionings


def generate_graphs():
    all_permutations = list(map(list, permutations([0, 1, 2, 3, 4])))
    perm_partitionings = []
    for perm in all_permutations:
        partitionings = all_partitionings(perm)
        perm_partitionings += partitionings
    perm_partitionings = list(
        map(lambda p: list(map(set, p)), perm_partitionings))
    c_dags = []
    three_var_graphs = [
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
        np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]),
        np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
    ]
    for p in perm_partitionings:
        c_dags_ord = []
        for g in three_var_graphs:
            c_dags_ord.append((p, g))
        c_dags.append(c_dags_ord)
    return c_dags


def compute_nll(data, theta, Cov, C, G):
    C = clustering_to_matrix(C, 3)
    return -ClusterLinearGaussianNetwork(5).logpmf(data, theta, Cov, C, G)


def print_nll_graphs(pairs):
    for i, pair in enumerate(pairs):
        print(f'{i}.')
        print(pair[0])
        print(pair[1][0])
        print(pair[1][1])
        print()


def flatten(ll):
    l = []
    for l_ in ll:
        l += l_
    return l


def main():
    data, (graph, theta) = generate_data_continuous_5(1000)
    all_c_dags = generate_graphs()
    nlls = []
    ord_nlls = []
    Cov = np.eye(5) * 0.1
    for ord in tqdm(all_c_dags):
        ord_nll = 0
        for (C, E_C) in ord:
            nll = compute_nll(data, theta, Cov, C, E_C)
            ord_nll += nll
            nlls.append(nll)
        ord_nlls.append(ord_nll)
    nll_graph_pairs = sorted(
        list(zip(nlls, flatten(all_c_dags))), key=lambda x: x[0])
    nll_ord_pairs = sorted(
        list(zip(ord_nlls, all_c_dags)), key=lambda x: x[0])
    print_nll_graphs(nll_graph_pairs)
    print('===========')
    # idx = np.argmin(list(map(lambda x: x[0], nll_graph_pairs)))
    print(nll_graph_pairs[0][0])
    print(nll_graph_pairs[0][1][0])
    print(nll_graph_pairs[0][1][1])

    print(nll_graph_pairs[1][0])
    print(nll_graph_pairs[1][1][0])
    print(nll_graph_pairs[1][1][1])

    print(nll_graph_pairs[-2][0])
    print(nll_graph_pairs[-2][1][0])
    print(nll_graph_pairs[-2][1][1])

    print(nll_graph_pairs[-1][0])
    print(nll_graph_pairs[-1][1][0])
    print(nll_graph_pairs[-1][1][1])

    min_nll = nll_graph_pairs[0][0]
    for (nll, cdag) in nll_graph_pairs:
        if nll > min_nll:
            print(nll)
            print(cdag[0])
            print(cdag[1])
            break

    print('=================')
    print(nll_ord_pairs[0][0])
    print(nll_ord_pairs[0][0][1][0])
    print(nll_ord_pairs[0][0][1][1])

    print(nll_ord_pairs[1][0])
    print(nll_ord_pairs[1][0][1][0])
    print(nll_ord_pairs[1][0][1][1])

    print(nll_ord_pairs[-2][0])
    print(nll_ord_pairs[-2][0][1][0])
    print(nll_ord_pairs[-2][0][1][1])

    print(nll_ord_pairs[-1][0])
    print(nll_ord_pairs[-1][0][1][0])
    print(nll_ord_pairs[-1][0][1][1])

    min_nll = nll_ord_pairs[0][0]
    for (nll, cdag) in nll_ord_pairs:
        if nll > min_nll:
            print(nll)
            print(cdag[0])
            print(cdag[1])
            break


if __name__ == '__main__':
    main()
