import numpy as np

import networkx as nx

from data.group_faithful import GroupFaithfulDAG
from models.cluster_linear_gaussian_network import ClusterLinearGaussianNetwork
from utils.c_dag import clustering_to_matrix


def compute_metrics(B_est, B_true):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive

        Taken from https://github.com/xunzheng/notears
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        # if not is_dag(B_est):
        #    raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    shd_wc = shd + len(pred_und)
    prc = float(len(true_pos)) / \
        max(float(len(true_pos)+len(reverse) + len(false_pos)), 1.)
    rec = tpr
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'prc': prc, 'rec': rec, 'shd': shd, 'shd_wc': shd_wc, 'nnz': pred_size}


def is_dsep(graph, x, y, z):
    # Check if x and y are d-separated given z in the graph
    return not any([path for path in nx.all_simple_paths(graph, source=x, target=y) if is_blocked(graph, path, z)])


def is_blocked(graph, path, z):
    # Check if the path is blocked by a set of nodes z in the graph
    for i in range(len(path) - 2):
        if (path[i + 1], path[i + 2]) not in graph.edges() and path[i + 1] not in z:
            return True
        if path[i + 1] in z:
            return True
    return False


def essential_graph(adjacency_matrix):
    graph = nx.DiGraph(adjacency_matrix)
    essential_graph = graph.copy()

    nodes = list(graph.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if graph.has_edge(nodes[i], nodes[j]):
                essential_graph.remove_edge(nodes[i], nodes[j])
                if not is_dsep(graph, nodes[i], nodes[j], essential_graph.predecessors(nodes[i])):
                    essential_graph.add_edge(nodes[i], nodes[j])
                else:
                    essential_graph.add_edge(nodes[j], nodes[i])

    return essential_graph


def cluster_shd(g_true, g_c):
    def get_cdag(clusters, adj):
        edges = np.zeros((len(clusters), len(clusters)))

        for i, C_i in enumerate(clusters):
            for j, C_j in enumerate(clusters):
                if i != j:
                    for p, X_p in enumerate(C_i):
                        for q, X_q in enumerate(C_j):
                            edges[i, j] += adj[X_p, X_q]

        return (clusters, edges)

    g_c_true = get_cdag(clusters=g_c[0], adj=g_true)

    return compute_metrics(essential_graph(g_c_true[1]), essential_graph(g_c[1]))['shd']


def expected_cluster_shd(g_true, graphs):
    shds = [cluster_shd(g_true, g_c) for g_c in graphs]
    return np.mean(shds), np.std(shds)


def shd_expanded_graph(cdag, theta, true_dag):
    C, G_C = cdag
    C = clustering_to_matrix(C, k=len(C))
    # theta = np.where(theta > 0.5, 1, 0)
    G_expand = C@G_C@C.T
    # G = G_expand*theta
    G = G_expand

    true_dag = true_dag*G_expand

    return compute_metrics(G, np.array(true_dag))['shd']


def metrics_expanded_graph(cdag, theta, true_dag):
    C, G_C = cdag
    C = clustering_to_matrix(C, k=len(C))
    # theta = np.where(theta > 0.5, 1, 0)
    G_expand = C@G_C@C.T
    # G = G_expand*theta
    G = G_expand

    true_dag = true_dag*G_expand

    vcn_stats = compute_metrics(G, true_dag)

    return vcn_stats['shd'], vcn_stats['prc'], vcn_stats['rec']


def expected_shd(cdags, theta, true_dag):
    shds = [shd_expanded_graph(cdag, theta, true_dag) for cdag in cdags]
    return np.mean(shds), np.std(shds)


def expected_metrics(cdags, theta, true_dag):
    metrics = [metrics_expanded_graph(cdag, theta, true_dag) for cdag in cdags]
    shds = list(map(lambda x: x[0], metrics))
    prcs = list(map(lambda x: x[1], metrics))
    recs = list(map(lambda x: x[2], metrics))
    return np.mean(shds), np.std(shds), np.mean(prcs), np.mean(recs)


def compute_nlls(data, samples, theta):
    n, n = theta.shape
    clgn = ClusterLinearGaussianNetwork(n)
    nlls = [-clgn.logpmf(data, theta, clustering_to_matrix(C, len(C)), G)
            for (C, G) in samples]
    return np.mean(nlls), np.std(nlls)


def compute_mse_theta(G_C, theta, theta_true):
    C, G = G_C
    C = clustering_to_matrix(C, len(C))
    G_expand = C@G@C.T
    theta_masked = G_expand*theta
    return np.mean((theta_masked - theta_true)**2)


def compute_mse_theta_all(samples, theta, theta_true):
    mses = [compute_mse_theta(sample, theta, theta_true) for sample in samples]
    return np.mean(mses), np.std(mses)


def faithfulness_score(samples, g_true, *, key):
    group_faithful_model = GroupFaithfulDAG(key)
    count = 0
    for groups, group_dag in samples:
        group_names = list(map(lambda x: {x}, np.arange(group_dag.shape[0])))
        H_indeps = group_faithful_model.get_group_dag_indeps(
            group_dag, group_names)
        isFaithful, _ = group_faithful_model.has_same_indepencies(
            H_indeps, groups, g_true)
        if isFaithful:
            count += 1

    return count / len(samples)
