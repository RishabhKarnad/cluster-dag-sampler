import numpy as np

from cdt.metrics import SHD
import networkx as nx

from data.group_faithful import GroupFaithfulDAG
from utils.c_dag import clustering_to_matrix

# def shd_agn(B_true, B_est):
#     pred = np.flatnonzero(B_est == 1)
#     cond = np.flatnonzero(B_true)
#     cond_reversed = np.flatnonzero(B_true.T)
#     extra = np.setdiff1d(pred, cond, assume_unique=True)
#     reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
#     pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
#     cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
#     extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
#     missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
#     shd = len(extra_lower) + len(missing_lower) + len(reverse)
#     return shd


# def eshd(g_true, g_samples):
#     shd = 0
#     for g_sample in g_samples:
#         shd += shd(g_true, g_sample)
#     shd /= len(g_samples)
#     return shd


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

    return SHD(essential_graph(g_c_true[1]), essential_graph(g_c[1]))


def expected_cluster_shd(g_true, graphs):
    return np.sum([cluster_shd(g_true, g_c) for g_c in graphs]) / len(graphs)


def shd_expanded_graph(cdag, theta, true_dag):
    C, G_C = cdag
    C = clustering_to_matrix(C, k=len(C))
    theta = np.where(theta > 0.5, 1, 0)
    G_expand = C@G_C@C.T
    G = G_expand*theta

    true_dag = true_dag*G_expand

    return SHD(G, true_dag)


def expected_shd(cdags, theta, true_dag):
    return np.mean([shd_expanded_graph(cdag, theta, true_dag) for cdag in cdags])


def faithfulness_score(samples, g_true):
    group_faithful_model = GroupFaithfulDAG()
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


def test():
    a = np.array([[0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0]])

    clusters = [(0, 1), (2,), (3, 4, 5)]
    edges = np.array([[0, 1, 1],
                      [0, 0, 0],
                      [0, 0, 0]])
    cdag = (clusters, edges)

    print(cluster_shd(a, cdag))


if __name__ == '__main__':
    test()
