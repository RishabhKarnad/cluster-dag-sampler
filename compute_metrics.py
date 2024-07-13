from utils.metrics import nll, expected_nll, mse_theta, expected_mse_theta, rand_index, expected_rand_index, mutual_information_score, expected_mutual_information_score, shd_expanded_graph, expected_shd, shd_expanded_mixed_graph, expected_cpdag_shd
from utils.c_dag import clustering_to_matrix, get_graphs_by_count, matrix_to_clustering


def evaluate_samples(*, C_true, G_true, samples, scores, theta, theta_true, data):
    C_samples = list(
        map(lambda x: clustering_to_matrix(x[0]), samples))

    C_true = clustering_to_matrix(C_true)

    # G_expanded = C_true@G_true@C_true.T

    rand_index_mean, rand_index_stddev = expected_rand_index(C_true, C_samples)
    print(f'Rand-index: {rand_index_mean}+-{rand_index_stddev}')

    mi_mean, mi_stddev = expected_mutual_information_score(C_true, C_samples)
    print(f'Cluster MI: {mi_mean}+-{mi_stddev}')

    shd_mean, shd_stddev = expected_shd((C_true, G_true), samples)
    print(f'E-SHD: {shd_mean}+-{shd_stddev}')

    shd_cpdag_mean, shd_cpdag_stddev = expected_cpdag_shd(
        (C_true, G_true), samples)
    print(f'CPDAG E-SHD: {shd_cpdag_mean}+-{shd_cpdag_stddev}')

    nll_mean, nll_stddev = expected_nll(data, samples, theta)
    print(f'NLL: {nll_mean}+-{nll_stddev}')

    if theta_true is not None:
        mse_theta_mean, mse_theta_stddev = expected_mse_theta(
            samples, theta, theta_true)
        print(f'MSE (Theta): {mse_theta_mean}+-{mse_theta_stddev}')

    # Compute metrics for the CDAG with best score
    print('Best DAG metrics:')
    best_cdag = samples[np.argmax(scores)]

    C_best, G_best = best_cdag
    m, n = data.shape

    C_best_mat = clustering_to_matrix(C_best)

    rand_index_best = rand_index(C_true, C_best_mat)
    print(f'\tRand-index: {rand_index_best}')

    mi_best = mutual_information_score(C_true, C_best_mat)
    print(f'\tCluster MI: {mi_best}')

    shd_best = shd_expanded_graph((C_true, G_true), best_cdag)
    print(f'\tSHD: {shd_best}')

    shd_cpdag_best = shd_expanded_mixed_graph((C_true, G_true), best_cdag)
    print(f'\tCPDAG SHD: {shd_cpdag_best}')

    nll_best = nll(data, C_best, G_best, theta)
    print(f'\tNLL: {nll_best}')

    if theta_true is not None:
        mse_theta_best = mse_theta(best_cdag, theta, theta_true)
        print(f'\tMSE (Theta): {mse_theta_best}')

    # Compute metrics for most frequently sampled CDAG
    print('Most frequent DAG metrics:')

    graphs, graph_counts = get_graphs_by_count(samples)
    # np.save(f'{filepath}/clusterings_by_count.npy',
    #         list(map(lambda x: clustering_to_matrix(x[0], len(x[0])), graphs)))
    # np.save(f'{filepath}/graphs_by_count.npy',
    #         list(map(lambda x: x[1], graphs)))
    # np.save(f'{filepath}/cdag_counts.npy', graph_counts)

    mode_dag = graphs[0]

    C_mode, G_mode = mode_dag
    m, n = data.shape

    C_mode_mat = clustering_to_matrix(C_mode)

    rand_index_mode = rand_index(C_true, C_mode_mat)
    print(f'\tRand-index: {rand_index_mode}')

    mi_mode = mutual_information_score(C_true, C_mode_mat)
    print(f'\tCluster MI: {mi_mode}')

    shd_mode = shd_expanded_graph((C_true, G_true), mode_dag)
    print(f'\tSHD: {shd_mode}')

    shd_cpdag_mode = shd_expanded_mixed_graph((C_true, G_true), mode_dag)
    print(f'\tCPDAG SHD: {shd_cpdag_mode}')

    nll_mode = nll(data, C_mode, G_mode, theta)
    print(f'\tNLL: {nll_mode}')

    if theta_true is not None:
        mse_theta_mode = mse_theta(mode_dag, theta, theta_true)
        print(f'\tMSE (Theta): {mse_theta_mode}')
