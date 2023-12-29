import numpy as np

from scores.bayesian_cdag_score import BayesianCDAGScore

from data.continuous import generate_data_continuous_5, generate_data_continuous_faithful


def main():
    cdag_ideal = ([{0, 1, 2}, {3}, {4}], np.array([[0, 1, 0],
                                                   [0, 0, 1],
                                                   [0, 0, 0]]))
    cdag = ([{0}, {1}, {2, 3, 4}], np.array([[0, 0, 0],
                                             [0, 0, 1],
                                             [0, 0, 0]]))
    cdag_full = ([{0}, {1}, {2}, {3}, {4}], np.array([[0, 1, 1, 0, 0],
                                                      [0, 0, 1, 1, 0],
                                                      [0, 0, 0, 0, 0],
                                                      [0, 0, 0, 0, 1],
                                                      [0, 0, 0, 0, 0]]))
    cdag_1 = ([{1, 2}, {0}, {3, 4}], np.array([[0, 1, 0],
                                               [0, 0, 0],
                                               [0, 0, 0]]))
    cdag_2 = ([{1, 3}, {2, 4}, {0}], np.array([[0, 1, 1],
                                               [0, 0, 1],
                                               [0, 0, 0]]))

    data, (g_true, theta_true) = generate_data_continuous_faithful(10000)
    theta_true = theta_true*g_true

    Cov_true = data.T@data / data.shape[0]

    # score = BayesianCDAGScore(
    #     data, theta_true, Cov_true, min_clusters=2, mean_clusters=3, max_clusters=4)(cdag_ideal)
    # print(score)
    print(f'cov estimated {np.round(Cov_true, 2)}')
    score1 = BayesianCDAGScore(
        data, theta_true, Cov_true, min_clusters=3, mean_clusters=3, max_clusters=3)(cdag_1)
    score2 = BayesianCDAGScore(
        data, theta_true, Cov_true, min_clusters=3, mean_clusters=3, max_clusters=3)(cdag_2)

    print(score1, score2)


if __name__ == '__main__':
    main()
