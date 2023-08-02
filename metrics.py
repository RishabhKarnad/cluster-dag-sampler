import numpy as np

from cdt.metrics import SHD


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

    return SHD(g_c_true[1], g_c[1])


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
