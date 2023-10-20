import numpy as np
from itertools import combinations


class UpperTriangular:
    def __init__(self, n):
        self.n = n
        self.mask = np.zeros((self.n, self.n))
        for i in range(n):
            for j in range(i+1, n):
                self.mask[i, j] = 1

    def sample(self):
        return self.mask * np.random.randint(0, 2, (self.n, self.n))

    def sample_n(self, n_samples=1):
        return np.array([self.sample() for _ in range(n_samples)])

    def get_graph(self, g_ord):
        """
        Enumerates all upper triangular matrices from numbers 0 to 2**k - 1
        where k = (n**2 - n) / 2
        """
        # TODO: Optimize / rethink
        n = self.n
        k = int((n**2 - n) / 2)

        g_ord_bin = bin(g_ord).split('b')[1]
        g_ord_bin_padded = g_ord_bin.zfill(k)
        adj_flattened = [int(c) for c in list(g_ord_bin_padded)]
        g = np.zeros((self.n, self.n))

        k = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                g[i, j] = adj_flattened[k]
                k += 1

        return g

    def all_graphs(self):
        n = self.n
        k = (n**2 - n) / 2
        n_graphs = int(2 ** k)
        return [self.get_graph(i) for i in range(n_graphs)]

    def _check_k_parents(self, k, graph):
        n_parents = graph.sum(axis=0)
        return (n_parents <= k).all()

    # def generate_columns(self, i, max_parents):
    #     n = 10

    #     all_columns = []

    #     max_parents = min(i, max_parents)

    #     for k in range(max_parents+1):
    #         # In the kth iteration, generate all ith columns with exactly k 1's

    #         all_columns_k_ones = np.array([
    #             # in the ith column, there can be at most i 1's
    #             [1 if j in comb else 0 for j in range(i)]
    #             for comb in combinations(np.arange(i), k)
    #         ])
    #         all_columns.append(all_columns_k_ones)

    #     all_columns = np.vstack(all_columns)
    #     all_columns = np.hstack([all_columns, np.zeros(
    #         (all_columns.shape[0], n-all_columns.shape[1]))])
    #     return list(all_columns)

    def all_k_parent_graphs(self, k):
        all_graphs = self.all_graphs()
        k_parent_graphs = list(
            filter(lambda g: self._check_k_parents(k, g), all_graphs))
        return k_parent_graphs

        # n = self.n

        # # ith element is an array containing all possible ith columns
        # # of upper triangular adjacency matrix with upto k 1's
        # columns = [self.generate_columns(i, k) for i in range(n)]


def test():
    UT = UpperTriangular(5)
    print(UT.mask)
    ut = UT.sample_n(10)
    print(ut)
    graphs = UT.all_graphs()
    # print(graphs)
    print(len(graphs))
    graphs = UT.all_k_parent_graphs(2)
    # print(graphs)
    print(len(graphs))


if __name__ == '__main__':
    test()
