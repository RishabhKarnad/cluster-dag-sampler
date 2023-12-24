import networkx as nx
import numpy as np
import random
import itertools

from generate_graphs import generate_graph

random.seed(42)


class GroupFaithfulDAG:
    def get_cluster_matrix(self, W, N, n):
        C = np.zeros((N, n))
        for i, W_i in enumerate(W):
            for w_i in list(W_i):
                C[w_i, i] = 1
        return C

    def acyclicity(self, mat):
        n_vars = mat.shape[0]
        alpha = 1.0 / n_vars
        M = np.eye(n_vars) + alpha * mat

        M_mult = np.linalg.matrix_power(M, n_vars)
        h = np.trace(M_mult) - n_vars
        return h

    def conditional_independence_DAG(self, dag, groups, setA, setB, setC):
        if len(setC) == 0:
            cond_set = set()
        else:
            cond_set = set()
            for i in range(len(setC)):
                cond_set = cond_set.union(groups[list(setC)[i]])
        for i in [groups[a] for a in list(setA)]:
            for j in [groups[b] for b in list(setB)]:
                # print(f'{i} {j} {cond_set}')
                if not nx.d_separated(dag, i, j, cond_set):
                    return False
        return True

    def get_group_dag_indeps(self, dag, W):
        group_dag_nx = nx.from_numpy_array(dag, create_using=nx.MultiDiGraph())

        n = len(W)

        indeps = []

        ord = 0
        done = False
        G = np.ones((n, n)) * (1 - np.eye(n))
        G_nx = nx.from_numpy_array(G, create_using=nx.MultiDiGraph())
        G_ = G.copy()

        while not done:
            done = True
            indices = np.argwhere(G_ > 0)
            X, Y = indices[:, 0].flatten(), indices[:, 1].flatten()
            for i in range(X.size):
                x = X[i]
                y = Y[i]

                nbrs = set(nx.neighbors(G_nx, y)).difference({x})

                if len(nbrs) >= ord:
                    done = False
                    SS = [set(comb)
                          for comb in itertools.combinations(nbrs, ord)]
                    for S in SS:
                        if self.conditional_independence_DAG(group_dag_nx, W, {x}, {y}, S):
                            indeps.append((x, y, S))
                            G[x, y] = 0
                            G[y, x] = 0
            ord = ord + 1
        return indeps

    def has_same_indepencies(self, H_indeps, W, G):
        indeps = self.get_group_dag_indeps(G, W)
        return len(indeps) == len(H_indeps), len(indeps)

    def gen_group_faithful_dag(self, N=10, k=3, p=0.2):
        n = int(np.ceil(N / k))

        group_dag = generate_graph(n, p)

        W = list(map(set, np.array_split(np.random.permutation(N), n)))

        indices = [np.random.randint(len(W_i)) for W_i in W]
        W_i = [list(W_i)[indices[i]] for i, W_i in enumerate(W)]

        C = self.get_cluster_matrix(W, N, n)

        G = np.zeros((N, N))
        for edge in np.argwhere(group_dag > 0):
            G[W_i[edge[0]], W_i[edge[1]]] = 1

        group_names = list(map(lambda x: {x}, np.arange(group_dag.shape[0])))
        H_indeps = self.get_group_dag_indeps(group_dag, group_names)

        for _ in range(1000):
            [u, v] = random.sample(range(N), 2)

            G_prime = G.copy()

            if G_prime[u, v] == 1:
                G_prime[u, v] = 0
            else:
                G_prime[u, v] = 1

            if self.acyclicity(G_prime) > 0:
                continue

            flag, length = self.has_same_indepencies(H_indeps, W, G_prime)
            if not flag:
                continue

            G = G_prime

        return W, group_dag, G
