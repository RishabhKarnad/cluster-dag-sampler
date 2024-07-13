import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig
from jax import random
import jax.numpy as jnp
import jax.scipy.stats as stats
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD
import scipy.sparse as sparse
import scipy
from functools import reduce
import pandas as pd

from data.group_faithful import GroupFaithfulDAG
from utils.c_dag import clustering_to_matrix

from rng import random_state


class DataGen:
    def __init__(self, obs_noise):
        self.obs_noise = obs_noise
        self.adjacency_matrix = None
        self.theta = None

    def generate_scm_data(self, n_samples=100):
        G = np.array([[0.,  0.,  0., 1., 1.,  0.,  0.],
                      [0.,  0.,  0., 1.,  1.,  0.,  0.],
                      [0.,  0.,  0.,  1., 1.,  0.,  0.],
                      [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [0.,  0.,  0., 1.,  1.,  0.,  0.],
                      [0.,  0.,  0., 1.,  1.,  0.,  0.]])

        theta = np.array([[0.,  0.,  0., -7., 12.,  0.,  0.],
                          [0.,  0.,  0., 10.,  2.,  0.,  0.],
                          [0.,  0.,  0.,  9., -4.,  0.,  0.],
                          [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [0.,  0.,  0., 26.,  4.,  0.,  0.],
                          [0.,  0.,  0., -15.,  2.,  0.,  0.]])

        mu = np.zeros(7)
        sigma = self.obs_noise*np.eye(7)

        subkey = random_state.get_key()
        Z = random.multivariate_normal(
            subkey, mu, sigma, shape=(n_samples,))

        W = theta * G

        X = Z @ np.linalg.inv(np.eye(7) - W)

        return X, (G, theta, sigma, [{0}, {1}, {2}, {3}, {4}, {5}, {6}], G)

    def generate_group_scm_data(self, n_samples=100, *,
                                vstruct=True, confounded=True, zero_centered=True):
        clus = [{0, 1, 2}, {3, 4}, {5, 6}]
        C = clustering_to_matrix(clus)

        if vstruct:
            G_C = np.array([[0, 1, 0],
                            [0, 0, 0],
                            [0, 1, 0]])
        else:
            G_C = np.array([[0, 1, 1],
                            [0, 0, 0],
                            [0, 0, 0]])

        ks = list(map(int, C.sum(axis=0)))
        n_vars = C.shape[0]

        if vstruct:
            theta = np.array([[0.,  0.,  0., -7., 12.,  0.,  0.],
                              [0.,  0.,  0., 10.,  2.,  0.,  0.],
                              [0.,  0.,  0.,  9., -4.,  0.,  0.],
                              [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                              [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                              [0.,  0.,  0., 26.,  4.,  0.,  0.],
                              [0.,  0.,  0., -15.,  2.,  0.,  0.]])
            # theta = np.array([[0.,  0.,  0., -7., -7.,   0.,  0.],
            #                   [0.,  0.,  0., 10.,  10.,   0.,  0.],
            #                   [0.,  0.,  0.,  9.,   9.,   0.,  0.],
            #                   [0.,  0.,  0.,  0.,   0.,   0.,  0.],
            #                   [0.,  0.,  0.,  0.,   0.,   0.,  0.],
            #                   [0.,  0.,  0., 26.,   26.,   0.,  0.],
            #                   [0.,  0.,  0., -15.,  -15.,  0.,  0.]])
        else:
            theta = np.array([[0.,  0.,  0.,  7., 12., 26.,  4.],
                              [0.,  0.,  0., 10.,  2., 15.,  2.],
                              [0.,  0.,  0.,  9.,  4., 11., -6.],
                              [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                              [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                              [0.,  0.,  0.,  0.,  0.,  0.,  0.],
                              [0.,  0.,  0.,  0.,  0.,  0.,  0.]])

        if zero_centered:
            mu_1 = np.zeros(ks[0])
            mu_2 = np.zeros(ks[1])
            mu_3 = np.zeros(ks[2])
        else:
            mu_1 = np.array([1, 2, -1])
            mu_2 = np.array([-2, 3])
            mu_3 = np.array([3.2, 1.5])

        if confounded:
            sigma_1 = 0.1*np.array([[1, 0.99, 0.99],
                                    [0.99, 1, 0.99],
                                    [0.99, 0.99, 1]])
            sigma_2 = 0.1*np.array([[1, 0.99],
                                    [0.99, 1]])
            sigma_3 = 0.1*np.array([[1, 0.99],
                                    [0.99, 1]])
        else:
            sigma_1 = np.array([[1, 0.5, 0.2],
                                [0.5, 1, 0.5],
                                [0.2, 0.5, 1]])
            sigma_2 = np.array([[1, 0.3],
                                [0.3, 1]])
            sigma_3 = np.array([[1, 0.45],
                                [0.45, 1]])

        subkey = random_state.get_key()
        Z_1 = random.multivariate_normal(
            subkey, mu_1, sigma_1, shape=(n_samples,))

        # subkey = random_state.get_key()
        # f1 = random.exponential(subkey, shape=(n_samples,))
        # subkey = random_state.get_key()
        # z1 = random.exponential(subkey, shape=(n_samples,)) + f1
        # subkey = random_state.get_key()
        # z2 = random.exponential(subkey, shape=(n_samples,)) + f1
        # subkey = random_state.get_key()
        # z3 = random.exponential(subkey, shape=(n_samples,)) + f1
        # Z_1 = jnp.vstack([z1, z2, z3]).T

        subkey = random_state.get_key()
        Z_2 = random.multivariate_normal(
            subkey, mu_2, sigma_2, shape=(n_samples,))

        # subkey = random_state.get_key()
        # f2 = random.exponential(subkey, shape=(n_samples,))
        # subkey = random_state.get_key()
        # z4 = random.exponential(subkey, shape=(n_samples,)) + f2
        # subkey = random_state.get_key()
        # z5 = random.exponential(subkey, shape=(n_samples,)) + f2
        # Z_2 = jnp.vstack([z4, z5]).T

        subkey = random_state.get_key()
        Z_3 = random.multivariate_normal(
            subkey, mu_3, sigma_3, shape=(n_samples,))

        # subkey = random_state.get_key()
        # f3 = random.exponential(subkey, shape=(n_samples,))
        # subkey = random_state.get_key()
        # z6 = random.exponential(subkey, shape=(n_samples,)) + f3
        # subkey = random_state.get_key()
        # z7 = random.exponential(subkey, shape=(n_samples,)) + f3
        # Z_3 = jnp.vstack([z6, z7]).T

        Z = np.hstack([Z_1, Z_2, Z_3])

        G_expand = C@G_C@C.T
        W = theta * G_expand

        X = Z @ np.linalg.inv(np.eye(n_vars) - W)
        Cov_true = np.zeros((7, 7))
        Cov_true[0:3, 0:3] = sigma_1
        Cov_true[3:5, 3:5] = sigma_2
        Cov_true[5:7, 5:7] = sigma_3
        return X, (G_expand, W, Cov_true, clus, G_C)

    def generate_group_scm_data_small_dag(self, n_samples=100, *,
                                          zero_centered=True):
        clus = [{0, 1}, {2}]
        C = clustering_to_matrix(clus)

        G_C = np.array([[0, 1],
                        [0, 0]])

        ks = list(map(int, C.sum(axis=0)))
        n_vars = C.shape[0]

        theta = np.array([[0, 0, 2],
                          [0, 0, 3],
                          [0, 0, 0]])

        if zero_centered:
            mu_1 = np.zeros(ks[0])
            mu_2 = np.zeros(ks[1])
        else:
            mu_1 = np.array([11, 23])
            mu_2 = np.array([31])

        sigma_1 = self.obs_noise*np.array([[1, 0.99], [0.99, 1]])
        sigma_2 = self.obs_noise*np.array([[1]])

        subkey = random_state.get_key()
        Z_1 = random.multivariate_normal(
            subkey, mu_1, sigma_1, shape=(n_samples,))

        subkey = random_state.get_key()
        Z_2 = random.multivariate_normal(
            subkey, mu_2, sigma_2, shape=(n_samples,))

        Z = np.hstack([Z_1, Z_2])

        G_expand = C@G_C@C.T
        W = theta * G_expand

        X = Z @ np.linalg.inv(np.eye(n_vars) - W)
        Cov_true = np.zeros((3, 3))
        Cov_true[0:2, 0:2] = sigma_1
        Cov_true[2:3, 2:3] = sigma_2
        return X, (G_expand, W, Cov_true, clus, G_C)

    def generate_group_scm_data_small_dag_4vars(self, n_samples=100, *,
                                                zero_centered=True):
        clus = [{0, 1}, {2, 3}]
        C = clustering_to_matrix(clus)

        G_C = np.array([[0, 1],
                        [0, 0]])

        ks = list(map(int, C.sum(axis=0)))
        n_vars = C.shape[0]

        theta = np.array([[0., 0., 2., 5.],
                          [0., 0., 3., 4.],
                          [0., 0., 0., 0.],
                          [0., 0., 0., 0.]])

        if zero_centered:
            mu_1 = np.zeros(ks[0])
            mu_2 = np.zeros(ks[1])
        else:
            mu_1 = np.array([11., 23.])
            mu_2 = np.array([31., 41.])

        sigma_1 = 0.1*np.array([[1, 0.99], [0.99, 1]])
        sigma_2 = 0.1*np.array([[1, 0.99], [0.99, 1]])

        subkey = random_state.get_key()
        Z_1 = random.multivariate_normal(
            subkey, mu_1, sigma_1, shape=(n_samples,))

        subkey = random_state.get_key()
        Z_2 = random.multivariate_normal(
            subkey, mu_2, sigma_2, shape=(n_samples,))

        Z = np.hstack([Z_1, Z_2])

        G_expand = C@G_C@C.T
        W = theta * G_expand

        X = Z @ np.linalg.inv(np.eye(n_vars) - W)
        Cov_true = np.zeros((n_vars, n_vars))
        Cov_true[0:2, 0:2] = sigma_1
        Cov_true[2:4, 2:4] = sigma_2
        return X, (G_expand, W, Cov_true, clus, G_C)

    def get_sachs_data(self,
                       center: bool = True,
                       print_labels: bool = False,
                       normalize=True,
                       n_data=853,
                       ):
        data = pd.read_csv('./data/sachs_observational.csv')
        data = data.drop(columns=['PIP3', 'PKC'])
        data_out = data.values
        if center:
            if normalize:
                data_out = (data_out - np.mean(data_out, axis=0)) / \
                    np.std(data_out, axis=0)
            else:
                data_out = data_out - np.mean(data_out, axis=0)

        # self.key, subkey = random.split(self.key)
        # idxs = random.choice(
        #     self.key, data_out.shape[0], shape=(n_data,), replace=False)
        # data_out = data_out[idxs]

        # C = [{2, 3, 4}, {0, 1, 5, 6, 7, 8, 9, 10}]
        # G_C = np.array([[0, 0], [0, 0]])

        C = [{2, 3}, {0, 1, 6, 7, 8}, {4, 5}]
        G_C = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 0, 0]])

        return data_out, (self.get_sachs_ground_truth(), None, None, C, G_C)

    def get_sachs_data_full(self,
                            center: bool = True,
                            print_labels: bool = False,
                            normalize=True,
                            n_data=853,
                            ):
        data = pd.read_csv('./data/sachs_observational.csv')
        data_out = data.values
        if center:
            if normalize:
                data_out = (data_out - np.mean(data_out, axis=0)) / \
                    np.std(data_out, axis=0)
            else:
                data_out = data_out - np.mean(data_out, axis=0)

        # self.key, subkey = random.split(self.key)
        # idxs = random.choice(
        #     self.key, data_out.shape[0], shape=(n_data,), replace=False)
        # data_out = data_out[idxs]

        # C = [{2, 3, 4}, {0, 1, 5, 6, 7, 8, 9, 10}]
        # G_C = np.array([[0, 0], [0, 0]])

        C = [{5, 2, 3}, {6}, {7, 8, 0, 1}, {4}]
        G_C = np.array([[0, 0, 1, 0],
                        [1, 0, 1, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0]])

        return data_out, (self.get_sachs_ground_truth(), None, None, C, G_C)

    def get_sachs_ground_truth(self):
        """Labels are ['praf', 'pmek', 'plcg', 'PIP2', 'PIP3', 'p44/42', 'pakts473',
        'PKA', 'PKC', 'P38', 'pjnk']."""
        W = np.load("./data/sachs_ground_truth.npy")
        return W

    def load_housing_data(self):
        df = pd.read_csv(
            './data/housing.csv', delim_whitespace=True)
        df = df.drop(['CHAS'], axis=1)
        data = df.to_numpy()

        true_graph = np.zeros((14, 14))

        C = [{3}, {8}, {0}, {6, 7}, {9}, {1, 2}, {4, 5}, {10, 11}, {12}]

        G_C = np.zeros((9, 9))
        G_C[0, [2, 3, 5, 6]] = 1
        G_C[1, [3, 5, 8]] = 1
        G_C[2, [3, 4, 6, 7]] = 1
        G_C[3, [4, 5, 7]] = 1
        G_C[4, 8] = 1
        G_C[5, 6] = 1
        G_C[6, [7, 8]] = 1
        G_C[7, 8] = 1

        return data, (true_graph, None, None, C, G_C)

    def viz_graph(adjacency_matrix, graph_lib='igraph'):
        G = nx.from_numpy_array(
            adjacency_matrix, create_using=nx.MultiDiGraph())
        g = ig.Graph.from_networkx(G)
        g.vs['label'] = g.vs['_nx_name']

        if graph_lib == 'igraph':
            fig, ax = plt.subplots()
            ig.plot(g, target=ax)
        elif graph_lib == 'nx':
            nx.draw_circular(G, with_labels=True)

    def compute_fullcov(self):
        n_vars = self.theta.shape[0]
        lambda_ = np.linalg.inv(np.eye(n_vars) - self.theta)
        D = self.obs_noise * np.eye(n_vars)
        Cov = lambda_.T@D@lambda_
        return Cov

    def compute_noisycov(self, C):
        # C: nvars x nclus
        nvars, nclus = C.shape
        Cov_true = self.compute_fullcov()
        G_cov = C@C.T
        G_anticov = np.ones((nvars, nvars))-G_cov
        Cov_mask_true = G_cov*Cov_true
        Cov_noise = stats.wishart.rvs(nvars, 100, size=(5, 5))
        Cov_mask_true_noise = Cov_mask_true+G_anticov*Cov_noise
        return Cov_mask_true_noise, Cov_mask_true

    def get_index(self, arr):
        len_ = len(arr)
        idx = np.zeros((len_*len_, 2))
        idx[:, 0] = np.reshape(np.tile(np.c_[arr], len_), len_*len_)
        idx[:, 1] = np.tile(arr, len_)
        idx = idx.astype(np.int64)
        return idx

    def compute_joint2condcov(self, adj, C_true, Cov_joint):
        C_group = [np.where(C_j > 0)[0].tolist() for C_j in C_true.T]
        G = nx.from_numpy_array(adj, create_using=nx.MultiDiGraph())
        Cond_Cov = []
        for node in G.nodes:
            parents = list(G.predecessors(node))
            if parents:
                node_group = C_group[node]
                parent_group = sorted(sum([C_group[i] for i in parents], []))
                arr = np.array(node_group+parent_group)
                len_ = len(arr)
                split_index = len(node_group)
                idx = self.get_index(arr)
                Cov_full = Cov_joint[(idx[:, 0], idx[:, 1])]
                Cov_full = Cov_full.reshape((len_, len_))
                # CovX|Y=CovX-beta*CovY*beta_T
                # beta=CovXY*CovY-1
                beta = Cov_full[:split_index, split_index:]@np.linalg.inv(
                    Cov_full[split_index:, split_index:])
                Cov_ = Cov_full[:split_index, :split_index] - \
                    beta@Cov_full[split_index:, split_index:]@beta.T
            else:
                node_group = C_group[node]
                arr = np.array(node_group)
                len_ = len(arr)
                idx = self.get_index(arr)
                Cov_full = Cov_joint[(idx[:, 0], idx[:, 1])]
                Cov_full = Cov_full.reshape((len_, len_))
                Cov_ = Cov_full

            Cond_Cov.append(Cov_)
            Cond_Cov_mat = reduce(
                lambda a, b: scipy.linalg.block_diag(a, b), Cond_Cov)

        return Cond_Cov_mat
