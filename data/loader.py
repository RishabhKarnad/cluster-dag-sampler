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
import pandas as pd

from data.group_faithful import GroupFaithfulDAG
from utils.c_dag import clustering_to_matrix

from rng import random_state


class DataGen:
    def __init__(self, obs_noise):
        self.obs_noise = obs_noise
        self.adjacency_matrix = None
        self.theta = None

    def generate_group_scm_data(self, n_samples=100, *,
                                vstruct=True, confounded=True, zero_centered=True):
        clus = [{0, 1, 2}, {3, 4}, {5, 6}]
        C = clustering_to_matrix(clus, k=3)

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
            sigma_1 = np.array([[1, 0.99, 0.99],
                                [0.99, 1, 0.99],
                                [0.99, 0.99, 1]])
            sigma_2 = np.array([[1, 0.99],
                                [0.99, 1]])
            sigma_3 = np.array([[1, 0.99],
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

        subkey = random_state.get_key()
        Z_2 = random.multivariate_normal(
            subkey, mu_2, sigma_2, shape=(n_samples,))

        subkey = random_state.get_key()
        Z_3 = random.multivariate_normal(
            subkey, mu_3, sigma_3, shape=(n_samples,))

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
        C = clustering_to_matrix(clus, k=2)

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

        sigma_1 = 0.1*np.array([[1, 0.99], [0.99, 1]])
        sigma_2 = np.array([[0.1]])

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
        C = clustering_to_matrix(clus, k=2)

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
                       normalize=False,
                       n_data=None,
                       ):
        data = pd.read_csv('./sachs_observational.csv')
        data_out = data.values
        if center:
            if normalize:
                data_out = (data_out - np.mean(data_out, axis=0)) / \
                    np.std(data_out, axis=0)
            else:
                data_out = data_out - np.mean(data_out, axis=0)

        if self.key != None:
            self.key, subkey = random.split(self.key)
            idxs = random.choice(
                self.key, data_out.shape[0], shape=(n_data,), replace=False)
            data_out = data_out[idxs]

        C = [{2, 3, 4}, {0, 1, 5, 6, 7, 8, 9, 10}]
        G_C = np.array([[0, 0], [0, 0]])

        return data_out, (self.get_sachs_ground_truth(), None, None, C, G_C)

    def get_sachs_ground_truth(self):
        """Labels are ['praf', 'pmek', 'plcg', 'PIP2', 'PIP3', 'p44/42', 'pakts473',
        'PKA', 'PKC', 'P38', 'pjnk']."""
        W = np.load("./sachs_ground_truth.npy")
        return W

    def load_housing_data(self):
        df = pd.read_csv(
            'housing.csv', delim_whitespace=True)
        df = df.drop(['CHAS'], axis=1)
        data = df.to_numpy()

        true_graph = np.zeros((14, 14))

        C = [{3}, {8}, {0}, {6, 7}, {9}, {1}, {4, 5}, {10, 11}, {12}]

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
