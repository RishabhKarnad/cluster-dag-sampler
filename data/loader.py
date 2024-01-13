import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig
from jax import random
import jax.numpy as jnp
import jax.scipy.stats as stats
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD
import scipy
import pandas
import cdt.data

from data.group_faithful import GroupFaithfulDAG
from utils.c_dag import clustering_to_matrix


class DataGen:
    def __init__(self, key, obs_noise):
        self.key = key
        self.obs_noise = obs_noise
        self.adjacency_matrix = None
        self.theta = None

    def generate_data_continuous_5(self, n_samples=100, vstruct=True, faithful=True):
        self.adjacency_matrix = np.zeros((5, 5))
        self.vstruct = vstruct
        self.faithful = faithful

        if not self.vstruct:
            print('Graph with no v-struct')
            self.adjacency_matrix[0, 1] = 1
            self.adjacency_matrix[0, 2] = 1
            self.adjacency_matrix[1, 2] = 1
            self.adjacency_matrix[1, 3] = 1
            self.adjacency_matrix[3, 4] = 1

            grouping = [{0, 1, 2}, {3}, {4}]
            group_dag = np.array([[0, 1, 0],
                                  [0, 0, 1],
                                  [0, 0, 0]])
        else:
            print('Graph with v-struct')
            self.adjacency_matrix[1, 0] = 1
            self.adjacency_matrix[1, 2] = 1
            self.adjacency_matrix[2, 0] = 1
            self.adjacency_matrix[3, 2] = 1
            self.adjacency_matrix[3, 4] = 1

            grouping = [{1, 3}, {2}, {0, 4}]
            group_dag = np.array([[0, 1, 1],
                                  [0, 0, 1],
                                  [0, 0, 0]])

        G = nx.from_numpy_array(self.adjacency_matrix,
                                create_using=nx.MultiDiGraph())
        g = ig.Graph.from_networkx(G)
        g.vs['label'] = g.vs['_nx_name']

        nvars = len(g.vs)

        self.key, subkey = random.split(self.key)
        self.theta = jnp.zeros((nvars, nvars))

        if self.vstruct:
            if self.faithful:
                print('Faithful with v-structure')
                self.theta = self.theta.at[1, 0].set(5)
                self.theta = self.theta.at[2, 0].set(6)
                self.theta = self.theta.at[0, 3].set(0)
                self.theta = self.theta.at[2, 3].set(0)
                self.theta = self.theta.at[1, 2].set(7)
                self.theta = self.theta.at[3, 2].set(2)
                self.theta = self.theta.at[3, 4].set(3)
            else:
                print('Unfaithful with v-structure')
                self.theta = self.theta.at[1, 0].set(1)
                self.theta = self.theta.at[2, 0].set(1)
                self.theta = self.theta.at[0, 3].set(0)
                self.theta = self.theta.at[2, 3].set(0)
                self.theta = self.theta.at[1, 2].set(-1)
                self.theta = self.theta.at[3, 2].set(2)
                self.theta = self.theta.at[3, 4].set(3)
        else:
            if self.faithful:
                print('Faithful no v-structure')
                self.theta = self.theta.at[0, 1].set(5)
                self.theta = self.theta.at[0, 2].set(6)
                self.theta = self.theta.at[0, 3].set(0)
                self.theta = self.theta.at[2, 3].set(0)
                self.theta = self.theta.at[1, 2].set(7)
                self.theta = self.theta.at[1, 3].set(2)
                self.theta = self.theta.at[3, 4].set(3)
            else:
                print('Unfaithful without v-structure')
                self.theta = self.theta.at[0, 1].set(1)
                self.theta = self.theta.at[0, 2].set(1)
                self.theta = self.theta.at[0, 3].set(0)
                self.theta = self.theta.at[2, 3].set(0)
                self.theta = self.theta.at[1, 2].set(-1)
                self.theta = self.theta.at[1, 3].set(2)
                self.theta = self.theta.at[3, 4].set(3)

        noise = np.sqrt(self.obs_noise) * \
            random.normal(subkey, shape=(n_samples, nvars))

        toporder = g.topological_sorting()

        X = jnp.zeros((n_samples, nvars))

        for j in toporder:
            parent_edges = g.incident(j, mode='in')
            parents = list(g.es[e].source for e in parent_edges)

            if parents:
                mean = X[:, jnp.array(
                    parents)] @ self.theta[jnp.array(parents), j]
                X = X.at[:, j].set(mean + noise[:, j])

            else:
                X = X.at[:, j].set(noise[:, j])

        return X, (self.adjacency_matrix, self.theta, self.compute_fullcov(), grouping, group_dag)

    def generate_data_continuous_3(self, theta, Cov, n_samples=100):
        self.adjacency_matrix = np.zeros((3, 3))
        self.adjacency_matrix[0, 1] = 1
        self.adjacency_matrix[0, 2] = 1
        self.adjacency_matrix[1, 2] = 1

        G = nx.from_numpy_array(self.adjacency_matrix,
                                create_using=nx.MultiDiGraph())
        g = ig.Graph.from_networkx(G)
        g.vs['label'] = g.vs['_nx_name']

        nvars = len(g.vs)

        self.key, subkey = random.split(self.key)
        self.theta = theta
        self.Cov = Cov

        n1 = np.sqrt(Cov[0, 0])*np.random.normal(size=(n_samples, 1))
        n2 = np.sqrt(Cov[1, 1])*np.random.normal(size=(n_samples, 1))
        n3 = np.sqrt(Cov[2, 2])*np.random.normal(size=(n_samples, 1))
        noise = np.concatenate((n1, n2, n3), axis=1)

        toporder = g.topological_sorting()

        X = jnp.zeros((n_samples, nvars))

        for j in toporder:
            parent_edges = g.incident(j, mode='in')
            parents = list(g.es[e].source for e in parent_edges)

            if parents:
                mean = X[:, jnp.array(
                    parents)] @ self.theta[jnp.array(parents), j]
                X = X.at[:, j].set(mean + noise[:, j])

            else:
                X = X.at[:, j].set(noise[:, j])

        return X, self.theta

    def generate_group_scm_data(self, n_samples=100):

        C = clustering_to_matrix([{0, 1, 2}, {3, 4}, {5, 6}], k=3)

        G_C = np.array([[0, 1, 0],
                        [0, 0, 0],
                        [0, 1, 0]])

        ks = list(map(int, C.sum(axis=0)))
        n_vars = C.shape[0]

        self.key, subkey = random.split(self.key)
        theta = random.normal(subkey, shape=(n_vars, n_vars))

        self.key, subkey = random.split(self.key)
        mu_1 = np.zeros(ks[0])
        sigma_1 = np.array([[1, 0.5, 0.2],
                            [0.5, 1, 0.5],
                            [0.2, 0.5, 1]])
        Z_1 = random.multivariate_normal(
            subkey, mu_1, sigma_1, shape=(n_samples,))

        self.key, subkey = random.split(self.key)
        mu_2 = np.zeros(ks[1])
        sigma_2 = np.array([[1, 0.3],
                            [0.3, 1]])
        Z_2 = random.multivariate_normal(
            subkey, mu_2, sigma_2, shape=(n_samples,))

        self.key, subkey = random.split(self.key)
        mu_3 = np.zeros(ks[2])
        sigma_3 = np.array([[1, 0.45],
                            [0.45, 1]])
        Z_3 = random.multivariate_normal(
            subkey, mu_3, sigma_3, shape=(n_samples,))

        Z = np.hstack([Z_1, Z_2, Z_3])

        G_expand = C@G_C@C.T
        W = theta * G_expand

        X = Z @ np.linalg.inv(np.eye(n_vars) - W)

        return X

    def load_sachs_data(self):
        data, true_graph = cdt.data.load_dataset('sachs')

        true_graph = nx.adjacency_matrix(true_graph).todense()

        return data, (true_graph,)

    def load_housing_data(self):
        data = pandas.read_csv(
            'housing.csv', delim_whitespace=True).to_numpy()

        true_graph = np.zeros((14, 14))

        true_theta = np.zeros((14, 14))
        true_cov = np.zeros((14, 14))

        return data, (true_graph, true_theta, true_cov)

    def generate_data_group_faithful(self, *, n_samples=100, N, k, p, var_params=10):
        grouping, group_dag, dag = GroupFaithfulDAG(
            self.key).gen_group_faithful_dag(N, k, p)

        G = nx.from_numpy_array(dag,
                                create_using=nx.MultiDiGraph())
        g = ig.Graph.from_networkx(G)
        g.vs['label'] = g.vs['_nx_name']

        toporder = g.topological_sorting()

        self.key, subkey = random.split(self.key)

        noise = np.sqrt(self.obs_noise) * \
            random.normal(subkey, shape=(n_samples, N))

        self.key, subkey = random.split(self.key)

        self.theta = random.normal(subkey, shape=(N, N)) * var_params

        X = jnp.zeros((n_samples, N))

        for j in toporder:
            parent_edges = g.incident(j, mode='in')
            parents = list(g.es[e].source for e in parent_edges)

            if parents:
                mean = X[:, jnp.array(
                    parents)] @ self.theta[jnp.array(parents), j]
                X = X.at[:, j].set(mean + noise[:, j])

            else:
                X = X.at[:, j].set(noise[:, j])

        return X, (dag, self.theta, self.compute_fullcov(), grouping, group_dag)

    def obtain_truecluster(self):
        """Define cluster"""
        C = np.zeros((5, 3))
        C[0:3, 0] = 1
        C[3, 1] = 1
        C[4, 2] = 1
        return C

    def obtain_truegraph(self):
        Gc = np.zeros((3, 3))
        Gc[0, 1] = 1
        Gc[1, 2] = 1
        return Gc

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
        # G = nx.from_numpy_array(self.adjacency_matrix,
        #                         create_using=nx.MultiDiGraph())
        # G_LBN = nx.from_numpy_array(
        #     self.adjacency_matrix, create_using=LinearGaussianBayesianNetwork)

        # factors = []
        # for node in G_LBN.nodes:
        #     print(node)
        #     parents = list(G.predecessors(node))
        #     if parents:
        #         theta_ = [0.0]
        #         theta_ += self.theta[jnp.array(parents), node].tolist()
        #         print(theta_)
        #         factor = LinearGaussianCPD(
        #             node, theta_, self.obs_noise, parents)
        #     else:
        #         print('here')
        #         factor = LinearGaussianCPD(
        #             node, [0.0], self.obs_noise, parents)
        #     factors.append(factor)

        # G_LBN.add_cpds(*factors)
        # joint_dist = G_LBN.to_joint_gaussian()
        # Cov_true = joint_dist.covariance
        # return Cov_true

        n_vars = self.theta.shape[0]
        lambda_ = np.linalg.inv(np.eye(n_vars) - self.theta)
        D = self.obs_noise * np.eye(n_vars)
        Cov = lambda_.T@D@lambda_
        return Cov

    # def compute_jointcov(self):
    #     G = nx.from_numpy_array(self.adjacency_matrix,
    #                             create_using=nx.MultiDiGraph())
    #     G_LBN = nx.from_numpy_array(
    #         self.adjacency_matrix, create_using=LinearGaussianBayesianNetwork)

    #     factors = []
    #     for node in G_LBN.nodes:
    #         print(node)
    #         parents = list(G.predecessors(node))
    #         if parents:
    #             theta_ = [0.0]
    #             theta_ += self.theta[jnp.array(parents), node].tolist()
    #             print(theta_)
    #             factor = LinearGaussianCPD(
    #                 node, theta_, self.Cov[node, node], parents)
    #         else:
    #             print('here')
    #             factor = LinearGaussianCPD(
    #                 node, [0.0], self.Cov[node, node], parents)
    #         factors.append(factor)

    #     G_LBN.add_cpds(*factors)
    #     joint_dist = G_LBN.to_joint_gaussian()
    #     Cov_true = joint_dist.covariance
    #     return Cov_true

    def compute_noisycov(self, C):
        # C:nvarsXnclus
        nvars, nclus = C.shape
        Cov_true = self.compute_fullcov()
        G_cov = C@C.T
        G_anticov = np.ones((nvars, nvars))-G_cov
        Cov_mask_true = G_cov*Cov_true
        Cov_noise = scipy.stats.wishart.rvs(nvars, 100, size=(5, 5))
        Cov_mask_true_noise = Cov_mask_true+G_anticov*Cov_noise
        return Cov_mask_true_noise, Cov_mask_true
