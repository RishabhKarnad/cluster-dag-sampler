import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig
from jax import random
import jax.numpy as jnp
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD
import scipy
import pandas


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
        else:
            print('Graph with v-struct')
            self.adjacency_matrix[1, 0] = 1
            self.adjacency_matrix[1, 2] = 1
            self.adjacency_matrix[2, 0] = 1
            self.adjacency_matrix[3, 2] = 1
            self.adjacency_matrix[3, 4] = 1

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
            np.random.normal(size=(n_samples, nvars))

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

        return X, (self.adjacency_matrix, self.theta, self.compute_fullcov())

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

    def load_sachs_data(self):
        data = pandas.read_csv('sachs.data.txt', delimiter='\t').to_numpy()

        true_graph = np.zeros((11, 11))

        true_theta = np.zeros((11, 11))
        true_cov = np.zeros((11, 11))

        return data, (true_graph, true_theta, true_cov)

    def load_housing_data(self):
        data = pandas.read_csv(
            'housing.csv', delim_whitespace=True, header=None).to_numpy()

        true_graph = np.zeros((14, 14))

        true_theta = np.zeros((14, 14))
        true_cov = np.zeros((14, 14))

        return data, (true_graph, true_theta, true_cov)

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
        G = nx.from_numpy_array(self.adjacency_matrix,
                                create_using=nx.MultiDiGraph())
        G_LBN = nx.from_numpy_array(
            self.adjacency_matrix, create_using=LinearGaussianBayesianNetwork)

        factors = []
        for node in G_LBN.nodes:
            print(node)
            parents = list(G.predecessors(node))
            if parents:
                theta_ = [0.0]
                theta_ += self.theta[jnp.array(parents), node].tolist()
                print(theta_)
                factor = LinearGaussianCPD(
                    node, theta_, self.obs_noise, parents)
            else:
                print('here')
                factor = LinearGaussianCPD(
                    node, [0.0], self.obs_noise, parents)
            factors.append(factor)

        G_LBN.add_cpds(*factors)
        joint_dist = G_LBN.to_joint_gaussian()
        Cov_true = joint_dist.covariance
        return Cov_true

    def compute_jointcov(self):
        G = nx.from_numpy_array(self.adjacency_matrix,
                                create_using=nx.MultiDiGraph())
        G_LBN = nx.from_numpy_array(
            self.adjacency_matrix, create_using=LinearGaussianBayesianNetwork)

        factors = []
        for node in G_LBN.nodes:
            print(node)
            parents = list(G.predecessors(node))
            if parents:
                theta_ = [0.0]
                theta_ += self.theta[jnp.array(parents), node].tolist()
                print(theta_)
                factor = LinearGaussianCPD(
                    node, theta_, self.Cov[node, node], parents)
            else:
                print('here')
                factor = LinearGaussianCPD(
                    node, [0.0], self.Cov[node, node], parents)
            factors.append(factor)

        G_LBN.add_cpds(*factors)
        joint_dist = G_LBN.to_joint_gaussian()
        Cov_true = joint_dist.covariance
        return Cov_true

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
