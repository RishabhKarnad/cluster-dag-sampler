import igraph as ig
import random as pyrandom
import jax.numpy as jnp
from jax import random, jit, vmap
import functools


def graph_to_mat(g):
    """Returns adjacency matrix of ``ig.Graph`` object

    Args:
        g (igraph.Graph): graph

    Returns:
        ndarray:
        adjacency matrix

    """
    return jnp.array(g.get_adjacency().data)


def mat_to_graph(mat):
    """Returns ``ig.Graph`` object for adjacency matrix

    Args:
        mat (ndarray): adjacency matrix

    Returns:
        igraph.Graph:
        graph
    """
    return ig.Graph.Weighted_Adjacency(mat.tolist())


class ScaleFreeDAGDistribution:
    """
    Randomly-oriented scale-free random graph with power-law degree distribution.
    The pmf is defined as

    :math:`p(G) \\propto \\prod_j (1 + \\text{deg}(j))^{-3}`

    where :math:`\\text{deg}(j)` denotes the in-degree of node :math:`j`

    Args:
        n_vars (int): number of variables in DAG
        n_edges_per_node (int): number of edges sampled per variable

    """

    def __init__(self, n_vars, verbose=False, n_edges_per_node=2):
        self.n_vars = n_vars
        self.n_edges_per_node = n_edges_per_node
        self.verbose = verbose

    def sample_G(self, key, return_mat=False):
        """Samples DAG

        Args:
            key (ndarray): rng
            return_mat (bool): if ``True``, returns adjacency matrix of shape ``[n_vars, n_vars]``

        Returns:
            ``iGraph.graph`` / ``jnp.array``:
            DAG
        """

        pyrandom.seed(int(key.sum()))
        perm = random.permutation(key, self.n_vars).tolist()
        g = ig.Graph.Barabasi(
            n=self.n_vars, m=self.n_edges_per_node, directed=True).permute_vertices(perm)

        if return_mat:
            return graph_to_mat(g)
        else:
            return g

    def unnormalized_log_prob_single(self, *, g, j):
        """
        Computes :math:`\\log p(G_j)` up the normalization constant

        Args:
            g (iGraph.graph): graph
            j (int): node index:

        Returns:
            unnormalized log probability of node family of :math:`j`

        """
        parent_edges = g.incident(j, mode='in')
        n_parents = len(parent_edges)
        return -3 * jnp.log(1 + n_parents)

    def unnormalized_log_prob(self, *, g):
        """
        Computes :math:`\\log p(G)` up the normalization constant

        Args:
            g (iGraph.graph): graph

        Returns:
            unnormalized log probability of :math:`G`

        """
        return jnp.array([self.unnormalized_log_prob_single(g=g, j=j) for j in range(self.n_vars)]).sum()

    def unnormalized_log_prob_soft(self, *, soft_g):
        """
        Computes :math:`\\log p(G)` up the normalization constant
        where :math:`G` is the matrix of edge probabilities

        Args:
            soft_g (ndarray): graph adjacency matrix, where entries
                may be probabilities and not necessarily 0 or 1

        Returns:
            unnormalized log probability corresponding to edge probabilities in :math:`G`

        """
        soft_indegree = soft_g.sum(0)
        return jnp.sum(-3 * jnp.log(1 + soft_indegree))
