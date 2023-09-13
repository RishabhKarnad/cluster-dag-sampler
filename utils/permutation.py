import numpy as np


def make_permutation_matrix(seq):
    """
    Generates a permutation matrix that sorts a given sequence
    """
    n = len(seq)
    P = np.zeros((n, n), dtype=int)
    target = sorted(list(zip(range(len(seq)), seq)), key=lambda x: x[1])
    for i, x in enumerate(target):
        j = x[0]
        P[i, j] = 1
    return P
