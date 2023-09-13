import numpy as np

if __name__ != '__main__':
    from utils.permutation import make_permutation_matrix


def stringify_cdag(G_C):
    C, E_C = G_C
    C_indexed = list(zip(range(len(C)), C))
    C_sorted = sorted(C_indexed, key=lambda s: min(s[1]))
    C_stringified = []
    for elem in C_sorted:
        c = map(str, sorted(elem[1]))
        C_stringified.append(','.join(c))
    C_stringified = '-'.join(C_stringified)
    P = make_permutation_matrix(list(map(lambda x: x[0], C_sorted)))
    E_C_sorted = P.T @ E_C @ P
    E_stringified = ','.join(map(str, list(E_C_sorted.flatten())))
    cdag_stringified = C_stringified + ':' + E_stringified
    return cdag_stringified


def unstringify_cdag(cdag_string):
    C_stringified, E_C_stringified = cdag_string.split(':')
    C = C_stringified.split('-')
    C = [set(map(int, c_i.split(','))) for c_i in C]
    n = len(C)
    E_C = list(map(int, E_C_stringified.split(',')))
    E_C = np.array(E_C).reshape(n, n)
    return C, E_C


def test():
    cdag = ([{1, 4, 3}, {5, 2}, {0, 14}], np.array(
        [[0,  0, 0], [0, 0, 1], [0, 0, 0]]))
    stringified = stringify_cdag(cdag)
    unstringified = unstringify_cdag(stringified)


if __name__ == '__main__':
    from permutation import make_permutation_matrix

    test()
