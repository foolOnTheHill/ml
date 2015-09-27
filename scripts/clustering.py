
def computePrototypes(E, U, m, q, k, d):
    """ Computes the prototype G_k which minimizes the clustering criterion J. (Proposition 2.1)"""

    n = len(E)
    # Will be the prototype
    G = []
    # Auxiliar list
    candidates = []
    for h in range(n):
        tmp = []
        for i in range(n):
            r = (U[i][k] ** m) * d(E[i], E[h])
            tmp.append(r)
        J = sum(tmp)
        c = (h, J)
        candidates.append(c)

    # Sorts the candidates according to the adequacy criterion
    candidates.sort(lambda (i, J_i), (k, J_k) : -1 if (J_i < J_k) else 1 if (J_i > J_k) else 0)

    # Sets the prototype to have the elements such that the criterion is minimum
    i = 0
    while i < q:
        c = candidates[i][0]
        G.append(c)
        i += 1

    return G

def fuzzyClustering(E, K, T, m, q, d, epsilon):
    """ Partitioning Fuzzy K-Medoids Clustering Algorithm Based on a Single Dissimilarity Matrix. (Section 2.1) """
    pass
