import random

def computePrototypes(U, D, m, n, q, k):
    """ Computes the prototype G_k which minimizes the clustering criterion J. (Proposition 2.1) """
    Gk = [] # Will be the prototype
    candidates = [] # Auxiliar list

    for h in range(n):
        tmp = []
        for i in range(n):
            r = (U[i][k] ** m) * D[i][h]
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
        Gk.append(c)
        i += 1

    return Gk

def _selectRandomPrototypes(K, n, q):
    """ Creates K random prototypes with cardinality q. """
    G = [] # Vector of prototypes
    elements = range(n) # Index of the elements of E
    for k in range(K):
        Gk = random.sample(elements, q)
        G.append(Gk)
    return G

def _extendedDissimilarity(D, Gk, e):
    """ Extended dissimilarity function. """
    tmp = []
    for g in Gk:
        tmp.append(D[e][g])
    return sum(tmp)

def fuzzyClustering(E, D, K, T, m, q, epsilon):
    """ Partitioning Fuzzy K-Medoids Clustering Algorithm Based on a Single Dissimilarity Matrix. (Section 2.1)
        - E: Set/List of elements;
        - D: Dissimilarity matrix;
        - K: Number of clusters;
        - T: Maximum number of iterations;
        - m: parameter of fuzziness of membership of elements;
        - q: cardinality of the prototypes;
        - epsilon: threshold for the goal function. """

    n = len(E)
    G = _selectRandomPrototypes(K, n, q) # Initial prototypes
    U = [] # Membership degree Matrix
    J = 0 # Homogeneity / Goal function
    t = 0 # Current Iteration step

    # Initialization
    exp = 1.0 / (m-1)
    for i in range(n):
        U_i = []
        for k in range(K):
            tmp = []
            num = 1.0 * _extendedDissimilarity(D, G[k], i) # Converts to float
            for h in range(K):
                r = (num / _extendedDissimilarity(D, G[h], i)) ** exp
                tmp.append(r)
            u_i_k = sum(tmp) ** -1
            U_i.append(u_i_k)
        U.append(U_i)

    for k in range(K):
        for i in range(n):
            u = U[i][k] ** m
            d = _extendedDissimilarity(D, G[k], i)
            J += u * d

    while t < T:
        t += 1
        # TODO:
            # Step 1
            # Step 2
