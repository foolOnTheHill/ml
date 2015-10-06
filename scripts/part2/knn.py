def estimatePosterior(X, Y, W, k):
    """ Uses KNN to estimate P(w_j | x). """
    P = []
    n = len(X)
    c = len(W)
    ids = range(n)
    indexed = zip(ids, X)

    delta = lambda (x_ik, x_jk) : 0 if (x_ik == x_jk) else 1
    d = lambda x_i, x_j : sum(map(delta, zip(x_i, x_j))) # Dissimmilarity function

    for j in range(c):
        P.append([])

    for i in range(n):
        # TODO: since d(-, -) is symmetrical, we can speed up and avoid re-doing some calculations to compute N.
        mp = lambda (k, x) : (k, d(X[i], x))
        ft = lambda (k, x) : not (k == i)
        N = map(mp, filter(ft, indexed)) # Neighbors
        N.sort(lambda (l, a), (m, b) : -1 if (a < b) else 1 if (a > b) else 0)
        N = N[:k] # K Nearest Neighbors

        freq = [0 for i in range(c)]
        for nb in N:
            n = nb[0]
            freq[Y[n]] += 1

        for j in range(c):
            p = freq[j] / float(k)
            P[j].append(p)

    return P
