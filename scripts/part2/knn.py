def estimatePosterior(X, Y, W, k):
    """ Uses KNN to estimate P(w_j | x). """
    P = []
    n = len(X)
    c = len(W)
    ids = range(n)

    delta = lambda (x_ik, x_jk) : 0 if (x_ik == x_jk) else 1
    d = lambda x_i, x_j : sum(map(delta, zip(x_i, x_j))) # Dissimmilarity function

    for j in range(c):
        P.append([])

    ND = [[0 for j in range(n)] for i in range(n)] # Matrix filled with 0s

    for i in range(n):
        for j in range(i+1, n):
            ND[i][j] = d(X[i], X[j])
            ND[j][i] = ND[i][j]

    for i in range(n):
        ft = lambda (k, x) : not (k == i) # Drops the element with index eq. to i
        N = filter(ft, zip(ids, ND[i])) # Neighbors
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
