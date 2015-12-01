def estimatePosterior(trainX, trainY, testX, k):
    """ Uses KNN to estimate P(w_j | x). """
    P = []

    n = len(testX)
    m = len(trainX)
    c = 2 # Number of classes
    ids = range(m)

    delta = lambda (x_ik, x_jk) : 0 if (x_ik == x_jk) else 1
    d = lambda x_i, x_j : sum(map(delta, zip(x_i, x_j))) # Dissimmilarity function

    ND = [[0 for j in range(m)] for i in range(n)] # Matrix filled with 0s

    for i in range(n):
        for j in range(m):
            ND[i][j] = d(testX[i], trainX[j])

    for i in range(n):
        P.append([])

        N = zip(ids, ND[i]) # Neighbors
        N.sort(lambda (l, a), (m, b) : -1 if (a < b) else 1 if (a > b) else 0) # Sorts by distance
        N = N[:k] # K Nearest Neighbors

        freq = [0 for i in range(c)]
        for nb in N:
            n = nb[0]
            freq[Y[n]] += 1

        for j in range(c):
            p = freq[j] / float(k)
            P[i].append(p)

    return P
