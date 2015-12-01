def fit(testX, P):
    """ Uses the posterior probabilities to classify a unseen example. """
    E = []
    n = len(testX)
    c = 2
    for i in range(n):
        y = 0
        tmp = P[i][0]
        for j in range(1, c):
            if P[i][j] > tmp:
                y = j
                tmp = P[i][j]
        E.append(y)
    return E
