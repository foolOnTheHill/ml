
def sum_rule(probabilities, num_data):
    """ Defines the sum rule to combine classifiers. """
    c = 2
    P = [[0 for j in range(c)] for i in range(num_data)]

    for i in range(num_data):
        for j in range(c):
            for k in range(len(probabilities)):
                P[i][j] += probabilities[k][i][j]

    return P
