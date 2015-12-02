from fit import fit
import error

def sum_rule(probabilities, num_data):
    """ Defines the sum rule to combine classifiers. """
    c = 2
    P = [[0 for j in range(c)] for i in range(num_data)]

    for i in range(num_data):
        for j in range(c):
            for k in range(len(probabilities)):
                P[i][j] += probabilities[k][i][j]

    return P

def classify(probabilities, testX, testY):
    """ Uses the sum rule to combine two predictions and then classify the data. """
    n = len(testX)
    P = sum_rule(probabilities, n)
    E = fit(testX, P)
    (e_rate, se, interval) = error.confidenceInterval(testY, E)
    return (P, E, e_rate, se, interval)
