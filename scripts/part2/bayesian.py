from operator import mul
import error
from data_proccessing import getClasses
from fit import fit

def prod(lst):
    """ Returns the product of the elements on the list. """
    return reduce(mul, lst, 1)

def estimateProbabilities(X, C):
    """ Estimates the conditional probabilities using the approximation given at the documentation. """
    p = []
    q = []
    r = []
    n = len(X)
    d = 9 # Dimension of the elements
    c = 2 # Number of classes
    for i in range(d):
        p_i = []
        q_i = []
        r_i = []
        for j in range(c):
            p_tmp = []
            q_tmp = []
            r_tmp = []
            elements = C[j] # Elements on the training set for class w_j
            nj = len(C[j])
            for k in elements:
                # p_i_j
                res = X[k][i]*(X[k][i] + 1)*0.5
                p_tmp.append(res)
                # q_i_j
                res = 1 - X[k][i]*X[k][i]
                q_tmp.append(res)
                # r_i_j
                res = X[k][i]*(X[k][i] - 1)*0.5
                r_tmp.append(res)
            p_i.append(sum(p_tmp) / float(nj))
            q_i.append(sum(q_tmp) / float(nj))
            r_i.append(sum(r_tmp) / float(nj))
        p.append(p_i)
        q.append(q_i)
        r.append(r_i)
    return (p, q, r)

def conditional(x, j, p, q, r):
    """ Computes the conditional probability P(x | w_j). """
    tmp = []
    d = 9
    for i in range(d):
        res = p[i][j] ** (x[i] * (x[i] + 1) * 0.5)
        res *= q[i][j] ** (1 - x[i]*x[i])
        res *= r[i][j] ** (x[i] * (x[i] - 1) * 0.5)
        tmp.append(res)
    return prod(tmp)

def bayes(j, x, p, q, r):
    """ Uses the Bayes Theorem to compute P(w_j | x). """
    tmp = []
    P = [0.653, 0.347]
    c = 2
    for k in range(c):
        res = conditional(x, k, p, q, r) * P[k]
        tmp.append(res)
    num = conditional(x, j, p, q, r) * P[j] * 1.0
    denom = sum(tmp)
    bt = num / denom
    return bt

def estimatePosterior(trainX, trainC, testX):
    """ Uses Maximum Likelihood and Bayes Theorem to estimate P(w_j | x). """
    P = [] # Classification computed by the algorithm
    (p, q, r) = estimateProbabilities(trainX, trainC)
    n = len(testX)
    c = 2 # Number of classes
    for i in range(n):
        P.append([])
        for j in range(c):
            res = bayes(j, testX[i], p, q, r)
            P[i].append(res)
    return P

def classify(trainX, trainY, testX, testY):
    """ Uses the Bayesian Classifier to classify the test data. """
    trainC = getClasses(trainY)
    P = estimatePosterior(trainX, trainC, testX)
    E = fit(testX, P)
    (e_rate, se, interval) = error.confidenceInterval(testY, E)
    return (P, E, e_rate, se, interval)
