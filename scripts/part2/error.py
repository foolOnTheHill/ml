from math import sqrt

def estimateError(testY, predictedY):
    """ Estimates the error on the classification. """
    num_errors = 0
    n = len(testY)
    for i in range(n):
        if testY[i] != predictedY[i]:
            num_errors += 1
    e_rate = num_errors / float(n)
    tmp = e_rate*(1 - e_rate)
    se = sqrt(tmp / float(n))
    return (e_rate, se)

def confidenceInterval(testY, predictedY):
    """ Computes a confidence interval for the error using alpha=0.05. """
    (e_rate, se) = estimateError(testY, predictedY)
    tmp  = 1.96*se
    interval = [e_rate - tmp, e_rate + tmp]
    return (e_rate, se, interval)
