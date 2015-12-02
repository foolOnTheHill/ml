import data_proccessing
import bayesian
import knn
import sum_rule
from math import sqrt

def prepareSets(H, testIndex):
    trainX = []
    trainY = []
    testX = []
    testY = []

    for i in range(len(H)):
        (X, Y) = H[i]
        if i == testIndex:
            testX = testX + X
            testY = testY + Y
        else:
            trainX = trainX + X
            trainY = trainY + Y

    return (trainX, trainY, testX, testY)

def writeResults(f, e_rate, se, interval):
    f.write("(a) Error rate: %f\n" % e_rate)
    f.write("(b) SE: %f\n" % se)
    f.write("(c) Confidence Interval: [%f, %f]\n\n" % (interval[0], interval[1]))

def run():
    H = data_proccessing.loadData() # Holdouts
    K = [30, 20, 10, 5, 1]

    resultsBay = open('part2-results-bayesian.txt', 'w')
    resultsKn = open('part2-results-knn.txt', 'w')
    resultsSum = open('part2-results-sum.txt', 'w')

    for i in range(10):
        resultsBay.write("Round %d\n\n" % (i+1))
        resultsKn.write("Round %d\n\n" % (i+1))
        resultsSum.write("Round %d\n\n" % (i+1))

        (trainX, trainY, testX, testY) = prepareSets(H, i)
        (P_bay, E_bay, e_rate_bay, se_bay, interval_bay) = bayesian.classify(trainX, trainY, testX, testY)

        resultsBay.write("- Bayesian\n")
        writeResults(resultsBay, e_rate_bay, se_bay, interval_bay)

        for k in K:
            (P_kn, E_kn, e_rate_kn, se_kn, interval_kn) = knn.classify(trainX, trainY, testX, testY, k)
            (P_sum, E_sum, e_rate_sum, se_sum, interval_sum) = sum_rule.classify([P_bay, P_kn], testX, testY)
            #
            resultsKn.write("- KNN (n = %d)\n" % k)
            writeResults(resultsKn, e_rate_kn, se_kn, interval_kn)
            #
            resultsSum.write("- Sum (n = %d)\n" % k)
            writeResults(resultsSum, e_rate_sum, se_sum, interval_sum)

    resultsBay.close()
    resultsKn.close()
    resultsSum.close()

if __name__ == '__main__':
    run()
