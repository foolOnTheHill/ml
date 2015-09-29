import sys
import data_processing
from clustering import fuzzyClustering

def run(E, D, K, T, m, q, epsilon):
    """ Runs the fuzzy clustering 100 times and selects the best solution based on the heterogeneity parameter. """
    optimal = None
    for t in range(100):
        current = fuzzyClustering(E, D, K, T, m, q, epsilon)
        if (optimal == None) or (optimal[2] > current[2]):
            optimal = current
    return optimal

def writeMatrix(filename, M):
    """ Prints the matrix M to a .txt file named `filename`. """
    f = open(filename+'.txt', 'w')
    f.write(filename+':\n')
    m = len(M)
    n = len(M[0])
    for i in range(m):
        for j in range(n):
            e = M[i][j]
            s = str(e) + ' '
            f.write(s)
        f.write('\n')

def computeHardPartition(U):
    """ Given the membership degree matrix U, computes the hard partitioning. """
    H = [] # Will be the hard partitioning
    m = len(U)
    n = len(U[0])
    for i in range(m):
        k = 0
        u_max = U[i][0]
        for j in range(1, n):
            if U[i][j] > u_max:
                k = j
                u_max = U[i][j]
        H.append(k)

def computeRandIndex(E, U, G):
    """ Computes the Adjusted Rand Index. """
    pass

if __name__ == "__main__":
    FILENAME = 'tic-tac-toe.data.txt'
    E = data_processing.readData(FILENAME) # Elements from the data set
    D = data_processing.computeDissimilarityMatrix(E) # Dissimilarity matrix
    K = 2
    T = 150
    m = 2
    q = 2
    epsilon = 10 ** -10
    (U, G, J) = run(E, D, K, T, m, q, epsilon)
    H = computeHardPartition(U)
    computeRandIndex(E, U, G)
    writeMatrix(U, 'fuzzy_partition')
    writeMatrix(G, 'medoids')
    writeMatrix(H, 'hard_partition')