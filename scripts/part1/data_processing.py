
def readData(filename):
    """ Returns a matrix with the data from the file. """
    f = open(filename, 'r')
    lines = f.readlines()
    # Pre-proccesses the data by removing the semicolons
    mp = lambda l : l.split(',')
    data = map(mp, lines)
    drp = lambda l : l[:len(l)-1]
    data = map(drp, data) # Removes the last element of every line (the class parameter)
    return data

def computeDissimilarityMatrix(data):
    """ Computes the Dissimilarity Matrix. The data should be pre-proccessed. """
    # Dissimmilarity function
    delta = lambda (x_ik, x_jk) : 0 if (x_ik == x_jk) else 1
    d = lambda x_i, x_j : sum(map(delta, zip(x_i, x_j)))

    # Number of examples
    n = len(data)

    matrix = [[0 for j in range(n)] for i in range(n)] # Matrix filled with 0s

    for i in range(n):
        for j in range(i, n): # The dissimilarity matrix is symmetrical
          matrix[i][j] = d(data[i], data[j])
          matrix[j][i] = matrix[i][j]

    return matrix

def proccessData(filename):
    data = readData(filename)
    return computeDissimilarityMatrix(data)
